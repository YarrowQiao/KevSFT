# 用 Docker 部署微调后的 Kev

以下命令在仓库根目录执行。需要 Linux x86-64、Bash、Git、curl、Docker Engine、NVIDIA 驱动和 NVIDIA Container Toolkit。使用与锁定 PyTorch CUDA 运行时兼容的 GPU 和驱动。首次构建需要能下载容器镜像和 Python 依赖，首次加载还要能下载检查点指定的基座和 tokenizer。为依赖、模型缓存和 Docker 镜像预留磁盘空间。

## 启动

完成快速微调并生成 `private/runs/demo/best` 后，运行：

```bash
bash deploy/kevsft/up.sh
```

使用其他权重目录、GPU 或端口时：

```bash
KEVSFT_GPU=1 KEVSFT_PORT=8010 KEVSFT_CONTAINER=kevsft-candidate \
  bash deploy/kevsft/up.sh private/runs/candidate/best
```

脚本在本机 Docker Engine 上构建服务镜像，默认选择第 0 张 GPU，发布到 `127.0.0.1:8008`。启动单一服务进程后等待模型就绪，默认等待 600 秒，首次下载较慢时可设置 `KEVSFT_WAIT_SECONDS`。如果指定名称已经存在，脚本会提示检查现有容器。升级时可以显式指定新容器名和端口，再切换调用地址。两个实例同时运行时应确保显存充足。

默认从官方 `python:3.12-slim-bookworm` 构建，按 `uv.lock` 安装服务依赖。若团队通过镜像缓存分发相同的 Python 基础镜像，可将 `KEVSFT_PYTHON_IMAGE` 设为缓存中的完整镜像名。对于已维护推理运行时的团队，可显式设置 `KEVSFT_RUNTIME_IMAGE` 为经验证的运行时镜像；此时脚本使用镜像中的 Python、CUDA/PyTorch、Transformers、PEFT、FastAPI、Uvicorn 等依赖，版本兼容与更新由镜像维护方负责。建议使用固定版本或镜像摘要，并用下方的实际决策请求检查兼容性。

## 发起一次决策

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8008/ready
curl --noproxy '*' -fsS http://127.0.0.1:8008/v1/models
curl --noproxy '*' -fsS http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @private/runs/demo/best/request.json
```

就绪接口应返回 `{"ready":true}`。模型信息中检查 `run` 为 `/checkpoint` 和 `device` 为 `cuda`。最后一条命令使用训练导出的请求模板。对新输入，使用同一数据配置生成请求，例如：

```bash
python -m kevsft.data --data examples/kevsft --request 'May I renew my loan?' \
  --model kev-latest | curl --noproxy '*' -fsS http://127.0.0.1:8008/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @-
```

从 `answers.route.choice` 读取标签键，从 `answers.route.probabilities` 读取各候选概率。输入格式、分类指令和候选顺序应与训练保持一致。`/ready` 检查模型是否已装入内存；分类质量通过团队自己的留出集验收。

## 私有数据、缓存与访问

Docker 构建上下文由脚本临时组装，只发送当前 Git 提交的 `kev/*.py`、`pyproject.toml`、`uv.lock`、`README.md`、`LICENSE` 和部署所用的两个文件。检查点目录以只读方式挂载到 `/checkpoint`。服务读入检查点的 `head.pt`、`adapter_config.json`、`adapter_model.safetensors` 并按其中记录的基座加载权重。默认部署路径使用公开基座的模型 ID。使用自行训练的检查点时，训练机专用的绝对基座路径必须在容器中同样可解析，否则请使用可解析的公开模型 ID 重新导出。

基座和 tokenizer 缓存在 Docker 卷 `kevsft-hf-cache` 中，独立于训练机的个人模型缓存。可用 `KEVSFT_CACHE_VOLUME` 指定已有的 Hugging Face 缓存卷，卷根目录应含 `hub/`。基座和 tokenizer 对应版本已缓存后，可用 `HF_HUB_OFFLINE=1 bash deploy/kevsft/up.sh` 在模型加载阶段禁用 Hub 下载。镜像构建所需依赖应提前缓存。

接口默认仅发布在本机回环地址。如果设置 `KEV_API_KEY`，脚本按环境变量名传给容器，`/v1/*` 请求需带 Bearer 头。交互式 shell 可如下输入密钥并启动：

```bash
read -rs -p 'Service API key: ' KEV_API_KEY
echo
export KEV_API_KEY
bash deploy/kevsft/up.sh
```

服务密钥和业务样本应保存在团队的密钥管理与私有数据目录中。需要跨机器访问时，由受控网关提供 TLS 和访问控制。公开示例用于验证格式和接口。

## 日常操作

```bash
docker logs --tail 80 kevsft-local
docker stop kevsft-local
docker start kevsft-local
```

启动失败或等待超时后，容器保留以便查看日志。端口占用、驱动不兼容、基座下载失败或显存不足都可能阻止服务就绪。修正配置后，可移除自己创建的旧容器并重新执行脚本：

```bash
docker rm -f kevsft-local
bash deploy/kevsft/up.sh
```

回滚时用同一脚本指定此前验收通过的检查点目录，并按需回到对应的代码版本。移除容器后，宿主机检查点和模型缓存卷继续保留。
