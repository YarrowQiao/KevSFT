# 用自己的业务样本微调 KEV

两家公司都可能把某条路线叫作“活动”，受理范围却不同。KevSFT 用标签表说明每条路线受理什么，再用带标签问句展示实际边界。训练后，服务对每次请求返回一个路由键，业务系统根据这个键分发任务。信息不足或所有路线均不适用时，可以返回 `other`。

## 先跑通公开样例

公开样例描述一家虚构图书馆，共有借阅 `loans`、活动 `events` 和兜底 `other` 三类。33 条样本按来源分为训练 18 条、验证 6 条、测试 9 条。命令均在仓库根目录执行。训练默认使用单张 NVIDIA GPU；CPU 可用 `--device cpu --batch-size 1` 做小规模检查。首次运行需要能下载依赖、公开 Kev 检查点及其指定版本的基座。

```bash
git clone https://github.com/YarrowQiao/KevSFT.git
cd KevSFT
uv sync --frozen --no-dev --extra serve --python 3.12
.venv/bin/python -m kevsft.data --data examples/kevsft
.venv/bin/python -m kevsft.train --data examples/kevsft --output private/runs/demo
```

每次实验使用新的输出目录。要做最短的流程检查，可在训练命令后加 `--epochs 1 --batch-size 1`。训练代码和推理请求生成器共用输入格式。[训练入口](../kevsft/train.py) · [样例与标签](../examples/kevsft)

## 换成自己的领域

把样例复制到默认受 Git 忽略的私有目录，再替换四个文件。标签说明、用户问句和历史均支持中文。

```bash
mkdir -p private
cp -R examples/kevsft private/data
```

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `labels.json` | 路由键到职责说明的有序映射，最后一项为 `other` | 训练、评测和线上推理共用同一份定义 |
| `train.jsonl` | 每行一个带标签请求，覆盖全部路线 | 更新模型参数 |
| `validation.jsonl` | 独立来源的带标签请求 | 比较不同轮次，选择导出的模型 |
| `test.jsonl` | 提前固定的独立来源样本 | 比较原始 KEV 与选中模型 |

例如，下面两个请求都提到了读书会。第一条需要处理书的归还，第二条需要处理活动报名。这样的近似问句能直接表达两个路由的边界。

```jsonl
{"group":"library-example-01","query":"读书会发给我的书，周五能还吗？","label":"loans"}
{"group":"library-example-02","query":"周五的读书会还有位置吗？","label":"events"}
{"group":"library-example-02","history":["周五的读书会还有位置吗？"],"query":"我想再报一个人。","label":"events"}
{"group":"library-example-03","query":"能帮我算一下这道几何题吗？","label":"other"}
```

1. 先定义业务边界。每个标签用一句话描述职责，给容易混淆的两条路线分别准备正例。收集自然问法、追问、换话题、超出服务范围和信息不足的请求。
2. 给每条记录加 `group`。同一会话、同一原始请求的改写或同一模板生成的近似样本共用组，整组进入同一个集合。校验器会发现跨集合的组和标准化后相同的输入；语义近似的改写需要制作者归组。
3. 私有语料在组织内部筛选、去标识、标注。使用合成或教师模型草拟的数据时，让熟悉业务的人复核边界和标签，再将同源改写分组。测试集保留真实语言分布，覆盖长尾类别及易混淆的业务线。
4. 按实际服务输入准备上下文。`history` 只接收按时间正序排列的历史用户发言，`query` 放本轮问题。优先保留能解释指代、补充条件或话题切换的历史。训练与调用阶段使用同样的历史筛选规则。

```bash
.venv/bin/python -m kevsft.data --data private/data
.venv/bin/python -m kevsft.train --data private/data --output private/runs/domain-v1 \
  --epochs 3 --batch-size 2 --lr 2e-5 --seed 42
```

## 训练在做什么

默认从公开的 `jaredpalmer/kev-0.8b` 继续训练。模型对同一请求中的候选路线打分。训练冻结基座，只更新 LoRA 适配器和输出各候选分数的指针头，用人工标注的目标路线计算交叉熵。训练每轮打乱样本和候选次序，同时重映射正确答案的位置，以降低依赖候选固定位置的风险。

每轮结束后，模型在验证集上按标签表的候选顺序打分。优先选择准确率最高的轮次，同分选验证交叉熵较低的轮次，两者相同取较早轮次。最后分别加载原始模型和选中的检查点，在相同的测试集与推理精度下评测。新模型使用温度 1；如果产品按置信度决定是否自动执行，还需独立校准并验证该策略。

`--max-row-tokens` 默认 2048，计入文本、指令及候选的编码长度。超出时会报出长度问题。业务需要长输入时，应先确定部署时可用的显存和上下文，再设置实验预算并单独评测长输入。这里的单选任务支持 2 到 255 个路由键，含 `other`。

## 如何判断微调有没有帮助

训练完成后打开 `private/runs/domain-v1/report.json`。其中 `test.baseline` 与 `test.candidate` 分别是原始模型和候选模型的结果。报告里比率的范围为 0 到 1。

| 字段 | 如何阅读 |
| --- | --- |
| `test_accuracy_delta` | 候选准确率减原始准确率。乘以 100 即百分点变化 |
| `support`、`recall` | 每条路线的测试样本数、正确识别占该类的比例。缺少该类时召回为 null |
| `fallback_recall` | 所有标注为 `other` 的请求中，正确兜底的比例 |
| `fallback_false_positive_rate` | 所有原本有业务路线可走的请求中，被模型送去兜底的比例 |
| `confusion_matrix` | 行是真实路线，列是预测路线，顺序见报告的 `labels` |
| `history`、`selected_epoch` | 各轮训练交叉熵及验证结果，和最终选用的轮次 |

例如，总准确率上升时，仍需查看高代价路线的召回和两条相近路线之间的误分。只有 9 条的样例测试集，每错一道题就会改变约 11.1 个百分点；它适合确认流程。业务验收应事先确定每条关键路线的最低可接受指标与样本覆盖。若据测试错例调整训练数据，下一次验收应使用新留出的独立数据。

### 公开样例的一次实测

使用仓库内未经修改的图书馆样例、公开 Kev-0.8B 检查点、单卡 CUDA 和 bf16，设 `--epochs 2 --batch-size 1 --lr 2e-5 --seed 42`。每轮的训练交叉熵分别为 0.4408、0.00138。两个轮次均答对 6/6 条验证样本，第二轮的验证交叉熵更低，因而入选。[公开适配器](https://huggingface.co/jaredpalmer/kev-0.8b/blob/9a45d25/adapter_model.safetensors) · [公开指针头](https://huggingface.co/jaredpalmer/kev-0.8b/blob/9a45d25/head.pt)

| 独立测试集指标 | 原始 KEV | 微调后 |
| --- | ---: | ---: |
| 总体正确 | 6/9 | 8/9 |
| 借阅召回 | 0/3 | 2/3 |
| 活动召回 | 3/3 | 3/3 |
| 兜底召回 | 3/3 | 3/3 |
| 业务请求误走兜底 | 2/6 | 0/6 |

两条原先误送兜底的借阅请求分类正确，仍有一条借阅请求被分到活动。加载导出的模型后，三种路线各发一条 HTTP 请求，均返回对应路由键。要在新环境复跑，请在快速开始的训练命令末尾加入上述实验参数，并选择一个新输出目录。

## 一键部署和调用

训练通过后，`best/` 包含适配器、指针头、模型元数据、分词器文件及请求模板。部署机需要 Git 检出、Docker、NVIDIA 驱动和 NVIDIA Container Toolkit，并能取得检查点指定的基座。默认绑定回环地址并等待权重加载就绪。

```bash
bash deploy/kevsft/up.sh private/runs/domain-v1/best
curl --noproxy '*' -fsS http://127.0.0.1:8008/ready
.venv/bin/python -m kevsft.data --data private/data \
  --history '周五有一个读书会。' --request '还能报名吗？' |
  curl --noproxy '*' -fsS http://127.0.0.1:8008/v1/systemone \
    -H 'Content-Type: application/json' --data-binary @-
```

从 `answers.route.choice` 读取路由键，从 `answers.route.probabilities` 读取候选分布。训练和服务分开部署时，私下传递已验收的 `best/`，并给调用方同一版 `labels.json`。权重以只读方式挂载，样本和训练报告无需进入服务镜像。端口、GPU、鉴权、离线缓存、升级及回滚详见[部署手册](../deploy/kevsft/README.md)。
