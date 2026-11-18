#!/usr/bin/env bash
# Build the locked serving environment and start a single local GPU endpoint.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git -C "$script_dir" rev-parse --show-toplevel)
if (( $# > 1 )); then
    echo "Usage: bash deploy/kevsft/up.sh [checkpoint-directory]" >&2
    exit 2
fi
checkpoint=$(cd -- "${1:-$root/private/runs/demo/best}" && pwd -P)
for artifact in head.pt adapter_config.json adapter_model.safetensors; do
    if [[ ! -s "$checkpoint/$artifact" ]]; then
        echo "Incomplete LoRA checkpoint: missing $artifact" >&2
        exit 2
    fi
done
port=${KEVSFT_PORT:-8008}
name=${KEVSFT_CONTAINER:-kevsft-local}
cache_volume=${KEVSFT_CACHE_VOLUME:-kevsft-hf-cache}
gpu=${KEVSFT_GPU:-0}
wait_seconds=${KEVSFT_WAIT_SECONDS:-600}
[[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo "Invalid KEVSFT_PORT" >&2; exit 2; }
[[ $wait_seconds =~ ^[1-9][0-9]{0,4}$ ]] || { echo "Invalid KEVSFT_WAIT_SECONDS" >&2; exit 2; }
[[ $gpu =~ ^[0-9]+$ ]] || { echo "KEVSFT_GPU must be one device index" >&2; exit 2; }
[[ $name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid container name" >&2; exit 2; }
[[ $cache_volume =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "Invalid model cache volume name" >&2; exit 2; }
[[ $checkpoint != *,* ]] || { echo "Use a checkpoint path without commas" >&2; exit 2; }
if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists. Inspect or remove it, or choose another name and port." >&2
    exit 1
fi

# Send only committed package sources and metadata; no checkout or data directory
# enters the Docker build context. Weights are mounted at runtime.
context=$(mktemp -d)
trap 'rm -rf -- "$context"' EXIT
git -C "$root" archive HEAD -- kev/'*.py' pyproject.toml uv.lock README.md LICENSE | tar -x -C "$context"
cp -- "$script_dir/Dockerfile" "$script_dir/server.py" "$context/"
image="kevsft-local:$(git -C "$root" rev-parse --short=12 HEAD)"
build_args=()
if [[ -n ${KEVSFT_PYTHON_IMAGE:-} ]]; then
    build_args+=(--build-arg "PYTHON_IMAGE=$KEVSFT_PYTHON_IMAGE")
fi
if [[ -n ${KEVSFT_RUNTIME_IMAGE:-} ]]; then
    build_args+=(--build-arg "RUNTIME_IMAGE=$KEVSFT_RUNTIME_IMAGE")
fi
docker build "${build_args[@]}" --tag "$image" "$context"
cid=$(docker run --detach --name "$name" --gpus "device=$gpu" \
    --publish "127.0.0.1:$port:8008" \
    --mount "type=bind,src=$checkpoint,dst=/checkpoint,readonly" \
    --mount "type=volume,src=$cache_volume,dst=/cache/huggingface" \
    --env HF_HUB_OFFLINE --env KEV_API_KEY \
    "$image")

deadline=$((SECONDS + wait_seconds))
while (( SECONDS < deadline )); do
    if [[ $(docker inspect --format '{{.State.Running}}' "$cid") != true ]]; then
        echo "Model server exited. Inspect it with: docker logs $name" >&2
        exit 1
    fi
    if curl --noproxy '*' --silent --fail --max-time 2 "http://127.0.0.1:$port/ready" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
        echo "Ready at http://127.0.0.1:$port (container: $name)"
        exit 0
    fi
    sleep 2
done
echo "Readiness wait expired after ${wait_seconds}s. Inspect downloads and startup with: docker logs $name" >&2
exit 1
