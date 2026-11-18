ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG RUNTIME_IMAGE=locked
FROM ${PYTHON_IMAGE} AS locked

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN python -m pip install --no-cache-dir uv==0.9.7 \
    && uv sync --frozen --extra serve --no-dev --no-install-project --no-cache
ENV PATH="/app/.venv/bin:${PATH}"

FROM ${RUNTIME_IMAGE} AS service
WORKDIR /app
COPY kev ./kev
COPY LICENSE README.md server.py ./
ENV HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 KEV_FUSED=0 KEV_CUDA_GRAPHS=0
HEALTHCHECK NONE
EXPOSE 8008
ENTRYPOINT ["python", "server.py", "--run", "/checkpoint", "--fallback", "/checkpoint", "--device", "cuda", "--host", "0.0.0.0", "--port", "8008"]
