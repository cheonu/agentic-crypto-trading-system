FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

# Force CPU-only torch and block all NVIDIA/CUDA packages
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN poetry config virtualenvs.create false \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && poetry install --no-interaction --no-ansi --no-root 2>&1 | grep -v "nvidia\|cuda\|triton" || true \
    && pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 triton cuda-bindings cuda-pathfinder 2>/dev/null || true \
    && rm -rf /root/.cache

COPY src/ src/
COPY config/ config/
COPY README.md paper_trading.py run_continuous.py ./

RUN poetry install --no-interaction --no-ansi \
    && rm -rf /root/.cache

ENV TRADING_INTERVAL=300
ENV TRADING_SYMBOL=BTC/USDT
ENV TRADING_MODE=langgraph

CMD ["python", "run_continuous.py"]
