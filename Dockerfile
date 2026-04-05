# Stage 1: Install dependencies (CUDA junk gets downloaded here but stays in this stage)
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

# Install CPU-only torch first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all deps — poetry may pull CUDA torch as transitive dep
RUN pip freeze | grep -i "^torch==" > /tmp/constraints.txt \
    && poetry config virtualenvs.create false \
    && PIP_CONSTRAINT=/tmp/constraints.txt poetry install --no-interaction --no-ansi --no-root \
    && rm -rf /root/.cache /tmp/*

# Force CPU torch again and nuke any NVIDIA packages that snuck in
RUN pip install --no-cache-dir --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu \
    && pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
       nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
       nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
       nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 triton 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.13/site-packages/nvidia /root/.cache

# Pre-download HF model
ENV HF_HOME=/app/.cache/huggingface
ARG SENTIMENT_MODEL=ElKulako/cryptobert
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${SENTIMENT_MODEL}', device='cpu')"

COPY src/ src/
COPY config/ config/
COPY README.md paper_trading.py run_continuous.py ./

RUN poetry install --only-root --no-interaction --no-ansi \
    && rm -rf /root/.cache

# Stage 2: Clean runtime image — only copy what we need, no CUDA bloat
FROM python:3.13-slim

WORKDIR /app

# Copy installed packages from builder (without CUDA layers)
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy HF model cache
COPY --from=builder /app/.cache /app/.cache

# Copy application code
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config
COPY --from=builder /app/README.md /app/paper_trading.py /app/run_continuous.py /app/
COPY --from=builder /app/pyproject.toml /app/

ENV HF_HOME=/app/.cache/huggingface
ENV TRADING_INTERVAL=300
ENV TRADING_SYMBOL=BTC/USDT
ENV TRADING_MODE=langgraph

CMD ["python", "run_continuous.py"]
