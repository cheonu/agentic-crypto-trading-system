FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

# Install CPU-only torch FIRST — this is the only torch we want
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Tell poetry to skip torch entirely — it's already installed via pip above
# This prevents poetry from downloading the 2GB+ CUDA version
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root \
    --without finetune \
    && pip install --no-cache-dir --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu \
    && pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
       nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
       nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
       nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 triton 2>/dev/null || true \
    && rm -rf /usr/local/lib/python3.13/site-packages/nvidia /root/.cache /tmp/*

# Pre-download HF model
ENV HF_HOME=/app/.cache/huggingface
ARG SENTIMENT_MODEL=ElKulako/cryptobert
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${SENTIMENT_MODEL}', device='cpu')"

COPY src/ src/
COPY config/ config/
COPY README.md paper_trading.py run_continuous.py ./

RUN poetry install --only-root --no-interaction --no-ansi \
    && rm -rf /root/.cache

# Stage 2: Clean runtime image
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /app/src /app/src
COPY --from=builder /app/config /app/config
COPY --from=builder /app/README.md /app/paper_trading.py /app/run_continuous.py /app/
COPY --from=builder /app/pyproject.toml /app/

ENV HF_HOME=/app/.cache/huggingface
ENV TRADING_INTERVAL=300
ENV TRADING_SYMBOL=BTC/CAD
ENV TRADING_MODE=langgraph

CMD ["python", "run_continuous.py"]
