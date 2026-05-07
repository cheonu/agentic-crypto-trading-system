FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry poetry-plugin-export

COPY pyproject.toml poetry.lock ./

# Install CPU-only torch FIRST so poetry doesn't pull the 2GB+ CUDA version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Export poetry deps to requirements.txt, strip torch (already installed), then pip install
RUN poetry config virtualenvs.create false \
    && poetry export --without finetune --without dev -o requirements.txt --without-hashes \
    && sed -i '/^torch[>=<! ]/d' requirements.txt \
    && sed -i '/^nvidia-/d' requirements.txt \
    && sed -i '/^triton/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache /tmp/*

# Verify torch is working
RUN python -c "import torch; print(f'PyTorch {torch.__version__} OK')"

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
