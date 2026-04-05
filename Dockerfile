FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

# Install CPU-only torch FIRST to prevent CUDA deps from being pulled
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Prevent poetry from reinstalling GPU torch — pin torch to the installed version
RUN pip freeze | grep -i "^torch==" > /tmp/constraints.txt

# Install project deps with torch constrained to CPU version
RUN poetry config virtualenvs.create false \
    && PIP_CONSTRAINT=/tmp/constraints.txt poetry install --no-interaction --no-ansi --no-root \
    && rm -rf /root/.cache /tmp/*

# HuggingFace model cache — pre-download during build so pods start offline
ENV HF_HOME=/app/.cache/huggingface
ARG SENTIMENT_MODEL=ElKulako/cryptobert
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='${SENTIMENT_MODEL}', device='cpu')"

COPY src/ src/
COPY config/ config/
COPY README.md paper_trading.py run_continuous.py ./

RUN poetry install --only-root --no-interaction --no-ansi \
    && rm -rf /root/.cache

ENV TRADING_INTERVAL=300
ENV TRADING_SYMBOL=BTC/USDT
ENV TRADING_MODE=langgraph

CMD ["python", "run_continuous.py"]
