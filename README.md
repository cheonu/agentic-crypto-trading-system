# Agentic Crypto Trading System

An autonomous crypto trading system powered by multi-agent AI debate, HuggingFace sentiment analysis, and real-time market data. Runs on Kubernetes with live exchange execution via Kraken.

## Architecture

Every 5 minutes, the system runs a 10-step pipeline:

1. Fetch market data (price, volume, candles) from Kraken via ccxt
2. Analyze intraday signals (EMA9/21 crossovers, RSI, VWAP, momentum)
3. Detect market regime (bull/bear/sideways)
4. Fetch crypto news headlines from RSS feeds (CoinDesk, CoinTelegraph, Decrypt)
5. Score headlines with CryptoBERT (HuggingFace NLP model)
6. Check stop-losses and trailing stops on open positions
7. Strategy evaluates all signals and produces BUY/SELL/HOLD
8. Fee filter validates profitability after exchange fees
9. CrewAI agents debate the trade, then execute on Kraken
10. Update position state and log results

## Tech Stack

- **AI/ML:** CrewAI, LangGraph, HuggingFace Transformers (CryptoBERT), LoRA/PEFT fine-tuning
- **Exchange:** Kraken (live), configurable via ccxt (supports any exchange)
- **Infrastructure:** GKE, Helm, Docker, Jenkins CI/CD, Google Cloud Build
- **Language:** Python 3.13
- **Data:** RSS news feeds, Binance/Kraken market data, technical indicators

## Strategy

Confidence-weighted signal evaluation:
- 30% regime detection (bull/bear transitions)
- 50% intraday signals (EMA cross, RSI, VWAP, momentum)
- 20% news sentiment (CryptoBERT score from live headlines)

Risk management:
- 0.8% stop-loss, 2.0% take-profit (2.5:1 reward/risk)
- 1.2% trailing stop to lock in gains
- One position at a time per symbol
- Fee-aware filtering (blocks trades where profit < 1.5x fees)

## Results (Paper Trading)

After 2 weeks of paper trading across 8 pods with different LLM providers:

| Agent | P&L | Trades | Win Rate |
|-------|-----|--------|----------|
| crewai-grok | +$12.91 | 11 | 73% |
| crewai-deepseek | +$10.60 | 12 | 58% |
| crewai-openai | +$10.32 | 10 | 70% |
| crewai-gemini | +$8.57 | 12 | 67% |

## Project Structure

```
├── run_continuous.py              # Main trading loop (5-min cycles)
├── paper_trading.py               # Market data, execution, agent orchestration
├── Dockerfile                     # Multi-stage build (2.7GB, CPU-only)
├── Jenkinsfile                    # CI/CD: Cloud Build + Helm deploy
├── helm/crypto-trader/            # Helm chart for GKE deployment
│   ├── templates/deployment.yaml
│   ├── templates/pvc.yaml
│   └── values.yaml
├── k8s/fine-tune-job.yaml         # K8s Job for model fine-tuning
├── terraform/                     # GKE cluster provisioning
├── src/agentic_crypto_trading_system/
│   ├── agents/                    # CrewAI + LangGraph agent frameworks
│   │   ├── crewai_framework.py
│   │   └── langgraph_framework.py
│   ├── day_trading/
│   │   ├── strategy.py            # Entry/exit signal logic
│   │   ├── position_manager.py    # Position tracking + P&L
│   │   ├── stop_loss_monitor.py   # Stop-loss + trailing stop
│   │   ├── news_provider.py       # RSS fetcher + CryptoBERT sentiment
│   │   ├── fee_filter.py          # Fee-aware trade filtering
│   │   ├── config.py              # Strategy parameters
│   │   └── fine_tune_sentiment.py # HF fine-tuning script
│   ├── debate/                    # Multi-agent debate mechanism
│   ├── exchange/                  # Exchange connector (Kraken/Binance)
│   └── regime/                    # Market regime detection
└── tests/
```

## Setup

```bash
# Install dependencies
poetry install

# Run locally (paper mode)
EXCHANGE=kraken EXCHANGE_API_KEY=xxx EXCHANGE_API_SECRET=xxx \
  python run_continuous.py

# Run live
TRADING_LIVE=true EXCHANGE=kraken EXCHANGE_API_KEY=xxx EXCHANGE_API_SECRET=xxx \
  python run_continuous.py
```

## Deploy to GKE

```bash
# Build and push
docker build --platform linux/amd64 -t $REGISTRY/crypto-trader:v1 .
docker push $REGISTRY/crypto-trader:v1

# Deploy
helm upgrade --install crypto-trader helm/crypto-trader -n crypto-trader \
  --set image.tag=v1
```

## Fine-Tuning

```bash
# Install fine-tuning deps
poetry install --with finetune

# Fine-tune FinBERT on financial sentiment data
python src/agentic_crypto_trading_system/day_trading/fine_tune_sentiment.py \
  --dataset-name "financial_phrasebank" \
  --epochs 3 --batch-size 8 --use-lora --evaluate
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXCHANGE` | Exchange name (ccxt) | `kraken` |
| `EXCHANGE_API_KEY` | Exchange API key | — |
| `EXCHANGE_API_SECRET` | Exchange API secret | — |
| `TRADING_LIVE` | Enable real orders | `false` |
| `TRADING_SYMBOL` | Trading pair | `BTC/USDT` |
| `TRADING_INTERVAL` | Seconds between cycles | `300` |
| `TRADING_MODE` | Agent framework | `langgraph` |
| `SENTIMENT_MODEL_NAME` | HuggingFace model | `ElKulako/cryptobert` |
| `ENSEMBLE_MODEL_NAME` | Optional second model for ensemble scoring | — |
| `PORTFOLIO_VALUE` | Account balance | `100.0` |

## License

MIT
