#!/bin/bash
# Create/update Kubernetes secrets for crypto-trader
# THIS FILE IS GITIGNORED — contains real API keys
#
# Usage: ./deploy/create_secrets.sh

set -e

kubectl create secret generic api-keys \
  --namespace crypto-trader \
  --from-literal=openai-api-key=sk-proj-REPLACE_ME \
  --from-literal=anthropic-api-key=sk-ant-REPLACE_ME \
  --from-literal=google-ai-key=REPLACE_ME \
  --from-literal=binance-testnet-key=REPLACE_ME \
  --from-literal=binance-testnet-secret=REPLACE_ME \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Secrets created/updated in crypto-trader namespace"
