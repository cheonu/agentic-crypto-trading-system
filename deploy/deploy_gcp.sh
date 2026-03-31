#!/bin/bash
# Deploy to GCP Cloud Run
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project set up
#
# Usage:
#   ./deploy/deploy_gcp.sh <project-id> <mode> [llm-provider]
#
# Examples:
#   ./deploy/deploy_gcp.sh my-project langgraph
#   ./deploy/deploy_gcp.sh my-project crewai openai
#   ./deploy/deploy_gcp.sh my-project crewai anthropic
#   ./deploy/deploy_gcp.sh my-project crewai google

set -e

PROJECT_ID=${1:?"Usage: $0 <project-id> <mode> [llm-provider]"}
MODE=${2:?"Usage: $0 <project-id> <mode> [llm-provider]"}
LLM_PROVIDER=${3:-"openai"}
REGION="us-central1"
SERVICE_NAME="crypto-trader-${MODE}-${LLM_PROVIDER}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== Deploying ${SERVICE_NAME} ==="
echo "  Project:  ${PROJECT_ID}"
echo "  Mode:     ${MODE}"
echo "  LLM:      ${LLM_PROVIDER}"
echo "  Region:   ${REGION}"
echo ""

# Build and push Docker image
echo "Building Docker image..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"

# Deploy to Cloud Run (always-on with min instances = 1)
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --platform managed \
    --no-allow-unauthenticated \
    --min-instances 1 \
    --max-instances 1 \
    --memory 1Gi \
    --cpu 1 \
    --timeout 3600 \
    --set-env-vars "TRADING_MODE=${MODE},TRADING_SYMBOL=BTC/USDT,TRADING_INTERVAL=300" \
    --update-secrets "OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,GOOGLE_API_KEY=google-api-key:latest"

echo ""
echo "=== Deployed ${SERVICE_NAME} ==="
echo "View logs: gcloud run services logs read ${SERVICE_NAME} --project ${PROJECT_ID} --region ${REGION}"
