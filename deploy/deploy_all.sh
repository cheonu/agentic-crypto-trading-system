#!/bin/bash
# Deploy multiple trading configurations to compare performance.
#
# This deploys 3 instances:
#   1. LangGraph (deterministic, no LLM cost)
#   2. CrewAI + OpenAI (GPT-4o)
#   3. CrewAI + Anthropic (Claude)
#
# Usage:
#   ./deploy/deploy_all.sh <project-id>

set -e

PROJECT_ID=${1:?"Usage: $0 <project-id>"}

echo "Deploying 3 trading configurations to compare..."
echo ""

# 1. LangGraph — deterministic baseline (no LLM cost)
./deploy/deploy_gcp.sh "${PROJECT_ID}" langgraph none

# 2. CrewAI + OpenAI
./deploy/deploy_gcp.sh "${PROJECT_ID}" crewai openai

# 3. CrewAI + Anthropic
./deploy/deploy_gcp.sh "${PROJECT_ID}" crewai anthropic

echo ""
echo "=== All 3 instances deployed ==="
echo ""
echo "Monitor logs:"
echo "  gcloud run services logs read crypto-trader-langgraph-none --project ${PROJECT_ID} --region us-central1"
echo "  gcloud run services logs read crypto-trader-crewai-openai --project ${PROJECT_ID} --region us-central1"
echo "  gcloud run services logs read crypto-trader-crewai-anthropic --project ${PROJECT_ID} --region us-central1"
echo ""
echo "Compare results after a few days by downloading the JSONL files from each instance."
