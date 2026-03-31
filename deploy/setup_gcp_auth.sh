#!/bin/bash
set -e

PROJECT_ID=$(gcloud config get-value project)
echo "Using project: ${PROJECT_ID}"

echo ""
echo "Step 1: Creating service account..."
gcloud iam service-accounts create terraform-admin \
    --display-name="Terraform Admin" \
    --project="${PROJECT_ID}"

echo ""
echo "Step 2: Granting owner role..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:terraform-admin@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/owner"

echo ""
echo "Step 3: Creating key file..."
gcloud iam service-accounts keys create ~/terraform-key.json \
    --iam-account="terraform-admin@${PROJECT_ID}.iam.gserviceaccount.com"

echo ""
echo "Step 4: Setting environment variable..."
export GOOGLE_APPLICATION_CREDENTIALS=~/terraform-key.json

echo ""
echo "Done! Run this before using terraform:"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=~/terraform-key.json"
echo ""
echo "Then: terraform plan"
