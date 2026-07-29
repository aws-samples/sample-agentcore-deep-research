#!/bin/bash
# Build and push the RL training container to ECR.
# Usage: ./training/build_and_push.sh

set -e

REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="deep-research-rl-training"
IMAGE_TAG=${1:-latest}
FULL_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Use finch if available, otherwise docker
CONTAINER_CLI=$(command -v finch 2>/dev/null || command -v docker 2>/dev/null)
if [ -z "$CONTAINER_CLI" ]; then
    echo "Error: neither finch nor docker found"
    exit 1
fi
echo "Using: ${CONTAINER_CLI}"

echo "Building training container..."
echo "  Region:  ${REGION}"
echo "  Account: ${ACCOUNT}"
echo "  Image:   ${FULL_URI}"
echo ""

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

# Login to ECR
aws ecr get-login-password --region "${REGION}" | \
    ${CONTAINER_CLI} login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Build (force amd64 — SageMaker GPU instances are x86_64)
${CONTAINER_CLI} build --platform linux/amd64 -t "${REPO_NAME}:${IMAGE_TAG}" -f training/Dockerfile .

# Tag and push
${CONTAINER_CLI} tag "${REPO_NAME}:${IMAGE_TAG}" "${FULL_URI}"
${CONTAINER_CLI} push "${FULL_URI}"

echo ""
echo "✓ Pushed: ${FULL_URI}"
echo ""
echo "Use with:"
echo "  uv run test-scripts/rl_train.py train --image-uri ${FULL_URI} --data <training.jsonl> ..."
