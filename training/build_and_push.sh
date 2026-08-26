#!/bin/bash
# Build and push training containers to ECR.
# Usage:
#   ./training/build_and_push.sh         # RL container (default)
#   ./training/build_and_push.sh sft     # SFT container

set -e

REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="deep-research-rl-training"
IMAGE_TAG=${1:-latest}

# Select Dockerfile based on tag
if [ "${IMAGE_TAG}" = "sft" ]; then
    DOCKERFILE="training/Dockerfile.sft"
else
    DOCKERFILE="training/Dockerfile"
fi

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
${CONTAINER_CLI} build --platform linux/amd64 -t "${REPO_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" .

# Tag and push
${CONTAINER_CLI} tag "${REPO_NAME}:${IMAGE_TAG}" "${FULL_URI}"
${CONTAINER_CLI} push "${FULL_URI}"

echo ""
echo "✓ Pushed: ${FULL_URI}"
echo ""
echo "Use with:"
if [ "${IMAGE_TAG}" = "sft" ]; then
    echo "  uv run test-scripts/sft_train.py --image-uri ${FULL_URI} \\"
    echo "      --data <traces.jsonl> --s3-bucket <bucket> --role-arn <role> \\"
    echo "      --hf-model-id Qwen/Qwen3.5-9B --lora-rank 32 --epochs 2 \\"
    echo "      --max-seq-length 32768 --instance-type ml.g6e.12xlarge"
else
    echo "  uv run test-scripts/rl_train.py --image-uri ${FULL_URI} \\"
    echo "      --data <rl-prompts.jsonl> --agent-arn <RLAgentRuntimeArn> \\"
    echo "      --s3-bucket <RLBucketName> --sft-job-name <sft-job> --model-type qwen3.5-9B"
fi
echo ""
echo "Note: the image must be in the same region as the training job."
