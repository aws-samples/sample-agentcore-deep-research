#!/bin/bash
# Build and push a training container to ECR.
#
# Usage:
#   ./training/build_and_push.sh rl      # agentic RL (verl, FSDP engine)
#   ./training/build_and_push.sh sft     # trajectory SFT (TRL)
#
# The stage is required rather than defaulted. The two images share a repository
# but nothing else -- different base, different dependencies, different entry
# point -- and a mistaken default overwrites the tag the other pipeline resolves,
# which only shows up as a confusing runtime failure in the wrong stage.

set -e

REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="deep-research-rl-training"
IMAGE_TAG=${1:-}

case "${IMAGE_TAG}" in
    rl)  DOCKERFILE="training/Dockerfile" ;;
    sft) DOCKERFILE="training/Dockerfile.sft" ;;
    *)
        echo "ERROR: specify the stage to build: rl or sft" >&2
        echo "  ./training/build_and_push.sh rl    # agentic RL (verl)" >&2
        echo "  ./training/build_and_push.sh sft   # trajectory SFT (TRL)" >&2
        exit 1
        ;;
esac

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
    echo "      --s3-bucket <RLBucketName>"
    echo ""
    echo "  Starting policy defaults to Qwen3.5-9B from the Hub. To start from your"
    echo "  own SFT run instead, add ONE of:"
    echo "      --sft-job-name <completed-sft-job>   (resolves its artifact for you)"
    echo "      --model-id <hf-id or s3://.../model.tar.gz>"
fi
echo ""
echo "Note: the image must be in the same region as the training job."
