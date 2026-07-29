# Training container for agentic RL fine-tuning with AgentCore RL Toolkit (slime).
# Runs on SageMaker with GPU (ml.g5.12xlarge = 4x A10G for 3B models).
#
# Build and push:
#   ./training/build_and_push.sh

# AWAITING SOURCE FIX: slimerl/slime:latest is incompatible with agentcore-rl-toolkit's SlimeRunner
# (missing sglang_data_parallel_size arg, [slime] extra missing ray dep).
# Pinned to nightly-dev-20260701a until upstream fixes land.
FROM slimerl/slime:nightly-dev-20260701a

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/ml/code

# Install agentcore-rl-toolkit
RUN pip install --no-cache-dir \
    "agentcore-rl-toolkit[slime]" \
    boto3 \
    huggingface_hub

# Copy training entry point
COPY training/train_entry.py /opt/ml/code/train_entry.py

ENTRYPOINT ["python3", "/opt/ml/code/train_entry.py"]
