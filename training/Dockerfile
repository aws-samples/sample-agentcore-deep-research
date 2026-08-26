# Training container for agentic RL fine-tuning with AgentCore RL Toolkit (slime).
# Runs on SageMaker with GPU (ml.g5.12xlarge = 4x A10G for 3B models).
#
# Build and push:
#   ./training/build_and_push.sh

# Pinned to slime's stable release rather than a nightly or :latest, both of which
# move underneath the build.
#
# This previously had to sit on nightly-dev-20260701a because SlimeRunner did not
# pass --sglang-data-parallel-size, which newer slime requires in its sglang_utils
# validation. That is fixed upstream (agentcore-rl-toolkit #92), so the newer slime
# is usable — but the fix is only on main, not in the published 0.1.3, hence the
# git install below.
FROM slimerl/slime:v0.3.1

ENV PYTHONUNBUFFERED=1

WORKDIR /opt/ml/code

# Install agentcore-rl-toolkit
# Installed from a pinned commit rather than PyPI. PyPI 0.1.3 (2026-07-01) predates
# several SlimeRunner options this pipeline needs and which only exist on main:
#   --sglang-data-parallel-size  required by newer slime's validation (#92)
#   --sglang-tool-call-parser    tool-call parsing must match the served model,
#                                without which Qwen tool calls do not parse
#   --sglang-reasoning-parser    splits <think> blocks in the gateway
#   --sglang-context-length      caps context for long agentic episodes
# Pinning the SHA keeps this reproducible; move to a release tag once one ships
# containing these.
RUN pip install --no-cache-dir \
    "agentcore-rl-toolkit[slime] @ git+https://github.com/awslabs/agentcore-rl-toolkit.git@3dbb414b313b7e808977dbec9664d175994c8295" \
    boto3 \
    huggingface_hub

# Copy training entry point
COPY training/train_entry.py /opt/ml/code/train_entry.py

ENTRYPOINT ["python3", "/opt/ml/code/train_entry.py"]
