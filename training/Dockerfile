# Training container for agentic RL with the AgentCore RL Toolkit (verl backend,
# FSDP engine). Built for one node of 4x L40S 48GB.
#
#   ./training/build_and_push.sh rl
#
# FSDP, not Megatron: attention comes from HuggingFace transformers, the path that
# already trains this 9B at 32K. Megatron goes through Transformer Engine, which
# found no fused kernel for head_dim 256 and fell back to unfused O(n^2) attention.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Tracks main, not a SHA: the toolkit is pre-release and its verl backend moves.
# Override to freeze a build:  --build-arg TOOLKIT_REF=<sha>
ARG TOOLKIT_REF=main

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git build-essential ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# pip, not uv: uv is a Rust binary and segfaults under the QEMU emulation this
# linux/amd64 image is built with on arm64.
#
# verl is pinned to a git rev, NOT PyPI 0.9.0. 0.9.0 (with vllm 0.24.0) OOMs during
# update_actor at 8 episodes/step on 4x L40S, where this rev completes 25+ steps on
# an identical config -- the only difference being the image. Do not "upgrade" without
# re-measuring peak memory.
#
# transferqueue/orjson/aiohttp are listed because verl declares runtime deps in
# requirements.txt rather than install_requires, so a git install misses them.
#
# flash-attn from Astral's cu130 index, the same source the toolkit uses. verl
# hardcodes attn_implementation="flash_attention_2", so it is required; PyPI ships
# only a source dist, which needs nvcc and will not build under QEMU.
#
# The toolkit installs without its [verl] extra, which is what pins verl and vllm;
# its base deps are required since it imports bedrock_agentcore at module scope.
RUN pip install --no-cache-dir --break-system-packages \
        "verl @ git+https://github.com/volcengine/verl.git@78bba31d1a6b95084d83f8bddd59c190ca704144" \
        transformers==5.12.1 \
        transferqueue==0.1.8 \
        orjson \
        "aiohttp>=3.9" \
        vllm==0.23.0 \
        ray \
        flash-linear-attention \
        qwen-vl-utils \
        --extra-index-url https://wheels.astral.sh/simple/cu130/ \
        "flash-attn==2.8.3+cu.13.0.torch.2.11" \
        "rllm-model-gateway>=0.1.0" \
        boto3 \
        huggingface_hub \
    && pip install --no-cache-dir --break-system-packages \
        "agentcore-rl-toolkit @ git+https://github.com/awslabs/agentcore-rl-toolkit.git@${TOOLKIT_REF}" \
    && rm -rf /root/.cache

# The OS pip/wheel packages are only patched in Ubuntu Pro ESM builds, which
# apt-get upgrade cannot reach, so they show up as unfixable CVEs on every scan.
# Nothing needs them once the installs above are done -- training runs python3 only --
# so purge them and the finding goes away rather than needing a standing exception.
RUN apt-get purge -y --auto-remove python3-pip python3-wheel \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c "import verl, vllm, agentcore_rl_toolkit; print('imports OK after purge')"

COPY training/train_entry.py /opt/ml/code/train_entry.py

ENTRYPOINT ["python3", "/opt/ml/code/train_entry.py"]
