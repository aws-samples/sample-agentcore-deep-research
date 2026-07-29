# RL-adapted Deep Research Agent — for use with AgentCore RL Toolkit training.
# Same dependencies as production agent, different entrypoint (rl_app.py).

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    DOCKER_CONTAINER=1 \
    PYTHONUNBUFFERED=1

# Copy pyproject.toml and shared packages
COPY pyproject.toml .
COPY gateway/ gateway/
COPY tools/ tools/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    libcairo2 libffi-dev libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install agent requirements + RL toolkit
COPY patterns/strands-deep-research/requirements.txt requirements.txt
RUN uv pip install --no-cache -r requirements.txt && \
    uv pip install --no-cache agentcore-rl-toolkit openai

# Install package
RUN uv pip install --no-cache -e . --no-deps && \
    uv pip install --no-cache requests>=2.31.0

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080

# Copy agent code
COPY patterns/strands-deep-research/rl_app.py .
COPY patterns/strands-deep-research/system_prompt.txt .
COPY patterns/utils/ utils/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping', timeout=2)" || exit 1

CMD ["python", "-m", "rl_app"]
