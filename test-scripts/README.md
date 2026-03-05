# Test Scripts

Utility scripts for deployment verification and operational tasks.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

## Available Scripts

### test-agent.py

Interactive chat interface for testing agent invocations with conversation continuity. Automatically detects the agent pattern from `infra-cdk/config.yaml`.

**Modes:**

- **Remote** (default): Test deployed agent via Cognito authentication
- **Local** (`--local`): Test agent running on localhost:8080

**Usage:**

```bash
# Remote mode (prompts for credentials, tests deployed agent)
uv run test-scripts/test-agent.py

# Local mode (auto-starts agent if not running, uses pattern from config.yaml)
uv run test-scripts/test-agent.py --local

# Override pattern for local testing
uv run test-scripts/test-agent.py --local --pattern strands-deep-research
```

**Supported Patterns:**

- `strands-deep-research` - Deep Research agent with multi-source search

**Prerequisites:**

- Remote: Deployed stack with Cognito user
- Local: Memory ID from deployed stack

---

### test-agent-docker.py

Test the agent by building and running the actual Docker image locally. This validates the production artifact before deployment.

**Usage:**

```bash
# Build and run (uses pattern from config.yaml)
python test-scripts/test-agent-docker.py

# Build only (verify Dockerfile)
python test-scripts/test-agent-docker.py --build-only

# Skip build, use existing image
python test-scripts/test-agent-docker.py --skip-build

# Test specific pattern
python test-scripts/test-agent-docker.py --pattern strands-deep-research
```

**What it tests:**

- Dockerfile builds correctly
- Dependencies install properly in container
- Container starts and responds to health checks
- Agent works in containerized environment (same as production)

**Prerequisites:**

- Docker installed and running
- Deployed stack (for Memory ID, Gateway, SSM parameters)
- AWS credentials in environment

**Note:** On x86/amd64 machines, you may need to enable ARM64 emulation:

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

See [Local Docker Testing Guide](../docs/LOCAL_DOCKER_TESTING.md) for detailed documentation.

---

### test-memory.py

Tests AgentCore Memory operations (create, list, get events and pagination).

**Usage:**

```bash
# Auto-discover memory from stack
uv run test-scripts/test-memory.py

# Use specific memory ARN
uv run test-scripts/test-memory.py --memory-arn <arn>
```

**Tests:**

1. Create conversation events
2. List events with pagination
3. Get specific events by ID
4. Session ID validation
5. Error handling

---

### test-feedback-api.py

Tests the deployed Feedback API endpoint with Cognito authentication.

**Prerequisites:**

- Stack deployed to AWS
- Cognito user created (see [Deployment Guide](../docs/DEPLOYMENT.md))

**Usage:**

```bash
uv run test-scripts/test-feedback-api.py
```

**What it does:**

1. Fetches configuration from SSM Parameter Store
2. Authenticates with Cognito (prompts for credentials)
3. Runs API tests (positive/negative feedback, validation)
4. Displays test results

---

### test-gateway.py

Tests AgentCore Gateway directly without frontend.

**Usage:**

```bash
uv run test-scripts/test-gateway.py
```

**What it does:**

1. Fetches Gateway configuration from SSM Parameter Store
2. Authenticates with Cognito (prompts for credentials)
3. Tests Gateway endpoints and authentication
4. Displays test results

---

### test-arxiv-tool.py

Tests the ArXiv search tool via AgentCore Gateway.

**Usage:**

```bash
uv run test-scripts/test-arxiv-tool.py
```

**What it does:**

1. Authenticates with Gateway using machine credentials
2. Lists available tools and finds arxiv_search
3. Executes a test search for "large language models agents"
4. Displays formatted search results

---

### test-deep-research-agent.py

Tests the Deep Research Agent via AgentCore Runtime with streaming.

**Usage:**

```bash
# Default research query
uv run test-scripts/test-deep-research-agent.py

# Custom query
uv run test-scripts/test-deep-research-agent.py "What are the latest advances in AI agents?"
```

**Prerequisites:**

- Deployed stack with `strands-deep-research` pattern
- `ACCESS_TOKEN` environment variable set with valid Cognito token

**What it does:**

1. Sends research query to the Deep Research Agent
2. Streams the response including tool calls
3. Shows file operations and data retrieval in real-time

---

## Shared Utilities

The test scripts use shared utilities from `../scripts/utils.py`:

- Stack configuration and SSM parameter retrieval
- Cognito authentication
- AWS client creation
- Session ID generation

## Troubleshooting

- **Authentication fails**: Verify Cognito user exists and credentials are correct
- **Stack not found**: Ensure stack is deployed and `config.yaml` has correct `stack_name_base`
- **Local agent fails**: Check Memory ID and AWS credentials are configured
- **Port 8080 in use**: Stop other services or use remote mode
