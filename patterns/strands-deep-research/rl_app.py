#!/usr/bin/env python3
"""
RL-adapted Deep Research Agent for AgentCore RL Toolkit.

This is the training variant of deep_research_agent.py. Key differences:
- Uses AgentCoreRLApp + @rollout_entrypoint (fire-and-forget, saves to S3)
- Uses OpenAIModel pointed at the training backend's inference server
- Computes and returns rubric-based rewards for GRPO training
- No streaming, no memory, no S3 upload hooks (training rollouts are stateless)

The agent keeps the same tools (Gateway MCP) and system prompt, so training
behavior matches production. Only the model interface and entrypoint change.

Deploy with:
    agentcore configure --entrypoint rl_app.py --name deep-research-rl \
        --requirements-file requirements.txt --deployment-type container \
        --non-interactive
    agentcore deploy --agent deep-research-rl
"""

import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["BYPASS_TOOL_CONSENT"] = "true"

import research_rubric
import strands_compat
from agentcore_rl_toolkit import AgentCoreRLApp, RewardFunction
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient
from strands_tools import editor, file_read, file_write
from utils.auth import get_gateway_access_token
from utils.ssm import get_ssm_parameter

from tools.code_interpreter.execute_python_tool import execute_python

app = AgentCoreRLApp()

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# Default data sources for training rollouts (web search only — fast and general)
DEFAULT_RL_SOURCES = ["tavily", "nova"]

# Tool name mapping (same as production agent)
DATA_SOURCES = {
    "tavily": {"tool": "tavily_web_search"},
    "nova": {"tool": "nova_web_search"},
    "arxiv": {"tool": "arxiv_search"},
    "openfda": {"tool": "openfda_drug_search"},
    "pubmed": {"tool": "pubmed_search"},
    "edgar": {"tool": "edgar_search"},
}


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------


class DeepResearchReward(RewardFunction):
    """
    Rubric-based reward for deep research reports.

    The rubric is defined once in research_rubric.py and shared with the
    offline eval (test-scripts/eval-agent.py). Do not re-implement it here:
    the training reward and the reported metric must be the same measurement,
    and the previous copy-paste arrangement is how they drift apart.

    Note for RL specifically: the citation and format components are
    string-matching heuristics and are therefore the most attackable part of
    this reward. Before using it for RL, confirm a vacuous report scores far
    below a real one
    showed that section headings plus fabricated citation-shaped links scored
    0.412 under the original weighting while the LLM judge rated the same text
    2/10 on every criterion. research_rubric.py now validates citation URLs,
    requires substantive content beneath headings, and down-weights both
    heuristics to 10% each. Re-run that probe after any reward change.
    """

    def __call__(
        self,
        response_text: str = "",
        ground_truth: str = "",
        user_input: str = "",
        retrieved_urls: set | None = None,
        **kwargs,
    ) -> float:
        """Compute scalar reward for the report."""
        if not response_text or response_text.startswith("ERROR"):
            return 0.0

        rubric_reward = self._judge_rubric(user_input, response_text)
        citation_reward = research_rubric.score_citations(response_text, retrieved_urls)
        format_reward = research_rubric.score_format(response_text)
        return research_rubric.combine(rubric_reward, citation_reward, format_reward)

    def _judge_rubric(self, question: str, report: str) -> float:
        """Score report against the shared rubric using an LLM judge call."""
        import boto3

        try:
            bedrock = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
            score, _ = research_rubric.score_rubric_with_judge(
                question,
                report[:24000],
                bedrock,
                "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            )
            return score
        except Exception as e:
            print(f"[REWARD] Judge failed: {e}, defaulting to 0")
            return 0.0


strands_compat.apply_shims()

reward_fn = DeepResearchReward()


# ---------------------------------------------------------------------------
# Agent Creation (simplified for RL — no memory, no streaming hooks)
# ---------------------------------------------------------------------------


def load_system_prompt(enabled_sources: list[str]) -> str:
    """Load and customize system prompt for enabled sources."""
    with open(SYSTEM_PROMPT_PATH) as f:
        base_prompt = f.read()

    tools_section = "### Data Retrieval (via Gateway)\n"
    tools_section += (
        "The following Gateway tools are available (prefixed with `gateway___`):\n"
    )
    for key in enabled_sources:
        if key in DATA_SOURCES:
            tools_section += f"- {key}\n"

    pattern = r"### Data Retrieval \(via Gateway\)\n(?:- .*\n)*"
    base_prompt = re.sub(pattern, tools_section, base_prompt)

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    return f"Today is {today}.\n\n{base_prompt}"


def create_gateway_client(enabled_sources: list[str]) -> MCPClient:
    """Create MCP client for Gateway tools."""
    stack_name = os.environ.get("STACK_NAME")
    if not stack_name:
        raise ValueError("STACK_NAME required")

    gateway_url = get_ssm_parameter(f"/{stack_name}/gateway_url")
    access_token = get_gateway_access_token()

    # Filter to enabled tools only
    allowed_tools = [
        DATA_SOURCES[k]["tool"] for k in enabled_sources if k in DATA_SOURCES
    ]
    tool_filter = re.compile(
        r"^.*___(" + "|".join(re.escape(n) for n in allowed_tools) + r")$"
    )

    return MCPClient(
        lambda: streamablehttp_client(
            url=gateway_url, headers={"Authorization": f"Bearer {access_token}"}
        ),
        tool_filters={"allowed": [tool_filter]},
        prefix="gateway",
    )


# ---------------------------------------------------------------------------
# RL Entrypoint
# ---------------------------------------------------------------------------


@app.rollout_entrypoint
def invoke_agent(payload: dict):
    """
    RL rollout entrypoint — runs one research episode and returns rewards.

    During training, the training backend injects _rollout config with:
    - base_url: inference server URL (vLLM via model-gateway)
    - model_id: model being trained
    - sampling_params: temperature, top_p, etc.

    The agent runs its full research workflow, then scores its own output.
    """
    # Extract rollout config (injected by training backend)
    cfg = payload.get("_rollout", {})
    base_url = cfg.get(
        "base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    )
    model_id = cfg.get(
        "model_id", os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    )
    sampling_params = cfg.get(
        "sampling_params", {"temperature": 0.7, "max_tokens": 4096}
    )

    # Get prompt and metadata
    prompt = payload.get("prompt", "")
    answer = payload.get("answer", "")
    enabled_sources = payload.get("enabled_sources", DEFAULT_RL_SOURCES)

    print(f"[RL] Rollout start: prompt={prompt[:80]}...")
    print(f"[RL] Model: {model_id} @ {base_url}")

    # Create model pointing at training infrastructure's inference server
    model = OpenAIModel(
        client_args={"api_key": "EMPTY", "base_url": base_url},
        model_id=model_id,
        params=sampling_params,
    )

    # Create agent with same tools as production
    system_prompt = load_system_prompt(enabled_sources)
    tools = [file_read, file_write, editor, execute_python]

    try:
        gateway_client = create_gateway_client(enabled_sources)
        tools.append(gateway_client)
    except Exception as e:
        print(f"[RL] Gateway unavailable ({e}), proceeding with local tools only")

    agent = Agent(
        name="DeepResearchRL",
        system_prompt=system_prompt,
        tools=tools,
        model=model,
    )

    # Run the agent
    try:
        response = agent(prompt)
        # Defensive extraction: response.message may be None, content may be empty or
        # contain non-text blocks (e.g. toolUse). Guard against IndexError/TypeError.
        response_text = ""
        if response.message and response.message.get("content"):
            for block in response.message["content"]:
                if isinstance(block, dict) and block.get("text"):
                    response_text = block["text"]
                    break
    except Exception as e:
        print(f"[RL] Agent failed: {e}")
        traceback.print_exc()
        response_text = f"ERROR: {e}"

    # Compute reward
    reward = reward_fn(
        response_text=response_text, ground_truth=answer, user_input=prompt
    )
    print(
        f"[RL] Rollout complete: reward={reward:.3f}, report_len={len(response_text)}"
    )

    return {"rewards": reward}


if __name__ == "__main__":
    app.run()
