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
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import AfterToolCallEvent, HookProvider
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient
from strands_tools import editor, file_read, file_write
from utils.auth import get_gateway_access_token
from utils.ssm import get_ssm_parameter

from tools.code_interpreter.execute_python_tool import execute_python

app = AgentCoreRLApp()

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"

# Where the system prompt instructs the agent to write its report. The reward is
# computed from this file, not from the agent's closing chat message.
REPORT_PATH = "/tmp/research_report.md"

# Cap on each tool result. Matches --observation-chars used to build the SFT
# trajectories; larger values overflow max_model_len mid-episode.
OBSERVATION_CHARS = int(os.environ.get("OBSERVATION_CHARS", "1200"))

# Default data sources for training rollouts. Key-free tools only: a source that
# needs an API key fails intermittently if the key is absent or rate-limited, and
# an intermittent tool failure is indistinguishable from a bad policy, so it adds
# pure noise to the reward.
DEFAULT_RL_SOURCES = ["nova", "arxiv", "pubmed"]

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
        observations: list[str] | None = None,
        **kwargs,
    ) -> float:
        """Compute scalar reward for the report."""
        total, _ = self.score(
            response_text=response_text,
            user_input=user_input,
            retrieved_urls=retrieved_urls,
            observations=observations,
        )
        return total

    def score(
        self,
        response_text: str = "",
        user_input: str = "",
        retrieved_urls: set | None = None,
        observations: list[str] | None = None,
    ) -> tuple[float, dict]:
        """
        Reward plus its individual components.

        GRPO only needs the scalar, but a scalar hides which part of the reward is
        moving. That distinction is the difference between a genuine gain and
        reward hacking: published results on judge-scored long-form generation
        report citation and format scores climbing while overall report quality
        falls, so the aggregate can rise for the wrong reason. Tracking the parts
        separately makes that visible while a run is still in progress.
        """
        if not response_text or response_text.startswith("ERROR"):
            return 0.0, {"rubric": 0.0, "citation": 0.0, "format": 0.0}

        rubric_reward = self._judge_rubric(user_input, response_text, observations)
        citation_reward = research_rubric.score_citations(response_text, retrieved_urls)
        format_reward = research_rubric.score_format(response_text)
        total = research_rubric.combine(rubric_reward, citation_reward, format_reward)
        return total, {
            "rubric": rubric_reward,
            "citation": citation_reward,
            "format": format_reward,
        }

    def _judge_rubric(
        self, question: str, report: str, observations: list[str] | None = None
    ) -> float:
        """
        Score report against the shared rubric using an LLM judge call.

        Deliberately does not catch judge failures. A throttled or unparseable
        judge is missing data, and returning 0.0 would hand GRPO a false label
        saying this report is worthless — which is worse than a failed rollout,
        because it is indistinguishable from signal. Throttling also correlates
        with rollout concurrency, so the noise would be systematic rather than
        random. Let it raise and lose the rollout instead.
        """
        import boto3

        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        )
        score, _ = research_rubric.score_rubric_with_judge(
            question,
            report,
            bedrock,
            os.environ.get(
                "JUDGE_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
            ),
            observations=observations,
        )
        return score


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


class TruncateObservations(HookProvider):
    """Cap every tool result, matching the SFT training distribution.

    SFT trajectories were built with --observation-chars 1200. Untruncated, one web
    search returns ~2900 chars, so the trajectory exhausts max_model_len before the
    report is written: the agent emits its skeleton, researches, then dies on the
    token limit leaving a 290-char report that scores 0. Hooked rather than wrapping
    tool functions because Gateway tools are MCP-backed, not decorated functions.
    """

    def __init__(self, limit: int):
        self.limit = limit

    def register_hooks(self, registry):
        registry.add_callback(AfterToolCallEvent, self._cap)

    def _cap(self, event) -> None:
        for block in (event.result or {}).get("content", []):
            text = block.get("text")
            if not isinstance(text, str) or len(text) <= self.limit:
                continue
            # Tools append their URLs in a trailing "## Sources" block. Head-truncating
            # would drop every URL, so the model could not cite anything and the
            # citation metric would read as a model regression rather than a cap.
            body, sep, sources = text.partition("\n\n## Sources")
            kept = body[: self.limit]
            block["text"] = (
                kept + f"\n[prose truncated at {self.limit} chars]" + sep + sources
            )


def read_report(path: str = REPORT_PATH) -> str:
    """
    Read the report the agent wrote.

    AgentCore gives each session its own microVM, so this path is private to the
    rollout and cannot be clobbered by concurrent rollouts.
    """
    # Only a missing file is a legitimate "no report". Any other error (encoding,
    # permissions) is a real fault and should surface rather than masquerade as an
    # empty report and a near-zero reward.
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


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
    api_key = cfg.get("api_key") or "EMPTY"
    sampling_params = cfg.get(
        "sampling_params", {"temperature": 0.7, "max_tokens": 4096}
    )

    # Get prompt and metadata
    prompt = payload.get("prompt", "")
    enabled_sources = payload.get("enabled_sources", DEFAULT_RL_SOURCES)

    print(f"[RL] Rollout start: prompt={prompt[:80]}...")
    print(f"[RL] Model: {model_id} @ {base_url}")

    # Create model pointing at training infrastructure's inference server
    model = OpenAIModel(
        # The gateway keys trajectory capture off the api-key slot, so the session
        # key the trainer supplies must be forwarded. With a placeholder the
        # gateway captures no token ids, the rollout is treated as degenerate and
        # scored 0 regardless of the reward the agent computed.
        client_args={"api_key": api_key, "base_url": base_url},
        model_id=model_id,
        params=sampling_params,
    )

    # Create agent with same tools as production
    system_prompt = load_system_prompt(enabled_sources)
    tools = [file_read, file_write, editor, execute_python]

    # No fallback to local-only tools. Without Gateway the agent cannot retrieve
    # anything, so it would write from parametric memory, score badly, and teach
    # the policy from a broken environment rather than a bad decision.
    gateway_client = create_gateway_client(enabled_sources)
    tools.append(gateway_client)

    agent = Agent(
        name="DeepResearchRL",
        system_prompt=system_prompt,
        tools=tools,
        model=model,
        # NullConversationManager, not the Strands default. The default is
        # SlidingWindowConversationManager(window_size=40); measured across 1,833 real
        # episodes 86% exceed 40 messages (p50=45, max=75), so the default truncates
        # mid-episode and can split a tool_use from its tool_result.
        conversation_manager=NullConversationManager(),
        hooks=[TruncateObservations(OBSERVATION_CHARS)],
    )

    # Run the agent
    try:
        response = agent(prompt)
        # The final assistant message is NOT the report. The agent writes the
        # report incrementally into REPORT_PATH via file_write/editor and then
        # says something like "The report is complete." — measured across 400 real
        # trajectories, that closing message has a median length of 21 characters
        # while the report itself is ~15,000. Scoring the message instead of the
        # file gives every rollout the same near-floor reward (~0.08), which
        # flattens GRPO advantages to zero and produces no gradient at all.
        # So read the artefact, and keep the message only as a fallback.
        response_text = read_report()
        if not response_text:
            if response.message and response.message.get("content"):
                for block in response.message["content"]:
                    if isinstance(block, dict) and block.get("text"):
                        response_text = block["text"]
                        break
            print(
                f"[RL] WARNING: no report at {REPORT_PATH}; "
                f"falling back to final message ({len(response_text)} chars). "
                "Reward will be near zero.",
                flush=True,
            )
    except Exception as e:
        print(f"[RL] Agent failed: {e}")
        traceback.print_exc()
        response_text = f"ERROR: {e}"

    # Compute reward
    reward, components = reward_fn.score(
        response_text=response_text, user_input=prompt
    )
    # One greppable line per episode, so component trends can be recovered from
    # training logs without re-running rollouts.
    print(
        f"[RL] Rollout complete: reward={reward:.3f} "
        f"rubric={components['rubric']:.3f} "
        f"citation={components['citation']:.3f} "
        f"format={components['format']:.3f} "
        f"report_len={len(response_text)}"
    )

    # Consumers read the scalar via .get("rewards"), so the breakdown rides along
    # into the saved rollout data without affecting training.
    return {"rewards": reward, "reward_components": components}


if __name__ == "__main__":
    app.run()
