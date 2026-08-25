#!/usr/bin/env python3
"""
Generate SFT training data by running the production agent on research questions.

Invokes the deployed agent (teacher) with each training question, captures the
full research report (via presigned S3 URL from the stream), and formats it
for SFT training.

Handles:
- Token refresh (Cognito tokens expire after ~1 hour)
- Parallel execution with adaptive exponential backoff
- Resume from partial runs
- Report extraction from S3 presigned URLs in the stream

Usage:
    export EVAL_USERNAME=<cognito-username>
    export EVAL_PASSWORD=<cognito-password>
    export AWS_DEFAULT_REGION=us-west-2
    uv run test-scripts/sft_generate_data.py --max-concurrent 2

Prerequisites:
    - Deployed deep research stack (npm run deploy from infra-cdk/)
    - EVAL_USERNAME and EVAL_PASSWORD environment variables set
    - Training questions generated (sft_generate_questions.py)
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from trajectory_format import build_sft_example

# Add scripts directory for shared utils
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from utils import authenticate_cognito, generate_session_id, get_stack_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / backoff configuration
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
MAX_BACKOFF = 120.0
BACKOFF_MULTIPLIER = 2.0
JITTER_FACTOR = 0.3

# Refresh token 5 minutes before the typical 1-hour expiry
TOKEN_REFRESH_INTERVAL = 55 * 60  # 55 minutes


def extract_stream_error(stream_text: str) -> str:
    """
    Pull the agent's error message out of an SSE stream, if it reported one.

    The runtime emits {"status": "error", "error": "..."} rather than failing the
    HTTP request, so a rollout that produced no report may still carry a precise
    explanation.
    """
    for match in re.finditer(r'\{"status":\s*"error".*?\}', stream_text, re.DOTALL):
        try:
            return json.loads(match.group(0)).get("error", "")
        except json.JSONDecodeError:
            continue
    return ""


def backoff_delay(attempt: int) -> float:
    """Compute exponential backoff with jitter."""
    delay = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER**attempt), MAX_BACKOFF)
    jitter = delay * JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.5, delay + jitter)


# ---------------------------------------------------------------------------
# Token manager — auto-refreshes Cognito tokens before expiry
# ---------------------------------------------------------------------------


class TokenManager:
    """Thread-safe token manager that refreshes Cognito tokens before expiry."""

    def __init__(self, user_pool_id: str, client_id: str, username: str, password: str):
        self._user_pool_id = user_pool_id
        self._client_id = client_id
        self._username = username
        self._password = password
        self._lock = threading.Lock()
        self._token: str | None = None
        self._obtained_at: float = 0

    def get_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        with self._lock:
            now = time.time()
            if (
                self._token is None
                or (now - self._obtained_at) > TOKEN_REFRESH_INTERVAL
            ):
                self._refresh()
            return self._token

    def _refresh(self):
        """Re-authenticate with Cognito."""
        logger.info("Refreshing Cognito access token...")
        access_token, _, _ = authenticate_cognito(
            self._user_pool_id, self._client_id, self._username, self._password
        )
        self._token = access_token
        self._obtained_at = time.time()
        logger.info("Token refreshed.")

    def get_headers(self) -> dict[str, str]:
        """Get auth headers with a fresh token."""
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Content-Type": "application/json",
        }


# ---------------------------------------------------------------------------
# Agent invocation with retries + report URL extraction
# ---------------------------------------------------------------------------


def invoke_agent_with_retry(
    url: str,
    prompt: str,
    token_manager: TokenManager,
    enabled_sources: list[str] | None = None,
    timeout: int = 900,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Invoke the deployed agent with exponential backoff and token refresh.

    Extracts the actual research report by:
    1. Streaming the response to find [REPORT_URL:...] tags
    2. Downloading the report markdown from the presigned S3 URL
    3. Falling back to stream text if it contains a valid report
    """
    last_error = ""

    for attempt in range(max_retries):
        session_id = generate_session_id()
        payload = {"prompt": prompt, "runtimeSessionId": session_id}
        if enabled_sources:
            payload["enabledSources"] = enabled_sources

        # Get fresh headers (auto-refreshes token if near expiry)
        headers = token_manager.get_headers()
        start = time.time()

        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=True, timeout=timeout
            )

            # Auth failure — force token refresh and retry immediately
            if response.status_code in (401, 403):
                last_error = f"HTTP {response.status_code} (auth)"
                # Force refresh on next get_token() call
                token_manager._obtained_at = 0
                delay = backoff_delay(0)  # Short delay
                if attempt < max_retries - 1:
                    logger.info(
                        f"  Auth expired, refreshing token and retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
                    continue

            # Retryable HTTP errors
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                delay = backoff_delay(attempt)
                if attempt < max_retries - 1:
                    logger.debug(
                        f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s ({last_error})"
                    )
                    time.sleep(delay)
                    continue

            if response.status_code != 200:
                return {
                    "response": "",
                    "elapsed_seconds": time.time() - start,
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "attempts": attempt + 1,
                }

            # Stream response — collect text AND look for report URL
            stream_text = ""
            report_url = None
            # Full Bedrock-style messages, which carry toolUse/toolResult blocks.
            # Strands emits one `message` event per completed message, so the
            # entire multi-step trajectory is recoverable from the stream.
            trajectory: list[dict] = []

            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                raw_line = line[6:]
                try:
                    chunk = json.loads(raw_line)
                    if isinstance(chunk.get("data"), str):
                        stream_text += chunk["data"]
                    elif chunk.get("type") == "AIMessageChunk" and isinstance(
                        chunk.get("content"), list
                    ):
                        for block in chunk["content"]:
                            if block.get("type") == "text" and block.get("text"):
                                stream_text += block["text"]
                    msg = chunk.get("message")
                    if isinstance(msg, dict) and msg.get("role"):
                        trajectory.append(msg)
                except (json.JSONDecodeError, KeyError):
                    pass

                # Look for report URL anywhere in the raw line
                if "[REPORT_URL:" in raw_line:
                    match = re.search(r"\[REPORT_URL:(https://[^\]]+)\]", raw_line)
                    if match:
                        report_url = match.group(1)

            # Also scan accumulated stream text for report URL
            if not report_url and "[REPORT_URL:" in stream_text:
                match = re.search(r"\[REPORT_URL:(https://[^\]]+)\]", stream_text)
                if match:
                    report_url = match.group(1)

            elapsed = time.time() - start

            # Strategy 1: Download report from presigned URL
            if report_url:
                try:
                    report_resp = requests.get(report_url, timeout=30)
                    if (
                        report_resp.status_code == 200
                        and len(report_resp.text.strip()) > 500
                    ):
                        return {
                            "response": report_resp.text.strip(),
                            "trajectory": trajectory,
                            "elapsed_seconds": round(elapsed, 2),
                            "success": True,
                            "attempts": attempt + 1,
                            "source": "s3_url",
                        }
                except Exception as e:
                    logger.debug(f"  Failed to fetch report URL: {e}")

            # Strategy 2: Stream text looks like a report (has markdown structure)
            if stream_text.strip().startswith("#") and len(stream_text.strip()) > 1000:
                return {
                    "response": stream_text.strip(),
                    "trajectory": trajectory,
                    "elapsed_seconds": round(elapsed, 2),
                    "success": True,
                    "attempts": attempt + 1,
                    "source": "stream",
                }

            # Not enough content — retry. Surface the agent's own error if it
            # emitted one: the runtime reports the real cause in the stream (for
            # example Bedrock's "maximum tokens you requested exceeds the model
            # limit of N"), and reporting only "no report" hides it, which turns
            # a self-explanatory failure into a debugging session.
            agent_error = extract_stream_error(stream_text)
            if agent_error:
                last_error = f"Agent error: {agent_error[:300]}"
            else:
                last_error = f"No report (stream={len(stream_text)} chars, url={'found' if report_url else 'none'})"
            delay = backoff_delay(attempt)
            if attempt < max_retries - 1:
                logger.debug(
                    f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s ({last_error})"
                )
                time.sleep(delay)
                continue

        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.ConnectionError:
            last_error = "ConnectionError"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            delay = backoff_delay(attempt)
            logger.debug(
                f"  Retry {attempt + 1}/{max_retries} after {delay:.1f}s ({last_error})"
            )
            time.sleep(delay)

    return {
        "response": "",
        "elapsed_seconds": 0,
        "success": False,
        "error": f"Failed after {max_retries} attempts: {last_error}",
        "attempts": max_retries,
    }


# ---------------------------------------------------------------------------
# Trace formatting
# ---------------------------------------------------------------------------


def format_sft_example(
    question: str,
    report: str,
    trajectory: list[dict] | None = None,
    observation_chars: int = 2000,
) -> dict:
    """
    Format one collected run into a TRL SFT example.

    When the full agent trajectory is available it is emitted in TRL's
    tool-calling format (assistant `tool_calls` + `tool` role observations),
    because training on the final report alone teaches report prose without the
    tool use that produces it — the failure mode behind a base model that makes
    zero tool calls. The report is retained under `report` for rubric scoring.

    Falls back to the legacy question -> report pair if no trajectory was
    captured, so collection never silently drops a successful run.
    """
    if trajectory:
        example = build_sft_example(question, trajectory, observation_chars)
        example["report"] = report
        return example

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": report},
        ],
        "tools": [],
        "report": report,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_questions(path: Path) -> list[dict]:
    """Load questions from JSONL file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row.get("prompt"), list):
                    prompt_text = row["prompt"][0]["content"]
                elif isinstance(row.get("prompt"), str):
                    prompt_text = row["prompt"]
                else:
                    prompt_text = row.get("metadata", {}).get("prompt", "")

                questions.append(
                    {
                        "prompt": prompt_text,
                        "enabled_sources": row.get(
                            "enabled_sources",
                            row.get("metadata", {}).get("tools", ["tavily", "nova"]),
                        ),
                        "metadata": row.get("metadata", {}),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed line: {e}")
    return questions


# ---------------------------------------------------------------------------
# Parallel trace generation
# ---------------------------------------------------------------------------


def generate_traces(
    questions: list[dict],
    url: str,
    token_manager: TokenManager,
    output_path: Path,
    max_concurrent: int = 2,
    max_questions: int | None = None,
    timeout: int = 900,
    observation_chars: int = 2000,
) -> dict:
    """
    Generate SFT traces with parallel execution, backoff, and token refresh.

    Resume-safe: writes results incrementally, skips already-completed questions.
    """
    # Resume support
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    completed.add(row.get("question", ""))
                except json.JSONDecodeError:
                    continue
        if completed:
            logger.info(f"Resuming: {len(completed)} traces already generated")

    remaining = [q for q in questions if q["prompt"] not in completed]
    if max_questions:
        remaining = remaining[:max_questions]

    if not remaining:
        logger.info("All questions already have traces!")
        return {"total": len(completed), "new": 0, "errors": 0}

    logger.info(
        f"Generating {len(remaining)} traces "
        f"(max_concurrent={max_concurrent}, max_retries={MAX_RETRIES}, "
        f"token_refresh={TOKEN_REFRESH_INTERVAL}s)"
    )

    success_count = 0
    error_count = 0
    total_attempts = 0
    lock = threading.Lock()

    def process_one(q: dict) -> dict | None:
        result = invoke_agent_with_retry(
            url=url,
            prompt=q["prompt"],
            token_manager=token_manager,
            enabled_sources=q.get("enabled_sources"),
            timeout=timeout,
        )

        if result["success"]:
            sft_example = format_sft_example(
                q["prompt"],
                result["response"],
                trajectory=result.get("trajectory"),
                observation_chars=observation_chars,
            )
            return {
                **sft_example,
                "question": q["prompt"],
                "metadata": q.get("metadata", {}),
                "elapsed_seconds": result["elapsed_seconds"],
                "attempts": result["attempts"],
                "source": result.get("source", "unknown"),
            }
        return None

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_q = {executor.submit(process_one, q): q for q in remaining}

        for i, future in enumerate(as_completed(future_to_q), 1):
            q = future_to_q[future]
            try:
                result = future.result()
                with lock:
                    if result:
                        with open(output_path, "a") as f:
                            f.write(json.dumps(result) + "\n")
                        success_count += 1
                        total_attempts += result.get("attempts", 1)
                        # Assistant messages carrying tool_calls may have no
                        # "content" key, so report length comes from the report
                        # field rather than a positional message lookup.
                        report_len = len(result.get("report") or "")
                        src = result.get("source", "?")
                        retries_info = (
                            f" r={result['attempts'] - 1}"
                            if result.get("attempts", 1) > 1
                            else ""
                        )
                        logger.info(
                            f"  [{i}/{len(remaining)}] ✓ {q['prompt'][:55]}... "
                            f"({result['elapsed_seconds']:.0f}s, {report_len} chars, {src}){retries_info}"
                        )
                    else:
                        error_count += 1
                        logger.warning(
                            f"  [{i}/{len(remaining)}] ✗ {q['prompt'][:55]}... (failed)"
                        )

                    # Progress every 25
                    done = success_count + error_count
                    if done % 25 == 0:
                        rate = success_count / done if done > 0 else 0
                        logger.info(
                            f"  >>> Progress: {done}/{len(remaining)}, "
                            f"{success_count} ok, {error_count} failed ({rate:.0%})"
                        )
            except Exception as e:
                with lock:
                    error_count += 1
                logger.error(f"  [{i}/{len(remaining)}] ✗ Exception: {e}")

    return {
        "total": len(completed) + success_count,
        "new": success_count,
        "errors": error_count,
        "avg_attempts": round(total_attempts / max(success_count, 1), 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT training traces from deployed production agent",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="test-scripts/results/rl_train_data.jsonl",
        help="Input questions JSONL (default: test-scripts/results/rl_train_data.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test-scripts/results/sft_traces.jsonl",
        help="Output traces JSONL (default: test-scripts/results/sft_traces.jsonl)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Max concurrent agent invocations (default: 2)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions to process (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-question timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--observation-chars",
        type=int,
        default=2000,
        help="Truncate each tool observation to this many characters "
        "(default: 2000). Observations are masked from the loss but still "
        "consume sequence length; an uncapped trajectory runs ~37K tokens.",
    )

    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        logger.error("Run sft_generate_questions.py first.")
        sys.exit(1)

    questions = load_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions from {questions_path}")

    # Set up connection
    stack_cfg = get_stack_config()
    outputs = stack_cfg["outputs"]
    region = stack_cfg["region"]

    username = os.environ.get("EVAL_USERNAME")
    password = os.environ.get("EVAL_PASSWORD")
    if not username or not password:
        logger.error("EVAL_USERNAME and EVAL_PASSWORD environment variables required")
        sys.exit(1)

    # Token manager handles refresh automatically
    token_manager = TokenManager(
        outputs["CognitoUserPoolId"], outputs["CognitoClientId"], username, password
    )
    # Force initial auth
    token_manager.get_token()

    runtime_arn = outputs["RuntimeArn"]
    endpoint = f"https://bedrock-agentcore.{region}.amazonaws.com"
    escaped_arn = requests.utils.quote(runtime_arn, safe="")
    url = f"{endpoint}/runtimes/{escaped_arn}/invocations?qualifier=DEFAULT"

    logger.info(f"Agent: {runtime_arn}")
    logger.info(f"Region: {region}")
    logger.info(f"Concurrency: {args.max_concurrent}")
    logger.info("")

    # Generate traces
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    stats = generate_traces(
        questions=questions,
        url=url,
        token_manager=token_manager,
        output_path=output_path,
        max_concurrent=args.max_concurrent,
        max_questions=args.max_questions,
        timeout=args.timeout,
        observation_chars=args.observation_chars,
    )
    total_time = time.time() - start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("SFT Data Generation Complete")
    logger.info("=" * 60)
    logger.info(f"Total traces:     {stats['total']}")
    logger.info(f"New this run:     {stats['new']}")
    logger.info(f"Errors:           {stats['errors']}")
    logger.info(f"Avg attempts/q:   {stats.get('avg_attempts', 1)}")
    logger.info(f"Time:             {total_time / 60:.1f} min")
    if stats["new"] > 0:
        logger.info(
            f"Throughput:       {stats['new'] / (total_time / 60):.1f} traces/min"
        )
    logger.info(f"Output:           {output_path}")
    logger.info("")
    logger.info("Next step: train with LoRA")
    logger.info(f"  uv run test-scripts/sft_train.py --data {output_path}")


if __name__ == "__main__":
    main()
