#!/usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""
Benchmark evaluation for the deployed deep research agent.

Evaluates the agent against two benchmarks from the TTD-DR paper (arXiv:2507.16075):
- GAIA (validation set): Real-world questions requiring reasoning + tool use
- HLE-search: Subset of Humanity's Last Exam requiring extensive web search

Both benchmarks use correctness evaluation (answer extraction + comparison to
ground-truth), making them objective and reproducible.

Usage:
    # Run both benchmarks (default)
    uv run test-scripts/eval-agent.py

    # Run specific benchmark
    uv run test-scripts/eval-agent.py --benchmark gaia
    uv run test-scripts/eval-agent.py --benchmark hle-search

    # Limit number of questions (for quick testing)
    uv run test-scripts/eval-agent.py --max-questions 10

    # Specify tools and model
    uv run test-scripts/eval-agent.py --model global.anthropic.claude-sonnet-4-5-20250929-v1:0 --tools tavily,arxiv,nova

    # Resume from a previous run
    uv run test-scripts/eval-agent.py --resume results/eval_gaia_20260707_113000.jsonl

    # Local mode (agent running on localhost:8080)
    uv run test-scripts/eval-agent.py --local

Prerequisites:
    pip install datasets requests colorama pyyaml boto3
"""

import argparse
import getpass
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from colorama import Fore, Style
from datasets import load_dataset

# Add scripts directory to path for reliable imports
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from utils import (
    authenticate_cognito,
    create_mock_jwt,
    generate_session_id,
    get_stack_config,
    print_msg,
    print_section,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

# Standard evaluation prompt for answer extraction (from Scale AI HLE leaderboard)
ANSWER_EXTRACTION_PROMPT = """You are an evaluation judge. Given a model's response to a question, extract the final answer.

Question: {question}

Model response:
{response}

Extract the single final answer from the model's response. Return ONLY the answer, nothing else.
If the model did not provide a clear answer, return "NO_ANSWER".
"""

# Correctness judge prompt (following the TTD-DR paper's approach)
CORRECTNESS_JUDGE_PROMPT = """You are an evaluation judge comparing a predicted answer against a ground truth answer.

Question: {question}
Ground truth answer: {ground_truth}
Predicted answer: {predicted}

Determine if the predicted answer is correct. The predicted answer does not need to match
the ground truth exactly, but must be semantically equivalent or contain the correct answer.
For numerical answers, minor formatting differences are acceptable.
For multiple choice, the letter must match.

Respond with ONLY one word: "CORRECT" or "INCORRECT"
"""

# HLE-search categorization prompt (from TTD-DR paper appendix A.4)
HLE_SEARCH_CATEGORIZATION_PROMPT = """Categorize this question into one of two categories:
[a] Pure reasoning: Can be answered using only logical reasoning, mathematical computation, or general knowledge without needing to search for specific facts.
[b] Requiring search: Requires looking up specific facts, data, recent events, or domain-specific information that would need external search tools.

Question: {question}

Respond with ONLY the letter: a or b
"""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_gaia_dataset() -> list[dict]:
    """
    Load GAIA validation set from HuggingFace.

    Returns list of dicts with keys: question, answer, level, task_id
    """
    print_msg("Loading GAIA validation set from HuggingFace...")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

    questions = []
    for row in ds:
        # Skip questions that require file attachments (we only have search tools)
        if row.get("file_name") and row["file_name"].strip():
            continue
        questions.append(
            {
                "id": row.get("task_id", str(uuid.uuid4())),
                "question": row["Question"],
                "answer": row["Final answer"],
                "level": row.get("Level", 0),
                "metadata": {"source": "gaia", "level": row.get("Level", 0)},
            }
        )

    print_msg(f"Loaded {len(questions)} GAIA questions (excluding file-based)", "success")
    return questions


def load_hle_dataset(categorize_fn=None, max_search_questions: int = 200) -> list[dict]:
    """
    Load HLE-search subset from HuggingFace.

    The TTD-DR paper filters HLE to questions requiring search (category [b]).
    If categorize_fn is provided, it's used to classify questions. Otherwise,
    a heuristic pre-filter is applied and all text-only questions are included.

    Parameters
    ----------
    categorize_fn : callable, optional
        Function that takes a question string and returns 'a' or 'b'
    max_search_questions : int
        Maximum number of search-requiring questions to include

    Returns list of dicts with keys: question, answer, id, metadata
    """
    print_msg("Loading HLE dataset from HuggingFace...")
    ds = load_dataset("cais/hle", split="test")

    questions = []
    for row in ds:
        # Skip multimodal questions (image-based)
        # The image field contains base64 strings; long ones indicate actual images
        img = row.get("image")
        if img is not None and len(str(img)) > 100:
            continue

        q = {
            "id": row.get("id", str(uuid.uuid4())),
            "question": row["question"],
            "answer": row["answer"],
            "metadata": {
                "source": "hle-search",
                "subject": row.get("subject", "unknown"),
                "answer_type": row.get("answer_type", "unknown"),
            },
        }

        if categorize_fn is not None:
            category = categorize_fn(row["question"])
            if category == "b":
                questions.append(q)
        else:
            # Without a categorizer, include all text-only questions
            # (user should ideally provide a categorizer or pre-filtered IDs)
            questions.append(q)

        if len(questions) >= max_search_questions:
            break

    print_msg(f"Loaded {len(questions)} HLE-search questions", "success")
    return questions


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


def invoke_agent_sync(
    url: str,
    prompt: str,
    session_id: str,
    headers: dict[str, str],
    enabled_sources: list[str] | None = None,
    timeout: int = 600,
) -> str:
    """
    Invoke the deployed agent and collect the full response.

    Parameters
    ----------
    url : str
        Agent endpoint URL
    prompt : str
        The question to send
    session_id : str
        Unique session ID for this invocation
    headers : dict
        HTTP headers including auth
    enabled_sources : list[str] | None
        List of enabled data sources
    timeout : int
        Request timeout in seconds

    Returns
    -------
    str
        Complete agent response text
    """
    payload = {
        "prompt": prompt,
        "runtimeSessionId": session_id,
    }
    if enabled_sources:
        payload["enabledSources"] = enabled_sources

    headers = {**headers, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)

        if response.status_code != 200:
            return f"ERROR: HTTP {response.status_code}: {response.text[:500]}"

        # Collect full text response from streaming
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])

                # Strands: text token
                if isinstance(chunk.get("data"), str):
                    full_text += chunk["data"]

                # LangGraph: AIMessageChunk with content array
                elif chunk.get("type") == "AIMessageChunk" and isinstance(
                    chunk.get("content"), list
                ):
                    for block in chunk["content"]:
                        if block.get("type") == "text" and block.get("text"):
                            full_text += block["text"]

            except (json.JSONDecodeError, KeyError):
                continue

        return full_text.strip()

    except requests.exceptions.Timeout:
        return "ERROR: Request timed out"
    except requests.exceptions.ConnectionError:
        return "ERROR: Connection failed"
    except Exception as e:
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Evaluation (correctness judge)
# ---------------------------------------------------------------------------


def judge_correctness(
    question: str,
    ground_truth: str,
    predicted: str,
    judge_model: str = "anthropic.claude-sonnet-4-20250514-v1:0",
) -> dict:
    """
    Use an LLM judge to determine if the predicted answer is correct.

    Parameters
    ----------
    question : str
        Original question
    ground_truth : str
        Ground truth answer
    predicted : str
        Model's predicted/extracted answer
    judge_model : str
        Bedrock model ID for the judge

    Returns
    -------
    dict with keys: correct (bool), raw_judgment (str)
    """
    import boto3

    # First extract the answer from the full response
    extraction_prompt = ANSWER_EXTRACTION_PROMPT.format(question=question, response=predicted)

    bedrock = boto3.client("bedrock-runtime")

    # Extract answer
    extract_response = bedrock.invoke_model(
        modelId=judge_model,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": extraction_prompt}],
            }
        ),
    )
    extract_body = json.loads(extract_response["body"].read())
    extracted_answer = extract_body["content"][0]["text"].strip()

    if extracted_answer == "NO_ANSWER":
        return {"correct": False, "extracted_answer": "NO_ANSWER", "raw_judgment": "NO_ANSWER"}

    # Judge correctness
    judge_prompt = CORRECTNESS_JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, predicted=extracted_answer
    )

    judge_response = bedrock.invoke_model(
        modelId=judge_model,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": judge_prompt}],
            }
        ),
    )
    judge_body = json.loads(judge_response["body"].read())
    judgment = judge_body["content"][0]["text"].strip().upper()

    return {
        "correct": "CORRECT" in judgment and "INCORRECT" not in judgment,
        "extracted_answer": extracted_answer,
        "raw_judgment": judgment,
    }


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def save_result(filepath: Path, result: dict) -> None:
    """Append a single result as a JSON line."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_completed_ids(filepath: Path) -> set[str]:
    """Load IDs of already-completed questions from a results file."""
    if not filepath.exists():
        return set()
    completed = set()
    with open(filepath) as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                completed.add(result["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def compute_metrics(filepath: Path) -> dict:
    """Compute aggregate metrics from a results file."""
    results = []
    with open(filepath) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    errors = sum(1 for r in results if r.get("response", "").startswith("ERROR"))

    metrics = {
        "total": total,
        "correct": correct,
        "errors": errors,
        "accuracy": correct / total if total > 0 else 0.0,
        "accuracy_excl_errors": correct / (total - errors) if (total - errors) > 0 else 0.0,
    }

    # Per-level breakdown for GAIA
    levels = {}
    for r in results:
        level = r.get("metadata", {}).get("level")
        if level is not None:
            if level not in levels:
                levels[level] = {"total": 0, "correct": 0}
            levels[level]["total"] += 1
            if r.get("correct", False):
                levels[level]["correct"] += 1

    if levels:
        metrics["per_level"] = {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0}
            for k, v in sorted(levels.items())
        }

    return metrics


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    questions: list[dict],
    benchmark_name: str,
    url: str,
    headers: dict[str, str],
    enabled_sources: list[str] | None,
    results_file: Path,
    judge_model: str,
    max_questions: int | None = None,
    parallel: int = 1,
) -> dict:
    """
    Run evaluation over a list of questions.

    Parameters
    ----------
    questions : list[dict]
        Questions to evaluate
    benchmark_name : str
        Name of the benchmark (for display)
    url : str
        Agent invocation URL
    headers : dict
        Auth headers
    enabled_sources : list[str] | None
        Enabled data sources
    results_file : Path
        Path to write JSONL results
    judge_model : str
        Model ID for the correctness judge
    max_questions : int | None
        Maximum number of questions to evaluate
    parallel : int
        Number of parallel workers (1 = sequential)

    Returns
    -------
    dict with aggregate metrics
    """
    # Load already-completed questions (for resume support)
    completed_ids = load_completed_ids(results_file)
    if completed_ids:
        print_msg(f"Resuming: {len(completed_ids)} questions already completed", "info")

    # Filter to remaining questions
    remaining = [q for q in questions if q["id"] not in completed_ids]
    if max_questions is not None:
        remaining = remaining[: max_questions - len(completed_ids)]

    total_to_run = len(remaining)
    if total_to_run == 0:
        print_msg("All questions already completed!", "success")
        if results_file.exists():
            return compute_metrics(results_file)
        return {"total": 0, "correct": 0, "errors": 0, "accuracy": 0.0, "accuracy_excl_errors": 0.0}

    print_section(f"Running {benchmark_name} Evaluation")
    print(f"Questions to evaluate: {total_to_run}")
    print(f"Results file: {results_file}")
    print(f"Enabled sources: {enabled_sources or 'all'}")
    print(f"Parallel workers: {parallel}")
    print()

    if parallel <= 1:
        # Sequential execution (original behavior)
        _run_sequential(remaining, total_to_run, url, headers, enabled_sources,
                        results_file, judge_model)
    else:
        # Parallel execution with ThreadPoolExecutor
        _run_parallel(remaining, total_to_run, url, headers, enabled_sources,
                      results_file, judge_model, parallel)

    # Final metrics
    metrics = compute_metrics(results_file)
    return metrics


def _evaluate_single_question(
    question: dict,
    url: str,
    headers: dict[str, str],
    enabled_sources: list[str] | None,
    judge_model: str,
) -> dict:
    """Evaluate a single question (thread-safe). Returns result dict."""
    session_id = generate_session_id()
    q_text = question["question"]

    start_time = time.time()
    response = invoke_agent_sync(
        url=url,
        prompt=q_text,
        session_id=session_id,
        headers=headers,
        enabled_sources=enabled_sources,
    )
    elapsed = time.time() - start_time

    if response.startswith("ERROR"):
        judgment = {"correct": False, "extracted_answer": "", "raw_judgment": response}
    else:
        judgment = judge_correctness(
            question=q_text,
            ground_truth=question["answer"],
            predicted=response,
            judge_model=judge_model,
        )

    return {
        "id": question["id"],
        "question": q_text,
        "ground_truth": question["answer"],
        "response": response[:5000],
        "extracted_answer": judgment["extracted_answer"],
        "correct": judgment["correct"],
        "raw_judgment": judgment["raw_judgment"],
        "elapsed_seconds": round(elapsed, 2),
        "metadata": question.get("metadata", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _run_sequential(
    remaining: list[dict],
    total_to_run: int,
    url: str,
    headers: dict[str, str],
    enabled_sources: list[str] | None,
    results_file: Path,
    judge_model: str,
) -> None:
    """Run evaluation sequentially."""
    correct_count = 0

    for i, question in enumerate(remaining, 1):
        q_text = question["question"]
        q_display = q_text[:100] + "..." if len(q_text) > 100 else q_text
        print(f"[{i}/{total_to_run}] {q_display}")

        result = _evaluate_single_question(question, url, headers, enabled_sources, judge_model)
        save_result(results_file, result)

        if result["correct"]:
            correct_count += 1

        status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if result["correct"] else f"{Fore.RED}✗{Style.RESET_ALL}"
        print(
            f"  {status} [{result['elapsed_seconds']:.1f}s] "
            f"Extracted: {result['extracted_answer'][:60]} | "
            f"GT: {question['answer'][:60]}"
        )
        print(
            f"  Running accuracy (this session): {correct_count}/{i} "
            f"({correct_count/i*100:.1f}%)\n"
        )


def _run_parallel(
    remaining: list[dict],
    total_to_run: int,
    url: str,
    headers: dict[str, str],
    enabled_sources: list[str] | None,
    results_file: Path,
    judge_model: str,
    parallel: int,
) -> None:
    """Run evaluation in parallel with ThreadPoolExecutor."""
    import threading

    correct_count = 0
    completed_count = 0
    lock = threading.Lock()

    print(f"Launching {parallel} parallel workers...\n")

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_question = {
            executor.submit(
                _evaluate_single_question, question, url, headers, enabled_sources, judge_model
            ): question
            for question in remaining
        }

        for future in as_completed(future_to_question):
            question = future_to_question[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "id": question["id"],
                    "question": question["question"],
                    "ground_truth": question["answer"],
                    "response": f"ERROR: {e}",
                    "extracted_answer": "",
                    "correct": False,
                    "raw_judgment": f"ERROR: {e}",
                    "elapsed_seconds": 0,
                    "metadata": question.get("metadata", {}),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            with lock:
                save_result(results_file, result)
                completed_count += 1
                if result["correct"]:
                    correct_count += 1

                status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if result["correct"] else f"{Fore.RED}✗{Style.RESET_ALL}"
                q_display = question["question"][:80] + "..." if len(question["question"]) > 80 else question["question"]
                print(
                    f"  {status} [{completed_count}/{total_to_run}] [{result['elapsed_seconds']:.1f}s] "
                    f"{result['extracted_answer'][:50]} | GT: {question['answer'][:50]}"
                )
                if completed_count % 5 == 0 or completed_count == total_to_run:
                    print(
                        f"  >>> Running accuracy: {correct_count}/{completed_count} "
                        f"({correct_count/completed_count*100:.1f}%)\n"
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deployed deep research agent against GAIA and HLE-search benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both benchmarks
  uv run test-scripts/eval-agent.py

  # Quick test with 5 questions per benchmark
  uv run test-scripts/eval-agent.py --max-questions 5

  # Run only GAIA with specific tools
  uv run test-scripts/eval-agent.py --benchmark gaia --tools tavily,nova

  # Resume a previous run
  uv run test-scripts/eval-agent.py --resume results/eval_gaia_20260707.jsonl

  # Use local agent
  uv run test-scripts/eval-agent.py --local --benchmark gaia --max-questions 10

Benchmark comparison from TTD-DR paper (arXiv:2507.16075):
  ┌─────────────────────────────┬────────────┬────────────┐
  │ System                      │ HLE-Search │    GAIA    │
  ├─────────────────────────────┼────────────┼────────────┤
  │ TTD-DR                      │   33.9%    │   69.1%    │
  │ OpenAI Deep Research        │   29.1%    │   67.4%    │
  │ Perplexity Deep Research    │   14.5%    │   54.5%    │
  │ Grok DeeperSearch           │   19.3%    │   47.9%    │
  │ GPT-Researcher              │    2.0%    │   37.7%    │
  │ Open Deep Search            │    3.0%    │   20.9%    │
  └─────────────────────────────┴────────────┴────────────┘
  Metrics: Correctness (%). Source: Table 1, arXiv:2507.16075
        """,
    )

    parser.add_argument(
        "--benchmark",
        choices=["gaia", "hle-search", "both"],
        default="both",
        help="Which benchmark to run (default: both)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions per benchmark (default: all)",
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Comma-separated list of enabled tools (e.g., tavily,nova,arxiv). Default: all configured.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID override (informational, logged in results metadata). "
        "Actual model is configured on the deployed agent.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="anthropic.claude-sonnet-4-20250514-v1:0",
        help="Bedrock model ID for the correctness judge (default: claude-sonnet-4)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous results JSONL file to resume from",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local agent on localhost:8080 (default: remote deployed agent)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results files (default: test-scripts/results/)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel agent invocations (default: 1, sequential)",
    )

    return parser.parse_args()


def setup_remote_connection(stack_cfg: dict) -> tuple[str, dict[str, str]]:
    """
    Set up remote agent connection with Cognito auth.

    Returns (url, headers)
    """
    outputs = stack_cfg["outputs"]
    required = ["CognitoUserPoolId", "CognitoClientId", "RuntimeArn"]
    missing = [k for k in required if k not in outputs]
    if missing:
        print_msg(f"Missing required stack outputs: {', '.join(missing)}", "error")
        sys.exit(1)

    runtime_arn = outputs["RuntimeArn"]
    region = stack_cfg["region"]

    # Authenticate
    print_section("Authentication")
    username = os.environ.get("EVAL_USERNAME") or input("Enter username: ").strip()
    if not username:
        print_msg("Username is required", "error")
        sys.exit(1)
    password = os.environ.get("EVAL_PASSWORD") or getpass.getpass(f"Enter password for {username}: ")

    access_token, _, _ = authenticate_cognito(
        outputs["CognitoUserPoolId"], outputs["CognitoClientId"], username, password
    )

    # Build URL
    endpoint = f"https://bedrock-agentcore.{region}.amazonaws.com"
    escaped_arn = requests.utils.quote(runtime_arn, safe="")
    url = f"{endpoint}/runtimes/{escaped_arn}/invocations?qualifier=DEFAULT"

    headers = {"Authorization": f"Bearer {access_token}"}

    print(f"Runtime ARN: {runtime_arn}")
    print(f"Region: {region}\n")

    return url, headers


def setup_local_connection() -> tuple[str, dict[str, str]]:
    """Set up local agent connection."""
    url = "http://localhost:8080/invocations"
    mock_token = create_mock_jwt("eval-user")
    headers = {"Authorization": f"Bearer {mock_token}"}
    return url, headers


def main():
    print("=" * 60)
    print("AgentCore Deep Research - Benchmark Evaluation")
    print("=" * 60 + "\n")

    args = parse_arguments()

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Parse enabled tools
    enabled_sources = args.tools.split(",") if args.tools else None

    # Set up connection
    if args.local:
        print_msg("Using LOCAL agent (localhost:8080)", "info")
        url, headers = setup_local_connection()
    else:
        print_msg("Using REMOTE deployed agent", "info")
        stack_cfg = get_stack_config()
        url, headers = setup_remote_connection(stack_cfg)

    # Timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run config metadata
    run_config = {
        "timestamp": run_timestamp,
        "model": args.model or "deployed-default",
        "tools": enabled_sources or "all",
        "judge_model": args.judge_model,
        "local": args.local,
        "max_questions": args.max_questions,
    }

    # Save run config
    config_file = output_dir / f"eval_config_{run_timestamp}.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(run_config, f, indent=2)

    all_metrics = {}

    # --- GAIA Benchmark ---
    if args.benchmark in ("gaia", "both"):
        if args.resume and "gaia" in args.resume:
            results_file = Path(args.resume)
        else:
            results_file = output_dir / f"eval_gaia_{run_timestamp}.jsonl"

        questions = load_gaia_dataset()

        metrics = run_evaluation(
            questions=questions,
            benchmark_name="GAIA",
            url=url,
            headers=headers,
            enabled_sources=enabled_sources,
            results_file=results_file,
            judge_model=args.judge_model,
            max_questions=args.max_questions,
            parallel=args.parallel,
        )
        all_metrics["gaia"] = metrics

    # --- HLE-search Benchmark ---
    if args.benchmark in ("hle-search", "both"):
        if args.resume and "hle" in args.resume:
            results_file = Path(args.resume)
        else:
            results_file = output_dir / f"eval_hle_search_{run_timestamp}.jsonl"

        questions = load_hle_dataset(max_search_questions=200)

        metrics = run_evaluation(
            questions=questions,
            benchmark_name="HLE-search",
            url=url,
            headers=headers,
            enabled_sources=enabled_sources,
            results_file=results_file,
            judge_model=args.judge_model,
            max_questions=args.max_questions,
            parallel=args.parallel,
        )
        all_metrics["hle-search"] = metrics

    # --- Final Summary ---
    print_section("EVALUATION RESULTS")

    print(f"Run: {run_timestamp}")
    print(f"Model: {args.model or 'deployed-default'}")
    print(f"Tools: {enabled_sources or 'all configured'}")
    print()

    print("Reference scores (TTD-DR paper, arXiv:2507.16075):")
    print("  ┌─────────────────────────────┬────────────┬────────────┐")
    print("  │ System                      │ HLE-Search │    GAIA    │")
    print("  ├─────────────────────────────┼────────────┼────────────┤")
    print("  │ TTD-DR                      │   33.9%    │   69.1%    │")
    print("  │ OpenAI Deep Research        │   29.1%    │   67.4%    │")
    print("  │ Perplexity Deep Research    │   14.5%    │   54.5%    │")
    print("  │ Grok DeeperSearch           │   19.3%    │   47.9%    │")
    print("  │ GPT-Researcher              │    2.0%    │   37.7%    │")
    print("  │ Open Deep Search            │    3.0%    │   20.9%    │")
    print("  └─────────────────────────────┴────────────┴────────────┘")
    print("  Metrics: Correctness (%). Source: Table 1, arXiv:2507.16075")
    print()

    for benchmark, metrics in all_metrics.items():
        print(f"{'─' * 50}")
        print(f"  {benchmark.upper()}")
        print(f"{'─' * 50}")
        print(f"  Total questions:  {metrics['total']}")
        print(f"  Correct:          {metrics['correct']}")
        print(f"  Errors:           {metrics['errors']}")
        print(f"  Accuracy:         {metrics['accuracy']*100:.1f}%")
        if metrics["errors"] > 0:
            print(f"  Accuracy (excl.): {metrics['accuracy_excl_errors']*100:.1f}%")

        if "per_level" in metrics:
            print(f"  Per level:")
            for level, level_metrics in metrics["per_level"].items():
                print(
                    f"    Level {level}: {level_metrics['correct']}/{level_metrics['total']} "
                    f"({level_metrics['accuracy']*100:.1f}%)"
                )
        print()

    # Save final summary
    summary_file = output_dir / f"eval_summary_{run_timestamp}.json"
    summary = {"config": run_config, "metrics": all_metrics}
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
