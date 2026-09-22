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
import math
import os
import re
import statistics
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

Determine if the predicted answer is correct. The predicted answer does not need to match the ground truth exactly, but must be semantically equivalent or contain the correct answer. For numerical answers, minor formatting differences are acceptable. For multiple choice, the letter must match.

You MUST respond with ONLY the following XML tag and nothing else:
<judgement>correct</judgement>
or
<judgement>incorrect</judgement>
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

    print_msg(
        f"Loaded {len(questions)} GAIA questions (excluding file-based)", "success"
    )
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
) -> tuple[str, list[str]]:
    """
    Invoke the deployed agent; return (report, tool_observations).

    Strands emits one `message` event per completed message, so tool results are
    recoverable from the stream. They are needed to verify grounding: without them
    the judge can only assess whether claims *look* attributed, and the citation
    component can only check that URLs are well-formed.

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
        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=timeout
        )

        if response.status_code != 200:
            return f"ERROR: HTTP {response.status_code}: {response.text[:500]}", []

        # Collect full text response from streaming + look for report URL
        full_text = ""
        report_url = None
        observations: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw_line = line[6:]
            try:
                chunk = json.loads(raw_line)

                # Strands: text token
                if isinstance(chunk.get("data"), str):
                    full_text += chunk["data"]

                # Strands: one completed message per event. Tool results arrive as
                # role="user" messages carrying toolResult blocks.
                msg = chunk.get("message")
                if isinstance(msg, dict):
                    for blk in msg.get("content") or []:
                        tr = blk.get("toolResult") if isinstance(blk, dict) else None
                        for inner in (tr or {}).get("content") or []:
                            txt = inner.get("text") if isinstance(inner, dict) else None
                            if isinstance(txt, str) and txt:
                                observations.append(txt)

                # LangGraph: AIMessageChunk with content array
                elif chunk.get("type") == "AIMessageChunk" and isinstance(
                    chunk.get("content"), list
                ):
                    for block in chunk["content"]:
                        if block.get("type") == "text" and block.get("text"):
                            full_text += block["text"]

            except (json.JSONDecodeError, KeyError):
                pass

            # Look for report URL in raw line
            if "[REPORT_URL:" in raw_line:
                match = re.search(r"\[REPORT_URL:(https://[^\]]+)\]", raw_line)
                if match:
                    report_url = match.group(1)

        # If we found a report URL, download the actual report
        if report_url:
            try:
                report_resp = requests.get(report_url, timeout=30)
                if (
                    report_resp.status_code == 200
                    and len(report_resp.text.strip()) > 500
                ):
                    return report_resp.text.strip(), observations
            except Exception:
                pass

        # Fallback: stream text if it looks like a report
        if full_text.strip().startswith("#") and len(full_text.strip()) > 1000:
            return full_text.strip(), observations

        # Also check stream text for report URL
        if "[REPORT_URL:" in full_text:
            match = re.search(r"\[REPORT_URL:(https://[^\]]+)\]", full_text)
            if match:
                try:
                    report_resp = requests.get(match.group(1), timeout=30)
                    if (
                        report_resp.status_code == 200
                        and len(report_resp.text.strip()) > 500
                    ):
                        return report_resp.text.strip(), observations
                except Exception:
                    pass

        return full_text.strip(), observations

    except requests.exceptions.Timeout:
        return "ERROR: Request timed out", []
    except requests.exceptions.ConnectionError:
        return "ERROR: Connection failed", []
    except Exception as e:
        return f"ERROR: {e}", []


# ---------------------------------------------------------------------------
# Evaluation (correctness judge)
# ---------------------------------------------------------------------------


def judge_correctness(
    question: str,
    ground_truth: str,
    predicted: str,
    judge_model: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0",
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
    extraction_prompt = ANSWER_EXTRACTION_PROMPT.format(
        question=question, response=predicted
    )

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
        return {
            "correct": False,
            "extracted_answer": "NO_ANSWER",
            "raw_judgment": "NO_ANSWER",
        }

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
                "max_tokens": 100,
                "temperature": 0.0,
                "messages": [
                    {"role": "user", "content": judge_prompt},
                    {"role": "assistant", "content": "<judgement>"},
                ],
            }
        ),
    )
    judge_body = json.loads(judge_response["body"].read())
    judgment = "<judgement>" + judge_body["content"][0]["text"].strip()

    # Extract verdict from <judgement> tags
    match = re.search(r"<judgement>(.*?)</judgement>", judgment, re.IGNORECASE)
    if match:
        verdict = match.group(1).strip().lower()
    else:
        # Fallback: check if response contains "incorrect" or "correct" as keywords
        text_lower = judgment.lower()
        if "incorrect" in text_lower:
            verdict = "incorrect"
        elif "correct" in text_lower:
            verdict = "correct"
        else:
            verdict = "unknown"

    return {
        "correct": verdict == "correct",
        "extracted_answer": extracted_answer,
        "raw_judgment": judgment,
    }


# ---------------------------------------------------------------------------
# Rubric-based evaluation (report quality scoring for SFT/RL training eval)
#
# The rubric itself lives in patterns/strands-deep-research/research_rubric.py
# so that this offline eval metric and the RL training reward in rl_app.py are
# the same code. It is loaded by path because rl_app.py runs inside the agent
# container (where it is a sibling module) while this script runs locally.
# ---------------------------------------------------------------------------

_SHARED_RUBRIC = None


def _load_shared_rubric():
    """Load the shared rubric module from the agent pattern directory."""
    global _SHARED_RUBRIC
    if _SHARED_RUBRIC is None:
        import importlib.util

        path = (
            Path(__file__).parent.parent
            / "patterns"
            / "strands-deep-research"
            / "research_rubric.py"
        )
        spec = importlib.util.spec_from_file_location("research_rubric", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SHARED_RUBRIC = module
    return _SHARED_RUBRIC


def score_report_rubric(
    question: str,
    report: str,
    judge_model: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    retrieved_urls: set | None = None,
    observations: list[str] | None = None,
) -> dict:
    """
    Score a research report using the shared rubric (see research_rubric.py).

    The rubric lives in patterns/strands-deep-research/research_rubric.py so
    that this offline metric and the RL training reward are literally the same
    code rather than two copies kept in sync by hand.

    Pass `retrieved_urls` (URLs actually returned by tool calls) to enable the
    grounding gate, which discards fabricated citations.

    Pass `observations` (raw tool results) to let the judge check grounding against
    what the tools actually returned. Without them the judge sees only the report,
    so its grounding criterion measures whether claims *look* attributed rather than
    whether they are -- the weakness DR Tulu (arXiv 2511.19399) addresses by showing
    the judge its search context. `retrieved_urls` is derived from them when not
    supplied explicitly.

    Returns dict with total (0-1), rubric, citation, format, and per_criterion.
    """
    import boto3 as _boto3

    rubric_mod = _load_shared_rubric()

    if observations and retrieved_urls is None:
        retrieved_urls = set(
            re.findall(r"https?://[^\s)\]\">]+", "\n".join(observations))
        )

    if not report or report.startswith("ERROR") or len(report) < 100:
        return {
            "total": 0.0,
            "rubric": 0.0,
            "citation": 0.0,
            "format": 0.0,
            "per_criterion": {},
        }

    bedrock = _boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    # The WHOLE report goes to the judge. An earlier version truncated at 6K
    # chars, which hid most of a 13-19K char report and penalised length; a later
    # 24K cap was equally silent, just further out. If a report ever exceeds the
    # judge's context the model raises, which is visible — unlike quietly scoring
    # a partial report and reporting the number as if it were complete.
    #
    # Judge failures are likewise NOT caught. Scoring them 0.0 would silently
    # depress a model's mean with no trace in the results, the same class of
    # problem as counting an empty response as a quality score of zero. Letting
    # it raise records the question as an error and the run reports fewer
    # questions than requested, which is visible.
    rubric_reward, per_criterion = rubric_mod.score_rubric_with_judge(
        question, report, bedrock, judge_model, observations=observations
    )

    citation_reward = rubric_mod.score_citations(report, retrieved_urls)
    format_reward = rubric_mod.score_format(report)
    total = rubric_mod.combine(rubric_reward, citation_reward, format_reward)

    return {
        "total": round(total, 4),
        "rubric": round(rubric_reward, 4),
        "citation": round(citation_reward, 4),
        "format": round(format_reward, 4),
        "per_criterion": per_criterion,
    }


def write_comparison_plot(
    rows: list[tuple[str, float, list[dict]]], out_path: Path
) -> None:
    """Bar chart of mean rubric score per model, with 95% CI error bars.

    Error bars are not decoration: at n=98 the resolvable difference is ~0.065, so
    bars without them invite reading noise as a result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    errs = []
    for _, _, recs in rows:
        vals = [x["scores"]["total"] for x in recs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        errs.append(1.96 * sd / math.sqrt(len(vals)) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(1.5 * len(rows) + 3, 5))
    bars = ax.bar(
        labels,
        means,
        yerr=errs,
        capsize=5,
        color="#4A7EBB",
        edgecolor="black",
        linewidth=0.6,
    )
    for b, m, n in zip(bars, means, [len(r[2]) for r in rows], strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{m:.3f}\nn={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Rubric score (0-1)")
    ax.set_ylim(0, max(m + e for m, e in zip(means, errs, strict=False)) * 1.25)
    ax.set_title("Deep research report quality (identical harness, same judge)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print_msg(f"Plot saved to: {out_path}", "success")


def write_compute_plot(
    rows: list[tuple[str, float, list[dict]]], hours: dict, out_path: Path
) -> None:
    """Eval score against cumulative training compute, in measured GPU-hours.

    One continuous line: base (no training) -> SFT -> successive RL checkpoints, so
    the marginal return of each stage is visible rather than implied.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for label, mean, recs in rows:
        if label not in hours:
            continue
        vals = [x["scores"]["total"] for x in recs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        err = 1.96 * sd / math.sqrt(len(vals)) if vals else 0.0
        pts.append((float(hours[label]), mean, err, label))
    pts.sort()
    if not pts:
        print_msg(
            "No --compute-hours mapping matched any tag; skipping compute plot",
            "warning",
        )
        return

    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    e = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x,
        y,
        yerr=e,
        marker="o",
        capsize=4,
        color="#4A7EBB",
        linewidth=1.8,
        markersize=7,
    )
    for xi, yi, _, lab in pts:
        ax.annotate(
            lab, (xi, yi), textcoords="offset points", xytext=(6, -12), fontsize=8
        )
    ax.set_xlabel("Cumulative training compute (GPU-hours, measured)")
    ax.set_ylabel("Rubric score (0-1)")
    ax.set_title("Report quality vs training compute")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print_msg(f"Compute plot saved to: {out_path}", "success")


def load_rubric_dataset(questions_path: str | None = None) -> list[dict]:
    """Load research questions for rubric-based evaluation."""
    # Comparing models is only meaningful on IDENTICAL questions. Generate this
    # file once with sft_generate_questions.py and then leave it alone for the
    # lifetime of a comparison — regenerating with a different seed silently
    # swaps the exam, which previously left different models scored on entirely
    # different question sets. Pass --rubric-questions to pin an explicit path.
    path = Path(questions_path or "test-scripts/results/sft_eval_questions.jsonl")
    if not path.exists():
        print_msg(f"Rubric eval questions not found: {path}", "error")
        print_msg(
            "Generate one with: uv run test-scripts/sft_generate_questions.py "
            "--count 400 --eval-count 100 --seed 42",
            "info",
        )
        print_msg(
            "Then keep it fixed: every model in a comparison must see the "
            "same questions, and check it does not overlap your training set.",
            "info",
        )
        sys.exit(1)

    questions = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                if isinstance(row.get("prompt"), list):
                    prompt_text = row["prompt"][0]["content"]
                else:
                    prompt_text = row.get("metadata", {}).get("prompt", "")
                questions.append(
                    {
                        "id": row.get("metadata", {}).get("prompt", prompt_text)[:50],
                        "question": prompt_text,
                        "enabled_sources": row.get(
                            "enabled_sources", ["tavily", "nova"]
                        ),
                        "metadata": {
                            "source": "rubric",
                            "domain": row.get("metadata", {}).get("domain", "unknown"),
                        },
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

    print_msg(f"Loaded {len(questions)} rubric eval questions", "success")
    return questions


def rescore_from_file(
    source: Path,
    results_file: Path,
    judge_model: str,
    parallel: int = 8,
) -> dict:
    """
    Re-judge reports stored in a previous eval file, without new rollouts.

    Used to change judge model or rubric without paying for rollouts again. Records
    written before observations were persisted can only be re-scored blind, so the
    grounding criterion is not comparable across the two record formats; the count of
    records carrying observations is reported so that is visible rather than implicit.
    """
    rows = [json.loads(line) for line in source.open() if line.strip()]
    with_obs = sum(1 for r in rows if r.get("observations"))
    print_msg(
        f"Re-scoring {len(rows)} reports from {source.name} with {judge_model} "
        f"({with_obs}/{len(rows)} carry observations)",
        "info",
    )

    def score_one(row: dict) -> dict:
        scores = score_report_rubric(
            row["question"],
            row["response"],
            judge_model,
            observations=row.get("observations"),
        )
        return {**row, "scores": scores, "rescored_with": judge_model}

    out = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(score_one, r): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            out.append(fut.result())
            if i % 10 == 0 or i == len(rows):
                print_msg(f"  re-scored {i}/{len(rows)}", "info")

    with results_file.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    keys = ("total", "rubric", "citation", "format")
    return {
        "total_questions": len(out),
        "observations_present": with_obs,
        "judge_model": judge_model,
        **{f"mean_{k}": sum(r["scores"][k] for r in out) / len(out) for k in keys},
    }


def run_rubric_evaluation(
    questions: list[dict],
    url: str,
    headers: dict[str, str],
    results_file: Path,
    judge_model: str,
    max_questions: int | None = None,
    parallel: int = 1,
    auth_refresh_fn=None,
) -> dict:
    """Run rubric-based evaluation: invoke agent, score reports with rubric."""
    import threading as _threading

    # Token refresh state
    _token_lock = _threading.Lock()
    _current_headers = dict(headers)
    _last_refresh = [time.time()]

    def get_headers():
        """Get current auth headers, refreshing if needed (every 50 min)."""
        with _token_lock:
            if auth_refresh_fn and (time.time() - _last_refresh[0]) > 3000:
                try:
                    new_token = auth_refresh_fn()
                    _current_headers["Authorization"] = f"Bearer {new_token}"
                    _last_refresh[0] = time.time()
                    print_msg("Token refreshed", "info")
                except Exception as e:
                    print_msg(f"Token refresh failed: {e}", "error")
            return dict(_current_headers)

    # Resume support
    completed_questions = set()
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # A failed rollout is not a result. Counting one as complete bakes a
                # zero into the mean and makes an infrastructure failure -- expired
                # credentials, a dropped connection -- look like a bad model. Leave
                # these out of the completed set so a resume retries them.
                resp = r.get("response") or ""
                if resp.startswith("ERROR") or len(resp) < 500:
                    continue
                completed_questions.add(r.get("question", ""))
        if completed_questions:
            print_msg(f"Resuming: {len(completed_questions)} already scored", "info")

    remaining = [q for q in questions if q["question"] not in completed_questions]
    if max_questions:
        remaining = remaining[: max(0, max_questions - len(completed_questions))]

    if remaining:
        print_section("Running Rubric Evaluation")
        print(f"Questions: {len(remaining)}")
        print(f"Parallel: {parallel}\n")

        def eval_one(question):
            q_text = question["question"]
            session_id = generate_session_id()
            start_time = time.time()
            response, observations = invoke_agent_sync(
                url=url,
                prompt=q_text,
                session_id=session_id,
                headers=get_headers(),
                enabled_sources=question.get("enabled_sources"),
            )
            elapsed = time.time() - start_time
            scores = score_report_rubric(
                q_text, response, judge_model, observations=observations
            )
            return {
                "question": q_text,
                "response_length": len(response),
                # Persist the report itself. Without it, a rubric change forces
                # a full re-run of every baseline just to re-score, and there
                # is no way to audit qualitatively what the model produced.
                "response": response,
                # Persist tool observations too. The grounding criterion is unscoreable
                # without them, so a record lacking them can only ever be re-scored by a
                # blind judge -- which is the weakness the grounding gate exists to fix.
                "observations": observations,
                "scores": scores,
                "elapsed_seconds": round(elapsed, 2),
                "metadata": question.get("metadata", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(eval_one, q): q for q in remaining}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    save_result(results_file, result)
                    status = (
                        f"{Fore.GREEN}✓{Style.RESET_ALL}"
                        if result["scores"]["total"] > 0.5
                        else f"{Fore.YELLOW}○{Style.RESET_ALL}"
                    )
                    print(
                        f"  {status} [{i}/{len(remaining)}] score={result['scores']['total']:.3f} "
                        f"({result['metadata'].get('domain', '?')}) [{result['elapsed_seconds']:.0f}s]"
                    )
                except Exception as e:
                    print(
                        f"  {Fore.RED}✗{Style.RESET_ALL} [{i}/{len(remaining)}] Error: {e}"
                    )

    # Compute metrics
    results = []
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not results:
        return {"total": 0, "mean_score": 0.0}

    scores = [r["scores"]["total"] for r in results]
    rubrics = [r["scores"]["rubric"] for r in results]
    citations = [r["scores"]["citation"] for r in results]
    formats = [r["scores"]["format"] for r in results]

    metrics = {
        "total": len(results),
        "mean_score": round(sum(scores) / len(scores), 4),
        "mean_rubric": round(sum(rubrics) / len(rubrics), 4),
        "mean_citation": round(sum(citations) / len(citations), 4),
        "mean_format": round(sum(formats) / len(formats), 4),
        # Empty responses mean the agent produced nothing at all, which is an
        # infrastructure failure rather than a quality score of zero. Surfaced in
        # the summary so a failed run cannot masquerade as a real measurement.
        "empty_responses": sum(
            1 for r in results if not (r.get("response") or "").strip()
        ),
    }

    # Per-domain
    domains = {}
    for r in results:
        d = r.get("metadata", {}).get("domain", "unknown")
        domains.setdefault(d, []).append(r["scores"]["total"])
    metrics["per_domain"] = {
        d: round(sum(s) / len(s), 4) for d, s in sorted(domains.items())
    }

    return metrics


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
        "accuracy_excl_errors": correct / (total - errors)
        if (total - errors) > 0
        else 0.0,
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
        remaining = remaining[: max(0, max_questions - len(completed_ids))]

    total_to_run = len(remaining)
    if total_to_run == 0:
        print_msg("All questions already completed!", "success")
        if results_file.exists():
            return compute_metrics(results_file)
        return {
            "total": 0,
            "correct": 0,
            "errors": 0,
            "accuracy": 0.0,
            "accuracy_excl_errors": 0.0,
        }

    print_section(f"Running {benchmark_name} Evaluation")
    print(f"Questions to evaluate: {total_to_run}")
    print(f"Results file: {results_file}")
    print(f"Enabled sources: {enabled_sources or 'all'}")
    print(f"Parallel workers: {parallel}")
    print()

    if parallel <= 1:
        # Sequential execution (original behavior)
        _run_sequential(
            remaining,
            total_to_run,
            url,
            headers,
            enabled_sources,
            results_file,
            judge_model,
        )
    else:
        # Parallel execution with ThreadPoolExecutor
        _run_parallel(
            remaining,
            total_to_run,
            url,
            headers,
            enabled_sources,
            results_file,
            judge_model,
            parallel,
        )

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
    response, _observations = invoke_agent_sync(
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
        # Full response, not a 5,000-char slice: the saved record is what any
        # later re-scoring or inspection reads, so truncating it silently
        # discards evidence.
        "response": response,
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

        result = _evaluate_single_question(
            question, url, headers, enabled_sources, judge_model
        )
        save_result(results_file, result)

        if result["correct"]:
            correct_count += 1

        status = (
            f"{Fore.GREEN}✓{Style.RESET_ALL}"
            if result["correct"]
            else f"{Fore.RED}✗{Style.RESET_ALL}"
        )
        print(
            f"  {status} [{result['elapsed_seconds']:.1f}s] "
            f"Extracted: {result['extracted_answer'][:60]} | "
            f"GT: {question['answer'][:60]}"
        )
        print(
            f"  Running accuracy (this session): {correct_count}/{i} "
            f"({correct_count / i * 100:.1f}%)\n"
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
                _evaluate_single_question,
                question,
                url,
                headers,
                enabled_sources,
                judge_model,
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

                status = (
                    f"{Fore.GREEN}✓{Style.RESET_ALL}"
                    if result["correct"]
                    else f"{Fore.RED}✗{Style.RESET_ALL}"
                )
                print(
                    f"  {status} [{completed_count}/{total_to_run}] [{result['elapsed_seconds']:.1f}s] "
                    f"{result['extracted_answer'][:50]} | GT: {question['answer'][:50]}"
                )
                if completed_count % 5 == 0 or completed_count == total_to_run:
                    print(
                        f"  >>> Running accuracy: {correct_count}/{completed_count} "
                        f"({correct_count / completed_count * 100:.1f}%)\n"
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
        choices=["gaia", "hle-search", "rubric", "both"],
        default="both",
        help="Which benchmark to run: gaia, hle-search, rubric (report quality), or both (gaia+hle)",
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
        "--rescore",
        type=str,
        default=None,
        help="Path to an existing eval_rubric_*.jsonl; re-judge its stored reports "
        "instead of running the agent (no rollouts, no endpoint needed)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="Bedrock model ID for the correctness judge (default: claude-haiku-4.5)",
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
        "--tag",
        type=str,
        default=None,
        help="Tag for results filenames (e.g., 'baseline-qwen', 'frontier-haiku'). "
        "Appended to output filenames for easy comparison.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel agent invocations (default: 1, sequential)",
    )
    parser.add_argument(
        "--runtime-arn",
        type=str,
        default=None,
        help="Override the RuntimeArn to eval a different agent (e.g., fine-tuned agent)",
    )
    parser.add_argument(
        "--rubric-questions",
        type=str,
        default=None,
        help="Path to rubric eval questions JSONL (default: test-scripts/results/sft_eval_questions.jsonl). "
        "Only used with --benchmark rubric.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="With --compare: write a bar chart (with 95%% CIs) to this filename in results/.",
    )
    parser.add_argument(
        "--plot-compute",
        type=str,
        default=None,
        help="With --compare: write a score-vs-GPU-hours curve to this filename in results/.",
    )
    parser.add_argument(
        "--compute-hours",
        action="append",
        default=[],
        metavar="LABEL=HOURS",
        help="Cumulative GPU-hours for a plot label, used by --plot-compute. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="TAG=NAME",
        help="Display name for a tag in --plot output. Repeatable.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare rubric results by tags (comma-separated, e.g., 'haiku-baseline,sft-model'). "
        "Prints a comparison table and exits.",
    )

    return parser.parse_args()


def setup_remote_connection(
    stack_cfg: dict, runtime_arn_override: str | None = None
) -> tuple[str, dict[str, str]]:
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

    runtime_arn = runtime_arn_override or outputs["RuntimeArn"]
    region = stack_cfg["region"]

    # Authenticate
    print_section("Authentication")
    username = os.environ.get("EVAL_USERNAME") or input("Enter username: ").strip()
    if not username:
        print_msg("Username is required", "error")
        sys.exit(1)
    password = os.environ.get("EVAL_PASSWORD") or getpass.getpass(
        f"Enter password for {username}: "
    )

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

    # Compare mode (no agent invocation needed)
    if args.compare:
        tags = [t.strip() for t in args.compare.split(",")]
        args.labels = dict(kv.split("=", 1) for kv in args.label)
        plot_rows: list[tuple[str, float, list[dict]]] = []
        print(
            f"\n{'Tag':<25} {'Total':>8} {'Rubric':>8} {'Citation':>8} {'Format':>8} {'N':>5}"
        )
        print("-" * 65)
        for tag in tags:
            matches = sorted(output_dir.glob(f"eval_rubric_*_{tag}.jsonl"))
            if not matches:
                print(f"{tag:<25} (not found)")
                continue
            results = []
            with open(matches[-1]) as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not results:
                continue
            total = sum(r["scores"]["total"] for r in results) / len(results)
            rubric = sum(r["scores"]["rubric"] for r in results) / len(results)
            citation = sum(r["scores"]["citation"] for r in results) / len(results)
            fmt = sum(r["scores"]["format"] for r in results) / len(results)
            print(
                f"{tag:<25} {total:>8.3f} {rubric:>8.3f} {citation:>8.3f} {fmt:>8.3f} {len(results):>5}"
            )
            plot_rows.append((args.labels.get(tag, tag), total, results))
        print()
        if args.plot and plot_rows:
            write_comparison_plot(plot_rows, output_dir / args.plot)
        if args.plot_compute and plot_rows:
            write_compute_plot(
                plot_rows,
                dict(kv.split("=", 1) for kv in args.compute_hours),
                output_dir / args.plot_compute,
            )
        return

    # Parse enabled tools
    enabled_sources = args.tools.split(",") if args.tools else None

    # Set up connection. --rescore re-judges stored reports, so it needs no runtime and
    # no Cognito token; authenticating anyway would fail whenever creds have gone stale.
    auth_refresh_fn = None
    url, headers = "", {}
    if args.rescore:
        print_msg(f"Re-scoring stored reports from {args.rescore}", "info")
    elif args.local:
        print_msg("Using LOCAL agent (localhost:8080)", "info")
        url, headers = setup_local_connection()
    else:
        print_msg("Using REMOTE deployed agent", "info")
        stack_cfg = get_stack_config()
        url, headers = setup_remote_connection(stack_cfg, args.runtime_arn)

        # Create token refresh function for long-running evals
        _cognito_cfg = stack_cfg["outputs"]
        _username = os.environ.get("EVAL_USERNAME", "")
        _password = os.environ.get("EVAL_PASSWORD", "")
        if _username and _password:

            def auth_refresh_fn():
                token, _, _ = authenticate_cognito(
                    _cognito_cfg["CognitoUserPoolId"],
                    _cognito_cfg["CognitoClientId"],
                    _username,
                    _password,
                )
                return token

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

    # Construct tag suffix for filenames
    tag_suffix = f"_{args.tag}" if args.tag else ""

    # --- Re-score only: no rollouts, no dataset, no runtime ---
    if args.rescore:
        results_file = output_dir / f"eval_rubric_{run_timestamp}{tag_suffix}.jsonl"
        all_metrics["rubric"] = rescore_from_file(
            source=Path(args.rescore),
            results_file=results_file,
            judge_model=args.judge_model,
            parallel=args.parallel,
        )
        print_section("RE-SCORE RESULTS")
        for k, v in all_metrics["rubric"].items():
            print(f"  {k}: {v}")
        print(f"\nSaved to {results_file}")
        return

    # --- GAIA Benchmark ---
    if args.benchmark in ("gaia", "both"):
        if args.resume and "gaia" in args.resume:
            results_file = Path(args.resume)
        else:
            results_file = output_dir / f"eval_gaia_{run_timestamp}{tag_suffix}.jsonl"

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
            results_file = (
                output_dir / f"eval_hle_search_{run_timestamp}{tag_suffix}.jsonl"
            )

        questions = load_hle_dataset(max_search_questions=args.max_questions or 200)

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

    # --- Rubric Benchmark (report quality) ---
    elif args.benchmark == "rubric":
        # Resume matters here because a full run can outlast a credential lifetime.
        if args.resume and "rubric" in args.resume:
            results_file = Path(args.resume)
        else:
            results_file = output_dir / f"eval_rubric_{run_timestamp}{tag_suffix}.jsonl"
        questions = load_rubric_dataset(args.rubric_questions)

        metrics = run_rubric_evaluation(
            questions=questions,
            url=url,
            headers=headers,
            results_file=results_file,
            judge_model=args.judge_model,
            max_questions=args.max_questions,
            parallel=args.parallel,
            auth_refresh_fn=auth_refresh_fn,
        )
        all_metrics["rubric"] = metrics

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

        # A run where the agent returned nothing is an infrastructure failure,
        # not a score of zero. Reporting 0.000 here is how a meaningless number
        # ends up in a results table and gets compared against real ones, so say
        # so loudly instead. Causes seen in practice: the runtime still swapping
        # images after a deploy, an invalid model identifier, or a missing
        # environment variable on the runtime.
        empty = metrics.get("empty_responses", 0)
        n_total = metrics.get("total", 0)
        if empty and n_total:
            pct = 100 * empty / n_total
            label = "ALL" if empty == n_total else f"{empty}/{n_total}"
            print(
                f"  ⚠ WARNING: {label} responses were empty ({pct:.0f}%). "
                "Scores below are NOT valid."
            )
            print(
                "    Check: runtime status READY and fully swapped (idle ~15 "
                "min after deploy), model_id valid, runtime env vars present."
            )
            print()

        if benchmark == "rubric":
            # Rubric eval uses quality scores, not correctness
            print(f"  Total questions:  {metrics['total']}")
            print(f"  Mean score:       {metrics.get('mean_score', 0.0):.3f}")
            print(f"    Rubric:         {metrics.get('mean_rubric', 0.0):.3f}")
            print(f"    Citation:       {metrics['mean_citation']:.3f}")
            print(f"    Format:         {metrics['mean_format']:.3f}")
            if "per_domain" in metrics:
                print("  Per domain:")
                for domain, score in metrics["per_domain"].items():
                    print(f"    {domain:<15} {score:.3f}")
        else:
            # Correctness-based benchmarks (GAIA, HLE-search)
            print(f"  Total questions:  {metrics['total']}")
            print(f"  Correct:          {metrics['correct']}")
            print(f"  Errors:           {metrics['errors']}")
            print(f"  Accuracy:         {metrics['accuracy'] * 100:.1f}%")
            if metrics["errors"] > 0:
                print(
                    f"  Accuracy (excl.): {metrics['accuracy_excl_errors'] * 100:.1f}%"
                )
            if "per_level" in metrics:
                print("  Per level:")
                for level, level_metrics in metrics["per_level"].items():
                    print(
                        f"    Level {level}: {level_metrics['correct']}/{level_metrics['total']} "
                        f"({level_metrics['accuracy'] * 100:.1f}%)"
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
