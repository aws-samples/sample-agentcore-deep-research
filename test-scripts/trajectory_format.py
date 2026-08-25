#!/usr/bin/env python3
"""
Convert captured agent trajectories into TRL tool-calling SFT format.

The agent streams Bedrock Converse-style messages (see
test-scripts/sft_generate_data.py), which contain `toolUse` and
`toolResult` content blocks. TRL expects a different shape:

    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": ..., "arguments": {...}}}]}
    {"role": "tool", "name": ..., "content": "..."}

plus a `tools` column holding JSON schemas for the available tools.
Ref: https://huggingface.co/docs/trl/v1.10.0/en/dataset_formats#tool-calling

Why trajectories rather than final reports: every paper in this space
(DR-Venus 2604.19859, DeepSearch-World 2607.07820, DeepRubric 2606.17029,
R^2-Searcher 2606.28566) trains on full multi-step trajectories with
observation tokens masked from the loss. Training on final reports alone
teaches report-shaped prose without the research behaviour that produces it,
which is how you get a model that confidently fabricates citations.

Observations are truncated (DeepSearch-World caps them at 8,192 chars) because
a single uncapped trajectory here runs to ~37.6K tokens, and the truncation is
applied to tool *results* only — never to the assistant's own tokens, which are
what the model must learn to produce.
"""

import json
from typing import Any

# Cap on characters kept per tool observation. Observations are ~66% of a raw
# trajectory; the model is not trained on them (they are masked), so detail
# beyond what is needed for context is pure sequence-length cost.
DEFAULT_OBSERVATION_CHARS = 2000


def _extract_text(blocks: Any) -> str:
    """Concatenate text blocks from a Bedrock content list."""
    if isinstance(blocks, str):
        return blocks
    out = []
    for b in blocks or []:
        if isinstance(b, dict) and "text" in b:
            out.append(b["text"])
    return "\n".join(out)


def _tool_result_text(tool_result: dict, max_chars: int) -> str:
    """Flatten a toolResult's content into text, truncated."""
    parts = []
    for b in tool_result.get("content") or []:
        if isinstance(b, dict):
            if "text" in b:
                parts.append(b["text"])
            elif "json" in b:
                parts.append(json.dumps(b["json"]))
            else:
                parts.append(json.dumps(b))
        else:
            parts.append(str(b))
    text = "\n".join(parts)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


def convert_trajectory(
    messages: list[dict],
    observation_chars: int = DEFAULT_OBSERVATION_CHARS,
) -> list[dict]:
    """
    Convert Bedrock Converse messages to TRL tool-calling messages.

    Assistant turns carrying `toolUse` become `tool_calls`; the paired user turn
    carrying `toolResult` becomes a `tool` role message. Tool names are resolved
    via toolUseId, because the result block does not repeat the name.
    """
    id_to_name: dict[str, str] = {}
    converted: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        blocks = msg.get("content") or []
        if isinstance(blocks, str):
            blocks = [{"text": blocks}]

        tool_uses = [
            b["toolUse"] for b in blocks if isinstance(b, dict) and "toolUse" in b
        ]
        tool_results = [
            b["toolResult"] for b in blocks if isinstance(b, dict) and "toolResult" in b
        ]

        if role == "assistant":
            text = _extract_text(blocks)
            if tool_uses:
                calls = []
                for tu in tool_uses:
                    name = tu.get("name") or "unknown"
                    id_to_name[tu.get("toolUseId", "")] = name
                    calls.append(
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": tu.get("input") or {},
                            },
                        }
                    )
                entry: dict[str, Any] = {"role": "assistant", "tool_calls": calls}
                if text:
                    entry["content"] = text
                converted.append(entry)
            elif text:
                converted.append({"role": "assistant", "content": text})

        elif role == "user":
            if tool_results:
                for tr in tool_results:
                    converted.append(
                        {
                            "role": "tool",
                            "name": id_to_name.get(tr.get("toolUseId", ""), "unknown"),
                            "content": _tool_result_text(tr, observation_chars),
                        }
                    )
            else:
                text = _extract_text(blocks)
                if text:
                    converted.append({"role": "user", "content": text})

    return converted


def synthesize_tool_schemas(messages: list[dict]) -> list[dict]:
    """
    Build JSON schemas for the tools actually used in a trajectory.

    TRL's `tools` column feeds the chat template's system prompt. Ideally these
    come from the live MCP tool specs; derived from observed calls they capture
    the true names and argument keys, which is what the template needs to teach
    correct call syntax.
    """
    seen: dict[str, set[str]] = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            fn = c.get("function", {})
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments") or {}
            seen.setdefault(name, set()).update(
                args.keys() if isinstance(args, dict) else []
            )

    schemas = []
    for name, keys in sorted(seen.items()):
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool '{name}' available to the research agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in sorted(keys)},
                        "required": sorted(keys),
                    },
                },
            }
        )
    return schemas


def build_sft_example(
    question: str,
    messages: list[dict],
    observation_chars: int = DEFAULT_OBSERVATION_CHARS,
) -> dict:
    """
    Produce one TRL tool-calling SFT example: {"messages": [...], "tools": [...]}.

    The leading user turn is normalised to the research question so the example
    starts from the task rather than whatever framing the harness used.
    """
    converted = convert_trajectory(messages, observation_chars)
    # Drop any leading user turns from the captured transcript and prepend the
    # canonical question.
    while converted and converted[0]["role"] == "user":
        converted.pop(0)
    converted.insert(0, {"role": "user", "content": question})
    return {"messages": converted, "tools": synthesize_tool_schemas(converted)}


def trajectory_stats(example: dict) -> dict:
    """Summarise an example for filtering and reporting."""
    msgs = example["messages"]
    n_calls = sum(len(m.get("tool_calls") or []) for m in msgs)
    n_obs = sum(1 for m in msgs if m["role"] == "tool")
    chars = len(json.dumps(example))
    final = next(
        (
            m.get("content", "")
            for m in reversed(msgs)
            if m["role"] == "assistant" and m.get("content") and not m.get("tool_calls")
        ),
        "",
    )
    return {
        "messages": len(msgs),
        "tool_calls": n_calls,
        "observations": n_obs,
        "chars": chars,
        "approx_tokens": int(chars / 3.6),
        "final_answer_chars": len(final),
        "tools": len(example.get("tools") or []),
    }


def has_malformed_tool_call(example: dict) -> str | None:
    """
    Detect a tool call the tool itself rejected as malformed.

    Returns a short reason string, or None if the trajectory is clean.

    Why this matters: a malformed tool call sits in an ASSISTANT turn, so it is
    trained on, while the corrective error sits in the masked tool result. The
    model therefore learns to emit the broken call and never sees why it was
    wrong. Measured on 1,963 teacher trajectories, 130 (6.6%) contain an
    `editor(str_replace)` call missing the required `path`, and the student
    subsequently omitted required arguments from `file_write` — 60 failures in a
    98-question eval, two of which stalled the run entirely.

    Detection is on the tool's own validation error rather than a schema we
    maintain, so it stays correct as tools change.
    """
    for msg in example.get("messages", []):
        if msg.get("role") != "tool":
            continue
        head = (msg.get("content") or "")[:400].lower()
        if "validation failed for input parameters" in head or (
            "field required" in head and head.startswith("error")
        ):
            return f"tool '{msg.get('name')}' rejected a malformed call"
    return None


def retruncate_observations(example: dict, max_chars: int) -> dict:
    """
    Shrink already-collected tool observations to fit a sequence-length budget.

    Observations are masked from the loss, so tightening them costs no training
    signal — unlike truncating the sequence itself, which would cut the END of a
    trajectory, exactly where the report is finalised and verified.

    Used to fit trajectories collected at a looser cap into a shorter
    max_length without re-running collection.
    """
    out = json.loads(json.dumps(example))
    for msg in out.get("messages", []):
        if msg.get("role") == "tool":
            content = msg.get("content") or ""
            if len(content) > max_chars:
                msg["content"] = content[:max_chars] + "\n...[truncated]"
    return out
