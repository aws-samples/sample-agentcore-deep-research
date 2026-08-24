"""
Compatibility shims for the Strands Agents SDK.

Applied at import time by the agent entry points. Each shim is guarded so it
becomes a no-op once upstream fixes the underlying issue, and logs what it did
so a silently-rotted patch is visible in the runtime logs.

This replaces an earlier approach that rewrote the installed
`strands/event_loop/streaming.py` during the Docker build. That was worse in
three ways: it mutated a third-party package inside site-packages, it required
deleting and recompiling `__pycache__` because BuildKit normalises layer mtimes
so Python would otherwise load stale bytecode, and it failed silently if the
upstream source changed.
"""

import logging

logger = logging.getLogger(__name__)


def _patch_tool_use_input_concat() -> bool:
    """
    Tolerate non-string tool-use input deltas while streaming.

    `strands.event_loop.streaming.handle_content_block_delta` accumulates
    tool-call arguments with:

        state["current_tool_use"]["input"] += tool_use_delta.get("input", "")

    Some models (observed with Claude Haiku 4.5) emit an int rather than a JSON
    string fragment, which raises:

        TypeError: can only concatenate str (not "int") to str

    and aborts the whole run. Coerce both sides to str instead.

    Returns True if the shim was applied, False if it was not needed.
    """
    try:
        from strands.event_loop import streaming
    except ImportError:  # pragma: no cover - SDK always present in the container
        return False

    original = getattr(streaming, "handle_content_block_delta", None)
    if original is None or getattr(original, "_adr_shimmed", False):
        return False

    def handle_content_block_delta(event, state, **kwargs):  # type: ignore[no-untyped-def]
        delta = (event or {}).get("delta", {})
        if "toolUse" in delta:
            tool_use_delta = delta["toolUse"]
            if "input" in tool_use_delta and not isinstance(
                tool_use_delta["input"], str
            ):
                tool_use_delta["input"] = str(tool_use_delta["input"])
            current = state.get("current_tool_use")
            if isinstance(current, dict) and not isinstance(
                current.get("input", ""), str
            ):
                current["input"] = str(current["input"])
        return original(event, state, **kwargs)

    handle_content_block_delta._adr_shimmed = True  # type: ignore[attr-defined]
    streaming.handle_content_block_delta = handle_content_block_delta
    return True


def apply_shims() -> None:
    """Apply all compatibility shims, logging which were needed."""
    applied = []
    if _patch_tool_use_input_concat():
        applied.append("tool_use_input_concat")
    if applied:
        logger.info("[COMPAT] Applied Strands shims: %s", ", ".join(applied))
        print(f"[COMPAT] Applied Strands shims: {', '.join(applied)}", flush=True)
    else:
        print("[COMPAT] No Strands shims needed", flush=True)
