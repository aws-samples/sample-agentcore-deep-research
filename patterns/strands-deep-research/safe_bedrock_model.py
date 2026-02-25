"""Safe wrapper around strands BedrockModel that fixes streaming bugs with large tool inputs.

This module addresses two issues in the strands-agents SDK (tested against v1.26.0):

1. Silent JSON parse failure: When large tool inputs (e.g., editor writing report sections)
   arrive via streaming and the JSON is truncated (due to max_tokens or EventStream errors),
   the SDK silently defaults to {}, losing all tool parameters.
   Fix: Monkey-patch handle_content_block_stop to attempt JSON repair before falling back.

2. Unhandled BotoCoreError: The SDK's _stream method only catches ClientError, but
   ConnectionClosedError, ReadTimeoutError, and other BotoCoreError subclasses can occur
   during large streaming responses. These propagate unhandled with no logging.
   Fix: Override _stream to catch and log BotoCoreError.

Usage:
    from safe_bedrock_model import SafeBedrockModel
    model = SafeBedrockModel(model_id="...", streaming=True, ...)
"""

import json
import logging
from typing import Any, Callable, Optional

from botocore.exceptions import BotoCoreError, ClientError
from strands.event_loop import streaming as _streaming_module
from strands.models.bedrock import BedrockModel
from strands.types.content import Messages, SystemContentBlock
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.tools import ToolChoice, ToolSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON Repair for Truncated Tool Inputs
# ---------------------------------------------------------------------------


def _repair_truncated_json(raw: str) -> dict | None:
    """Attempt to repair truncated JSON by closing unclosed strings and brackets.

    This handles the common case where the model hits max_tokens mid-JSON,
    producing syntactically valid JSON except for missing closing characters.

    Args:
        raw: The raw accumulated JSON string that failed json.loads().

    Returns:
        Parsed dict if repair succeeded, None if repair also failed.
    """
    repaired = raw.rstrip()
    if not repaired:
        return None

    # Step 1: Close unclosed string literal
    in_string = False
    i = 0
    while i < len(repaired):
        ch = repaired[i]
        if in_string:
            if ch == "\\":
                i += 2  # skip escaped character
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
        i += 1

    if in_string:
        repaired += '"'

    # Step 2: Close unmatched braces and brackets using a stack
    stack: list[str] = []
    in_str = False
    i = 0
    while i < len(repaired):
        ch = repaired[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]") and stack and stack[-1] == ch:
                stack.pop()
        i += 1

    # Close in LIFO order
    repaired += "".join(reversed(stack))

    # Step 3: Try parsing the repaired JSON
    try:
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
        return None
    except (ValueError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Monkey-Patch for handle_content_block_stop
# ---------------------------------------------------------------------------

_original_handle_content_block_stop = _streaming_module.handle_content_block_stop
_PATCH_APPLIED = False


def _safe_handle_content_block_stop(state: dict[str, Any]) -> dict[str, Any]:
    """Enhanced content block stop handler with JSON repair for truncated tool inputs.

    Wraps the original handle_content_block_stop. Before the original function runs,
    checks if tool input JSON would fail to parse. If so, attempts repair.
    """
    current_tool_use = state.get("current_tool_use", {})

    if current_tool_use and isinstance(current_tool_use.get("input"), str):
        raw_input = current_tool_use["input"]
        if raw_input:
            try:
                json.loads(raw_input)
            except (ValueError, json.JSONDecodeError):
                tool_name = current_tool_use.get("name", "unknown")
                input_len = len(raw_input)
                preview_start = raw_input[:200]
                preview_end = raw_input[-200:] if input_len > 200 else ""

                logger.warning(
                    "[SafeBedrockModel] Tool input JSON parse failed for tool '%s' "
                    "(length: %d chars). Attempting repair. "
                    "Start: %.200s | End: %.200s",
                    tool_name,
                    input_len,
                    preview_start,
                    preview_end,
                )

                repaired = _repair_truncated_json(raw_input)
                if repaired is not None:
                    # Replace the raw string with the repaired JSON string
                    # so the original function's json.loads() will succeed
                    current_tool_use["input"] = json.dumps(repaired)
                    logger.info(
                        "[SafeBedrockModel] Successfully repaired truncated JSON for tool '%s'",
                        tool_name,
                    )
                else:
                    logger.error(
                        "[SafeBedrockModel] Could not repair tool input JSON for '%s'. "
                        "Tool will receive empty input {}.",
                        tool_name,
                    )

    return _original_handle_content_block_stop(state)


def _apply_streaming_patch() -> None:
    """Apply the monkey-patch to handle_content_block_stop. Idempotent."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _streaming_module.handle_content_block_stop = _safe_handle_content_block_stop
    _PATCH_APPLIED = True
    logger.info("[SafeBedrockModel] Applied streaming patch for tool input JSON repair")


# Apply patch at import time
_apply_streaming_patch()


# ---------------------------------------------------------------------------
# SafeBedrockModel Class
# ---------------------------------------------------------------------------


class SafeBedrockModel(BedrockModel):
    """Drop-in replacement for BedrockModel with improved streaming error handling.

    - Catches BotoCoreError and other non-ClientError exceptions in _stream with logging
    - Activates the handle_content_block_stop monkey-patch for JSON repair (via import)
    """

    def _stream(
        self,
        callback: Callable[..., None],
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        tool_choice: ToolChoice | None = None,
    ) -> None:
        """Override _stream to catch broader exceptions with proper logging.

        The parent only catches ClientError. BotoCoreError subclasses like
        ConnectionClosedError and ReadTimeoutError can occur during large
        streaming responses and would otherwise propagate with no context.
        """
        try:
            super()._stream(callback, messages, tool_specs, system_prompt_content, tool_choice)
        except (ModelThrottledException, ContextWindowOverflowException):
            # Already handled by parent, let propagate
            raise
        except ClientError:
            # Already handled by parent with annotations, let propagate
            raise
        except BotoCoreError as e:
            logger.error(
                "[SafeBedrockModel] BotoCoreError during streaming: %s (%s)",
                str(e),
                type(e).__name__,
            )
            raise
        except Exception as e:
            logger.error(
                "[SafeBedrockModel] Unexpected error during streaming: %s (%s)",
                str(e),
                type(e).__name__,
            )
            raise
