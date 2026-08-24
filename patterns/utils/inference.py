# Copyright Amazon.com, Inc. or its affiliates.
# AWS Content under the AWS Enterprise Agreement or
# AWS Customer Agreement (as applicable).

import os
from typing import Any

import botocore

BEDROCK_READ_TIMEOUT = 600
BEDROCK_CONNECT_TIMEOUT = 600
BEDROCK_MAX_ATTEMPTS = 3
BEDROCK_MAX_CONNECTIONS = 10

MAX_TOKENS = 64_000
THINKING_TOKENS = 2_000
TEMPERATURE = 0.0

# Per-model output-token ceilings, matched as substrings of the model id.
#
# Bedrock rejects any request whose maxTokens exceeds the model's limit with
# ValidationException ("The maximum tokens you requested exceeds the model
# limit of N"). The agent then returns nothing, which looks identical to a
# model that is simply incapable — Nova Micro scored a spurious 0.000 across a
# whole eval run for this reason. Clamp instead of failing.
MODEL_MAX_OUTPUT_TOKENS = {
    "nova-micro": 10_000,
    "nova-lite": 10_000,
    "nova-2-lite": 10_000,
    "nova-pro": 10_000,
    "nova-premier": 32_000,
    "llama3-1": 8_192,
    "llama3-2": 8_192,
    "llama3-3": 8_192,
    "llama4": 8_192,
    "mistral": 8_192,
    "deepseek": 32_768,
    # Claude 3 generation caps output at 4,096 tokens (Claude 3.5 at 8,192).
    # Without these entries the default (64,000) is sent and Bedrock rejects the
    # request outright, which is indistinguishable from an incapable model. The
    # agent writes reports incrementally — median 929 tokens per turn — so a
    # 4,096 per-call ceiling is workable.
    "claude-3-haiku": 4_096,
    "claude-3-sonnet": 4_096,
    "claude-3-opus": 4_096,
    "claude-3-5-haiku": 8_192,
    "claude-3-5-sonnet": 8_192,
}


def get_max_output_tokens(model_id: str | None) -> int:
    """
    Resolve a safe maxTokens for the given model.

    Order of precedence: MAX_OUTPUT_TOKENS env override, then the per-model
    ceiling above, then the default. Always returns a value the model accepts.
    """
    override = os.environ.get("MAX_OUTPUT_TOKENS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    if model_id:
        lowered = model_id.lower()
        for fragment, limit in MODEL_MAX_OUTPUT_TOKENS.items():
            if fragment in lowered:
                return limit
    return MAX_TOKENS


# Model families that reject tool use when streaming is enabled. Bedrock raises
# ValidationException("This model doesn't support tool use in streaming mode"),
# which zeroes an entire eval run. These models still work with tool use when
# streaming is off, so fall back rather than treating them as incapable.
MODELS_WITHOUT_STREAMING_TOOL_USE = (
    "llama3-1",
    "llama3-2",
    "llama3-3",
    "mistral",
)


def supports_streaming_tool_use(model_id: str | None) -> bool:
    """Whether the model can use tools while streaming."""
    if not model_id:
        return True
    lowered = model_id.lower()
    return not any(f in lowered for f in MODELS_WITHOUT_STREAMING_TOOL_USE)

VALID_SERVICE_TIERS = {"default", "priority", "flex"}

INFERENCE_CONFIG = {
    "stopSequences": [],  # words after which the generation is stopped
    "maxTokens": MAX_TOKENS,  # max tokens to be generated
    "temperature": TEMPERATURE,  # randomness of the model's output
}

REASONING_CONFIG = {
    "thinking": {
        "type": "enabled",  # whether extended thinking is enabled
        "budget_tokens": THINKING_TOKENS,  # max tokens for thinking budget
    }
}


def get_inference_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Get inference and reasoning parameters for Bedrock language models.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Tuple containing:
        - Inference config dict with temperature, maxTokens,
          topP and stopSequences parameters
        - Reasoning config dict with thinking settings
    """

    inference_config = INFERENCE_CONFIG.copy()
    reasoning_config = REASONING_CONFIG.copy()

    if reasoning_config["thinking"]["type"] == "enabled":
        inference_config["temperature"] = 1.0  # required in thinking mode
    else:
        reasoning_config = {
            "thinking": {
                "type": "disabled",
            }
        }

    return inference_config, reasoning_config


def get_service_tier() -> str:
    """
    Get the Bedrock service tier from the SERVICE_TIER environment variable.

    Returns
    -------
    str
        Service tier value (default, priority, or flex). Falls back to "default"
        when the env var is unset or contains an unrecognised value.
    """

    tier = os.environ.get("SERVICE_TIER", "default").lower()
    if tier not in VALID_SERVICE_TIERS:
        print(f"[INFERENCE] Unknown SERVICE_TIER '{tier}', falling back to 'default'")
        tier = "default"
    return tier


def get_bedrock_config() -> botocore.config.Config:
    """
    Get botocore configuration for Bedrock API calls.

    Returns
    -------
    botocore.config.Config
        Configuration object with read timeout and retry settings for Bedrock client
    """
    return botocore.config.Config(
        read_timeout=BEDROCK_READ_TIMEOUT,
        connect_timeout=BEDROCK_CONNECT_TIMEOUT,
        retries={
            "max_attempts": BEDROCK_MAX_ATTEMPTS,
            "mode": "adaptive",
        },
        max_pool_connections=BEDROCK_MAX_CONNECTIONS,
    )
