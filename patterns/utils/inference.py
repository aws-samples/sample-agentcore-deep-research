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

MAX_TOKENS = 8_192
THINKING_TOKENS = 2_000
TEMPERATURE = 0.0


def get_max_output_tokens(model_id: str | None = None) -> int:
    """
    Output-token cap for the agent, from MAX_OUTPUT_TOKENS or the default.

    Deliberately NOT a per-model lookup table. Bedrock already reports the
    authoritative limit when a request exceeds it:

        ValidationException: The maximum tokens you requested exceeds the model
        limit of 10000. Try again with a maximum tokens value that is lower.

    That message is always correct, needs no maintenance, and covers every model,
    whereas a hand-maintained table silently goes stale and only covers the
    families someone thought to list. The default below is well above what the
    agent actually emits — reports are written incrementally, and the largest
    single turn measured across 300 real trajectories was 2,602 tokens — so it
    fits comfortably inside every current model's ceiling.
    """
    override = os.environ.get("MAX_OUTPUT_TOKENS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    return MAX_TOKENS


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
