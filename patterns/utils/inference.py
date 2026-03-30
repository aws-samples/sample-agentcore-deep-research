# Copyright Amazon.com, Inc. or its affiliates.
# AWS Content under the AWS Enterprise Agreement or
# AWS Customer Agreement (as applicable).

from typing import Any

import botocore

BEDROCK_READ_TIMEOUT = 600
BEDROCK_CONNECT_TIMEOUT = 600
BEDROCK_MAX_ATTEMPTS = 3
BEDROCK_MAX_CONNECTIONS = 10

MAX_TOKENS = 64_000
THINKING_TOKENS = 2_000
TEMPERATURE = 0.0

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
