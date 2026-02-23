#!/usr/bin/env python3
"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

"""
Test the Deep Research Agent via AgentCore Runtime.

This script tests the full research workflow by sending a research query
and streaming the response, including tool calls for file operations
and data retrieval.

Usage:
    uv run test-scripts/test-deep-research-agent.py [query]

Examples:
    uv run test-scripts/test-deep-research-agent.py
    uv run test-scripts/test-deep-research-agent.py "What are the latest advances in AI agents?"
"""

import json
import os
import sys
from pathlib import Path

import boto3

# Add scripts directory to path for reliable imports
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from utils import get_ssm_params, get_stack_config, print_msg, print_section


def get_runtime_client(region: str):
    """Create AgentCore Runtime client."""
    return boto3.client("bedrock-agentcore-runtime", region_name=region)


def invoke_agent_streaming(
    client,
    runtime_arn: str,
    prompt: str,
    session_id: str,
    access_token: str,
) -> None:
    """
    Invoke the agent with streaming response.

    Parameters
    ----------
    client : boto3.client
        AgentCore Runtime client
    runtime_arn : str
        ARN of the agent runtime
    prompt : str
        User's research query
    session_id : str
        Session ID for conversation continuity
    access_token : str
        Cognito access token for authentication
    """
    print(f"\n{'=' * 60}")
    print("RESEARCH QUERY:")
    print(f"{'=' * 60}")
    print(prompt)
    print(f"{'=' * 60}\n")

    payload = {
        "prompt": prompt,
        "runtimeSessionId": session_id,
    }

    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        payload=json.dumps(payload),
        payloadContentType="application/json",
        sessionId=session_id,
        authContext={
            "authToken": access_token,
            "authTokenType": "ACCESS_TOKEN",
        },
    )

    print("AGENT RESPONSE (streaming):")
    print("-" * 60)

    # Process streaming response
    event_stream = response.get("payloadStream")
    if event_stream:
        current_text = ""
        for event in event_stream:
            if "chunk" in event:
                chunk_data = event["chunk"]["bytes"].decode("utf-8")
                try:
                    data = json.loads(chunk_data)

                    # Handle different event types
                    if "data" in data:
                        text = data["data"]
                        if text:
                            print(text, end="", flush=True)
                            current_text += text

                    elif "event" in data:
                        event_type = data["event"]
                        if event_type == "tool_use":
                            tool_name = data.get("name", "unknown")
                            print(f"\n[Tool: {tool_name}]", flush=True)
                        elif event_type == "tool_result":
                            print("\n[Tool completed]", flush=True)

                except json.JSONDecodeError:
                    # Raw text chunk
                    print(chunk_data, end="", flush=True)

    print("\n" + "-" * 60)
    print_msg("Research completed!")


def main():
    """Main entry point."""
    print_section("Deep Research Agent Test")

    # Get research query from args or use default
    if len(sys.argv) > 1:
        research_query = " ".join(sys.argv[1:])
    else:
        research_query = (
            "What are the key challenges and recent advances in building "
            "AI agents that can reason and plan? Focus on academic research "
            "from the last 2 years."
        )

    # Get stack configuration
    stack_cfg = get_stack_config()
    print(f"Stack: {stack_cfg['stack_name']}")

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    print(f"Region: {region}\n")

    # Fetch SSM parameters
    print("Fetching configuration...")
    params = get_ssm_params(
        stack_cfg["stack_name"],
        "runtime-arn",
        "cognito-user-pool-id",
        "cognito-user-pool-client-id",
    )
    print_msg("Configuration fetched")

    runtime_arn = params.get("runtime-arn")
    if not runtime_arn:
        print_msg("Runtime ARN not found in SSM parameters", "error")
        sys.exit(1)

    print(f"Runtime ARN: {runtime_arn}")

    # For testing, we need a valid access token
    # In production, this would come from Cognito authentication
    print_section("Authentication")
    print(
        "Note: This test requires a valid Cognito access token.\n"
        "For local testing, use test-agent-docker.py instead."
    )

    # Check if access token is provided via environment
    access_token = os.environ.get("ACCESS_TOKEN")
    if not access_token:
        print_msg(
            "ACCESS_TOKEN environment variable not set.\n"
            "Please authenticate and set ACCESS_TOKEN, or use test-agent-docker.py",
            "error",
        )
        sys.exit(1)

    # Create client and invoke agent
    print_section("Invoking Deep Research Agent")
    client = get_runtime_client(region)

    import uuid

    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")

    invoke_agent_streaming(
        client=client,
        runtime_arn=runtime_arn,
        prompt=research_query,
        session_id=session_id,
        access_token=access_token,
    )


if __name__ == "__main__":
    main()
