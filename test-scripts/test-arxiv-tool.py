#!/usr/bin/env python3


"""
Test ArXiv search tool via AgentCore Gateway.

Usage:
    uv run test-scripts/test-arxiv-tool.py
"""

import json
import os
import sys
from pathlib import Path

import boto3
import requests

# Add scripts directory to path for reliable imports
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from utils import get_ssm_params, get_stack_config, print_msg, print_section


def get_secret(secret_name: str) -> str:
    """Fetch secret from AWS Secrets Manager."""
    region = os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    secrets_client = boto3.client("secretsmanager", region_name=region)
    response = secrets_client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]


def fetch_access_token(client_id: str, client_secret: str, token_url: str) -> str:
    """Fetch access token using client credentials flow."""
    response = requests.post(
        token_url,
        data=f"grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )

    if response.status_code != 200:
        print_msg(
            f"Token request failed: {response.status_code} - {response.text}", "error"
        )
        sys.exit(1)

    return response.json()["access_token"]


def list_tools(gateway_url: str, access_token: str) -> dict:
    """List available tools via gateway."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    payload = {"jsonrpc": "2.0", "id": "list-tools-request", "method": "tools/list"}

    response = requests.post(gateway_url, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        print_msg(
            f"Gateway request failed: {response.status_code} - {response.text}", "error"
        )
        sys.exit(1)

    return response.json()


def call_tool(
    gateway_url: str, access_token: str, tool_name: str, arguments: dict
) -> dict:
    """Call a specific tool via gateway."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    payload = {
        "jsonrpc": "2.0",
        "id": "call-tool-request",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }

    response = requests.post(gateway_url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        print_msg(
            f"Gateway request failed: {response.status_code} - {response.text}", "error"
        )
        sys.exit(1)

    return response.json()


def main():
    """Main entry point."""
    print_section("ArXiv Search Tool Test")

    # Get stack configuration
    stack_cfg = get_stack_config()
    print(f"Stack: {stack_cfg['stack_name']}\n")

    # Fetch SSM parameters
    print("Fetching configuration...")
    gateway_params = get_ssm_params(
        stack_cfg["stack_name"], "gateway_url", "machine_client_id", "cognito_provider"
    )

    # Get client secret from Secrets Manager
    client_secret = get_secret(f"/{stack_cfg['stack_name']}/machine_client_secret")
    print_msg("Configuration fetched")

    # Extract gateway configuration
    gateway_url = gateway_params["gateway_url"]
    client_id = gateway_params["machine_client_id"]
    cognito_domain = gateway_params["cognito_provider"]
    token_url = f"https://{cognito_domain}/oauth2/token"

    # Get access token
    print_section("Authentication")
    print("Fetching access token...")
    access_token = fetch_access_token(client_id, client_secret, token_url)
    print_msg("Access token obtained")

    # List tools to find arxiv_search
    print_section("Finding ArXiv Tool")
    tools = list_tools(gateway_url, access_token)

    tool_list = tools.get("result", {}).get("tools", [])
    target_tool = "arxiv_search"
    tool_name = None

    for t in tool_list:
        if t["name"].endswith(f"___{target_tool}"):
            tool_name = t["name"]
            break

    if not tool_name:
        print_msg(
            f"Tool '{target_tool}' not found. Available: {[t['name'] for t in tool_list]}",
            "error",
        )
        sys.exit(1)

    print(f"Found tool: {tool_name}")

    # Test ArXiv search
    print_section("ArXiv Search Test")
    print("Searching for 'large language models' papers...")

    tool_result = call_tool(
        gateway_url,
        access_token,
        tool_name,
        {
            "query": "large language models agents",
            "max_results": 5,
            "sort_by": "relevance",
            "categories": ["cs.AI", "cs.CL"],
        },
    )

    # Validate response
    if "error" in tool_result:
        print_msg(f"Tool returned error: {tool_result['error']}", "error")
        sys.exit(1)

    if "result" not in tool_result or "content" not in tool_result["result"]:
        print_msg(f"Unexpected response format: {json.dumps(tool_result)}", "error")
        sys.exit(1)

    print_msg("ArXiv search successful!")
    print("\nResults:")

    # Extract and print the text content
    content = tool_result["result"]["content"]
    for item in content:
        if item.get("type") == "text":
            print(item["text"])


if __name__ == "__main__":
    main()
