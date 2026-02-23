"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import os

import boto3


def test_nova_web_search():
    """Test Nova web search using Converse API with web grounding tool."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = "us.amazon.nova-pro-v1:0"

    print(f"Testing Nova web search in region: {region}")
    print(f"Model ID: {model_id}")

    client = boto3.client("bedrock-runtime", region_name=region)

    query = "What is self-attention in transformers?"

    # Web grounding tool configuration
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "web_search",
                    "description": "Search the web for current information",
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                }
            }
        ],
        "toolChoice": {"tool": {"name": "web_search"}},
    }

    # System prompt for web grounding
    system_prompt = """You are a helpful assistant with access to web search.
When asked a question, use the web_search tool to find current information.
Provide a comprehensive answer based on the search results."""

    messages = [{"role": "user", "content": [{"text": query}]}]

    print(f"\nQuery: {query}")
    print("Calling Converse API...")

    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": system_prompt}],
            toolConfig=tool_config,
        )

        print("\n=== Nova Converse Response ===")
        print(json.dumps(response, indent=2, default=str))

    except Exception as e:
        print(f"\nError with Converse API: {e}")
        print(f"Exception type: {type(e).__name__}")

        # Try alternative approach with InvokeModel
        print("\n\nTrying InvokeModel API instead...")
        try:
            invoke_response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "messages": messages,
                        "system": [{"text": system_prompt}],
                    }
                ),
                contentType="application/json",
                accept="application/json",
            )

            result = json.loads(invoke_response["body"].read())
            print("\n=== Nova InvokeModel Response ===")
            print(json.dumps(result, indent=2, default=str))

        except Exception as e2:
            print(f"\nError with InvokeModel API: {e2}")


def test_nova_grounding_tool():
    """Test Nova with the amazon.nova_grounding system tool."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = "us.amazon.nova-pro-v1:0"

    print(f"\n\nTesting Nova with amazon.nova_grounding tool in region: {region}")

    client = boto3.client("bedrock-runtime", region_name=region)

    query = "What is self-attention in transformers?"

    # Use the system grounding tool
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "amazon.nova_grounding",
                    "description": "Ground responses with web search results",
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                }
            }
        ]
    }

    messages = [{"role": "user", "content": [{"text": query}]}]

    print(f"Query: {query}")
    print("Calling Converse API with grounding tool...")

    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            toolConfig=tool_config,
        )

        print("\n=== Nova Grounding Tool Response ===")
        print(json.dumps(response, indent=2, default=str))

    except Exception as e:
        print(f"\nError: {e}")
        print(f"Exception type: {type(e).__name__}")


if __name__ == "__main__":
    test_nova_web_search()
    test_nova_grounding_tool()
