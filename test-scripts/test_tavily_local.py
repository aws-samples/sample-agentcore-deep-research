"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import os

import boto3


def get_tavily_api_key() -> str:
    """Retrieve Tavily API key from AWS Secrets Manager."""
    secret_name = os.environ.get("TAVILY_SECRET_NAME", "tavily-api-key")
    region = os.environ.get("AWS_REGION", "us-east-1")

    print(f"Fetching secret: {secret_name} from region: {region}")

    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)

    return response["SecretString"]


def test_tavily_sdk():
    """Test Tavily using the Python SDK."""
    from tavily import TavilyClient

    api_key = get_tavily_api_key()
    print(f"API key (first 10 chars): {api_key[:10]}...")

    client = TavilyClient(api_key=api_key)

    result = client.search(
        query="What is self-attention in transformers?",
        max_results=3,
        search_depth="basic",
        include_answer=True,
    )

    print("\n=== Tavily SDK Results ===")
    print(json.dumps(result, indent=2, default=str))


def test_tavily_rest():
    """Test Tavily using REST API."""
    from urllib.request import Request, urlopen

    api_key = get_tavily_api_key()
    print(f"API key (first 10 chars): {api_key[:10]}...")

    payload = {
        "api_key": api_key,
        "query": "What is self-attention in transformers?",
        "max_results": 3,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
    }

    req = Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    print("\n=== Tavily REST Results ===")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    print("Testing Tavily REST API...")
    test_tavily_rest()

    print("\n" + "=" * 50 + "\n")

    print("Testing Tavily SDK...")
    test_tavily_sdk()
