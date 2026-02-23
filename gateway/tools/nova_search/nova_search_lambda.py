"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import logging
import os

import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def search_web_with_nova(query: str) -> str:
    """
    Search the web using Amazon Nova 2 Lite's Web Grounding feature.

    Returns the full response with synthesized text and citations.

    Parameters
    ----------
    query : str
        Search query string

    Returns
    -------
    str
        Web search results with text and citations
    """
    region = os.environ.get("AWS_REGION", "us-east-1")

    # Create client with extended timeout for web grounding
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=120),
    )

    # Configure Nova Web Grounding as a system tool
    tool_config = {"tools": [{"systemTool": {"name": "nova_grounding"}}]}

    # Send request to Nova 2 Lite with web grounding enabled
    response = bedrock.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": query}]}],
        toolConfig=tool_config,
    )

    # Extract text with inline citations
    output = f"Web search results for: '{query}'\n\n"
    seen_urls = set()
    sources = []

    content_list = response.get("output", {}).get("message", {}).get("content", [])

    for content in content_list:
        # Add text content (skip thinking tags)
        if "text" in content:
            text = content["text"]
            if not text.startswith("<thinking>"):
                output += text

        # Collect unique citations
        if "citationsContent" in content:
            citations = content["citationsContent"].get("citations", [])
            for citation in citations:
                web_info = citation.get("location", {}).get("web", {})
                url = web_info.get("url", "")
                domain = web_info.get("domain", "")

                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"url": url, "domain": domain})

    # Add sources list at the end
    if sources:
        output += f"\n\n## Sources ({len(sources)})\n"
        for i, source in enumerate(sources, 1):
            output += f"{i}. [{source['domain']}] {source['url']}\n"

    return output


def handler(event, context):
    """
    Nova web search tool Lambda function for AgentCore Gateway.

    This Lambda searches the web using Amazon Nova's Web Grounding feature,
    which provides responses with citations to source material.

    Parameters
    ----------
    event : dict
        Tool arguments passed directly from gateway
    context : LambdaContext
        Lambda context with AgentCore metadata in client_context.custom

    Returns
    -------
    dict
        Response object with 'content' array or 'error' string
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # Get tool name from context and strip the target prefix
        delimiter = "___"
        original_tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = original_tool_name[
            original_tool_name.index(delimiter) + len(delimiter) :
        ]

        logger.info(f"Processing tool: {tool_name}")

        if tool_name == "nova_web_search":
            query = event.get("query", "")
            if not query:
                return {"error": "Missing required parameter: query"}

            result = search_web_with_nova(query=query)

            return {"content": [{"type": "text", "text": result}]}
        else:
            logger.error(f"Unexpected tool name: {tool_name}")
            return {
                "error": f"This Lambda only supports 'nova_web_search', "
                f"received: {tool_name}"
            }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
