"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import logging
import os
from urllib.request import Request, urlopen

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TAVILY_API_URL = "https://api.tavily.com/search"


def get_tavily_api_key() -> str:
    """
    Retrieve Tavily API key from AWS Secrets Manager.

    Returns
    -------
    str
        The Tavily API key
    """
    secret_name = os.environ.get("TAVILY_SECRET_NAME", "tavily-api-key")
    region = os.environ.get("AWS_REGION", "us-east-1")

    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)

    return response["SecretString"]


def search_web(
    query: str,
    max_results: int = 10,
    search_depth: str = "basic",
    include_answer: bool = True,
) -> str:
    """
    Search the web using Tavily REST API directly.

    Parameters
    ----------
    query : str
        Search query string
    max_results : int
        Maximum number of results to return (default 10)
    search_depth : str
        Search depth: "basic" for fast results, "advanced" for thorough search
    include_answer : bool
        Whether to include an AI-generated answer summary

    Returns
    -------
    str
        Formatted search results as string
    """
    api_key = get_tavily_api_key()

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": min(max_results, 20),
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_raw_content": False,
    }

    req = Request(
        TAVILY_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(req, timeout=30) as resp:
        response = json.loads(resp.read().decode("utf-8"))

    # Format results
    output = f"Web search results for: '{query}'\n\n"

    # Include AI answer if available
    if include_answer and response.get("answer"):
        output += f"## Summary\n{response['answer']}\n\n"

    output += "## Sources\n\n"

    results = response.get("results", [])
    if not results:
        return f"No results found for query: {query}"

    for i, result in enumerate(results, 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        content = result.get("content", "")[:500]
        score = result.get("score", 0)

        output += f"### {i}. {title}\n"
        output += f"**URL:** {url}\n"
        output += f"**Relevance:** {score:.2f}\n"
        output += f"**Content:** {content}...\n\n"

    return output


def handler(event, context):
    """
    Tavily web search tool Lambda function for AgentCore Gateway.

    This Lambda searches the web using Tavily API for current information
    on any topic.

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

        if tool_name == "tavily_web_search":
            query = event.get("query", "")
            if not query:
                return {"error": "Missing required parameter: query"}

            max_results = event.get("max_results", 10)
            search_depth = event.get("search_depth", "basic")
            include_answer = event.get("include_answer", True)

            result = search_web(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
            )

            return {"content": [{"type": "text", "text": result}]}
        else:
            logger.error(f"Unexpected tool name: {tool_name}")
            return {
                "error": f"This Lambda only supports 'tavily_web_search', "
                f"received: {tool_name}"
            }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
