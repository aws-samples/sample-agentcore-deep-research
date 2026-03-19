# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def search_knowledge_base(
    query: str,
    knowledge_base_id: str,
    max_results: int = 10,
) -> str:
    """
    Search a Bedrock Knowledge Base for relevant content.

    Parameters
    ----------
    query : str
        Search query string
    knowledge_base_id : str
        The ID of the knowledge base to search
    max_results : int
        Maximum number of results to return (default 10)

    Returns
    -------
    str
        Formatted search results as string
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-agent-runtime", region_name=region)

    response = client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": min(max_results, 25),
            }
        },
    )

    results = response.get("retrievalResults", [])

    if not results:
        return f"No results found in knowledge base {knowledge_base_id} for: {query}"

    # Format results
    output = f"Knowledge Base search results for: '{query}'\n"
    output += f"Knowledge Base ID: {knowledge_base_id}\n"
    output += f"Found {len(results)} relevant chunks\n\n"

    for i, result in enumerate(results, 1):
        content = result.get("content", {}).get("text", "No content")
        score = result.get("score", 0)

        # Extract location metadata
        location = result.get("location", {})
        location_type = location.get("type", "UNKNOWN")

        source_info = "[Knowledge Base] Unknown source"
        if location_type == "S3":
            s3_loc = location.get("s3Location", {})
            source_info = f"[Knowledge Base] {s3_loc.get('uri', 'Unknown S3 URI')}"
        elif location_type == "WEB":
            web_loc = location.get("webLocation", {})
            source_info = f"[Knowledge Base] {web_loc.get('url', 'Unknown URL')}"

        # Truncate content if too long
        if len(content) > 800:
            content = content[:800] + "..."

        output += f"### {i}. Chunk (Score: {score:.3f})\n"
        output += f"**Source:** {source_info}\n"
        output += f"**Content:**\n{content}\n\n"

    return output


def handler(event, context):
    """
    Knowledge Base search tool Lambda function for AgentCore Gateway.

    This Lambda searches a Bedrock Knowledge Base for relevant content
    based on the provided query.

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

        if tool_name == "knowledge_base_search":
            query = event.get("query", "")
            if not query:
                return {"error": "Missing required parameter: query"}

            knowledge_base_id = event.get(
                "knowledge_base_id",
                os.environ.get("DEFAULT_KNOWLEDGE_BASE_ID", ""),
            )
            if not knowledge_base_id:
                return {
                    "error": "Missing knowledge_base_id. "
                    "Configure it in config.yaml: tools.bedrock_kb.knowledge_base_id"
                }

            max_results = event.get("max_results", 10)

            result = search_knowledge_base(
                query=query,
                knowledge_base_id=knowledge_base_id,
                max_results=max_results,
            )

            return {"content": [{"type": "text", "text": result}]}
        else:
            logger.error(f"Unexpected tool name: {tool_name}")
            return {
                "error": f"This Lambda only supports 'knowledge_base_search', "
                f"received: {tool_name}"
            }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
