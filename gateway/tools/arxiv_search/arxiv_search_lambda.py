# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging

import arxiv

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def search_arxiv(
    query: str = "",
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: list[str] | None = None,
    arxiv_ids: list[str] | None = None,
) -> str:
    """
    Search ArXiv for academic papers matching the query or retrieve papers by ID.

    Parameters
    ----------
    query : str
        Search query string, supports arXiv query syntax
        (e.g., "ti:transformer AND cat:cs.LG")
    max_results : int
        Maximum number of results to return (default 10, max 200)
    sort_by : str
        Sort order: "relevance" or "date"
    categories : list[str] | None
        ArXiv categories to filter by (e.g., ["cs.AI", "cs.LG"])
    arxiv_ids : list[str] | None
        List of arXiv IDs to look up directly (e.g., ["2309.10668", "2406.04093"])

    Returns
    -------
    str
        Formatted search results as string
    """
    max_results = min(max_results, 200)

    # Determine sort criterion
    sort_criterion = (
        arxiv.SortCriterion.Relevance
        if sort_by == "relevance"
        else arxiv.SortCriterion.SubmittedDate
    )

    if arxiv_ids:
        # ID lookup mode
        search = arxiv.Search(id_list=arxiv_ids)
    else:
        # Wrap plain text queries with all: prefix for proper matching
        search_query = query
        has_field_prefix = any(
            f"{p}:" in query
            for p in ("ti", "au", "abs", "cat", "all", "co", "jr", "id")
        )
        if not has_field_prefix and search_query:
            search_query = f"all:{search_query}"
        if categories:
            cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({search_query}) AND ({cat_filter})"

        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

    client = arxiv.Client(
        page_size=max_results if not arxiv_ids else 20,
        delay_seconds=1.0,
        num_retries=1,
    )
    results = list(client.results(search))

    if not results:
        return f"No papers found for query: {query}"

    output = f"Found {len(results)} papers\n\n"

    for i, paper in enumerate(results, 1):
        paper_id = paper.entry_id.split("/abs/")[-1]
        authors_str = ", ".join(a.name for a in paper.authors)
        cats = ", ".join(paper.categories)

        output += f"## {i}. {paper.title}\n"
        output += f"**ID:** {paper_id}\n"
        output += f"**Authors:** {authors_str}\n"
        output += f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n"
        output += f"**Categories:** {cats}\n"
        output += f"**PDF:** {paper.pdf_url}\n"
        output += f"**Abstract:** {paper.summary}\n\n"

    return output


def handler(event, context):
    """
    ArXiv search tool Lambda function for AgentCore Gateway.

    This Lambda searches ArXiv for academic papers based on the provided query.
    It uses the official ArXiv API via the arxiv Python library.

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

        if tool_name == "arxiv_search":
            query = event.get("query", "")
            arxiv_ids = event.get("arxiv_ids")

            if not query and not arxiv_ids:
                return {"error": "Provide either 'query' or 'arxiv_ids' parameter"}

            max_results = event.get("max_results", 10)
            sort_by = event.get("sort_by", "relevance")
            categories = event.get("categories")

            result = search_arxiv(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                categories=categories,
                arxiv_ids=arxiv_ids,
            )

            return {"content": [{"type": "text", "text": result}]}
        else:
            logger.error(f"Unexpected tool name: {tool_name}")
            return {
                "error": f"This Lambda only supports 'arxiv_search', "
                f"received: {tool_name}"
            }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
