import json
import logging

import arxiv

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    categories: list[str] | None = None,
) -> str:
    """
    Search ArXiv for academic papers matching the query.

    Parameters
    ----------
    query : str
        Search query string
    max_results : int
        Maximum number of results to return (default 10, max 50)
    sort_by : str
        Sort order: "relevance" or "date"
    categories : list[str] | None
        ArXiv categories to filter by (e.g., ["cs.AI", "cs.LG"])

    Returns
    -------
    str
        Formatted search results as string
    """
    max_results = min(max_results, 50)

    # Build query with category filter if provided
    search_query = query
    if categories:
        category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
        search_query = f"({query}) AND ({category_filter})"

    # Determine sort criterion
    sort_criterion = (
        arxiv.SortCriterion.Relevance
        if sort_by == "relevance"
        else arxiv.SortCriterion.SubmittedDate
    )

    # Create search
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Execute search
    client = arxiv.Client()
    results = list(client.results(search))

    if not results:
        return f"No papers found for query: {query}"

    # Format results
    output = f"Found {len(results)} papers for query: '{query}'\n\n"

    for i, paper in enumerate(results, 1):
        # Extract paper ID from URL
        paper_id = paper.entry_id.split("/abs/")[-1]

        # Format authors (limit to first 3)
        authors = [a.name for a in paper.authors[:3]]
        if len(paper.authors) > 3:
            authors.append(f"et al. ({len(paper.authors)} total)")
        authors_str = ", ".join(authors)

        # Format categories
        cats = ", ".join(paper.categories[:3])

        output += f"## {i}. {paper.title}\n"
        output += f"**ID:** {paper_id}\n"
        output += f"**Authors:** {authors_str}\n"
        output += f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n"
        output += f"**Categories:** {cats}\n"
        output += f"**PDF:** {paper.pdf_url}\n"
        output += f"**Abstract:** {paper.summary[:500]}...\n\n"

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
            if not query:
                return {"error": "Missing required parameter: query"}

            max_results = event.get("max_results", 10)
            sort_by = event.get("sort_by", "relevance")
            categories = event.get("categories")

            result = search_arxiv(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
                categories=categories,
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
