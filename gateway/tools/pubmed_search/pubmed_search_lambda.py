# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical literature and return formatted results."""
    max_results = min(max_results, 50)

    # Step 1: ESearch to get PMIDs
    search_params = urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
    )
    search_url = f"{ESEARCH_URL}?{search_params}"
    req = Request(search_url, method="GET")  # noqa: S310
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            search_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Error searching PubMed: {e}"

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return f"No PubMed results found for: {query}"

    # Step 2: EFetch to get article details
    fetch_params = urlencode(
        {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract",
        }
    )
    fetch_url = f"{EFETCH_URL}?{fetch_params}"
    req = Request(fetch_url, method="GET")  # noqa: S310
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            xml_data = resp.read().decode("utf-8")
    except Exception as e:
        return f"Error fetching PubMed articles: {e}"

    # Parse XML minimally — extract key fields with simple string parsing
    # (avoids xml.etree dependency issues in Lambda)
    articles = xml_data.split("<PubmedArticle>")[1:]
    output = f"Found {len(articles)} PubMed articles\n\n"

    for i, article in enumerate(articles, 1):
        title = _extract_tag(article, "ArticleTitle") or "No title"
        abstract = _extract_tag(article, "AbstractText") or "No abstract available"
        pmid = _extract_tag(article, "PMID") or "Unknown"
        journal = _extract_tag(article, "Title") or "Unknown journal"
        year = _extract_tag(article, "Year") or ""

        # Truncate abstract
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."

        output += f"## {i}. {title}\n"
        output += f"**PMID:** {pmid}\n"
        output += f"**Journal:** {journal}\n"
        output += f"**Year:** {year}\n"
        output += f"**URL:** https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
        output += f"**Abstract:** {abstract}\n\n"

    return output


def _extract_tag(xml: str, tag: str) -> str | None:
    """Extract first occurrence of a simple XML tag value."""
    import re

    start = xml.find(f"<{tag}")
    if start == -1:
        return None
    start = xml.find(">", start) + 1
    end = xml.find(f"</{tag}>", start)
    if end == -1:
        return None
    text = xml[start:end].strip()
    # Strip nested XML tags and decode entities
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    return text


def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        delimiter = "___"
        tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = tool_name[tool_name.index(delimiter) + len(delimiter) :]

        if tool_name != "pubmed_search":
            return {"error": f"Expected 'pubmed_search', received: {tool_name}"}

        query = event.get("query", "")
        if not query:
            return {"error": "Missing required parameter: query"}

        result = search_pubmed(query=query, max_results=event.get("max_results", 10))
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": f"Internal server error: {e}"}
