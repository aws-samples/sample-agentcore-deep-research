# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions"
HEADERS = {"User-Agent": "AgentCoreDeepResearch/1.0 research@example.com"}
MAX_CONTENT_CHARS = 25000


def _fetch_text(url: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Fetch a filing document and extract readable text content."""
    req = Request(url, headers=HEADERS, method="GET")  # noqa: S310
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[Error fetching filing: {e}]"

    # Skip past XBRL/hidden metadata to real filing content
    for marker in [
        "UNITED STATES",
        "ANNUAL REPORT",
        "FORM 10-K",
        "FORM 10-Q",
        "Table of Contents",
    ]:
        idx = raw.find(marker)
        if idx > 0:
            raw = raw[idx:]
            break

    # Strip HTML tags and clean up
    text = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[Truncated at {max_chars} chars]"
    return text


def _resolve_filing_doc_url(cik: str, accession: str) -> str | None:
    """Resolve the primary document URL from a filing index."""
    acc_clean = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{accession}-index.json"
    req = Request(index_url, headers=HEADERS, method="GET")  # noqa: S310
    try:
        with urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        for item in data.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.endswith(".htm") or name.endswith(".txt"):
                return (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{name}"
                )
    except Exception:
        logger.debug("Failed to resolve filing doc URL for %s/%s", cik, accession)
    return None


def search_edgar(
    query: str = "",
    ticker: str = "",
    form_type: str = "",
    max_results: int = 5,
    fetch_content: bool = False,
) -> str:
    """Search SEC EDGAR for company filings, optionally fetching content."""
    max_results = min(max_results, 10)

    if ticker:
        return _get_company_filings(
            ticker.upper(), form_type, max_results, fetch_content
        )

    # Full-text search via EFTS
    params = {"q": query, "forms": form_type or "10-K,10-Q,8-K"}
    url = f"https://efts.sec.gov/LATEST/search-index?{urlencode(params)}&from=0&size={max_results}"
    req = Request(url, headers=HEADERS, method="GET")  # noqa: S310
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Error searching EDGAR: {e}"

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        return f"No EDGAR filings found for: {query}"

    output = f"Found {len(hits)} SEC filings\n\n"
    for i, hit in enumerate(hits[:max_results], 1):
        src = hit.get("_source", {})
        name = src.get("display_names", ["Unknown"])[0]
        form = src.get(
            "form",
            src.get("root_forms", ["N/A"])[0] if src.get("root_forms") else "N/A",
        )
        adsh = src.get("adsh", "")
        cik = src.get("ciks", [""])[0].lstrip("0") if src.get("ciks") else ""
        output += f"## {i}. {name}\n"
        output += f"**Form:** {form}\n"
        output += f"**Filed:** {src.get('file_date', 'N/A')}\n"
        if adsh and cik:
            acc_clean = adsh.replace("-", "")
            output += (
                f"**URL:** https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{acc_clean}/{adsh}-index.htm\n"
            )
        output += "\n"

    return output


def _get_company_filings(
    ticker: str, form_type: str, max_results: int, fetch_content: bool
) -> str:
    """Get filings for a specific company by ticker, optionally with content."""
    # Resolve ticker to CIK
    req = Request(  # noqa: S310
        "https://www.sec.gov/files/company_tickers.json", headers=HEADERS, method="GET"
    )
    try:
        with urlopen(req, timeout=15) as resp:  # noqa: S310
            tickers_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Error resolving ticker {ticker}: {e}"

    cik = None
    company_name = ticker
    for entry in tickers_data.values():
        if entry.get("ticker", "").upper() == ticker:
            cik = str(entry["cik_str"]).zfill(10)
            company_name = entry.get("title", ticker)
            break

    if not cik:
        return f"Could not find CIK for ticker: {ticker}"

    # Get recent filings
    url = f"{SUBMISSIONS_URL}/CIK{cik}.json"
    req = Request(url, headers=HEADERS, method="GET")  # noqa: S310
    try:
        with urlopen(req, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Error fetching filings for {ticker}: {e}"

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    output = f"SEC filings for {company_name} ({ticker})\n\n"
    count = 0
    cik_stripped = cik.lstrip("0")

    for j in range(len(forms)):
        if form_type and forms[j] != form_type:
            continue
        if count >= max_results:
            break
        count += 1
        acc_clean = accessions[j].replace("-", "")
        desc = descriptions[j] if j < len(descriptions) else ""

        # Build direct document URL
        doc_name = primary_docs[j] if j < len(primary_docs) else ""
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{acc_clean}/{doc_name}"
            if doc_name
            else None
        )
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_stripped}/{acc_clean}/{accessions[j]}-index.htm"
        )

        output += f"## {count}. {forms[j]} — {dates[j]}\n"
        if desc:
            output += f"**Description:** {desc}\n"
        output += f"**URL:** {index_url}\n"

        if fetch_content and doc_url:
            per_doc_limit = MAX_CONTENT_CHARS // max(max_results, 1)
            content = _fetch_text(doc_url, max_chars=per_doc_limit)
            output += f"\n### Filing Content (excerpt)\n{content}\n"

        output += "\n"

    if count == 0:
        output += f"No {form_type or 'recent'} filings found.\n"

    return output


def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        delimiter = "___"
        tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = tool_name[tool_name.index(delimiter) + len(delimiter) :]

        if tool_name != "edgar_search":
            return {"error": f"Expected 'edgar_search', received: {tool_name}"}

        query = event.get("query", "")
        ticker = event.get("ticker", "")
        if not query and not ticker:
            return {"error": "Provide either 'query' or 'ticker'"}

        result = search_edgar(
            query=query,
            ticker=ticker,
            form_type=event.get("form_type", ""),
            max_results=event.get("max_results", 5),
            fetch_content=event.get("fetch_content", False),
        )
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": f"Internal server error: {e}"}
