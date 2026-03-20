# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BASE_URL = "https://api.stlouisfed.org/fred"


def _get_api_key() -> str:
    """Retrieve FRED API key from Secrets Manager."""
    secret_name = os.environ.get("FRED_SECRET_NAME", "")
    if not secret_name:
        raise ValueError("FRED_SECRET_NAME environment variable not set")
    client = boto3.client("secretsmanager")
    try:
        resp = client.get_secret_value(SecretId=secret_name)
    except client.exceptions.ResourceNotFoundException:
        raise ValueError(
            "FRED API key not configured. Get a free key at https://fredaccount.stlouisfed.org "
            "and add it to config.yaml under api_keys.fred"
        )
    return resp["SecretString"]


def search_fred(
    query: str = "",
    series_id: str = "",
    max_results: int = 10,
) -> str:
    """Search FRED for economic data series or retrieve observations."""
    api_key = _get_api_key()
    max_results = min(max_results, 50)

    # If a specific series ID is provided, fetch its observations
    if series_id:
        params = urlencode(
            {
                "api_key": api_key,
                "series_id": series_id,
                "file_type": "json",
                "sort_order": "desc",
                "limit": max_results,
            }
        )
        url = f"{BASE_URL}/series/observations?{params}"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return f"Error fetching FRED series {series_id}: {e}"

        observations = data.get("observations", [])
        if not observations:
            return f"No observations found for series: {series_id}"

        output = (
            f"FRED Series: {series_id} (latest {len(observations)} observations)\n\n"
        )
        output += "| Date | Value |\n|------|-------|\n"
        for obs in observations:
            output += f"| {obs.get('date', '')} | {obs.get('value', 'N/A')} |\n"
        return output

    # Otherwise, search for series by keyword
    params = urlencode(
        {
            "api_key": api_key,
            "search_text": query,
            "file_type": "json",
            "limit": max_results,
            "order_by": "search_rank",
        }
    )
    url = f"{BASE_URL}/series/search?{params}"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Error searching FRED: {e}"

    series_list = data.get("seriess", [])
    if not series_list:
        return f"No FRED series found for: {query}"

    output = f"Found {len(series_list)} FRED series\n\n"
    for i, s in enumerate(series_list, 1):
        output += f"## {i}. {s.get('title', 'No title')}\n"
        output += f"**Series ID:** {s.get('id', '')}\n"
        output += f"**Frequency:** {s.get('frequency', 'N/A')}\n"
        output += f"**Units:** {s.get('units', 'N/A')}\n"
        output += f"**Last Updated:** {s.get('last_updated', 'N/A')}\n"
        notes = s.get("notes", "")
        if notes:
            output += (
                f"**Notes:** {notes[:300]}...\n"
                if len(notes) > 300
                else f"**Notes:** {notes}\n"
            )
        output += "\n"

    return output


def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        delimiter = "___"
        tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = tool_name[tool_name.index(delimiter) + len(delimiter) :]

        if tool_name != "fred_economic_search":
            return {"error": f"Expected 'fred_economic_search', received: {tool_name}"}

        query = event.get("query", "")
        series_id = event.get("series_id", "")
        if not query and not series_id:
            return {"error": "Provide either 'query' or 'series_id'"}

        result = search_fred(
            query=query,
            series_id=series_id,
            max_results=event.get("max_results", 10),
        )
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": f"Internal server error: {e}"}
