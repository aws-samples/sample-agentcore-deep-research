# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def search_clinicaltrials(
    condition: str = "",
    intervention: str = "",
    status: str = "",
    phase: str = "",
    max_results: int = 10,
) -> str:
    """Search ClinicalTrials.gov for clinical studies."""
    max_results = min(max_results, 50)

    params = {"pageSize": max_results}
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    if status:
        params["filter.overallStatus"] = status
    if phase:
        params["query.term"] = f"AREA[Phase]{phase}"

    url = f"{BASE_URL}?{urlencode(params)}"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"Error searching ClinicalTrials.gov: {e}"

    studies = data.get("studies", [])
    if not studies:
        return f"No clinical trials found for condition='{condition}', intervention='{intervention}'"

    total = data.get("totalCount", len(studies))
    output = f"Found {total} clinical trials (showing {len(studies)})\n\n"

    for i, study in enumerate(studies, 1):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        desc = proto.get("descriptionModule", {})

        nct_id = ident.get("nctId", "Unknown")
        title = ident.get("briefTitle", "No title")
        overall_status = status_mod.get("overallStatus", "Unknown")
        phases = ", ".join(design.get("phases", ["N/A"]))
        brief_summary = desc.get("briefSummary", "No summary available")

        if len(brief_summary) > 400:
            brief_summary = brief_summary[:400] + "..."

        output += f"## {i}. {title}\n"
        output += f"**NCT ID:** {nct_id}\n"
        output += f"**Status:** {overall_status}\n"
        output += f"**Phase:** {phases}\n"
        output += f"**URL:** https://clinicaltrials.gov/study/{nct_id}\n"
        output += f"**Summary:** {brief_summary}\n\n"

    return output


def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        delimiter = "___"
        tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = tool_name[tool_name.index(delimiter) + len(delimiter) :]

        if tool_name != "clinicaltrials_search":
            return {"error": f"Expected 'clinicaltrials_search', received: {tool_name}"}

        result = search_clinicaltrials(
            condition=event.get("condition", ""),
            intervention=event.get("intervention", ""),
            status=event.get("status", ""),
            phase=event.get("phase", ""),
            max_results=event.get("max_results", 10),
        )
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": f"Internal server error: {e}"}
