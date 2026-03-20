# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger()
logger.setLevel(logging.INFO)

OPENFDA_DRUG_LABEL_URL = "https://api.fda.gov/drug/label.json"


def search_openfda(drug_name: str, max_results: int = 10) -> str:
    """
    Search OpenFDA drug label API for information on a given drug.

    Searches across brand names, generic names, and active ingredients to find
    comprehensive drug information including indications, warnings, interactions,
    contraindications, dosage, and mechanism of action.

    Parameters
    ----------
    drug_name : str
        Name of the drug (brand, generic, or active ingredient) to search for.
        Examples: "Ozempic", "semaglutide", "acetaminophen"
    max_results : int
        Maximum number of results to return (default 10, max 50)

    Returns
    -------
    str
        Formatted results containing drug info
    """
    query = quote(drug_name)
    limit = min(max_results, 50)
    search = (
        f"openfda.brand_name:{query}"
        f"+openfda.generic_name:{query}"
        f"+openfda.substance_name:{query}"
    )
    url = f"{OPENFDA_DRUG_LABEL_URL}?search={search}&limit={limit}"

    if not url.startswith("https://"):
        return "Error: invalid URL scheme for OpenFDA request"

    req = Request(url, method="GET")  # noqa: S310  # nosec B310
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310  # nosec B310  # nosemgrep: dynamic-urllib-use-detected
            response = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"OpenFDA request failed: {str(e)}")
        return f"Error fetching data from OpenFDA: {str(e)}"

    results = response.get("results", [])
    if not results:
        return f"No OpenFDA results found for '{drug_name}'."

    output = f"OpenFDA drug label search results for '{drug_name}':\n\n"

    for i, drug in enumerate(results, 1):
        openfda = drug.get("openfda", {})
        brand_name = ", ".join(openfda.get("brand_name", ["Unknown"]))
        generic_name = ", ".join(openfda.get("generic_name", ["Unknown"]))
        manufacturer = ", ".join(openfda.get("manufacturer_name", ["Unknown"]))
        product_type = ", ".join(openfda.get("product_type", ["Unknown"]))

        indications = "\n".join(drug.get("indications_and_usage", ["N/A"]))[:500]
        warnings = "\n".join(drug.get("warnings", drug.get("boxed_warning", ["N/A"])))[
            :500
        ]
        contraindications = "\n".join(drug.get("contraindications", ["N/A"]))[:500]
        interactions = "\n".join(drug.get("drug_interactions", ["N/A"]))[:500]
        dosage = "\n".join(drug.get("dosage_and_administration", ["N/A"]))[:500]
        mechanism = "\n".join(drug.get("mechanism_of_action", ["N/A"]))[:300]

        output += f"### {i}. {brand_name}\n"
        output += f"**Generic Name:** {generic_name}\n"
        output += f"**Manufacturer:** {manufacturer}\n"
        output += f"**Product Type:** {product_type}\n"
        output += f"**Indications:** {indications}...\n"
        output += f"**Warnings:** {warnings}...\n"
        output += f"**Contraindications:** {contraindications}...\n"
        output += f"**Drug Interactions:** {interactions}...\n"
        output += f"**Dosage:** {dosage}...\n"
        output += f"**Mechanism of Action:** {mechanism}...\n\n"

    return output


def handler(event, context):
    """
    OpenFDA drug label search tool Lambda function.

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
        tool_name = context.client_context.custom.get("bedrockAgentCoreToolName", "")
        delimiter = "___"
        tool_name = tool_name[tool_name.index(delimiter) + len(delimiter) :]

        if tool_name != "openfda_drug_search":
            return {
                "error": (
                    "This Lambda only supports "
                    f"'openfda_drug_search', received: {tool_name}"
                )
            }

        drug_name = event.get("drug_name", "")
        if not drug_name:
            return {"error": "Missing required parameter: drug_name"}

        max_results = event.get("max_results", 10)
        result = search_openfda(drug_name=drug_name, max_results=max_results)

        return {"content": [{"type": "text", "text": result}]}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
