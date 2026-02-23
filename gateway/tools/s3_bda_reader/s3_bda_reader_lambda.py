"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import logging
import os
import time

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# BDA project ARN should be set as environment variable
BDA_PROJECT_ARN = os.environ.get("BDA_PROJECT_ARN")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """
    Parse S3 URI into bucket and key.

    Parameters
    ----------
    s3_uri : str
        S3 URI in format s3://bucket/key

    Returns
    -------
    tuple[str, str]
        Bucket name and object key
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    path = s3_uri[5:]  # Remove 's3://'
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    return parts[0], parts[1]


def read_document_with_bda(
    s3_uri: str,
    extract_tables: bool = True,
) -> str:
    """
    Read and parse a document from S3 using Bedrock Data Automation.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the document to read
    extract_tables : bool
        Whether to extract tables as structured data

    Returns
    -------
    str
        Extracted content from the document
    """
    region = os.environ.get("AWS_REGION", "us-east-1")
    bda_client = boto3.client("bedrock-data-automation-runtime", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    bucket, key = parse_s3_uri(s3_uri)

    # Create output location for BDA results
    output_bucket = bucket
    output_prefix = f"bda-output/{key.rsplit('/', 1)[-1]}-{int(time.time())}"
    output_uri = f"s3://{output_bucket}/{output_prefix}/"

    if not BDA_PROJECT_ARN:
        raise ValueError(
            "BDA_PROJECT_ARN environment variable not set. "
            "Please configure a Bedrock Data Automation project."
        )

    # Invoke BDA async processing
    response = bda_client.invoke_data_automation_async(
        inputConfiguration={"s3Uri": s3_uri},
        outputConfiguration={"s3Uri": output_uri},
        dataAutomationConfiguration={
            "dataAutomationProjectArn": BDA_PROJECT_ARN,
        },
    )

    invocation_arn = response["invocationArn"]
    logger.info(f"Started BDA job: {invocation_arn}")

    # Poll for completion (with timeout)
    max_wait_seconds = 120
    poll_interval = 5
    elapsed = 0

    while elapsed < max_wait_seconds:
        status_response = bda_client.get_data_automation_status(
            invocationArn=invocation_arn
        )
        status = status_response["status"]

        if status == "COMPLETED":
            break
        elif status in ["FAILED", "STOPPED"]:
            error_msg = status_response.get("errorMessage", "Unknown error")
            raise Exception(f"BDA processing failed: {error_msg}")

        time.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= max_wait_seconds:
        raise Exception("BDA processing timed out")

    # Read the output from S3
    output_response = s3_client.list_objects_v2(
        Bucket=output_bucket,
        Prefix=output_prefix,
    )

    content_parts = []

    for obj in output_response.get("Contents", []):
        obj_key = obj["Key"]
        if obj_key.endswith(".json"):
            result = s3_client.get_object(Bucket=output_bucket, Key=obj_key)
            result_data = json.loads(result["Body"].read().decode("utf-8"))

            # Extract text content from BDA output
            if "standardOutput" in result_data:
                std_output = result_data["standardOutput"]
                if "document" in std_output:
                    doc = std_output["document"]
                    if "text" in doc:
                        content_parts.append(doc["text"])
                    if extract_tables and "tables" in doc:
                        for table in doc["tables"]:
                            content_parts.append(f"\n[Table]\n{table}")

    if not content_parts:
        # Fallback: try to read raw text content
        return f"No structured content extracted from: {s3_uri}"

    # Format output
    output = f"Document content from: {s3_uri}\n\n"
    output += "\n\n".join(content_parts)

    return output


def handler(event, context):
    """
    S3 document reader tool Lambda function for AgentCore Gateway.

    This Lambda reads documents from S3 and extracts content using
    Bedrock Data Automation. Supports various document formats including
    PDF, DOCX, images, etc.

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

        if tool_name == "s3_document_reader":
            s3_uri = event.get("s3_uri", "")
            if not s3_uri:
                return {"error": "Missing required parameter: s3_uri"}

            extract_tables = event.get("extract_tables", True)

            result = read_document_with_bda(
                s3_uri=s3_uri,
                extract_tables=extract_tables,
            )

            return {"content": [{"type": "text", "text": result}]}
        else:
            logger.error(f"Unexpected tool name: {tool_name}")
            return {
                "error": f"This Lambda only supports 's3_document_reader', "
                f"received: {tool_name}"
            }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
