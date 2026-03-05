"""
Copyright Amazon.com and Affiliates
This code is being licensed under the terms of the Amazon Software License available at https://aws.amazon.com/asl/
"""

import json
import logging
import os
import tempfile

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")

MAX_LINES_LIMIT = 5000
PRESIGNED_URL_EXPIRY = 3600  # 1 hour
PDF_EXTENSIONS = {".pdf"}


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

    path = s3_uri[5:]
    parts = path.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    return parts[0], parts[1]


def _get_file_extension(key: str) -> str:
    """Get lowercase file extension from S3 key."""
    _, ext = os.path.splitext(key)
    return ext.lower()


def _generate_presigned_url(bucket: str, key: str) -> str:
    """Generate a pre-signed URL for an S3 object."""
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGNED_URL_EXPIRY,
    )


def _truncate_lines(text: str, max_lines: int) -> tuple[str, int, bool]:
    """Split text into lines and truncate. Returns (content, total, truncated)."""
    lines = text.splitlines()
    total = len(lines)
    truncated = total > max_lines
    content = "\n".join(lines[:max_lines])
    return content, total, truncated


def read_pdf_file(s3_uri: str, max_lines: int = 500) -> str:
    """
    Download a PDF from S3, convert to markdown, and return content.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the PDF file
    max_lines : int
        Maximum number of lines to return
    """
    import pymupdf4llm  # noqa: E402

    bucket, key = parse_s3_uri(s3_uri)
    max_lines = min(max_lines, MAX_LINES_LIMIT)

    # download PDF to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        s3_client.download_file(bucket, key, tmp.name)
        markdown = pymupdf4llm.to_markdown(tmp.name, show_progress=False)

    content, total, truncated = _truncate_lines(markdown, max_lines)
    presigned_url = _generate_presigned_url(bucket, key)

    header = f"File: {s3_uri} (PDF converted to markdown, {total} lines"
    if truncated:
        header += f", showing first {max_lines})"
    else:
        header += ")"
    header += f"\nDownload link: {presigned_url}"

    return f"{header}\n\n{content}"


def read_text_file(s3_uri: str, max_lines: int = 500) -> str:
    """
    Read a text file from S3 and return up to max_lines lines.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the text file to read
    max_lines : int
        Maximum number of lines to return
    """
    bucket, key = parse_s3_uri(s3_uri)
    max_lines = min(max_lines, MAX_LINES_LIMIT)

    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("latin-1")

    content, total, truncated = _truncate_lines(text, max_lines)
    presigned_url = _generate_presigned_url(bucket, key)

    header = f"File: {s3_uri} ({total} lines total"
    if truncated:
        header += f", showing first {max_lines})"
    else:
        header += ")"
    header += f"\nDownload link: {presigned_url}"

    return f"{header}\n\n{content}"


def handler(event, context):
    """
    S3 file reader Lambda for AgentCore Gateway.

    Reads text files and PDFs from S3. Text files are returned
    directly; PDFs are converted to markdown first using pymupdf4llm.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        delimiter = "___"
        original = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = original[original.index(delimiter) + len(delimiter) :]

        logger.info(f"Processing tool: {tool_name}")

        if tool_name == "s3_text_reader":
            s3_uri = event.get("s3_uri", "")
            if not s3_uri:
                return {"error": "Missing required parameter: s3_uri"}

            max_lines = event.get("max_lines", 500)
            if not isinstance(max_lines, int) or max_lines < 1:
                max_lines = 500

            _, key = parse_s3_uri(s3_uri)
            ext = _get_file_extension(key)

            if ext in PDF_EXTENSIONS:
                result = read_pdf_file(s3_uri=s3_uri, max_lines=max_lines)
            else:
                result = read_text_file(s3_uri=s3_uri, max_lines=max_lines)

            return {"content": [{"type": "text", "text": result}]}

        else:
            return {
                "error": (
                    f"This Lambda only supports 's3_text_reader', received: {tool_name}"
                )
            }

    except s3_client.exceptions.NoSuchKey:
        return {"error": f"File not found: {event.get('s3_uri', '')}"}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Error reading file: {str(e)}"}
