# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import json
import logging
import os
import tempfile

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")

MAX_LINES_LIMIT = 5000
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


def _slice_lines(text: str, start_line: int, end_line: int) -> tuple[str, int]:
    """Extract a range of lines. Returns (content, total_lines)."""
    lines = text.splitlines()
    total = len(lines)
    start = max(0, start_line - 1)  # convert to 0-indexed
    end = min(total, end_line)
    content = "\n".join(lines[start:end])
    return content, total


def read_pdf_file(s3_uri: str, start_line: int = 1, end_line: int = 500) -> str:
    """
    Download a PDF from S3, convert to markdown, and return a line range.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the PDF file
    start_line : int
        First line to return (1-indexed)
    end_line : int
        Last line to return (inclusive)
    """
    import pymupdf4llm  # noqa: E402

    bucket, key = parse_s3_uri(s3_uri)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        s3_client.download_file(bucket, key, tmp.name)
        markdown = pymupdf4llm.to_markdown(tmp.name, show_progress=False)

    content, total = _slice_lines(markdown, start_line, end_line)
    header = (
        f"File: {s3_uri} (PDF, {total} lines total, "
        f"showing lines {start_line}-{min(end_line, total)})"
    )
    return f"{header}\n\n{content}"


def read_text_file(s3_uri: str, start_line: int = 1, end_line: int = 500) -> str:
    """
    Read a text file from S3 and return a range of lines.

    Parameters
    ----------
    s3_uri : str
        S3 URI of the text file to read
    start_line : int
        First line to return (1-indexed)
    end_line : int
        Last line to return (inclusive)
    """
    bucket, key = parse_s3_uri(s3_uri)

    response = s3_client.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read()

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("latin-1")

    content, total = _slice_lines(text, start_line, end_line)
    header = (
        f"File: {s3_uri} ({total} lines total, "
        f"showing lines {start_line}-{min(end_line, total)})"
    )
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

            start_line = event.get("start_line", 1)
            end_line = event.get("end_line", 500)
            if not isinstance(start_line, int) or start_line < 1:
                start_line = 1
            if not isinstance(end_line, int) or end_line < start_line:
                end_line = start_line + 499
            # cap range to MAX_LINES_LIMIT
            if end_line - start_line + 1 > MAX_LINES_LIMIT:
                end_line = start_line + MAX_LINES_LIMIT - 1

            _, key = parse_s3_uri(s3_uri)
            ext = _get_file_extension(key)

            if ext in PDF_EXTENSIONS:
                result = read_pdf_file(s3_uri, start_line, end_line)
            else:
                result = read_text_file(s3_uri, start_line, end_line)

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
