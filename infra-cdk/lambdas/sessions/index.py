# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


"""Sessions API Lambda Handler"""

import base64
import gzip
import json
import os
import time
from typing import Any

import boto3
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.logging.correlation_paths import API_GATEWAY_REST
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

TABLE_NAME = os.environ["TABLE_NAME"]
REPORT_BUCKET = os.environ.get("REPORT_BUCKET", "")
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "*")

cors_origins = [
    origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",") if origin.strip()
]
primary_origin = cors_origins[0] if cors_origins else "*"
extra_origins = cors_origins[1:] if len(cors_origins) > 1 else None

cors_config = CORSConfig(
    allow_origin=primary_origin,
    extra_origins=extra_origins,
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=True,
)

dynamodb = boto3.client("dynamodb")
s3_client = boto3.client("s3")

tracer = Tracer()
logger = Logger()
app = APIGatewayRestResolver(cors=cors_config)

MAX_SESSION_ID_LENGTH = 100
MAX_TITLE_LENGTH = 200
MAX_MESSAGES_BYTES = 350_000  # gzipped limit, well under DynamoDB 400KB item limit
PRESIGNED_URL_EXPIRY = 3600  # 1 hour


class SaveSessionRequest(BaseModel):
    """Request payload for saving a session."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

    title: str = Field("", max_length=MAX_TITLE_LENGTH)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    enabled_sources: dict[str, bool] = Field(default_factory=dict)
    s3_file_input: str = Field("", max_length=2000)
    has_report: bool = Field(False)


def _extract_user_id(event: dict) -> str | None:
    """Extract user ID from Cognito authorizer claims."""
    request_context = event.get("requestContext", {})
    authorizer = request_context.get("authorizer", {})
    claims = authorizer.get("claims", {})
    return claims.get("sub")


def _validate_session_id(session_id: str) -> bool:
    """Validate session ID format."""
    if not session_id or len(session_id) > MAX_SESSION_ID_LENGTH:
        return False
    return session_id.replace("-", "").replace("_", "").isalnum()


def _compress_messages(messages: list[dict[str, Any]]) -> str:
    """Compress messages to gzipped base64 string for DynamoDB storage."""
    json_bytes = json.dumps(messages, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(json_bytes)
    return base64.b64encode(compressed).decode("ascii")


def _decompress_messages(data: str) -> list[dict[str, Any]]:
    """Decompress messages from gzipped base64 string."""
    compressed = base64.b64decode(data)
    json_bytes = gzip.decompress(compressed)
    return json.loads(json_bytes)


def _generate_report_urls(session_id: str) -> dict[str, str | None]:
    """Generate fresh pre-signed URLs for session reports in S3."""
    if not REPORT_BUCKET:
        return {"reportContent": None, "reportPdfUrl": None}

    report_content = None
    report_pdf_url = None

    # fetch report markdown content
    try:
        response = s3_client.get_object(
            Bucket=REPORT_BUCKET,
            Key=f"reports/{session_id}/report.md",
        )
        report_content = response["Body"].read().decode("utf-8")
    except ClientError:
        pass

    # generate pre-signed URL for PDF
    try:
        s3_client.head_object(
            Bucket=REPORT_BUCKET,
            Key=f"reports/{session_id}/report.pdf",
        )
        report_pdf_url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": REPORT_BUCKET,
                "Key": f"reports/{session_id}/report.pdf",
            },
            ExpiresIn=PRESIGNED_URL_EXPIRY,
        )
    except ClientError:
        pass

    return {"reportContent": report_content, "reportPdfUrl": report_pdf_url}


@app.get("/sessions")
def list_sessions() -> dict[str, Any]:
    """List all sessions for the authenticated user, sorted by updatedAt descending."""
    user_id = _extract_user_id(app.current_event.raw_event)
    if not user_id:
        return {"error": "Unauthorized"}, 401

    try:
        response = dynamodb.query(
            TableName=TABLE_NAME,
            IndexName="userId-updatedAt-index",
            KeyConditionExpression="userId = :uid",
            ExpressionAttributeValues={":uid": {"S": user_id}},
            ScanIndexForward=False,
            ProjectionExpression="sessionId, title, createdAt, updatedAt, hasReport",
        )

        sessions = []
        for item in response.get("Items", []):
            sessions.append(
                {
                    "sessionId": item["sessionId"]["S"],
                    "title": item.get("title", {}).get("S", "Untitled"),
                    "createdAt": int(item.get("createdAt", {}).get("N", "0")),
                    "updatedAt": int(item.get("updatedAt", {}).get("N", "0")),
                    "hasReport": item.get("hasReport", {}).get("BOOL", False),
                }
            )

        return {"sessions": sessions}

    except ClientError as e:
        logger.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return {"error": "Internal server error"}, 500


@app.get("/sessions/<session_id>")
def get_session(session_id: str) -> dict[str, Any]:
    """Load a full session with messages and fresh report URLs."""
    user_id = _extract_user_id(app.current_event.raw_event)
    if not user_id:
        return {"error": "Unauthorized"}, 401

    if not _validate_session_id(session_id):
        return {"error": "Invalid session ID"}, 400

    try:
        response = dynamodb.get_item(
            TableName=TABLE_NAME,
            Key={
                "userId": {"S": user_id},
                "sessionId": {"S": session_id},
            },
        )

        item = response.get("Item")
        if not item:
            return {"error": "Session not found"}, 404

        # decompress messages
        messages = []
        if "messages" in item:
            messages = _decompress_messages(item["messages"]["S"])

        # parse enabled sources
        enabled_sources = {}
        if "enabledSources" in item:
            enabled_sources = json.loads(item["enabledSources"]["S"])

        result = {
            "sessionId": item["sessionId"]["S"],
            "title": item.get("title", {}).get("S", "Untitled"),
            "messages": messages,
            "enabledSources": enabled_sources,
            "s3FileInput": item.get("s3FileInput", {}).get("S", ""),
            "createdAt": int(item.get("createdAt", {}).get("N", "0")),
            "updatedAt": int(item.get("updatedAt", {}).get("N", "0")),
            "reportContent": None,
            "reportPdfUrl": None,
        }

        # always check S3 for reports (hasReport flag may be stale)
        report_data = _generate_report_urls(session_id)
        result.update(report_data)

        return result

    except ClientError as e:
        logger.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return {"error": "Internal server error"}, 500


@app.put("/sessions/<session_id>")
def save_session(session_id: str) -> dict[str, Any]:
    """Create or update a session."""
    user_id = _extract_user_id(app.current_event.raw_event)
    if not user_id:
        return {"error": "Unauthorized"}, 401

    if not _validate_session_id(session_id):
        return {"error": "Invalid session ID"}, 400

    try:
        data = SaveSessionRequest(**app.current_event.json_body)
    except Exception as e:
        logger.warning(f"Validation error: {str(e)}")
        return {"error": str(e)}, 400

    # auto-generate title from first user message if not provided
    title = data.title
    if not title and data.messages:
        for msg in data.messages:
            if msg.get("role") == "user" and msg.get("content"):
                title = msg["content"][:MAX_TITLE_LENGTH]
                break
    if not title:
        title = "Untitled"

    now = int(time.time() * 1000)

    # compress messages
    compressed_messages = _compress_messages(data.messages)
    if len(base64.b64decode(compressed_messages)) > MAX_MESSAGES_BYTES:
        return {"error": "Session data too large"}, 413

    try:
        # check if session exists to preserve createdAt
        existing = dynamodb.get_item(
            TableName=TABLE_NAME,
            Key={
                "userId": {"S": user_id},
                "sessionId": {"S": session_id},
            },
            ProjectionExpression="createdAt",
        )
        created_at = now
        if "Item" in existing and "createdAt" in existing["Item"]:
            created_at = int(existing["Item"]["createdAt"]["N"])

        item = {
            "userId": {"S": user_id},
            "sessionId": {"S": session_id},
            "title": {"S": title},
            "messages": {"S": compressed_messages},
            "enabledSources": {"S": json.dumps(data.enabled_sources)},
            "s3FileInput": {"S": data.s3_file_input},
            "hasReport": {"BOOL": data.has_report},
            "createdAt": {"N": str(created_at)},
            "updatedAt": {"N": str(now)},
        }

        dynamodb.put_item(TableName=TABLE_NAME, Item=item)

        return {
            "success": True,
            "sessionId": session_id,
            "title": title,
            "createdAt": created_at,
            "updatedAt": now,
        }

    except ClientError as e:
        logger.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return {"error": "Internal server error"}, 500


@app.delete("/sessions/<session_id>")
def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session."""
    user_id = _extract_user_id(app.current_event.raw_event)
    if not user_id:
        return {"error": "Unauthorized"}, 401

    if not _validate_session_id(session_id):
        return {"error": "Invalid session ID"}, 400

    try:
        dynamodb.delete_item(
            TableName=TABLE_NAME,
            Key={
                "userId": {"S": user_id},
                "sessionId": {"S": session_id},
            },
        )

        return {"success": True}

    except ClientError as e:
        logger.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return {"error": "Internal server error"}, 500


@logger.inject_lambda_context(correlation_id_path=API_GATEWAY_REST)
def handler(event: dict, context: LambdaContext) -> dict:
    """
    Lambda handler for sessions API.

    Parameters
    ----------
    event : dict
        API Gateway event
    context : LambdaContext
        Lambda context

    Returns
    -------
    dict
        API Gateway response
    """
    return app.resolve(event, context)
