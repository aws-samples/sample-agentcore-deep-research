# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import threading
from pathlib import Path

import boto3
from botocore.config import Config
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

REPORT_FILE_PATH = "/tmp/research_report.md"  # noqa: S108  # nosec B108
REPORTS_BUCKET = os.environ.get("STAGING_BUCKET_NAME", "")
URL_EXPIRATION = 3600  # 1 hour


class ReportS3UploadHook(HookProvider):
    """
    Hook that uploads the research report to S3 after every file_write or editor call.

    After uploading, it modifies the tool result to include a pre-signed URL
    that the frontend can use to fetch and display the report.
    """

    def __init__(self):
        region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            config=Config(s3={"addressing_style": "virtual"}),
        )
        self._last_url: str | None = None
        self._lock = threading.Lock()

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.upload_report_to_s3)

    def _do_upload(self, session_id: str, content: str) -> str | None:
        """Upload report and any chart images to S3, return pre-signed URL for report."""
        try:
            s3_key = f"reports/{session_id}/report.md"

            self.s3_client.put_object(
                Bucket=REPORTS_BUCKET,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType="text/markdown",
            )
            print(f"[HOOK] Uploaded report to s3://{REPORTS_BUCKET}/{s3_key}")

            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": REPORTS_BUCKET, "Key": s3_key},
                ExpiresIn=URL_EXPIRATION,
            )
            with self._lock:
                self._last_url = presigned_url
            return presigned_url
        except Exception as e:
            print(f"[HOOK ERROR] Failed to upload report to S3: {e}")
            return None

    def upload_report_to_s3(self, event: AfterToolCallEvent) -> None:
        """Upload report to S3 after file_write/editor and add URL to result."""
        tool_name = event.tool_use.get("name", "")

        # Only process file write operations (not file_read)
        if tool_name not in ("file_write", "editor"):
            return

        # Check if the operation involves the report file
        tool_input = event.tool_use.get("input", {})
        # Different tools may use different param names: path, file_path, file, filename
        file_path = (
            tool_input.get("path", "")
            or tool_input.get("file_path", "")
            or tool_input.get("file", "")
            or tool_input.get("filename", "")
        )

        if "research_report" not in file_path:
            return

        # Get session_id from invocation state
        session_id = event.invocation_state.get("session_id", "default")

        # Read the current report file
        report_path = Path(REPORT_FILE_PATH)
        if not report_path.exists():
            print(f"[HOOK] Report file not found at {REPORT_FILE_PATH}")
            return

        if not REPORTS_BUCKET:
            print("[HOOK] STAGING_BUCKET_NAME not set, skipping S3 upload")
            return

        report_content = report_path.read_text()

        # Upload synchronously but quickly - S3 is fast
        presigned_url = self._do_upload(session_id, report_content)

        # Append the URL to the tool result so frontend can fetch it
        if presigned_url and "content" in event.result and event.result["content"]:
            original_text = event.result["content"][0].get("text", "")
            event.result["content"][0]["text"] = (
                f"{original_text}\n\n[REPORT_URL:{presigned_url}]"
            )
            print("[HOOK] Added pre-signed URL to tool result")
