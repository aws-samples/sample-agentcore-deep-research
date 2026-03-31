# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import threading
from pathlib import Path

import boto3
from botocore.config import Config
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry

REPORT_FILE_PATH = "/tmp/research_report.md"  # noqa: S108  # nosec B108
DATA_ANALYSIS_PLAN_PATH = "/tmp/data_analysis_plan.md"  # noqa: S108  # nosec B108
REPORTS_BUCKET = os.environ.get("STAGING_BUCKET_NAME", "")
URL_EXPIRATION = 3600  # 1 hour

# Map local file substrings to their S3 key suffix
_TRACKED_FILES = {
    "research_report": ("report.md", REPORT_FILE_PATH),
    "data_analysis_plan": ("data_analysis_plan.md", DATA_ANALYSIS_PLAN_PATH),
}


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

    def _do_upload(self, session_id: str, s3_suffix: str, content: str) -> str | None:
        """Upload file to S3, return pre-signed URL."""
        try:
            s3_key = f"reports/{session_id}/{s3_suffix}"

            self.s3_client.put_object(
                Bucket=REPORTS_BUCKET,
                Key=s3_key,
                Body=content.encode("utf-8"),
                ContentType="text/markdown",
            )
            print(f"[HOOK] Uploaded to s3://{REPORTS_BUCKET}/{s3_key}")

            presigned_url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": REPORTS_BUCKET, "Key": s3_key},
                ExpiresIn=URL_EXPIRATION,
            )
            with self._lock:
                self._last_url = presigned_url
            return presigned_url
        except Exception as e:
            print(f"[HOOK ERROR] Failed to upload to S3: {e}")
            return None

    def upload_report_to_s3(self, event: AfterToolCallEvent) -> None:
        """Upload tracked files to S3 after file_write/editor and add URL to result."""
        tool_name = event.tool_use.get("name", "")

        if tool_name not in ("file_write", "editor"):
            return

        tool_input = event.tool_use.get("input", {})
        file_path = (
            tool_input.get("path", "")
            or tool_input.get("file_path", "")
            or tool_input.get("file", "")
            or tool_input.get("filename", "")
        )

        # Find which tracked file this write targets
        matched = None
        for key, (s3_suffix, local_path) in _TRACKED_FILES.items():
            if key in file_path:
                matched = (key, s3_suffix, local_path)
                break

        if not matched:
            return

        file_key, s3_suffix, local_path = matched
        session_id = event.invocation_state.get("session_id", "default")

        local = Path(local_path)
        if not local.exists():
            print(f"[HOOK] File not found at {local_path}")
            return

        if not REPORTS_BUCKET:
            print("[HOOK] STAGING_BUCKET_NAME not set, skipping S3 upload")
            return

        content = local.read_text()
        presigned_url = self._do_upload(session_id, s3_suffix, content)

        # Only inject URL tag for the report (frontend uses it for real-time display)
        if (
            file_key == "research_report"
            and presigned_url
            and "content" in event.result
            and event.result["content"]
        ):
            original_text = event.result["content"][0].get("text", "")
            event.result["content"][0]["text"] = (
                f"{original_text}\n\n[REPORT_URL:{presigned_url}]"
            )
            print("[HOOK] Added REPORT_URL pre-signed URL to tool result")
