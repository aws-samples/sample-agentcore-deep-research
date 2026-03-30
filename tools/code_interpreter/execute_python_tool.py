# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Strands @tool wrapper for AgentCore Code Interpreter."""

import base64
import json
import os
import re

import boto3
from botocore.config import Config
from strands import tool

from tools.code_interpreter.code_interpreter_tools import CodeInterpreterTools

_interpreter = None
_s3_client = None

STAGING_BUCKET = os.environ.get("STAGING_BUCKET_NAME", "")
URL_EXPIRATION = 3600


def _get_interpreter() -> CodeInterpreterTools:
    global _interpreter
    if _interpreter is None:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        _interpreter = CodeInterpreterTools(region)
    return _interpreter


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        region = os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        _s3_client = boto3.client(
            "s3", region_name=region, config=Config(s3={"addressing_style": "virtual"})
        )
    return _s3_client


def _upload_chart(b64_data: str, chart_name: str) -> str:
    """Decode base64 PNG, upload to S3, return presigned URL."""
    img_bytes = base64.b64decode(b64_data)
    s3_key = f"reports/charts/{chart_name}.png"
    client = _get_s3_client()
    client.put_object(
        Bucket=STAGING_BUCKET, Key=s3_key, Body=img_bytes, ContentType="image/png"
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": STAGING_BUCKET, "Key": s3_key},
        ExpiresIn=URL_EXPIRATION,
    )


@tool
def execute_python(code: str, chart_name: str = "") -> str:
    """Execute Python code for data visualization and statistical analysis ONLY.

    Use this tool exclusively when the user clicks "Analyze" to generate charts
    and run statistical tests based on the Data Analysis Recommendations section.
    Do NOT use this tool during the research or report writing phases.

    The sandbox has numpy, pandas, matplotlib, seaborn, scipy, and scikit-learn.

    IMPORTANT: The sandbox is remote. To output a chart, save to BytesIO and print base64:

        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)

    Always set text/axis colors for visibility on both light and dark backgrounds:
        plt.rcParams.update({'text.color': '#333333', 'axes.labelcolor': '#333333',
                             'xtick.color': '#333333', 'ytick.color': '#333333'})
        plt.close()
        buf.seek(0)
        print(f"CHART_BASE64:{base64.b64encode(buf.read()).decode()}")

    The tool uploads the image to S3 and returns a presigned URL.
    Use editor to replace the CHART_PLACEHOLDER in the report with: ![Chart Title](returned_url)

    Args:
        code: Python code for data analysis or visualization. Include all imports.
        chart_name: Short filename for the chart (e.g., "gold_forecasts").
    """  # noqa: E501
    result = _get_interpreter().execute_python_securely(code)

    if STAGING_BUCKET and chart_name:
        try:
            match = re.search(r"CHART_BASE64:([A-Za-z0-9+/=]+)", result)
            if match:
                url = _upload_chart(match.group(1), chart_name)
                return json.dumps({"chart_url": url, "status": "success"})
        except Exception as e:
            return json.dumps(
                {"error": f"Chart upload failed: {e}", "raw_result": result}
            )

    return result
