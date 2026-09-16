#!/usr/bin/env python3
"""
Deploy the deep research agent with a fine-tuned model on a SageMaker endpoint.

Updates the RL CDK stack's finetuned agent runtime to point at the specified
SageMaker endpoint, then redeploys. The agent uses the same tools and system
prompt as production — just a different model backend.

Usage:
    uv run test-scripts/deploy_finetuned_agent.py --endpoint-name <sagemaker-endpoint>

Prerequisites:
    - Deployed RL stack (npm run deploy:rl)
    - Model deployed to SageMaker endpoint (deploy_model.py)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def wait_for_ready(runtime_arn: str, region: str, timeout: int = 900) -> None:
    """Block until the agent runtime reports READY, printing its version."""
    import boto3

    acc = boto3.client("bedrock-agentcore-control", region_name=region)
    runtime_id = runtime_arn.rsplit("/", 1)[-1]
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = acc.get_agent_runtime(agentRuntimeId=runtime_id)
        status, version = r.get("status"), r.get("agentRuntimeVersion")
        if (status, version) != last:
            print(f"  runtime v{version}: {status}")
            last = (status, version)
        if status == "READY":
            return
        if status in ("CREATE_FAILED", "UPDATE_FAILED", "DELETING"):
            raise RuntimeError(f"runtime is {status}")
        time.sleep(15)
    raise TimeoutError(f"runtime not READY after {timeout}s")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy the deep research agent with a fine-tuned SageMaker model"
    )
    parser.add_argument(
        "--endpoint-name", type=str, required=True, help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--region", type=str, default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Deploy Agent with Fine-tuned Model (SageMaker)")
    print("=" * 60)
    print(f"  Endpoint: {args.endpoint_name}")
    print(f"  Region:   {args.region}")
    print()

    # Set the endpoint name in config.yaml (the CDK stack reads it from there,
    # so infrastructure code stays free of environment-specific values).
    cdk_dir = Path(__file__).parent.parent / "infra-cdk"
    config_file = cdk_dir / "config.yaml"

    if not config_file.exists():
        print(f"ERROR: {config_file} not found. Copy .config_example.yaml first.")
        sys.exit(1)

    content = config_file.read_text()
    if re.search(r"^\s*finetuned_endpoint_name:", content, flags=re.MULTILINE):
        content = re.sub(
            r"^(\s*)finetuned_endpoint_name:.*$",
            rf"\g<1>finetuned_endpoint_name: {args.endpoint_name}",
            content,
            flags=re.MULTILINE,
        )
    elif re.search(r"^training:\s*$", content, flags=re.MULTILINE):
        content = re.sub(
            r"^(training:\s*)$",
            rf"\g<1>\n  finetuned_endpoint_name: {args.endpoint_name}",
            content,
            flags=re.MULTILINE,
        )
    else:
        content = content.rstrip("\n") + (
            f"\n\ntraining:\n  finetuned_endpoint_name: {args.endpoint_name}\n"
        )
    config_file.write_text(content)
    print(
        f"✓ Set training.finetuned_endpoint_name = {args.endpoint_name} in config.yaml"
    )

    # Also record the main stack's staging bucket. Without STAGING_BUCKET_NAME the
    # report upload hook skips S3 upload, no [REPORT_URL:...] is emitted, and the
    # eval silently scores the agent's narration instead of its report.
    try:
        import boto3

        cfn = boto3.client("cloudformation", region_name=args.region)
        outputs = cfn.describe_stacks(StackName="deep-research")["Stacks"][0]["Outputs"]
        bucket = next(
            o["OutputValue"] for o in outputs if o["OutputKey"] == "StagingBucketName"
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"⚠ Could not resolve StagingBucketName ({e}); report URLs will be absent"
        )
        bucket = None

    if bucket:
        content = config_file.read_text()
        if re.search(r"^\s*staging_bucket_name:", content, flags=re.MULTILINE):
            content = re.sub(
                r"^(\s*)staging_bucket_name:.*$",
                rf"\g<1>staging_bucket_name: {bucket}",
                content,
                flags=re.MULTILINE,
            )
        else:
            content = re.sub(
                r"^(\s*finetuned_endpoint_name:.*)$",
                rf"\g<1>\n  staging_bucket_name: {bucket}",
                content,
                flags=re.MULTILINE,
            )
        config_file.write_text(content)
        print(f"✓ Set training.staging_bucket_name = {bucket}")

    # Redeploy RL stack (use bash -lc to pick up user's PATH with nvm/node)
    print("\nDeploying...")
    result = subprocess.run(
        ["bash", "-lc", "npm run deploy:rl"],
        cwd=str(cdk_dir),
        check=False,
    )

    if result.returncode != 0:
        print("\n✗ Deploy failed.")
        sys.exit(1)

    # Get the finetuned agent ARN
    import boto3

    cfn = boto3.client("cloudformation", region_name=args.region)
    resp = cfn.describe_stacks(StackName="deep-research-rl")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}
    finetuned_arn = outputs.get("FinetunedAgentRuntimeArn", "unknown")

    print("\n✓ Fine-tuned agent deployed!")
    print(f"  Runtime ARN: {finetuned_arn}")
    print(f"  Endpoint:    {args.endpoint_name}")

    # Poll for readiness rather than sleeping a fixed interval. Existing sessions keep
    # the old image, but a new session gets the new version as soon as the runtime
    # leaves UPDATING -- so READY at the bumped version is the real signal. Evaluating
    # while it is still UPDATING is what scores an empty runtime.
    wait_for_ready(finetuned_arn, args.region)

    print("\nTo eval:")
    print(
        f"  uv run test-scripts/eval-agent.py --benchmark rubric --max-questions 98 \\\n"
        f"      --parallel 8 --tag finetuned --runtime-arn {finetuned_arn}"
    )
    print(
        "  (match --parallel across runs being compared: 12 produced 21 failures vs 11 at 8)"
    )


if __name__ == "__main__":
    main()
