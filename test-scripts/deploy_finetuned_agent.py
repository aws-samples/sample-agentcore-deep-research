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
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Deploy the deep research agent with a fine-tuned SageMaker model"
    )
    parser.add_argument("--endpoint-name", type=str, required=True, help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))

    args = parser.parse_args()

    print("=" * 60)
    print("Deploy Agent with Fine-tuned Model (SageMaker)")
    print("=" * 60)
    print(f"  Endpoint: {args.endpoint_name}")
    print(f"  Region:   {args.region}")
    print()

    # Update the endpoint name in the CDK stack
    cdk_dir = Path(__file__).parent.parent / "infra-cdk"
    stack_file = cdk_dir / "lib" / "rl-training-stack.ts"

    content = stack_file.read_text()
    content = re.sub(
        r'SAGEMAKER_ENDPOINT_NAME: ".*?"',
        f'SAGEMAKER_ENDPOINT_NAME: "{args.endpoint_name}"',
        content,
    )
    stack_file.write_text(content)
    print(f"✓ Updated endpoint name in rl-training-stack.ts")

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

    print(f"\n✓ Fine-tuned agent deployed!")
    print(f"  Runtime ARN: {finetuned_arn}")
    print(f"  Endpoint:    {args.endpoint_name}")
    print(f"\nTo eval:")
    print(f"  uv run test-scripts/eval-agent.py --benchmark hle-search --max-questions 10 --tag finetuned --runtime-arn {finetuned_arn}")


if __name__ == "__main__":
    main()
