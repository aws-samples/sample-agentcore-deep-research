#!/usr/bin/env python3
"""
Deploy a fine-tuned model as a SageMaker real-time endpoint.

After training completes, this script:
1. Locates the model checkpoint on S3
2. Creates a SageMaker model + endpoint config + endpoint
3. Waits for the endpoint to be InService
4. Outputs the endpoint URL for use with the agent

Usage:
    uv run test-scripts/deploy_model.py --job-name <sagemaker-training-job-name>
    uv run test-scripts/deploy_model.py --s3-uri s3://bucket/path/to/model/

Prerequisites:
    - Completed training job (checkpoint in S3) or model weights uploaded to S3
    - IAM permissions for SageMaker endpoint creation
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import boto3


def get_checkpoint_s3_uri(job_name: str, region: str) -> str:
    """Get the S3 output path from a completed SageMaker training job."""
    sagemaker = boto3.client("sagemaker", region_name=region)
    response = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]

    if status != "Completed":
        print(f"ERROR: Training job '{job_name}' status is '{status}', not 'Completed'")
        sys.exit(1)

    s3_output = response["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"Training job completed. Output: {s3_output}")
    return s3_output


def main():
    parser = argparse.ArgumentParser(
        description="Deploy fine-tuned model as a SageMaker endpoint"
    )
    parser.add_argument("--job-name", type=str, default=None, help="SageMaker training job name")
    parser.add_argument("--s3-uri", type=str, default=None, help="S3 URI of model weights (alternative to --job-name)")
    parser.add_argument("--endpoint-name", type=str, default=None, help="Endpoint name (default: auto-generated)")
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge", help="Endpoint instance type (default: ml.g5.2xlarge)")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for endpoint to be InService")

    args = parser.parse_args()

    if not args.job_name and not args.s3_uri:
        print("ERROR: Provide either --job-name or --s3-uri")
        sys.exit(1)

    region = args.region
    sagemaker = boto3.client("sagemaker", region_name=region)

    print("=" * 60)
    print("Deploy Fine-tuned Model — SageMaker Endpoint")
    print("=" * 60 + "\n")

    # Get model location
    if args.s3_uri:
        model_data_url = args.s3_uri
    else:
        model_data_url = get_checkpoint_s3_uri(args.job_name, region)

    # Get role ARN from RL stack
    try:
        cfn = boto3.client("cloudformation", region_name=region)
        resp = cfn.describe_stacks(StackName="deep-research-rl")
        outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}
        role_arn = outputs.get("RLTrainingRoleArn")
    except Exception:
        print("ERROR: Could not get role ARN from deep-research-rl stack")
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    endpoint_name = args.endpoint_name or f"deep-research-rl-{timestamp}"
    model_name = f"dr-rl-model-{timestamp}"
    config_name = f"dr-rl-config-{timestamp}"

    # Choose container: DJL LMI with vLLM backend
    is_training_output = model_data_url.endswith("model.tar.gz") or model_data_url.endswith("/output/")

    from sagemaker.core import image_uris
    image = image_uris.retrieve(framework="djl-lmi", region=region, version="0.31.0")

    # For training outputs, SageMaker extracts model.tar.gz to /opt/ml/model
    if is_training_output:
        if not model_data_url.endswith("model.tar.gz"):
            model_data_url = model_data_url.rstrip("/") + "/model.tar.gz"
        model_id = "/opt/ml/model"
    else:
        model_id = model_data_url

    env = {
        "HF_MODEL_ID": model_id,
        "OPTION_ROLLING_BATCH": "vllm",
        "OPTION_TENSOR_PARALLEL_DEGREE": "1",
        "OPTION_MAX_MODEL_LEN": "4096",
        "OPTION_DTYPE": "bf16",
        "OPTION_ENABLE_AUTO_TOOL_CHOICE": "true",
        "OPTION_TOOL_CALL_PARSER": "hermes",
    }

    print(f"  Model data:    {model_data_url}")
    print(f"  Endpoint:      {endpoint_name}")
    print(f"  Instance:      {args.instance_type}")
    print(f"  Image:         {image}")
    print(f"  Role:          {role_arn}")
    print()

    # Create SageMaker model
    print("Creating model...")
    container_config = {
        "Image": image,
        "Environment": env,
    }
    # For training outputs, use ModelDataUrl (SageMaker extracts .tar.gz to /opt/ml/model)
    if is_training_output:
        container_config["ModelDataUrl"] = model_data_url
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer=container_config,
        ExecutionRoleArn=role_arn,
    )

    # Create endpoint config with instance pools for capacity fallback
    # Single-GPU instances with 24GB+ VRAM, mix of sizes for best capacity odds (max 5 pools)
    instance_pool_types = [
        "ml.g5.xlarge",
        "ml.g5.4xlarge",
        "ml.g6.4xlarge",
        "ml.g5.8xlarge",
        "ml.g5.16xlarge",
    ]
    # Put requested type first, deduplicate
    ordered = [args.instance_type] + [t for t in instance_pool_types if t != args.instance_type]
    pools = [{"InstanceType": t, "Priority": i + 1} for i, t in enumerate(ordered[:5])]

    print(f"Creating endpoint config with instance pools:")
    for p in pools:
        print(f"  Priority {p['Priority']}: {p['InstanceType']}")
    print()

    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "primary",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstancePools": pools,
            "VariantInstanceProvisionTimeoutInSeconds": 1800,
        }],
    )

    # Create endpoint
    print("Creating endpoint...")
    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    if args.no_wait:
        print(f"\n✓ Endpoint creation started: {endpoint_name}")
        print(f"  Check status: aws sagemaker describe-endpoint --endpoint-name {endpoint_name} --region {region}")
    else:
        print("Waiting for endpoint to be InService (tries all instance pools)...")
        start = time.time()
        while True:
            resp = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]
            elapsed = int(time.time() - start)
            print(f"  [{elapsed}s] Status: {status}")

            if status == "InService":
                break
            elif status == "Failed":
                print(f"\n✗ Endpoint failed: {resp.get('FailureReason', 'unknown')}")
                sys.exit(1)

            time.sleep(30)

    endpoint_url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"
    print(f"\n✓ Endpoint ready: {endpoint_name}")
    print(f"  URL: {endpoint_url}")
    print(f"\n  Next: deploy the agent with this endpoint:")
    print(f"    uv run test-scripts/deploy_finetuned_agent.py --endpoint-name {endpoint_name}")


if __name__ == "__main__":
    main()
