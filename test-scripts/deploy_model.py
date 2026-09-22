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
    parser.add_argument(
        "--job-name", type=str, default=None, help="SageMaker training job name"
    )
    parser.add_argument(
        "--s3-uri",
        type=str,
        default=None,
        help="S3 URI of model weights (alternative to --job-name)",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="HuggingFace model ID to deploy directly from the Hub, e.g. Qwen/Qwen3.6-27B "
        "(alternative to --job-name/--s3-uri — for base/untrained model baselines)",
    )
    parser.add_argument(
        "--tensor-parallel-degree",
        type=int,
        default=1,
        help="Number of GPUs to shard the model across via tensor parallelism "
        "(default: 1). Use 4 for large models on ml.g5.12xlarge (4x A10G, 96GB total).",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["bitsandbytes", "awq", "gptq", "fp8"],
        help="On-the-fly quantization method (via vLLM) to shrink VRAM footprint. "
        "Use for large models on single-GPU instances, e.g. bitsandbytes to fit "
        "a 27B model on a single 24GB A10G without pre-quantized weights.",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=None,
        help="Endpoint name (default: auto-generated)",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g5.2xlarge",
        help="Endpoint instance type (default: ml.g5.2xlarge)",
    )
    parser.add_argument(
        "--region", type=str, default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max context length (default: 4096). vLLM reserves KV-cache memory "
        "up front based on this — reduce it if the endpoint OOMs during startup "
        "on a tightly-fitting GPU.",
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Don't wait for endpoint to be InService"
    )
    parser.add_argument(
        "--enable-capacity-fallback",
        action="store_true",
        help="Allow SageMaker to fall back to other single-GPU instance types if the "
        "requested type lacks capacity (only applies when tensor-parallel-degree=1). "
        "Off by default to avoid silently colliding with account-level quotas on "
        "instance types used elsewhere.",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Cap concurrent sequences. Required for hybrid models (Qwen3.5) whose "
        "recurrent-state budget is below vLLM's default of 256.",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=None,
        help="vLLM tool-call parser for the model family (e.g. qwen3_coder, "
        "pythonic, hermes). Omitted by default — set it to match the model.",
    )
    parser.add_argument(
        "--reasoning-parser",
        default=None,
        help="vLLM reasoning parser for the model family (e.g. qwen3). Omitted "
        "by default — a mismatched parser corrupts output.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Override the serving container image URI. Use a newer vLLM "
        "container when the default lacks support for a recent architecture.",
    )

    args = parser.parse_args()

    if not args.job_name and not args.s3_uri and not args.hf_model_id:
        print("ERROR: Provide --job-name, --s3-uri, or --hf-model-id")
        sys.exit(1)

    region = args.region
    sagemaker = boto3.client("sagemaker", region_name=region)

    print("=" * 60)
    print("Deploy Fine-tuned Model — SageMaker Endpoint")
    print("=" * 60 + "\n")

    # Get model location
    if args.hf_model_id:
        model_data_url = args.hf_model_id
    elif args.s3_uri:
        model_data_url = args.s3_uri
    else:
        model_data_url = get_checkpoint_s3_uri(args.job_name, region)

    # Get role ARN from RL stack
    try:
        cfn = boto3.client("cloudformation", region_name=region)
        resp = cfn.describe_stacks(StackName="deep-research-rl")
        outputs = {
            o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]
        }
        role_arn = outputs.get("RLTrainingRoleArn")
    except Exception:
        # Cross-region fallback: try the us-west-2 stack
        try:
            cfn = boto3.client("cloudformation", region_name="us-west-2")
            resp = cfn.describe_stacks(StackName="deep-research-rl")
            outputs = {
                o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]
            }
            role_arn = outputs.get("RLTrainingRoleArn")
        except Exception:
            print("ERROR: Could not get role ARN from deep-research-rl stack")
            sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    endpoint_name = args.endpoint_name or f"deep-research-rl-{timestamp}"
    model_name = f"dr-rl-model-{timestamp}"
    config_name = f"dr-rl-config-{timestamp}"

    # Choose container: native SageMaker vLLM container (not DJL-LMI, which has
    # rolling-batch bugs with tool-call streaming for Qwen3.6).
    # Ref: https://huggingface.co/blog/alvarobartt/agents-on-aws-sagemaker
    is_training_output = model_data_url.endswith(
        "model.tar.gz"
    ) or model_data_url.endswith("/output/")

    image = (
        args.image
        or f"763104351884.dkr.ecr.{region}.amazonaws.com/vllm:server-sagemaker-cuda-v2"
    )

    # For training outputs, SageMaker extracts model.tar.gz to /opt/ml/model
    if is_training_output:
        if not model_data_url.endswith("model.tar.gz"):
            model_data_url = model_data_url.rstrip("/") + "/model.tar.gz"
        model_id = "/opt/ml/model"
    else:
        model_id = model_data_url

    # Native vLLM container uses SM_VLLM_* env vars (not OPTION_*)
    env = {
        "SM_VLLM_MODEL": model_id,
        "SM_VLLM_TENSOR_PARALLEL_SIZE": str(args.tensor_parallel_degree),
        "SM_VLLM_MAX_MODEL_LEN": str(args.max_model_len),
        "SM_VLLM_ENABLE_AUTO_TOOL_CHOICE": "true",
        "SM_VLLM_ENABLE_LOG_REQUESTS": "true",
    }
    # Qwen3.5 is hybrid attention/Gated-DeltaNet, so vLLM reserves a recurrent state
    # block per sequence. When the model's block budget is below vLLM's default
    # max_num_seqs (256) the engine refuses to start with "exceeds available Mamba
    # cache blocks", which surfaces only as a failed ping health check.
    if args.max_num_seqs:
        env["SM_VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)
    # Parsers are model-family specific — omit when not applicable so vLLM
    # falls back to its defaults instead of misreading another family's format.
    if args.tool_call_parser:
        env["SM_VLLM_TOOL_CALL_PARSER"] = args.tool_call_parser
    if args.reasoning_parser:
        env["SM_VLLM_REASONING_PARSER"] = args.reasoning_parser
    if args.quantize:
        env["SM_VLLM_QUANTIZATION"] = args.quantize

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

    # Create endpoint config. Use a single, exact instance type by default —
    # avoids silently falling back to other instance types that may hit
    # unrelated account-level quota limits (e.g. an existing endpoint already
    # using that type elsewhere). Opt into multi-type capacity fallback with
    # --enable-capacity-fallback if desired.
    if args.enable_capacity_fallback and args.tensor_parallel_degree == 1:
        # All ml.g6e.{2,4,8,16}xlarge sizes carry exactly one L40S (48 GB), so
        # they are interchangeable for a TP=1 deployment and differ only in
        # host CPU/RAM. Larger g5 sizes (A10G, 24 GB) are last-resort fallbacks
        # and only fit smaller models. Keeping the GPU identical across the
        # pool matters when comparing models: the served weights, dtype and
        # context length stay the same regardless of which pool wins.
        instance_pool_types = [
            "ml.g6e.2xlarge",
            "ml.g6e.4xlarge",
            "ml.g6e.8xlarge",
            "ml.g6e.16xlarge",
            "ml.g5.12xlarge",
        ]
        ordered = [args.instance_type] + [
            t for t in instance_pool_types if t != args.instance_type
        ]
        pools = [
            {"InstanceType": t, "Priority": i + 1} for i, t in enumerate(ordered[:5])
        ]
    else:
        pools = [{"InstanceType": args.instance_type, "Priority": 1}]

    print("Creating endpoint config with instance pools:")
    for p in pools:
        print(f"  Priority {p['Priority']}: {p['InstanceType']}")
    print()

    variant = {
        "VariantName": "primary",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        # Required for native vLLM container (provides CUDA/driver layer)
        "InferenceAmiVersion": "al2-ami-sagemaker-inference-gpu-3-1",
    }
    if len(pools) >= 2:
        # Multiple candidate instance types: use InstancePools for capacity fallback
        variant["InstancePools"] = pools
        variant["VariantInstanceProvisionTimeoutInSeconds"] = 1800
    else:
        # Single instance type: SageMaker requires plain InstanceType (InstancePools needs >=2)
        variant["InstanceType"] = pools[0]["InstanceType"]

    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[variant],
    )

    # Create endpoint
    print("Creating endpoint...")
    sagemaker.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    if args.no_wait:
        print(f"\n✓ Endpoint creation started: {endpoint_name}")
        print(
            f"  Check status: aws sagemaker describe-endpoint --endpoint-name {endpoint_name} --region {region}"
        )
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
    print("\n  Next: deploy the agent with this endpoint:")
    print(
        f"    uv run test-scripts/deploy_finetuned_agent.py --endpoint-name {endpoint_name}"
    )


if __name__ == "__main__":
    main()
