#!/usr/bin/env python3
"""
Launch agentic RL training for the deep research agent.

Uploads training data to S3 and submits a SageMaker GRPO training job
using AgentCore RL Toolkit (slime backend). The agent learns to write
better research reports by using tools during training rollouts.

Usage:
    uv run test-scripts/rl_train.py \
        --data test-scripts/results/rl_train_data.jsonl \
        --agent-arn <RLAgentRuntimeArn> \
        --s3-bucket <RLBucketName>

Prerequisites:
    - Deployed RL stack: cd infra-cdk && npm run deploy:rl
    - Training container pushed: ./training/build_and_push.sh
    - Training data JSONL (see README for format)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Launch agentic RL training for the deep research agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data", type=str, required=True, help="Training JSONL file path"
    )
    parser.add_argument(
        "--agent-arn",
        type=str,
        default=os.environ.get("AGENT_RUNTIME_ARN"),
        help="AgentCore Runtime ARN of RL agent",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("RL_S3_BUCKET"),
        help="S3 bucket for rollout results",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Policy to start from: an HF Hub id or an S3 model.tar.gz "
        "(default: Qwen/Qwen3.5-4B). Use --sft-job-name to continue from an SFT "
        "run instead of naming the artifact by hand.",
    )
    parser.add_argument(
        "--sft-job-name",
        type=str,
        default=None,
        help="Continue RL from this completed SFT training job. Resolves the "
        "job's model artifact and overrides --hf-model-id. This is the SFT -> RL "
        "hand-off: the literature treats SFT purely as an RL cold start.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen3.5-4B",
        help="Model type for SlimeRunner (default: qwen3.5-4B)",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g5.12xlarge",
        help="SageMaker instance type (default: ml.g5.12xlarge)",
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        default=None,
        help="Training container ECR URI (default: auto-detect)",
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        default=None,
        help="SageMaker execution role ARN (default: from CDK stack)",
    )
    parser.add_argument("--exp-id", type=str, default=None, help="Experiment ID")
    parser.add_argument(
        "--num-rollout", type=int, default=30, help="Training iterations (default: 30)"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=4, help="Number of GPUs (default: 4)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=2, help="Tensor parallel size (default: 2)"
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help="Rollout batch size (default: 8)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Samples per prompt for GRPO (default: 4)",
    )
    parser.add_argument(
        "--max-response-len",
        type=int,
        default=16384,
        help="Max response tokens per episode (default: 16384). A report is ~4,200 "
        "tokens and a full episode generates ~10,300, so a small value truncates "
        "the rollout before a report exists and every episode scores at the floor.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, help="Learning rate (default: 1e-6)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent ACR sessions (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-rollout ACR timeout in seconds (default: 900)",
    )

    args = parser.parse_args()

    # Validate required args
    if not args.agent_arn:
        logger.error(
            "--agent-arn required (from `npm run deploy:rl` output: RLAgentRuntimeArn)"
        )
        sys.exit(1)
    if not args.s3_bucket:
        logger.error(
            "--s3-bucket required (from `npm run deploy:rl` output: RLBucketName)"
        )
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        sys.exit(1)

    num_records = sum(1 for _ in open(data_path))

    # Determine training image URI
    account = boto3.client("sts").get_caller_identity()["Account"]
    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    image_uri = (
        args.image_uri
        or f"{account}.dkr.ecr.{region}.amazonaws.com/deep-research-rl-training:latest"
    )

    logger.info(f"{'=' * 60}")
    logger.info("GRPO TRAINING — SageMaker Job")
    logger.info(f"{'=' * 60}")
    logger.info(f"Agent ARN:     {args.agent_arn}")
    logger.info(f"S3 bucket:     {args.s3_bucket}")
    logger.info(f"Model:         {args.hf_model_id}")
    logger.info(f"Data:          {data_path} ({num_records} prompts)")
    logger.info(f"Instance:      {args.instance_type}")
    logger.info(f"Image:         {image_uri}")
    logger.info(f"Num rollouts:  {args.num_rollout}")
    logger.info("")

    # Resolve an SFT job to its artifact so RL can continue from it.
    if args.sft_job_name:
        sm = boto3.client("sagemaker", region_name=region)
        job = sm.describe_training_job(TrainingJobName=args.sft_job_name)
        status = job["TrainingJobStatus"]
        if status != "Completed":
            logger.error(
                f"SFT job '{args.sft_job_name}' is '{status}', not 'Completed'"
            )
            sys.exit(1)
        args.hf_model_id = job["ModelArtifacts"]["S3ModelArtifacts"]
        logger.info(f"Continuing RL from SFT job {args.sft_job_name}")
        logger.info(f"  policy artifact: {args.hf_model_id}")

    # Upload training data to S3
    s3 = boto3.client("s3")
    s3_data_key = f"training-data/{data_path.name}"
    logger.info(f"Uploading training data to s3://{args.s3_bucket}/{s3_data_key}...")
    s3.upload_file(str(data_path), args.s3_bucket, s3_data_key)

    # Get config from CDK stack (role, VPC)
    outputs = {}
    training_role = args.role_arn
    try:
        cfn = boto3.client("cloudformation")
        resp = cfn.describe_stacks(StackName="deep-research-rl")
        outputs = {
            o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]
        }
        if not training_role:
            training_role = outputs.get("RLTrainingRoleArn")
    except Exception:
        pass

    if not training_role:
        logger.error(
            "--role-arn required (or deploy RL stack first: npm run deploy:rl)"
        )
        sys.exit(1)

    # Launch SageMaker training job
    exp_id = (
        args.exp_id or f"dr-rl-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    sagemaker = boto3.client("sagemaker", region_name=region)
    job_name = (
        f"deep-research-rl-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )

    hyperparameters = {
        "agent_runtime_arn": args.agent_arn,
        "s3_bucket": args.s3_bucket,
        "exp_id": exp_id,
        "model_type": args.model_type,
        "hf_model_id": args.hf_model_id,
        "num_rollout": str(args.num_rollout),
        "num_gpus": str(args.num_gpus),
        "tp_size": str(args.tp_size),
        "rollout_batch_size": str(args.rollout_batch_size),
        "n_samples_per_prompt": str(args.n_samples),
        "rollout_max_response_len": str(args.max_response_len),
        "rollout_temperature": str(args.temperature),
        "lr": str(args.lr),
        "max_concurrent": str(args.max_concurrent),
        "acr_timeout": str(args.timeout),
    }

    # Get VPC config from CDK stack outputs
    vpc_subnets = None
    vpc_sg = None
    try:
        vpc_subnets = outputs.get("RLVpcSubnets", "").split(",")
        vpc_sg = outputs.get("RLSecurityGroupId", "")
    except Exception:
        pass

    training_params = {
        "TrainingJobName": job_name,
        "RoleArn": training_role,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "train:loss", "Regex": r"'train/loss': ([0-9\.\-e]+)"},
                {"Name": "train:pg_loss", "Regex": r"'train/pg_loss': ([0-9\.\-e]+)"},
                {"Name": "train:kl_loss", "Regex": r"'train/kl_loss': ([0-9\.\-e]+)"},
                {
                    "Name": "train:grad_norm",
                    "Regex": r"'train/grad_norm': ([0-9\.\-e]+)",
                },
                {"Name": "train:ppo_kl", "Regex": r"'train/ppo_kl': ([0-9\.\-e]+)"},
                {"Name": "rollout:reward", "Regex": r"reward=([0-9\.\-]+)"},
                {"Name": "rollout:traces", "Regex": r"traces=([0-9]+)"},
            ],
        },
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{args.s3_bucket}/training-data/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{args.s3_bucket}/checkpoints/",
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 14400,  # 4 hours max
        },
    }

    # Add VPC config if available (required for agent <-> training connectivity)
    if vpc_subnets and vpc_sg:
        training_params["VpcConfig"] = {
            "SecurityGroupIds": [vpc_sg],
            "Subnets": [s for s in vpc_subnets if s],
        }
        logger.info(f"VPC config: subnets={vpc_subnets}, sg={vpc_sg}")

    logger.info(f"Launching SageMaker job: {job_name}")
    sagemaker.create_training_job(**training_params)
    logger.info(f"✓ Job submitted: {job_name}")
    logger.info(
        f"  Monitor: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
    )
    logger.info(f"  Output:  s3://{args.s3_bucket}/checkpoints/{job_name}/output/")
    logger.info("")
    logger.info("Once complete, deploy the fine-tuned model:")
    logger.info(
        f"  uv run test-scripts/deploy_model.py --job-name {job_name} --endpoint-name dr-finetuned --instance-type ml.g5.xlarge"
    )


if __name__ == "__main__":
    main()
