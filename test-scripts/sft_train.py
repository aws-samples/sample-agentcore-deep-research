#!/usr/bin/env python3
"""
Launch SFT (Supervised Fine-Tuning) training job on SageMaker.

Uploads SFT traces to S3 and launches a SageMaker training job that runs
LoRA fine-tuning on Qwen3.6-27B using the teacher traces from Sonnet 5.

Uses the same infrastructure as the RL pipeline (same S3 bucket, IAM role,
VPC). The trained model outputs in the same HF format, deployable via the
same deploy_model.py script.

Usage:
    uv run test-scripts/sft_train.py \
        --data test-scripts/results/sft_traces.jsonl \
        --s3-bucket <RLBucketName>

    # Custom model
    uv run test-scripts/sft_train.py \
        --data test-scripts/results/sft_traces.jsonl \
        --s3-bucket <RLBucketName> \
        --hf-model-id Qwen/Qwen3.6-27B

Prerequisites:
    - Deployed RL stack: cd infra-cdk && npm run deploy:train
    - SFT traces generated: uv run test-scripts/sft_generate_data.py
    - Training container pushed: ./training/build_and_push.sh sft
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
        description="Launch SFT training job on SageMaker (LoRA on Qwen3.6-27B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data", type=str, required=True, help="SFT traces JSONL file path"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.environ.get("RL_S3_BUCKET"),
        help="S3 bucket (from deploy:train output)",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="Qwen/Qwen3.6-27B",
        help="HuggingFace model ID (default: Qwen/Qwen3.6-27B)",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g5.12xlarge",
        help="SageMaker instance type (default: ml.g5.12xlarge, 4x A10G)",
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
    # LoRA hyperparameters
    parser.add_argument(
        "--lora-rank", type=int, default=64, help="LoRA rank (default: 64)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: 2x lora-rank, keeping alpha/r scaling constant)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-device batch size (default: 1)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed: controls shuffle order, dropout and LoRA init (default: 42)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save an adapter every N optimizer steps (0 = once per epoch). "
        "Enables mid-run evaluation of intermediate checkpoints.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to retain (default: 2)",
    )
    parser.add_argument(
        "--max-runtime",
        type=int,
        default=86400,
        help="Job runtime cap in seconds (default: 86400 = 24h). Trajectory SFT "
        "at 32K sequence length runs ~1200s/step on 4x L40S, so a 2-epoch run "
        "over 500 trajectories needs ~10h; the old 6h cap killed a run mid-training.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of traces held out for validation loss (default: 0.05). "
        "Held-out loss is what distinguishes learning from memorisation.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap total optimizer steps (default: -1 = full run). Set to a small "
        "value for a fast diagnostic run that exercises setup and a few steps.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=40960,
        help="Maximum sequence length (default: 40960). Measured on real "
        "tool-call trajectories with observations capped at 2000 chars: median "
        "~27K tokens, max observed 32.3K. 40960 leaves headroom so the end of a "
        "trajectory — where the report is finalised and verified — is never "
        "truncated. Fits ml.g6e.12xlarge (4x L40S 48GB); does NOT fit A10G 24GB.",
    )

    args = parser.parse_args()

    # Validate
    if not args.s3_bucket:
        logger.error("--s3-bucket required (from `npm run deploy:train` output: RLBucketName)")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.error("Run sft_generate_data.py first to generate traces.")
        sys.exit(1)

    num_records = sum(1 for _ in open(data_path))

    # Determine training image URI
    account = boto3.client("sts").get_caller_identity()["Account"]
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    image_uri = (
        args.image_uri
        or f"{account}.dkr.ecr.{region}.amazonaws.com/deep-research-rl-training:sft"
    )

    # Get role from CDK stack
    training_role = args.role_arn
    outputs = {}
    try:
        cfn = boto3.client("cloudformation")
        resp = cfn.describe_stacks(StackName="deep-research-rl")
        outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}
        if not training_role:
            training_role = outputs.get("RLTrainingRoleArn")
    except Exception:
        pass

    if not training_role:
        logger.error("--role-arn required (or deploy training stack first: npm run deploy:train)")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("SFT TRAINING — SageMaker Job (LoRA)")
    logger.info("=" * 60)
    logger.info(f"Model:         {args.hf_model_id}")
    logger.info(f"Data:          {data_path} ({num_records} traces)")
    logger.info(f"Instance:      {args.instance_type}")
    logger.info(f"Image:         {image_uri}")
    # LoRA scaling is alpha/r. A fixed alpha silently changes that scaling
    # whenever rank is overridden (rank 16 with alpha 128 gives 8x instead of
    # the intended 2x, inflating the effective adapter learning rate), so
    # derive alpha from rank unless the caller set it explicitly.
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_rank
    logger.info(f"LoRA scaling:  alpha/r = {args.lora_alpha / args.lora_rank:.1f}")
    logger.info(f"LoRA rank:     {args.lora_rank}")
    logger.info(f"LoRA alpha:    {args.lora_alpha}")
    logger.info(f"LR:            {args.lr}")
    logger.info(f"Epochs:        {args.epochs}")
    logger.info(f"Batch/GPU:     {args.batch_size}")
    logger.info(f"Grad accum:    {args.grad_accum}")
    logger.info(f"Max seq len:   {args.max_seq_length}")
    logger.info(f"Effective BS:  {args.batch_size * args.grad_accum * 4}")  # 4 GPUs
    logger.info("")

    # Upload training data to S3
    s3 = boto3.client("s3")
    job_name = f"deep-research-sft-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    s3_data_key = f"sft-data/{job_name}/{data_path.name}"
    logger.info(f"Uploading training data to s3://{args.s3_bucket}/{s3_data_key}...")
    s3.upload_file(str(data_path), args.s3_bucket, s3_data_key)

    # Launch SageMaker training job
    sagemaker = boto3.client("sagemaker", region_name=region)

    hyperparameters = {
        "hf_model_id": args.hf_model_id,
        "lora_rank": str(args.lora_rank),
        "lora_alpha": str(args.lora_alpha),
        "lr": str(args.lr),
        "epochs": str(args.epochs),
        "per_device_batch_size": str(args.batch_size),
        "gradient_accumulation_steps": str(args.grad_accum),
        "max_seq_length": str(args.max_seq_length),
        "max_steps": str(args.max_steps),
        "eval_fraction": str(args.eval_fraction),
        "seed": str(args.seed),
        "save_steps": str(args.save_steps),
        "save_total_limit": str(args.save_total_limit),
    }

    # VPC config from CDK stack
    vpc_subnets = None
    vpc_sg = None
    if outputs.get("RLVpcSubnets") and outputs.get("RLSecurityGroupId"):
        vpc_subnets = outputs["RLVpcSubnets"].split(",")
        vpc_sg = outputs["RLSecurityGroupId"]

    training_params = {
        "TrainingJobName": job_name,
        "RoleArn": training_role,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri,
            "TrainingInputMode": "File",
            "MetricDefinitions": [
                {"Name": "train:loss", "Regex": r"'loss': ([0-9\.\-e]+)"},
                {"Name": "train:grad_norm", "Regex": r"'grad_norm': ([0-9\.\-e]+)"},
                {"Name": "train:learning_rate", "Regex": r"'learning_rate': ([0-9\.\-e]+)"},
                {"Name": "train:epoch", "Regex": r"'epoch': ([0-9\.\-e]+)"},
            ],
        },
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{args.s3_bucket}/sft-data/{job_name}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{args.s3_bucket}/checkpoints/",
        },
        # Without this, /opt/ml/checkpoints is container-local and every
        # intermediate adapter is discarded when the job ends — only the final
        # merged model in /opt/ml/model is uploaded. Declaring it makes SageMaker
        # sync the directory to S3 continuously, so intermediate checkpoints can
        # be merged and evaluated while the run is still going.
        "CheckpointConfig": {
            "S3Uri": f"s3://{args.s3_bucket}/ckpt-sync/{job_name}/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 200,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": args.max_runtime,
        },
    }

    if vpc_subnets and vpc_sg:
        training_params["VpcConfig"] = {
            "SecurityGroupIds": [vpc_sg],
            "Subnets": [s for s in vpc_subnets if s],
        }

    logger.info(f"Launching SageMaker job: {job_name}")
    sagemaker.create_training_job(**training_params)

    logger.info(f"✓ Job submitted: {job_name}")
    logger.info(f"  Monitor: https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}")
    logger.info(f"  Output:  s3://{args.s3_bucket}/checkpoints/{job_name}/output/")
    logger.info("")
    logger.info("Once complete, deploy the fine-tuned model:")
    logger.info(f"  uv run test-scripts/deploy_model.py --job-name {job_name} --endpoint-name dr-finetuned")
    logger.info("")
    logger.info("Then run RL on top of the SFT checkpoint:")
    logger.info(f"  uv run test-scripts/rl_train.py --hf-model-id s3://{args.s3_bucket}/checkpoints/{job_name}/output/model.tar.gz ...")


if __name__ == "__main__":
    main()
