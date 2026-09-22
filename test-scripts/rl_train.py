#!/usr/bin/env python3
"""
Launch agentic RL fine-tuning for the deep research agent on SageMaker.

Rollouts are full research episodes run by the agent on AgentCore Runtime, using
the same code and Gateway tools as production. Reports are scored by the shared
rubric in research_rubric.py and GRPO updates the policy from each group.

    uv run test-scripts/rl_train.py \\
        --data test-scripts/results/rl_prompts_5k.jsonl \\
        --agent-arn <RLAgentRuntimeArn> \\
        --s3-bucket <bucket-in-the-training-region> \\
        --sft-job-name <completed-sft-job>

Training configuration lives in verl's defaults and the toolkit's
agentcore_grpo.yaml. Only what changes per run is exposed here.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 4x L40S 48GB = 192 GB, no NVLink. ml.p5.48xlarge (8x H100 80GB = 640 GB, NVLink)
INSTANCE_TYPE = "ml.g6e.12xlarge"  # smaller cards do not fit a 9B
VAL_PROMPTS = 32  # verl requires a validation file; this is enough to track drift


def to_parquet(jsonl: Path, out: Path) -> None:
    """
    Write the train/val parquet pair verl reads.

    One `payload` column holding the exact AgentCore invoke dict, authored against
    rl_app.py's contract. PayloadDataset forwards it verbatim and synthesises the
    chat-format `prompt` column verl's dataloader needs.
    """
    import pandas as pd

    rows = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
    df = pd.DataFrame(
        [
            {
                "payload": {
                    "prompt": r["prompt"][0]["content"],
                    "enabled_sources": r.get("enabled_sources", []),
                }
            }
            for r in rows
        ]
    ).sample(frac=1.0, random_state=42)

    out.mkdir(parents=True, exist_ok=True)
    df.iloc[:VAL_PROMPTS].to_parquet(out / "rl_prompts_val.parquet", index=False)
    df.iloc[VAL_PROMPTS:].to_parquet(out / "rl_prompts_train.parquet", index=False)
    logger.info(f"Prompts: {len(df) - VAL_PROMPTS} train / {VAL_PROMPTS} val")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Prompt JSONL")
    p.add_argument(
        "--agent-arn", required=True, help="AgentCore Runtime ARN of the RL agent"
    )
    p.add_argument(
        "--s3-bucket", required=True, help="Bucket for prompts and checkpoints"
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--sft-job-name", help="Start from this completed SFT job's checkpoint"
    )
    src.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-9B",
        help="Starting policy: HF Hub id or s3://.../model.tar.gz",
    )

    p.add_argument(
        "--steps", type=int, default=1000, help="Training steps (default: 1000)"
    )
    p.add_argument(
        "--group-size", type=int, default=8, help="Rollouts per prompt (default: 8)"
    )
    p.add_argument(
        "--prompts-per-step", type=int, default=8, help="Prompts per step (default: 8)"
    )
    p.add_argument("--max-prompt-len", type=int, default=24576)
    p.add_argument("--max-response-len", type=int, default=24576)
    p.add_argument("--max-tokens-per-turn", type=int, default=4096)
    # Matches train_entry.py. Each save gathers FSDP shards to materialise hf_model,
    # which is a memory spike on top of a step that already fits in 44.4 GB -- saving
    # more often than needed costs time and risks OOM.
    p.add_argument("--save-freq", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-6)
    # LoRA frees the optimiser state and gradients that force param/optimizer offload
    # in full-parameter mode, which is what caps us at 8 episodes/step. 0 = full FT.
    p.add_argument("--lora-rank", type=int, default=0)
    p.add_argument("--instance-type", default=INSTANCE_TYPE)
    p.add_argument(
        "--gpu-mem-fraction", type=float, default=0.4, help="vLLM's share of each GPU"
    )
    p.add_argument("--image-uri", help="Training image (default: :rl in this region)")
    args = p.parse_args()

    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(data)

    sm = boto3.client("sagemaker", region_name=region)
    model_id = args.model_id
    if args.sft_job_name:
        job = sm.describe_training_job(TrainingJobName=args.sft_job_name)
        if job["TrainingJobStatus"] != "Completed":
            raise ValueError(
                f"{args.sft_job_name} is {job['TrainingJobStatus']}, not Completed"
            )
        model_id = job["ModelArtifacts"]["S3ModelArtifacts"]
        logger.info(f"Policy from {args.sft_job_name}: {model_id}")

    job_name = f"deep-research-rl-{time.strftime('%Y%m%d-%H%M%S')}"
    account = boto3.client("sts").get_caller_identity()["Account"]
    image = args.image_uri or (
        f"{account}.dkr.ecr.{region}.amazonaws.com/deep-research-rl-training:rl"
    )
    stack = {
        o["OutputKey"]: o["OutputValue"]
        for o in boto3.client("cloudformation", region_name=region).describe_stacks(
            StackName="deep-research-rl"
        )["Stacks"][0]["Outputs"]
    }

    to_parquet(data, Path("/tmp/rl_parquet"))  # noqa: S108  # nosec B108
    prefix = f"rl-data/{job_name}"
    s3 = boto3.client("s3", region_name=region)
    for f in Path("/tmp/rl_parquet").glob("*.parquet"):  # noqa: S108  # nosec B108
        s3.upload_file(str(f), args.s3_bucket, f"{prefix}/{f.name}")

    logger.info(f"Launching {job_name}")
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        RoleArn=stack["RLTrainingRoleArn"],
        HyperParameters={
            "model_id": model_id,
            "agent_runtime_arn": args.agent_arn,
            "s3_bucket": args.s3_bucket,
            "group_size": str(args.group_size),
            "prompts_per_step": str(args.prompts_per_step),
            "total_steps": str(args.steps),
            "max_prompt_len": str(args.max_prompt_len),
            "max_response_len": str(args.max_response_len),
            "max_tokens_per_turn": str(args.max_tokens_per_turn),
            "save_freq": str(args.save_freq),
            "lr": str(args.lr),
            "lora_rank": str(args.lora_rank),
            "gpu_mem_fraction": str(args.gpu_mem_fraction),
        },
        InputDataConfig=[
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{args.s3_bucket}/{prefix}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{args.s3_bucket}/checkpoints/"},
        # Streams checkpoints to S3 during training. Without this they only appear
        # when the job terminates, so a mid-run checkpoint cannot be evaluated.
        CheckpointConfig={
            "S3Uri": f"s3://{args.s3_bucket}/live-ckpt/{job_name}/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        ResourceConfig={
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 500,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 432000},
        # The agent calls the rollout gateway on the trainer's private IP, so the
        # training job must sit in the same VPC as the agent runtime.
        VpcConfig={
            "SecurityGroupIds": [stack["RLSecurityGroupId"]],
            "Subnets": [x for x in stack["RLVpcSubnets"].split(",") if x],
        },
    )

    logger.info(
        f"✓ {job_name}  ({args.group_size * args.prompts_per_step} episodes/step)"
    )
    logger.info(
        f"  https://console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
    )
    logger.info("  Read the first step's wall clock before trusting the step budget.")


if __name__ == "__main__":
    main()
