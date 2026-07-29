#!/usr/bin/env python3
"""
SageMaker training entry point for agentic RL fine-tuning with AgentCore RL Toolkit.

Uses SlimeRunner (GRPO with full agent rollouts via AgentCore Runtime).
The agent runs with tools during training — learning from complete research
trajectories, not just text completions.

SageMaker passes hyperparameters via /opt/ml/input/config/hyperparameters.json.
Training data arrives at /opt/ml/input/data/training/
Output goes to /opt/ml/model/ (uploaded to S3 automatically).
"""

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from typing import Any

# SageMaker paths
OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
HP_FILE = os.environ.get("SM_HPS", "/opt/ml/input/config/hyperparameters.json")

# Load hyperparameters
if os.path.exists(HP_FILE):
    with open(HP_FILE) as f:
        hyperparameters = json.load(f)
else:
    hyperparameters = {}


def get_hp(
    key: str, default: Any = None, cast: Callable[[Any], Any] | None = None
) -> Any:
    """Get hyperparameter with optional type casting."""
    val = hyperparameters.get(key, os.environ.get(f"SM_HP_{key.upper()}", default))
    if val is not None and cast is not None:
        val = cast(val)
    return val


def resolve_data_path(data_path: str) -> str:
    """Resolve data path — SageMaker puts files in a directory."""
    if os.path.isdir(data_path):
        jsonl_files = [f for f in os.listdir(data_path) if f.endswith(".jsonl")]
        if jsonl_files:
            return os.path.join(data_path, jsonl_files[0])
        print(f"ERROR: No .jsonl files found in {data_path}")
        print(f"Contents: {os.listdir(data_path)}")
        sys.exit(1)
    return data_path


def main() -> None:
    from agentcore_rl_toolkit.backends.slime import SlimeRunner
    from huggingface_hub import snapshot_download

    # Required params
    agent_runtime_arn = get_hp("agent_runtime_arn")
    s3_bucket = get_hp("s3_bucket")
    data_path = get_hp("data_path", "/opt/ml/input/data/training/")
    exp_id = get_hp("exp_id", "dr-rl-sagemaker")

    # Model config
    model_type = get_hp("model_type", "qwen2.5-3B")
    hf_model_id = get_hp("hf_model_id", "Qwen/Qwen2.5-3B-Instruct")

    # Training hyperparameters (tuned for 4x A10G)
    num_rollout = get_hp("num_rollout", 20, int)
    num_gpus = get_hp("num_gpus", 4, int)
    tp_size = get_hp("tp_size", 2, int)
    rollout_gpus_per_engine = get_hp("rollout_gpus_per_engine", 2, int)
    rollout_batch_size = get_hp("rollout_batch_size", 8, int)
    n_samples_per_prompt = get_hp("n_samples_per_prompt", 4, int)
    rollout_max_response_len = get_hp("rollout_max_response_len", 1024, int)
    rollout_temperature = get_hp("rollout_temperature", 1.0, float)
    lr = get_hp("lr", 1e-6, float)
    max_concurrent = get_hp("max_concurrent", 10, int)
    acr_timeout = get_hp("acr_timeout", 900, int)
    sglang_mem_fraction_static = get_hp("sglang_mem_fraction_static", 0.7, float)

    # Resolve data path
    data_path = resolve_data_path(data_path)

    # Download model from HuggingFace
    model_dir = f"/opt/ml/model-cache/{hf_model_id.replace('/', '_')}"
    print(f"Downloading model from HuggingFace: {hf_model_id}")
    snapshot_download(repo_id=hf_model_id, local_dir=model_dir)
    print(f"Model downloaded to: {model_dir}")

    print("=" * 60)
    print("AgentCore Deep Research — Agentic RL Training (SlimeRunner)")
    print("=" * 60)
    print(f"Agent ARN:     {agent_runtime_arn}")
    print(f"S3 bucket:     {s3_bucket}")
    print(f"Model:         {hf_model_id} ({model_type})")
    print(f"Data:          {data_path}")
    print(f"Num rollouts:  {num_rollout}")
    print(f"GPUs:          {num_gpus}")
    print(f"TP size:       {tp_size}")
    print(f"Rollout GPUs:  {rollout_gpus_per_engine}")
    print(f"Batch size:    {rollout_batch_size}")
    print(f"N samples:     {n_samples_per_prompt}")
    print(f"Max resp len:  {rollout_max_response_len}")
    print(f"LR:            {lr}")
    print(f"SGLang mem:    {sglang_mem_fraction_static}")
    print(f"Output:        {OUTPUT_DIR}")
    print()

    if not agent_runtime_arn:
        print("ERROR: agent_runtime_arn hyperparameter required")
        sys.exit(1)
    if not s3_bucket:
        print("ERROR: s3_bucket hyperparameter required")
        sys.exit(1)

    # SlimeRunner doesn't pass --save/--save-hf by default (paths are user-specific).
    # We save in HF format directly to SageMaker output dir for seamless deployment.
    # --save: Megatron format (needed for slime's save machinery to trigger)
    # --save-hf: HF safetensors export (what we deploy to SageMaker)
    hf_save_path = os.path.join(OUTPUT_DIR, "hf")
    megatron_save_path = os.path.join(tempfile.gettempdir(), "megatron_ckpts")

    runner = SlimeRunner(
        exp_id=exp_id,
        agent_runtime_arn=agent_runtime_arn,
        s3_bucket=s3_bucket,
        model_dir=model_dir,
        data_path=data_path,
        model_type=model_type,
        num_gpus=num_gpus,
        tp_size=tp_size,
        rollout_gpus_per_engine=rollout_gpus_per_engine,
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
        rollout_max_response_len=rollout_max_response_len,
        rollout_temperature=rollout_temperature,
        lr=lr,
        max_concurrent=max_concurrent,
        acr_timeout=acr_timeout,
        reward_postprocessing="grpo",
        sglang_mem_fraction_static=sglang_mem_fraction_static,
        extra_flags=[
            "--save",
            megatron_save_path,
            "--save-interval",
            str(num_rollout),
            "--save-hf",
            f"{hf_save_path}/{{rollout_id}}",
        ],
    )

    print("Starting agentic GRPO training via SlimeRunner...")
    print("(Agent will use tools during rollouts on AgentCore Runtime)")
    print()
    runner.train(num_rollout=num_rollout)

    # Save trained model to SageMaker output dir.
    # Strategy: start with the complete original model (has correct tokenizer,
    # config, etc.), then overwrite with trained weights from slime's HF export.
    print("Preparing model for deployment...")

    # Step 1: Copy full original model to output (tokenizer, config, everything)
    for item in os.listdir(model_dir):
        src = os.path.join(model_dir, item)
        dst = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # Step 2: Overwrite with trained weights from --save-hf (only weight files)
    # slime's tokenizer export is incomplete (missing chat_template), so we
    # keep the original tokenizer files from step 1 and only take the weights.
    weight_extensions = {".safetensors", ".bin", ".pt"}
    weight_files = {"model.safetensors.index.json", "config.json"}

    if os.path.exists(hf_save_path):

        def _rollout_sort_key(name: str) -> int:
            """Parse rollout dir name as int, returning -1 for non-numeric dirs."""
            try:
                return int(name)
            except (ValueError, TypeError):
                return -1

        all_dirs = [
            d
            for d in os.listdir(hf_save_path)
            if os.path.isdir(os.path.join(hf_save_path, d))
        ]
        # Only consider directories with valid numeric names (rollout checkpoints)
        numeric_dirs = [d for d in all_dirs if _rollout_sort_key(d) >= 0]
        rollout_dirs = sorted(numeric_dirs, key=_rollout_sort_key)
        if rollout_dirs:
            latest = os.path.join(hf_save_path, rollout_dirs[-1])
            print(f"Overwriting with trained weights from rollout {rollout_dirs[-1]}")
            for item in os.listdir(latest):
                # Only copy weight files and model config, skip tokenizer files
                if (
                    any(item.endswith(ext) for ext in weight_extensions)
                    or item in weight_files
                ):
                    src = os.path.join(latest, item)
                    dst = os.path.join(OUTPUT_DIR, item)
                    shutil.copy2(src, dst)
            shutil.rmtree(hf_save_path, ignore_errors=True)
            print("Training complete. Model saved to SageMaker output.")
        else:
            print("WARNING: No rollout checkpoints found. Saving base model.")
    else:
        print("WARNING: No HF checkpoint found. Saving base model.")


if __name__ == "__main__":
    main()
