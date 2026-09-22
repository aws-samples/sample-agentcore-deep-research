"""
SageMaker entry point for agentic RL fine-tuning of the deep research agent.

Uses the AgentCore RL Toolkit's verl backend with the FSDP engine. Rollouts are
full research episodes run by the agent on AgentCore Runtime, with the same code
and Gateway tools as production; reports are scored by research_rubric.py.

A translation layer only: resolve the policy checkpoint, build hydra overrides,
hand off. Anything not set here is left to verl's defaults on purpose.
"""

import json
import os
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
MODEL_CACHE = "/opt/ml/model-cache"

# SageMaker writes hyperparameters to a JSON file, and only mirrors them into
# SM_HP_* env vars in some configurations. Read the file first.
HP_FILE = os.environ.get("SM_HPS", "/opt/ml/input/config/hyperparameters.json")
_HP = json.loads(Path(HP_FILE).read_text()) if os.path.exists(HP_FILE) else {}

# Fixed by the hardware and the data, not per run. One node of 4x L40S 48GB
# (44.4GB usable, no NVLink) and research episodes that reach ~32K tokens.
GPUS = 4
ROLLOUT_TP = 2  # TP=4 pays per-layer all-reduce over PCIe for no gain


def hp(name: str, default: Any = None, cast: Callable[[str], Any] = str) -> Any:
    """Read a SageMaker hyperparameter."""
    raw = _HP.get(name, os.environ.get(f"SM_HP_{name.upper()}"))
    return default if raw is None or raw == "" else cast(raw)


def resolve_policy(model_id: str) -> str:
    """
    Materialise the starting policy: an HF Hub id, or an S3 model.tar.gz so RL can
    continue from our own SFT run rather than only from a public checkpoint.
    """
    if not model_id.startswith("s3://"):
        from huggingface_hub import snapshot_download

        print(f"Downloading {model_id}", flush=True)
        return str(snapshot_download(repo_id=model_id, local_dir=MODEL_CACHE))

    if not model_id.endswith(".tar.gz"):
        raise ValueError(f"S3 policy must be a .tar.gz, got {model_id}")

    import boto3

    bucket, _, key = model_id[len("s3://") :].partition("/")
    os.makedirs(MODEL_CACHE, exist_ok=True)
    archive = os.path.join(MODEL_CACHE, "model.tar.gz")
    print(f"Downloading {model_id}", flush=True)
    boto3.client("s3").download_file(bucket, key, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(MODEL_CACHE, filter="data")  # refuses paths outside dest
    os.remove(archive)

    # Without config.json this fails much later inside verl, unhelpfully.
    if not os.path.exists(os.path.join(MODEL_CACHE, "config.json")):
        raise RuntimeError(
            f"No config.json after extracting {model_id}. "
            f"Contents: {sorted(os.listdir(MODEL_CACHE))[:20]}"
        )
    return MODEL_CACHE


def main() -> None:
    agent_arn = hp("agent_runtime_arn")
    s3_bucket = hp("s3_bucket")
    if not agent_arn or not s3_bucket:
        raise ValueError("agent_runtime_arn and s3_bucket are required")

    group_size = hp("group_size", 8, int)
    prompts_per_step = hp("prompts_per_step", 8, int)
    total_steps = hp("total_steps", 1000, int)
    # Memory-shaping knobs, tunable per run without rebuilding the image.
    # The initial prompt only: system prompt (~3.4K tokens) plus the question.
    max_prompt_len = hp("max_prompt_len", 8192, int)
    # CUMULATIVE assistant tokens per trajectory, not per call -- verl's response
    # storage width doubles as the gateway's whole-trajectory budget. Measured over
    # 1,833 teacher trajectories: median 13.6K, max 21.5K, so anything below 24576
    # kills rollouts mid-episode (4096 killed 100% of them).
    max_response_len = hp("max_response_len", 24576, int)
    # One model call's output. Measured per-turn p99.9 is 2,636 tokens.
    max_tokens_per_turn = hp("max_tokens_per_turn", 4096, int)
    # 0 = full-parameter. LoRA removes the optimiser state and gradient copies that
    # force param/optimizer offload, which is what caps full FT at 8 episodes/step.
    lora_rank = hp("lora_rank", 0, int)
    gpu_mem_fraction = hp("gpu_mem_fraction", 0.4, float)
    policy = resolve_policy(hp("model_id", "Qwen/Qwen3.5-9B"))
    job_name = os.environ.get("TRAINING_JOB_NAME", "local")

    # verl asserts ppo_max_token_len_per_gpu >= the padded sequence width, and this
    # is also the inference context (max_model_len).
    width = max_prompt_len + max_response_len

    # The AgentCore integration is one of verl's agent loops, configured by file
    # rather than hydra override. Generated here so the ARN and bucket come from
    # hyperparameters instead of committed config or environment plumbing.
    loop_config = Path("/tmp/agentcore_agent.yaml")  # noqa: S108  # nosec B108
    loop_config.write_text(
        "- name: agentcore_agent\n"
        "  _target_: agentcore_rl_toolkit.backends.verl.agent_loop.AgentCoreAgentLoop\n"
        f"  agent_runtime_arn: {agent_arn}\n"
        f"  s3_bucket: {s3_bucket}\n"
        f"  max_tokens_per_turn: {max_tokens_per_turn}\n"
        "  max_rollout_time: 1800\n"
    )

    overrides = [
        "trainer.use_v1=true",
        "trainer.v1.trainer_mode=sync",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=true",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files=['{DATA_DIR}/rl_prompts_train.parquet']",
        f"data.val_files=['{DATA_DIR}/rl_prompts_val.parquet']",
        f"data.train_batch_size={prompts_per_step}",
        f"data.max_prompt_length={max_prompt_len}",
        f"data.max_response_length={max_response_len}",
        # Rows reach the agent as the session payload verbatim.
        "data.custom_cls.path=pkg://agentcore_rl_toolkit.backends.verl.dataset",
        "data.custom_cls.name=PayloadDataset",
        f"actor_rollout_ref.model.path={policy}",
        "actor_rollout_ref.model.use_remove_padding=True",
        *(
            [
                f"actor_rollout_ref.model.lora_rank={lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={lora_rank * 2}",
                "actor_rollout_ref.model.target_modules=all-linear",
                # vLLM must serve the adapter during rollouts, else generation uses
                # base weights
                "actor_rollout_ref.rollout.load_format=safetensors",
                f"actor_rollout_ref.rollout.max_lora_rank={lora_rank}",
            ]
            if lora_rank
            else []
        ),
        # A 248,320-token vocabulary makes the lm_head logits ~16GB per micro-batch
        # in bf16. Fused kernels chunk the projection instead of materialising it.
        "actor_rollout_ref.model.use_fused_kernels=true",
        f"actor_rollout_ref.actor.optim.lr={hp('lr', 1e-6, float)}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={width}",
        # Full-parameter 9B carries 16 bytes/param of master and optimizer state,
        # 38GB per card unsharded. Offloading to host RAM (373GB) brings resident
        # use to ~33GB of 44.4GB; parameter offload frees the card during rollout.
        # verl defaults this to fp32, which holds fp32 master weights AND fp32
        # gradients: 19.2GB/card for a 9B, against ~26GB left after vLLM. bf16
        # halves it; mixed precision already reduces gradients in fp32.
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={str(lora_rank == 0).lower()}",  # noqa: E501
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(lora_rank == 0).lower()}",  # noqa: E501
        # No KL term, so no reference policy: a second 9B does not fit, and KL-free
        # GRPO is standard in DAPO and Dr.GRPO.
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        f"actor_rollout_ref.rollout.n={group_size}",
        f"actor_rollout_ref.rollout.prompt_length={max_prompt_len}",
        f"actor_rollout_ref.rollout.response_length={max_response_len}",
        # Otherwise vLLM reserves KV cache for max_position_embeddings (262144).
        f"actor_rollout_ref.rollout.max_model_len={width}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={ROLLOUT_TP}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_fraction}",
        # Qwen3.5 is a hybrid attention/Gated-DeltaNet model, so vLLM reserves a
        # recurrent-state cache block per sequence. The 1024 default exceeds what
        # fits at our memory fraction; we only need group_size * prompts_per_step.
        f"actor_rollout_ref.rollout.max_num_seqs={max(16, group_size * prompts_per_step)}",  # noqa: E501
        "actor_rollout_ref.rollout.agent.default_agent_loop=agentcore_agent",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={loop_config}",
        f"trainer.n_gpus_per_node={GPUS}",
        "trainer.nnodes=1",
        # /opt/ml/checkpoints is SageMaker's CheckpointConfig LocalPath: contents sync
        # to S3 during training, so a mid-run checkpoint is evaluable. Writing under
        # /opt/ml/model instead would only surface at job termination.
        "trainer.default_local_dir=/opt/ml/checkpoints",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.save_freq={hp('save_freq', 50, int)}",
        # Saves an HF-format copy alongside the sharded checkpoint, so
        # deploy_model.py has something servable without a conversion step.
        # No 'optimizer'/'extra': resume_mode is disable, so resumable state is dead
        # weight. Saving all four OOM'd at the first save_freq boundary -- gathering
        # FSDP shards spikes memory on top of a training step that already fits in
        # 44.4 GB. 'hf_model' is the one we need: it is directly deployable.
        "actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model']",
        # Without these, exp_id derives from verl's defaults and every run shares
        # one S3 rollout prefix, so runs cannot be told apart after the fact.
        "trainer.project_name=deep-research-rl",
        f"trainer.experiment_name={job_name}",
        "trainer.resume_mode=disable",
        "trainer.logger=['console']",
    ]

    print(f"Policy: {policy}")
    print(f"Episodes per step: {group_size * prompts_per_step}  Steps: {total_steps}")
    print(f"Agent loop config:\n{loop_config.read_text()}")
    for o in overrides:
        print(f"  {o}")
    print(flush=True)

    # verl's own entry point. The toolkit contributes a dataset class and an agent
    # loop through the overrides above rather than wrapping the trainer.
    subprocess.run(
        [sys.executable, "-m", "verl.trainer.main_ppo", *overrides],
        check=True,
        env={**os.environ, "HYDRA_FULL_ERROR": "1"},
    )


if __name__ == "__main__":
    main()
