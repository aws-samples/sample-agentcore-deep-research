#!/usr/bin/env python3
"""
SageMaker entry point for LoRA SFT with TRL.

Input: JSONL with {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
Output: Merged HF model (base + LoRA adapter) in /opt/ml/model/
"""

import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

# SageMaker paths
OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
HP_FILE = os.environ.get("SM_HPS", "/opt/ml/input/config/hyperparameters.json")
DATA_DIR = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")

# Load hyperparameters
hyperparameters = {}
if os.path.exists(HP_FILE):
    with open(HP_FILE) as f:
        hyperparameters = json.load(f)


def get_hp(
    key: str, default: Any = None, cast: Callable[[Any], Any] | None = None
) -> Any:
    val = hyperparameters.get(key, os.environ.get(f"SM_HP_{key.upper()}", default))
    if val is not None and cast is not None:
        val = cast(val)
    return val


def find_data_file(data_dir: str) -> str:
    data_path = Path(data_dir)
    if data_path.is_file():
        return str(data_path)
    jsonl_files = list(data_path.glob("*.jsonl"))
    if jsonl_files:
        return str(jsonl_files[0])
    print(f"ERROR: No .jsonl files in {data_dir}")
    sys.exit(1)


RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


def log_mem(tag: str) -> None:
    """Report per-GPU memory. Cheap, and the fastest way to diagnose an OOM."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(LOCAL_RANK).total_memory / 1e9
    print(
        f"[MEM rank{RANK}] {tag}: allocated={alloc:.2f}GB reserved={reserved:.2f}GB "
        f"peak={peak:.2f}GB capacity={total:.2f}GB",
        flush=True,
    )


def log_sharding(model: Any) -> None:
    """Confirm parameters are sharded across GPUs rather than replicated.

    With FSDP2/DTensor, `p.numel()` reports the GLOBAL shape and there is no
    FullyShardedDataParallel wrapper class, so counting parameters or looking
    for wrapper names both wrongly report "not sharded". The reliable signal is
    the size of each parameter's LOCAL storage.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()  # type: ignore[assignment]  # noqa: N806

    global_elems = local_elems = 0
    for p in model.parameters():
        global_elems += p.numel()
        lp = p.to_local() if DTensor and isinstance(p, DTensor) else p
        local_elems += lp.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = global_elems / local_elems if local_elems else 0
    print(
        f"[SHARD rank{RANK}] global={global_elems / 1e9:.2f}B "
        f"local={local_elems / 1e9:.2f}B ({ratio:.2f}x) "
        f"trainable={trainable / 1e6:.1f}M "
        f"verdict={'SHARDED' if ratio > 1.5 else 'REPLICATED'}",
        flush=True,
    )


def main() -> None:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Hyperparameters
    hf_model_id = get_hp("hf_model_id", "Qwen/Qwen3.5-9B")
    lora_rank = get_hp("lora_rank", 64, int)
    # Default alpha to 2x rank so LoRA scaling (alpha/r) stays constant when
    # rank changes. A hardcoded alpha silently rescales adapter updates.
    lora_alpha = get_hp("lora_alpha", 2 * lora_rank, int)
    lr = get_hp("lr", 2e-4, float)
    epochs = get_hp("epochs", 3, int)
    per_device_batch_size = get_hp("per_device_batch_size", 1, int)
    gradient_accumulation_steps = get_hp("gradient_accumulation_steps", 8, int)
    max_seq_length = get_hp("max_seq_length", 6144, int)
    max_steps = get_hp("max_steps", -1, int)
    eval_fraction = get_hp("eval_fraction", 0.05, float)
    save_steps = get_hp("save_steps", 0, int)
    seed = get_hp("seed", 42, int)
    save_total_limit = get_hp("save_total_limit", 2, int)

    print("=" * 60)
    print("AgentCore Deep Research — LoRA SFT")
    print("=" * 60)
    print(f"Model:          {hf_model_id}")
    print(f"Seed:           {seed}")
    print(
        "Fine-tuning:    "
        + (f"LoRA r={lora_rank}" if lora_rank > 0 else "FULL-PARAMETER (no adapters)")
    )
    print(f"LoRA alpha:     {lora_alpha} (alpha/r = {lora_alpha / lora_rank:.1f})")
    print(f"LR:             {lr}")
    print(f"Epochs:         {epochs}")
    print(f"Batch/GPU:      {per_device_batch_size}")
    print(f"Grad accum:     {gradient_accumulation_steps}")
    print(f"GPUs:           {torch.cuda.device_count()}")
    print(f"Output:         {OUTPUT_DIR}")
    print()

    # Find and load training data
    data_file = find_data_file(DATA_DIR)
    dataset = load_dataset("json", data_files=data_file, split="train")

    # Hold out a validation split. Without this we cannot tell learning from
    # memorisation — LoRA r=64 is ~205M trainable params against ~500 examples,
    # so held-out loss is the only honest signal for how long to train.
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(
        f"Dataset: {len(dataset)} examples from {data_file} "
        f"-> train={len(train_dataset)} eval={len(eval_dataset)}"
    )

    # Download model (with retries)
    model_cache = f"/opt/ml/model-cache/{hf_model_id.replace('/', '_')}"

    def shards_complete(path: str) -> bool:
        """Verify every shard named in the safetensors index is present."""
        index = Path(path) / "model.safetensors.index.json"
        if not index.exists():
            # Single-shard model: accept any .safetensors file.
            return any(Path(path).glob("*.safetensors"))
        try:
            wanted = set(json.loads(index.read_text())["weight_map"].values())
        except Exception:
            return False
        return all((Path(path) / shard).exists() for shard in wanted)

    # Only rank 0 downloads: concurrent snapshot_download into one local_dir races
    # and leaves partial shards, surfacing much later as a load-time FileNotFoundError.
    if RANK == 0:
        print(f"Downloading model: {hf_model_id}", flush=True)
        for attempt in range(5):
            try:
                snapshot_download(repo_id=hf_model_id, local_dir=model_cache)
                if shards_complete(model_cache):
                    print("Model downloaded and verified.", flush=True)
                    break
                print(
                    f"Download attempt {attempt + 1}: shards incomplete, retrying",
                    flush=True,
                )
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 4:
                raise RuntimeError(f"Could not fully download {hf_model_id}")
            time.sleep(30)
    else:
        print(f"[rank{RANK}] waiting for rank 0 to download model", flush=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        # Barrier not up yet (torchrun initialises it inside the Trainer), so
        # poll for the completed snapshot instead.
        for _ in range(240):
            if shards_complete(model_cache):
                break
            time.sleep(15)
        else:
            raise RuntimeError(f"Timed out waiting for model download on rank {RANK}")

    # exclude_modules keeps adapters off the 27-layer vision tower: we train on
    # text-only trajectories, so those adapters never receive gradient and merge as
    # identity, but still cost memory. lora_rank <= 0 means full-parameter, which
    # needs far more memory and likely optimizer offload at long sequences.
    use_lora = lora_rank > 0
    lora_config = (
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            exclude_modules=r".*\.?(visual|vision_tower|vision_model)\..*",
        )
        if use_lora
        else None
    )

    # assistant_only_loss masks the loss to assistant turns, but it requires the
    # chat template to mark them with {% generation %}. Not all model families
    # ship such a template (Gemma 4 does, Qwen3.5 does not), and enabling it
    # without the markers raises at dataset-prep time — so detect it.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_cache, trust_remote_code=True)

    # Loss masking needs {% generation %} markers in the chat template. Without
    # them assistant_only_loss does not error -- it silently trains on tool
    # observations, teaching the model to fabricate retrieval results. TRL swaps in
    # a marked-up template for recognised families; the assertion below covers the
    # unrecognised case, where the failure is silent.
    from trl.chat_template_utils import (
        get_training_chat_template,
        has_generation_markers,
    )

    # The published template is what gets saved with the merged checkpoint. The
    # training template is training-only: its markers would either break a
    # serving stack that does not know the tag, or silently change the prompt
    # format the model was tuned on.
    upstream_chat_template = tok.chat_template

    assistant_only = has_generation_markers(tok.chat_template or "")
    if assistant_only:
        source = "model's own template"
    else:
        # Not an error yet — this is the case TRL fixes for recognised families.
        # It raises rather than returning None for a model with no chat template
        # at all, so treat any failure as "masking unavailable" and let the guard
        # below report it clearly.
        try:
            assistant_only = get_training_chat_template(tok) is not None
        except Exception as exc:
            print(f"No TRL training template available ({exc})", flush=True)
            assistant_only = False
        source = "TRL training template" if assistant_only else "unavailable"
    print(f"assistant_only_loss: {assistant_only} (masking via {source})", flush=True)

    # Refuse rather than silently train on observations: invisible in the loss
    # curve, and it teaches fabrication. ALLOW_UNMASKED_OBSERVATIONS=1 to override.
    if not assistant_only and os.environ.get("ALLOW_UNMASKED_OBSERVATIONS") != "1":
        raise RuntimeError(
            "Refusing to train: chat template for "
            f"{hf_model_id} has no {{% generation %}} block, so tool "
            "observations cannot be masked out of the loss, and TRL has no "
            "training template for this family. Add {% generation %} markers to "
            "the template, or set ALLOW_UNMASKED_OBSERVATIONS=1 to override."
        )

    # Training config. FSDP shards the base model across GPUs; TRL/accelerate
    # auto-wraps the correct decoder layer class for the loaded architecture.
    training_args = SFTConfig(
        output_dir="/opt/ml/checkpoints",
        # Pinned explicitly rather than relying on the framework default, so a
        # re-run reproduces the same shuffle order, dropout masks and LoRA init.
        seed=seed,
        data_seed=seed,
        # No Liger: its fused kernel operates on plain tensors and raises under
        # FSDP, where lm_head weights are sharded DTensors. Measured peak without it
        # is 38.5-44.9GB of 47.7GB, which fits.
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=True,
        # TRL defaults max_length to 1024, which would truncate ~80% of each
        # teacher report and teach the model to stop early. Traces top out
        # around 5.3K tokens, so the default of 6144 covers every example.
        max_length=max_seq_length,
        max_steps=max_steps,
        logging_steps=5,
        # Evaluate each epoch so held-out loss reveals whether additional
        # epochs are still helping or starting to overfit.
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        # Intermediate checkpoints show mid-run whether the score is still climbing,
        # instead of waiting ~33h. Needs the job to declare a CheckpointConfig or
        # SageMaker never copies them to S3.
        **(
            {"save_strategy": "steps", "save_steps": save_steps}
            if save_steps > 0
            else {"save_strategy": "epoch"}
        ),
        save_total_limit=save_total_limit,
        # Gradient checkpointing is essential here: at 6K sequence length the
        # DeltaNet linear-attention layers fall back to plain PyTorch ops (the
        # fused kernels can't be built in this image), and their stored
        # intermediates OOM a 24GB A10G without it.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp="full_shard auto_wrap",
        # Gather shards into a full state dict on save, so the adapter can be
        # serialized and later merged outside the sharded context.
        fsdp_config={"state_dict_type": "FULL_STATE_DICT"},
        # Train only on assistant turns when the chat template supports it
        assistant_only_loss=assistant_only,
        packing=False,
        report_to="none",
    )

    log_mem("before trainer init")

    trainer = SFTTrainer(
        model=model_cache,
        args=training_args,
        processing_class=tok,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    # Echo the values TRL actually resolved. Several of these have silently
    # failed to apply in the past (max_length defaulting to 1024, fsdp ignored),
    # and each wasted a full training cycle to discover.
    ta = trainer.args
    print(
        f"[CONFIG] max_length={getattr(ta, 'max_length', None)} "
        f"max_steps={ta.max_steps} epochs={ta.num_train_epochs} "
        f"batch={ta.per_device_train_batch_size} "
        f"grad_accum={ta.gradient_accumulation_steps} "
        f"grad_ckpt={ta.gradient_checkpointing} "
        f"assistant_only_loss={getattr(ta, 'assistant_only_loss', None)} "
        f"fsdp={ta.fsdp} world_size={ta.world_size}",
        flush=True,
    )
    log_sharding(trainer.model)

    # Check every example, not a prefix: datasets concatenate collection batches, so
    # a prefix can miss a whole batch of longer trajectories. Truncation cuts the end
    # of the report and teaches the model to stop mid-report, so treat it as fatal.
    lens = sorted(len(x) for x in trainer.train_dataset["input_ids"])
    n = len(lens)
    at_cap = sum(1 for x in lens if x >= max_seq_length)
    print(
        f"[DATA] tokenized n={n} p50={lens[n // 2]} p95={lens[int(n * 0.95)]} "
        f"max={lens[-1]} at_cap={at_cap}",
        flush=True,
    )
    if at_cap:
        raise RuntimeError(
            f"Refusing to train: {at_cap}/{n} examples reach max_seq_length="
            f"{max_seq_length} and are therefore truncated mid-report. Raise "
            "--max-seq-length or shrink observations with "
            "retruncate_observations() in trajectory_format.py."
        )

    from transformers import TrainerCallback

    class MemCallback(TrainerCallback):
        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if state.global_step == 1 or state.global_step % 10 == 0:
                log_mem(f"step {state.global_step}")

    trainer.add_callback(MemCallback())

    print("Starting SFT training...", flush=True)
    trainer.train()
    print("Training complete.", flush=True)

    # Under FSDP the parameters are sharded DTensors, so merging in place fails with
    # safetensors "invalid python storage". Gather the adapter first, then merge into
    # the base model on CPU.
    save_dir = "/opt/ml/checkpoints/adapter" if use_lora else OUTPUT_DIR
    print(
        f"Saving {'LoRA adapter' if use_lora else 'full model'} "
        "(gathering sharded state)...",
        flush=True,
    )
    trainer.save_model(save_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if RANK != 0:
        print(f"[rank{RANK}] checkpoint saved by rank 0; exiting.", flush=True)
        return

    # Free GPU memory held by the sharded training model before the CPU merge
    processing_class = trainer.processing_class
    del trainer
    torch.cuda.empty_cache()

    # Full-parameter training already wrote real weights to OUTPUT_DIR, so the
    # merge step is LoRA-only.
    if not use_lora:
        print(
            "Full-parameter run: weights already written, no merge needed.", flush=True
        )
    import transformers
    from transformers import AutoConfig

    # Resolve the concrete model class from the checkpoint rather than assuming
    # AutoModelForCausalLM — current Qwen/Gemma checkpoints are vision-language
    # classes (e.g. Qwen3_5ForConditionalGeneration).
    if use_lora:
        print("Merging adapter into base weights on CPU...", flush=True)
        from peft import PeftModel

        cfg = AutoConfig.from_pretrained(model_cache, trust_remote_code=True)
        model_cls = getattr(transformers, cfg.architectures[0])
        base = model_cls.from_pretrained(
            model_cache,
            dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base, save_dir).merge_and_unload()
        merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    # Serve with the published template, not the training one (see above).
    if upstream_chat_template is not None:
        processing_class.chat_template = upstream_chat_template
        print("Restored upstream chat template for inference", flush=True)
    processing_class.save_pretrained(OUTPUT_DIR)

    # vLLM instantiates the full processor for a vision-language architecture and
    # fails with "Can't load image processor" without preprocessor_config.json. We
    # hold a plain tokenizer, so save the base model's processor alongside it.
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(
            model_cache, trust_remote_code=True
        ).save_pretrained(OUTPUT_DIR)
        print("Saved AutoProcessor (preprocessor configs) to output", flush=True)
    except Exception as exc:  # text-only base models have no processor
        print(
            f"No AutoProcessor for this base model ({exc}); copying configs", flush=True
        )
        for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
            src = Path(model_cache) / name
            if src.exists():
                shutil.copy2(src, Path(OUTPUT_DIR) / name)
                print(f"  copied {name}", flush=True)
    print(f"Merged model saved to {OUTPUT_DIR}", flush=True)

    # Both modes must leave the same artifact shape; check here rather than letting
    # vLLM fail to start.
    required = ("config.json", "tokenizer_config.json")
    missing = [n for n in required if not (Path(OUTPUT_DIR) / n).exists()]
    weights = list(Path(OUTPUT_DIR).glob("*.safetensors"))
    if missing or not weights:
        raise RuntimeError(
            f"Incomplete checkpoint in {OUTPUT_DIR}: "
            f"missing {missing or 'no .safetensors'}. "
            f"Contents: {sorted(f.name for f in Path(OUTPUT_DIR).iterdir())[:20]}"
        )

    output_files = list(Path(OUTPUT_DIR).glob("*"))
    total_size = sum(f.stat().st_size for f in output_files if f.is_file())
    print(f"Output: {len(output_files)} files, {total_size / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
