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
from pathlib import Path

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


def get_hp(key, default=None, cast=None):
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


def log_sharding(model) -> None:
    """Confirm parameters are sharded across GPUs rather than replicated.

    With FSDP2/DTensor, `p.numel()` reports the GLOBAL shape and there is no
    FullyShardedDataParallel wrapper class, so counting parameters or looking
    for wrapper names both wrongly report "not sharded". The reliable signal is
    the size of each parameter's LOCAL storage.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()  # type: ignore[assignment]

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


def main():
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
    use_liger = get_hp("use_liger_kernel", "1") in ("1", "true", "True")
    save_total_limit = get_hp("save_total_limit", 2, int)

    print("=" * 60)
    print("AgentCore Deep Research — LoRA SFT")
    print("=" * 60)
    print(f"Model:          {hf_model_id}")
    print(f"Seed:           {seed}")
    print(f"Liger kernel:   {use_liger}")
    print(f"LoRA rank:      {lora_rank}")
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

    # Only rank 0 downloads. All four ranks calling snapshot_download into the
    # same local_dir races and leaves partial/missing shards, which surfaces
    # much later as FileNotFoundError on a shard at model-load time — a failure
    # that looks like a flaky download but is actually concurrent writers.
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

    # LoRA config
    # Qwen3.5 is a vision-language architecture: a 27-layer vision tower sits
    # alongside the 32-layer text model. "all-linear" would adapt the vision
    # tower too, but we train on text-only trajectories, so those adapters never
    # receive gradient. They stay zero-initialised and merge as identity, so
    # results are unaffected — but they consume GPU memory and merge time for
    # nothing. Excluding them puts the whole memory budget behind the text model.
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        exclude_modules=r".*\.?(visual|vision_tower|vision_model)\..*",
    )

    # assistant_only_loss masks the loss to assistant turns, but it requires the
    # chat template to mark them with {% generation %}. Not all model families
    # ship such a template (Gemma 4 does, Qwen3.5 does not), and enabling it
    # without the markers raises at dataset-prep time — so detect it.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_cache, trust_remote_code=True)

    # Loss masking needs {% generation %} markers in the chat template so
    # transformers can build the assistant-token mask. Most published templates,
    # Qwen3.5's included, do not have them — and enabling assistant_only_loss
    # against such a template does NOT error, it silently trains on the whole
    # sequence including tool observations, teaching the model to fabricate
    # retrieval results.
    #
    # TRL handles this: SFTTrainer swaps in a marked-up training template for
    # recognised model families (see trl/chat_templates/). We rely on that rather
    # than vendoring a copy, which would only drift from upstream. What we add is
    # the assertion below, because TRL only covers listed families and the
    # failure is silent for anything else.
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

    # Training on tool observations teaches the model to fabricate retrieval
    # results, which is the single worst failure mode for a research agent and
    # is invisible in the loss curve. Without a {% generation %} block there is
    # no way to mask them, so refuse to run rather than silently produce a
    # model trained on the wrong tokens. Set ALLOW_UNMASKED_OBSERVATIONS=1 to
    # override deliberately (e.g. for a non-agentic dataset).
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
        # Liger replaces RMSNorm/SwiGLU/RoPE and the cross-entropy head with
        # fused Triton kernels. The fused linear cross-entropy is the large
        # win here: it avoids materialising the full logits tensor, which at
        # 32K sequence length dominates activation memory. Support is
        # per-architecture, so this fails fast rather than silently if the
        # model is unsupported — set --use-liger-kernel 0 in that case.
        use_liger_kernel=use_liger,
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
        # Save adapters during the run, not only at the end. Intermediate
        # checkpoints let us measure quality partway through instead of waiting
        # ~33h to learn whether the recipe worked, and they show whether the
        # score is still climbing or has plateaued. Requires the job to declare
        # a CheckpointConfig, otherwise SageMaker never copies this directory to
        # S3 and the adapters die with the container.
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
        f"batch={ta.per_device_train_batch_size} grad_accum={ta.gradient_accumulation_steps} "
        f"grad_ckpt={ta.gradient_checkpointing} "
        f"assistant_only_loss={getattr(ta, 'assistant_only_loss', None)} "
        f"fsdp={ta.fsdp} world_size={ta.world_size}",
        flush=True,
    )
    log_sharding(trainer.model)

    # Confirms whether reports are training in full or being truncated
    # Check every example, not a prefix. Datasets are usually concatenated from
    # several collection batches, so a prefix sample can miss an entire batch
    # whose trajectories are longer. A sequence at the cap has been truncated,
    # which silently cuts the end of the report and trains the model to stop
    # mid-report, so treat it as fatal rather than informational.
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
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1 or state.global_step % 10 == 0:
                log_mem(f"step {state.global_step}")

    trainer.add_callback(MemCallback())

    print("Starting SFT training...", flush=True)
    trainer.train()
    print("Training complete.", flush=True)

    # Saving under FSDP needs care: the trained parameters are sharded
    # DTensors, so calling merge_and_unload().save_pretrained() directly fails
    # with safetensors "Attempted to access the data pointer on an invalid
    # python storage". Instead: let the trainer gather the adapter to a full
    # state dict, then merge into the base model in a single process on CPU
    # (the instance has ample host RAM for a BF16 copy).
    adapter_dir = "/opt/ml/checkpoints/adapter"
    print("Saving LoRA adapter (gathering sharded state)...", flush=True)
    trainer.save_model(adapter_dir)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if RANK != 0:
        print(f"[rank{RANK}] adapter saved by rank 0; exiting.", flush=True)
        return

    # Free GPU memory held by the sharded training model before the CPU merge
    processing_class = trainer.processing_class
    del trainer
    torch.cuda.empty_cache()

    print("Merging adapter into base weights on CPU...", flush=True)
    import transformers
    from peft import PeftModel
    from transformers import AutoConfig

    # Resolve the concrete model class from the checkpoint rather than assuming
    # AutoModelForCausalLM — current Qwen/Gemma checkpoints are vision-language
    # classes (e.g. Qwen3_5ForConditionalGeneration).
    cfg = AutoConfig.from_pretrained(model_cache, trust_remote_code=True)
    model_cls = getattr(transformers, cfg.architectures[0])
    base = model_cls.from_pretrained(
        model_cache,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    # Serve with the published template, not the training one (see above).
    if upstream_chat_template is not None:
        processing_class.chat_template = upstream_chat_template
        print("Restored upstream chat template for inference", flush=True)
    processing_class.save_pretrained(OUTPUT_DIR)

    # Qwen3.5 declares a vision-language architecture, so a serving stack such
    # as vLLM instantiates the full processor and hard-fails with
    # "Can't load image processor for ..." if preprocessor_config.json and
    # video_preprocessor_config.json are absent. We train text-only and so hold
    # a plain tokenizer as processing_class, which does not emit those files.
    # Persist the base model's full processor alongside it so the merged
    # checkpoint is self-contained and directly servable.
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

    output_files = list(Path(OUTPUT_DIR).glob("*"))
    total_size = sum(f.stat().st_size for f in output_files if f.is_file())
    print(f"Output: {len(output_files)} files, {total_size / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
