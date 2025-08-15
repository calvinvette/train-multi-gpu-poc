#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supervised fine-tuning with LoRA on TRL's SFTTrainer, driven by Hugging Face Accelerate.
Supports three "modes":
  - pp   : naive pipeline/model-parallel via device_map='auto'
  - ddp  : DistributedDataParallel (multi-node capable) via Accelerate
  - fsdp : Fully Sharded Data Parallel via Accelerate (PEFT-aware wrapping)

Targets:
  - Two machines, each: 16-core x86_64 CPU, 200GB RAM, 1x H100 80GB
  - SLURM orchestrated; job launched with accelerate

Telemetry:
  - MLflow system + custom metrics (GPU util %, GPU VRAM used/total, step throughput)
  - Optional Prometheus exporter (for Grafana) via --prometheus-port

MFU (Model FLOPs Utilization):
  - We log an *estimate* based on tokens/sec and param count with the standard 6*N*T/sec formula.
  - Provide your GPU peak BF16 TFLOPS with --gpu-peak-tflops to make the MFU % meaningful.
    (H100 PCIe vs. SXM differs; don’t hardcode—pass the correct peak value.)

Notes:
  - FSDP + 4-bit QLoRA is often fragile. By default we disable 4-bit in FSDP mode.
  - For H100, default attention backend is flash_attention_2.
  - We auto-detect dataset text field; override with --dataset-text-field if needed.

Dependencies (conda/pip):
  accelerate>=0.24  transformers>=4.41  trl  peft  datasets  bitsandbytes
  mlflow  prometheus_client  pynvml  torch (CUDA build)
"""

import argparse
import os
import time
import math
import json
from contextlib import nullcontext

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer

# Telemetry
import mlflow

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

try:
    from prometheus_client import start_http_server, Gauge
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False


# ---------------------------
# Utilities
# ---------------------------

def pick_attn_backend(user_choice: str | None) -> str:
    """
    Transformers accepts: 'flash_attention_2', 'flash_attention_3' (Blackwell),
    'sdpa' (PyTorch scaled-dot product attention), or 'eager'.
    Default to flash_attention_2 for H100.
    """
    if user_choice:
        return user_choice
    return "flash_attention_2"


def resolve_device_map(mode: str):
    """
    For 'pp' (naive pipeline/model parallel), we let HF shard with device_map='auto'.
    For DDP/FSDP (with Accelerate), we let Accelerate handle device placement and DO NOT
    pass device_map at model load.
    """
    if mode == "pp":
        return "auto"
    return None


def build_bnb_config(enabled_4bit: bool):
    if not enabled_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def maybe_textify(example):
    """
    Convert various conversational/structured datasets into a 'text' field.
    You can customize this to your schema. We try common fields first.
    """
    if "text" in example and isinstance(example["text"], str):
        return {"text": example["text"]}

    # Common chat formats
    if "messages" in example and isinstance(example["messages"], list):
        # naive render: "[role]: content\n"
        chunks = []
        for m in example["messages"]:
            role = m.get("role", "user")
            content = m.get("content", "")
            chunks.append(f"{role}: {content}")
        return {"text": "\n".join(chunks)}

    if "instruction" in example and "output" in example:
        return {"text": f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['output']}"}

    # Fallback: stringify everything
    return {"text": json.dumps(example, ensure_ascii=False)}


class GPUAndMFUTracker:
    """
    Collect GPU utilization, VRAM, and estimate MFU.
    - Logs to MLflow
    - Optionally exposes Prometheus metrics
    """
    def __init__(self, gpu_peak_tflops=None, prometheus_port=None, world_size=1):
        self.gpu_peak_tflops = gpu_peak_tflops  # e.g. 990 for ~990 TFLOPS BF16 on certain H100 SKUs
        self.world_size = max(1, int(world_size))
        self.last_log_t = time.time()
        self.sample_tokens = 0
        self.sample_time = 0.0

        self.prom_enabled = _HAS_PROM and (prometheus_port is not None)
        self.nvml_enabled = _HAS_NVML

        if self.nvml_enabled:
            pynvml.nvmlInit()
            # Assume single GPU per process
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

        # Prometheus Gauges
        if self.prom_enabled:
            start_http_server(prometheus_port)
            self.gpu_util_g = Gauge('gpu_utilization_percent', 'GPU util %')
            self.gpu_mem_used_g = Gauge('gpu_mem_used_bytes', 'GPU memory used (bytes)')
            self.gpu_mem_total_g = Gauge('gpu_mem_total_bytes', 'GPU memory total (bytes)')
            self.mfu_g = Gauge('model_flops_utilization_percent', 'Estimated MFU %')
            self.tokps_g = Gauge('tokens_per_second', 'Tokens per second (aggregate)')

    def update_token_counters(self, tokens_this_step: int, step_time_sec: float):
        self.sample_tokens += tokens_this_step
        self.sample_time += max(1e-6, step_time_sec)

    def _log_gpu_now(self):
        if not self.nvml_enabled:
            return None

        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        gpu_util = float(util.gpu)
        mem_used = float(mem.used)
        mem_total = float(mem.total)

        mlflow.log_metric("gpu_util_percent", gpu_util)
        mlflow.log_metric("gpu_mem_used_bytes", mem_used)
        mlflow.log_metric("gpu_mem_total_bytes", mem_total)

        if self.prom_enabled:
            self.gpu_util_g.set(gpu_util)
            self.gpu_mem_used_g.set(mem_used)
            self.gpu_mem_total_g.set(mem_total)

        return gpu_util, mem_used, mem_total

    def _estimate_mfu(self, param_count: int):
        """
        MFU ≈ (6 * N_params * tokens/sec) / peak_flops
        - tokens/sec computed from sliding counters
        - peak_flops = gpu_peak_tflops * 1e12
        Result as percentage in [0,100].
        """
        if not self.gpu_peak_tflops:
            return None, None

        if self.sample_time <= 0:
            return None, None

        tok_per_sec = (self.sample_tokens / self.sample_time) * self.world_size
        achieved_flops = 6.0 * float(param_count) * tok_per_sec  # FLOPs/sec
        peak_flops = float(self.gpu_peak_tflops) * 1e12
        mfu = 100.0 * (achieved_flops / peak_flops)
        return mfu, tok_per_sec

    def periodic_log(self, param_count: int, every_sec: float = 5.0):
        now = time.time()
        if now - self.last_log_t < every_sec:
            return
        self.last_log_t = now

        self._log_gpu_now()
        mfu, tokps = self._estimate_mfu(param_count)
        if mfu is not None:
            mlflow.log_metric("mfu_percent", mfu)
        if tokps is not None:
            mlflow.log_metric("tokens_per_second", tokps)

        if self.prom_enabled:
            if mfu is not None:
                self.mfu_g.set(mfu)
            if tokps is not None:
                self.tokps_g.set(tokps)

    def close(self):
        if self.nvml_enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="TRL SFT with LoRA + Accelerate + MLflow/Grafana")
    # Core
    parser.add_argument("--mode", choices=["pp", "ddp", "fsdp"], default="ddp",
                        help="Training mode: naive pipeline (pp), DDP (ddp), or FSDP (fsdp)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dataset", default="Salesforce/xlam-function-calling-60k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-text-field", default=None, help="If set, use this field as text")
    parser.add_argument("--seed", type=int, default=42)

    # Training hyperparams
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.1)

    # Quantization and attention
    parser.add_argument("--qlora", action="store_true", default=True,
                        help="Enable 4-bit quantization + LoRA for pp/ddp. Forced off in fsdp.")
    parser.add_argument("--no-qlora", dest="qlora", action="store_false")
    parser.add_argument("--attn", choices=["flash_attention_2", "flash_attention_3", "sdpa", "eager"],
                        default=None, help="Override attention backend")

    # MLflow & Prometheus
    parser.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))  # if empty, file-based
    parser.add_argument("--mlflow-experiment", default="sft_lora")
    parser.add_argument("--gpu-peak-tflops", type=float, default=None,
                        help="GPU peak BF16 TFLOPS for MFU calc (e.g., 990).")
    parser.add_argument("--prometheus-port", type=int, default=None,
                        help="If set, expose Prometheus /metrics on this port")

    args = parser.parse_args()
    set_seed(args.seed)

    # --------------------------------------------------------------------------------
    # Environment: enable MLflow system metrics; set tracking URI if provided
    # --------------------------------------------------------------------------------
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment)

    # --------------------------------------------------------------------------------
    # Attention backend and device map
    # --------------------------------------------------------------------------------
    attn_implementation = pick_attn_backend(args.attn)  # default fa2 for H100
    device_map = resolve_device_map(args.mode)

    # FSDP + 4-bit: usually not supported. Force off 4-bit in FSDP.
    enable_4bit = args.qlora and (args.mode != "fsdp")
    bnb_config = build_bnb_config(enabled_4bit=enable_4bit)

    # Gradient checkpointing tuning per mode
    grad_ckpt_kwargs = {"use_reentrant": True}
    if args.mode == "ddp":
        # Faster & less VRAM overhead under DDP with reentrant=False
        grad_ckpt_kwargs = {"use_reentrant": False}

    # --------------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------------
    raw = load_dataset(args.dataset, split=args.dataset_split)

    if args.dataset_text_field:
        # Ensure the field exists
        if args.dataset_text_field not in raw.column_names:
            raise ValueError(f"--dataset-text-field '{args.dataset_text_field}' not found in: {raw.column_names}")
        dataset = raw.rename_column(args.dataset_text_field, "text")
    else:
        # Map to "text"
        dataset = raw.map(maybe_textify, remove_columns=raw.column_names)

    # --------------------------------------------------------------------------------
    # Tokenizer and Model
    # --------------------------------------------------------------------------------
    trust_remote_code = True
    tokenizer = AutoTokenizer.from_pretrained(args.model-name if hasattr(args, "model-name") else args.model_name,
                                              trust_remote_code=trust_remote_code)
    # accommodate models missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        trust_remote_code=trust_remote_code,
        use_cache=False,
        attn_implementation=attn_implementation,
    )

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        # dtype left to bnb compute dtype (bf16 above)
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # --------------------------------------------------------------------------------
    # PEFT (LoRA)
    # --------------------------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
    )

    # --------------------------------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------------------------------
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim=args.optim,
        bf16=args.bf16,
        group_by_length=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=grad_ckpt_kwargs,
        report_to=["mlflow"],       # <-- send trainer metrics to MLflow
    )

    # --------------------------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        args=training_arguments,
    )

    # Adjust FSDP auto-wrap policy so only PEFT params are wrapped/sharded.
    if args.mode == "fsdp":
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy
            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # --------------------------------------------------------------------------------
    # Metrics: MLflow run + optional Prometheus + MFU estimate
    # --------------------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tracker = GPUAndMFUTracker(
        gpu_peak_tflops=args.gpu_peak_tflops,
        prometheus_port=args.prometheus_port,
        world_size=world_size,
    )

    # Log run params
    with mlflow.start_run() as run:
        mlflow.log_params({
            "mode": args.mode,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "dataset_split": args.dataset_split,
            "qlora": enable_4bit,
            "attn_implementation": attn_implementation,
            "bf16": args.bf16,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "max_steps": args.max_steps,
            "world_size": world_size,
        })

        # count parameters
        total_params = sum(p.numel() for p in trainer.model.parameters())
        mlflow.log_metric("model_parameters", total_params)

        # Training loop wrapper to capture tokens/sec per step and GPU stats
        orig_training_step = trainer.training_step

        def wrapped_training_step(model, inputs):
            start_t = time.time()
            out = orig_training_step(model, inputs)
            step_t = max(1e-6, time.time() - start_t)

            # Estimate tokens processed this step
            # Best-effort: batch_size * seq_len; use the tokenizer pad length when present.
            bs = training_arguments.per_device_train_batch_size
            ga = training_arguments.gradient_accumulation_steps
            # Try infer real input length:
            input_ids = inputs.get("input_ids", None)
            if input_ids is not None:
                # average per-sample length * batch_size
                seq_len = int(torch.tensor([x.shape[-1] for x in input_ids]).float().mean().item()) if isinstance(input_ids, list) else int(input_ids.shape[-1])
            else:
                seq_len = args.max_seq_length

            effective_bs = bs  # per-process; tokens/sec later multiplied by world_size in tracker
            tokens_this_step = effective_bs * seq_len

            # Update token/time counters for MFU & throughput
            tracker.update_token_counters(tokens_this_step, step_t)

            # Periodically also log GPU + MFU
            tracker.periodic_log(param_count=total_params, every_sec=5.0)
            return out

        trainer.training_step = wrapped_training_step

        # Kick training
        trainer.model.print_trainable_parameters()
        trainer.train()

        # Save adapters (LoRA) and tokenizer
        adapters_dir = os.path.join(args.output_dir, "adapters")
        full_dir = os.path.join(args.output_dir, "full-model")
        os.makedirs(adapters_dir, exist_ok=True)
        os.makedirs(full_dir, exist_ok=True)

        # Save just LoRA adapters (PEFT)
        trainer.model.save_pretrained(adapters_dir)
        tokenizer.save_pretrained(adapters_dir)
        mlflow.log_artifacts(adapters_dir, artifact_path="adapters")

        # If not in FSDP sharded state, also save a full merged snapshot (optional)
        try:
            # For FSDP, best to use SHARDED_STATE_DICT in Accelerate’s plugin to recompose before saving full.
            trainer.save_model(full_dir)
            tokenizer.save_pretrained(full_dir)
            mlflow.log_artifacts(full_dir, artifact_path="full-model")
        except Exception as e:
            # It's fine if we can’t (e.g., FSDP with sharded state)
            mlflow.log_text(str(e), "save_full_model_error.txt")

        tracker.close()


if __name__ == "__main__":
    main()

