#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("SFT (tool-calling) with LoRA via Accelerate")
    p.add_argument("--mode", choices=["pp", "ddp", "fsdp"], default="ddp")
    p.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--dataset", default="Salesforce/xlam-function-calling-60k")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--seed", type=int, default=42)

    # training
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-seq-length", type=int, default=1414)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")

    # LoRA
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    # quant + attention
    p.add_argument("--qlora", action="store_true", default=True,
                   help="Enable 4-bit + LoRA for pp/ddp. Forced off in fsdp.")
    p.add_argument("--no-qlora", dest="qlora", action="store_false")
    p.add_argument("--attn", choices=["flash_attention_2", "flash_attention_3", "sdpa", "eager"],
                   default=None)

    # logging
    p.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))
    p.add_argument("--mlflow-experiment", default="sft_lora_tool_calling")
    return p.parse_args()

# ---------------------------
# Helpers
# ---------------------------
def pick_attn_backend(ch: str | None) -> str:
    return ch or "flash_attention_2"  # H100 default

def resolve_device_map(mode: str):
    return "auto" if mode == "pp" else None

def build_bnb_config(enabled_4bit: bool):
    if not enabled_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def to_prompt_completion(ex):
    # Normalize tools and answer to JSON strings
    tools = ex.get("tools", [])
    if isinstance(tools, str):
        try: tools = json.loads(tools)
        except Exception: pass
    tools_txt = json.dumps(tools, ensure_ascii=False, indent=2)

    ans = ex.get("answer")
    if isinstance(ans, str):
        try: ans = json.loads(ans)
        except Exception: pass
    answer_json = json.dumps(ans, ensure_ascii=False)

    prompt = (
        "You are a function-calling assistant. Return ONLY a JSON array of tool calls "
        'as [{"name": "<tool_name>", "arguments": {...}}, ...].\n\n'
        f"TOOLS:\n{tools_txt}\n\n"
        f"USER QUERY:\n{ex.get('query','')}\n"
    )

    # TRL 0.21: prompt-completion dataset => loss on completion by default
    return {"prompt": prompt, "completion": answer_json}
        # or tool_calls field if template supports it
        #    {"prompt": [{"role":"user","content": ex["query"]}],
        #     "completion": [{"role":"assistant","content": ex["answer"]}],  
        #     "tools": tools_schema_list}

# Separator where the **answer JSON** begins (loss is applied after this)
SEP = "\n### ASSISTANT JSON:\n"

def formatting_func(ex):
    """
    Expects xLAM-like fields: query (str), answer (str|list|dict), tools (str|list|dict).
    Produces:
      [PROMPT (tools + query) + SEP + ANSWER_JSON]
    """
    # Normalize tools to pretty JSON for context
    tools = ex.get("tools", [])
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            pass
    # tools_text = json.dumps(tools, ensure_ascii=False, indent=2)
    # Removing indent to compact tokens
    tools_text = json.dumps(tools, ensure_ascii=False)

    # Normalize label (answer) to compact JSON
    ans = ex.get("answer")
    if isinstance(ans, str):
        try:
            ans = json.loads(ans)
        except Exception:
            pass
    answer_json = json.dumps(ans, ensure_ascii=False)

    query = ex.get("query", "")

    prompt = (
        "You are a function-calling assistant. Return ONLY a JSON array of tool calls "
        "with objects {\"name\": ..., \"arguments\": {...}}.\n\n"
        f"TOOLS:\n{tools_text}\n\n"
        f"USER QUERY:\n{query}\n"
        "RESPONSE FORMAT:\n"
        "[{\"name\": \"<tool_name>\", \"arguments\": {\"<arg>\": \"<value>\"}}]\n"
        "NO EXPLANATION."
    )
    return [prompt + SEP + answer_json]

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # MLflow (optional; harmless if URI unset)
    if args.mlflow_tracking_uri:
        import mlflow
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    # dataset
    ds = load_dataset(args.dataset, split=args.dataset_split)
    dataset = ds.map(to_prompt_completion, remove_columns=[c for c in ds.column_names if c not in {"prompt","completion"}])

    # tokenizer
    trust_remote_code = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model (PP gets device_map='auto'; DDP/FSDP placement handled by Accelerate)
    attn_impl = pick_attn_backend(args.attn)
    device_map = resolve_device_map(args.mode)

    enable_4bit = args.qlora and (args.mode != "fsdp")
    bnb_config = build_bnb_config(enable_4bit)

    model_kwargs = dict(
        trust_remote_code=trust_remote_code,
        use_cache=False,
        attn_implementation=attn_impl,
    )

    if enable_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # LoRA (PEFT)
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
    )

    # SFT config (replaces plain TrainingArguments)
    # NOTE: completion_only_loss=True + response_template=SEP will mask loss on the prompt,
    # keeping loss only on the **answer JSON** that follows SEP.
    # sft_config = SFTConfig(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     gradient_accumulation_steps=args.grad_accum_steps,
    #     learning_rate=args.learning_rate,
    #     warmup_ratio=args.warmup_ratio,
    #     max_steps=args.max_steps,
    #     logging_steps=args.logging_steps,
    #     save_steps=args.save_steps,
    #     optim=args.optim,
    #     bf16=args.bf16,
    #     # max_seq_length=args.max_seq_length,   # migrated
    #     report_to=["mlflow"] if args.mlflow_tracking_uri else [],
    #     completion_only_loss=True,
    #     response_template=SEP,
    # )

    # Trainer
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=ds,
    #     tokenizer=tokenizer,            # required with formatting_func
    #     formatting_func=formatting_func,
    #     peft_config=peft_config,
    #     args=sft_config,
    #     max_seq_length=args.max_seq_length,   # ✅ pass here in 0.21.0 (or default to tokenizer.max_model_length)
    # )

    from trl import SFTTrainer, SFTConfig

    # Optional: enforce your own max length in 0.21 by setting it on the tokenizer
    # tokenizer.model_max_length = args.max_seq_length  # safest cross-version approach
    tokenizer.model_max_length = 2048  # Catering to Llama's 8k context window / 4

    sft_config = SFTConfig(
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
        report_to=["mlflow"] if args.mlflow_tracking_uri else [],
        # completion_only_loss: for prompt-completion datasets this is True by default in 0.21;
        # you can still pass completion_only_loss=True explicitly if you want.
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,               # now contains "prompt" and "completion"
        processing_class=tokenizer,          # <- in 0.21 use processing_class (or omit)
        peft_config=peft_config,
        args=sft_config,
        packing=False                        # Turn off while debugging.
        # DO NOT pass response_template or tokenizer here in 0.21
    )

    # Adjust LoRA layers to bf16 to avoid dtype issues:
    try:
        from peft.tuners.lora import LoraLayer
        for m in trainer.model.modules():
            if isinstance(m, LoraLayer):
                m.to(torch.bfloat16)
    except Exception as e:
        print("[warn] Could not cast LoRA layers to bf16:", e)


    # PEFT + FSDP: wrap only LoRA params
    if args.mode == "fsdp" and getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    # Train
    trainer.model.print_trainable_parameters()
    trainer.train()

    # Save adapters + tokenizer
    adapters_dir = os.path.join(args.output_dir, "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    trainer.model.save_pretrained(adapters_dir)
    tokenizer.save_pretrained(adapters_dir)

    # Optional: try to save a full snapshot (may be sharded under FSDP)
    full_dir = os.path.join(args.output_dir, "full-model")
    os.makedirs(full_dir, exist_ok=True)
    try:
        trainer.save_model(full_dir)
        tokenizer.save_pretrained(full_dir)
    except Exception as e:
        print(f"[warn] full save failed under FSDP/sharded state: {e}")

if __name__ == "__main__":
    main()

