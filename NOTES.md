# Project NOTES.md — Fine-tuning for Tool/Function Calling (xLAM → Llama 3.2‑3B)
# Project Notes: Multi-GPU Fine-Tuning with DDP/FSDP

---

This file is meant to keep ChatGPT, Continue.dev (Cline), and any other assistants in sync.
Update this file whenever the training scripts, configs, or datasets change.



> This file is a “memory bridge” so editor assistants (Continue.dev/Cline/Cursor/etc.) always have the essentials for this repo. Keep it short, current, and version‑accurate.

--- 

## Current State
- Training script (`train_sft.py`) works with HuggingFace Accelerate on SLURM and non-SLURM setups.
- Validated on H100 hardware (CUDA 12.6, FlashAttention 2.8.3 built from source).
- MLflow integration works: tracks loss, metrics, and system stats (GPU utilization, memory).
- QLoRA + bf16 training confirmed working after forcing LoRA modules into bf16.
- Produces saved model and adapter layers at end of training.

---

## Improvements Incorporated
1. **LoRA dtype fix**  
   - Cast all LoRA layers to bf16 when running with `--bf16 --qlora` to avoid `float != bfloat16` runtime errors.
   - Example:
     ```python
     from peft.tuners.lora import LoraLayer
     for m in model.modules():
         if isinstance(m, LoraLayer):
             m.to(torch.bfloat16)
     ```

2. **BitsAndBytesConfig improvements**  
   - Ensure quantized models use:
     ```python
     bnb_4bit_compute_dtype=torch.bfloat16
     ```

3. **Accelerate launcher cleanup**  
   - Removed stray `python` after `accelerate launch`.  
   - Fixed line-continuation with no trailing spaces after `\`.

4. **NUM_GPUs detection**  
   - Improved with `wc -l`:
     ```bash
     NUM_GPUs=$(nvidia-smi -L | wc -l)
     ```

5. **Dataset preprocessing**  
   - For Salesforce/xlam-function-calling-60k:
     ```python
     def format_example(example):
         return {
             "text": f"query:\n{example['query']}\n\nanswer:\n{example['answer']}"
         }
     ```
   - Tokenization handled during training with `completion_only_loss=True`.

---

## Open Questions / TODO
- [ ] Decide whether to log GPU VRAM/TFLOPs directly to MLflow or scrape via Prometheus/Grafana.
- [ ] Benchmark FlashAttention 2.8.3 vs. 3.x on H100s once stable.
- [ ] Test multi-node SLURM training (`MACHINE_RANK`, `MASTER_ADDR`) with >1 GPU/node.
- [ ] Add validation/eval dataset to monitor function-calling accuracy (not just training loss).
- [ ] Update README with exact Accelerate CLI invocations for each mode (DDP, FSDP, PP).

---

## Sync Checklist (ChatGPT â Continue.dev)
Whenever you change one of these, update NOTES.md:
- Dataset path or formatting
- Model name/version
- Training hyperparameters (batch size, lr, warmup, etc.)
- Accelerate/SLURM configs
- FlashAttention / CUDA versions
- Logging integrations (MLflow, Prometheus)

---

Previous Notes

---

## 0) TL;DR (What we’re doing)
- **Task:** SFT to generate **tool/function calls** (JSON array of `{name, arguments}`).
- **Dataset:** `Salesforce/xlam-function-calling-60k` (gated).
- **Model:** `meta-llama/Llama-3.2-3B-Instruct` (gated).
- **Frameworks:** `transformers`, `trl==0.21.0`, `accelerate`, `peft`, `datasets`.
- **Parallelism modes:** `--mode ddp` (single/multi‑node) and `--mode fsdp` (multi‑node).
- **Precision:** H100 → `bf16`, attention backend default `flash_attention_2` (custom compiled version 2.8.2 for minimum SMS of 87 (Amphere); consider version 3.x for Blackwell).
- **Adapters:** LoRA; **QLoRA off** for FSDP (use for DDP/PP only).
- **Data style:** **prompt/completion** (completion‑only loss by default in TRL 0.21).
- **Launch:** via HuggingFace Accelerate; SLURM scripts provided.
- **Telemetry:** MLflow (+ optional Prometheus/Grafana) — can be enabled later.

---

## 1) Access + Auth (gated assets)
1. Accept terms on HF Hub for:
   - `meta-llama/Llama-3.2-3B-Instruct`
   - `Salesforce/xlam-function-calling-60k`
2. Authenticate:
   ```bash
   huggingface-cli login
   # or
   export HUGGINGFACE_HUB_TOKEN=hf_xxx
   ```

---

## 2) Environment (conda/venv)
```bash
cp .env.example .env # Customize
uv init
. .venv/bin/activate
. .env
pip install --upgrade pip
pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.41" "trl==0.21.0" "accelerate>=0.24" "peft>=0.11.1" "datasets>=2.20"
# Optional (DDP/PP only): QLoRA
pip install bitsandbytes>=0.43
# Optional telemetry for later
pip install mlflow prometheus-client pynvml
```

---

## 3) Data preprocessing (xLAM → prompt/completion)
We convert each example:
- **PROMPT** includes the per‑example `tools` schema and the `query`.
- **COMPLETION** is **JSON‑only** array of tool calls.

```python
# scripts/preproc_xlam.py (snippet)
import json

def xlam_to_prompt_completion(ex):
    tools = ex.get("tools", [])
    if isinstance(tools, str):
        try: tools = json.loads(tools)
        except Exception: pass
    tools_txt = json.dumps(tools, ensure_ascii=False, indent=2)

    ans = ex.get("answer")
    if isinstance(ans, str):
        try: ans = json.loads(ans)
        except Exception: pass
    completion = json.dumps(ans, ensure_ascii=False)

    prompt = (
        "You are a function-calling assistant. Return ONLY a JSON array of tool calls "
        "as [{"name": "<tool>", "arguments": {...}}, ...].\n\n"
        f"TOOLS:\n{tools_txt}\n\n"
        f"USER QUERY:\n{ex.get('query','')}\n"
        "NO EXPLANATION. JSON ONLY."
    )
    return {"prompt": prompt, "completion": completion}
```
Usage in training: map this over the dataset and keep only `prompt`, `completion`.

---

## 4) Training wiring (TRL 0.21)
- **No** `response_template` or `max_seq_length` in `SFTConfig` (removed in 0.21).
- Use `processing_class=tokenizer` (not `tokenizer=`) in the trainer.
- Completion‑only loss is **on by default** when using prompt/completion data.
- Enforce length via `tokenizer.model_max_length` or `max_seq_length` on `SFTTrainer`.

```python
from trl import SFTTrainer, SFTConfig

tokenizer.model_max_length = 1024  # or set via args

sft_config = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    max_steps=1000,
    logging_steps=10,
    save_steps=200,
    optim="adamw_torch",
    bf16=True,
    completion_only_loss=True,  # explicit (default is True for prompt/completion)
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,            # contains {prompt, completion}
    processing_class=tokenizer,       # TRL 0.21 API
    peft_config=peft_config,          # LoRA config
    args=sft_config,
    # or: max_seq_length=1024 (supported on SFTTrainer in 0.21)
)
```

**LoRA (PEFT) targets** for Llama‑3.x typically include: `["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]`

---

## 5) Parallelism modes
- **DDP**: `--mode ddp` (QLoRA allowed). For single‑GPU node, `num_processes=1`. Multi‑GPU → one proc/GPU.
- **FSDP**: `--mode fsdp` (QLoRA off; BF16 + LoRA). Use `fsdp_auto_wrap_policy` to wrap only PEFT params.

**Accelerate config files** live in `configs/`:
- `accelerate_ddp.yaml` — single GPU per node (adjust `num_processes` for multi‑GPU).
- `accelerate_fsdp.yaml` — two nodes × one GPU each; `FULL_SHARD`, `SHARDED_STATE_DICT`, `use_orig_params: false`.

---

## 6) SLURM launch (examples)
**DDP, single node**
```bash
sbatch run_ddp_1node.sbatch
```
**FSDP, two nodes**
```bash
sbatch run_fsdp_2nodes.sbatch
```

Key envs used inside scripts:
- `MASTER_ADDR`, `MASTER_PORT`
- `NUM_MACHINES=$SLURM_NNODES`, `MACHINE_RANK=$SLURM_NODEID`
- NCCL settings (`NCCL_DEBUG=WARN`, etc.)

---

## 7) Inference reminders
- Use greedy or low‑temperature decoding; parse with `json.loads`.
- Optionally add constrained decoding later (Outlines / Lm-format-enforcer) for strict JSON.
- Always return an **array** (even single tool call).

---

## 8) Telemetry (later)
- **MLflow**: set `MLFLOW_TRACKING_URI`, `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true`.
- **Prometheus/Grafana**: training script can expose `/metrics` (enable via CLI flag). Add static or file_sd targets in Prometheus.

---

## 9) Troubleshooting cheats
- **403 on HF assets** → accept terms + `huggingface-cli login`.
- **TRL arg errors** → you’re on `trl==0.21.0`: don’t pass `response_template` or `max_seq_length` into `SFTConfig`; use `processing_class`, not `tokenizer=`.
- **OOM** → reduce `per_device_train_batch_size` / `max_seq_length`, increase `grad_accum_steps`.
- **NCCL failures** → verify `MASTER_ADDR/PORT`, fabric (`NCCL_SOCKET_IFNAME`), and that `MACHINE_RANK/LOCAL_RANK` are correct.
- **FSDP save** → use `SHARDED_STATE_DICT`; for full save, recombine or save adapters only.

---

## 10) Open tasks / Decisions log
- [ ] Confirm final `max_seq_length`: ______
- [ ] Decide if we enable packing: true/false (TRL 0.21 supports it).
- [ ] Add evaluation split & simple JSON validity metric.
- [ ] Wire MLflow params/metrics fully (tokens/sec, MFU; optional Prometheus).

---

## 11) Quick context (for editor assistants)
- Repo files of interest:
  - `train_sft.py` — main entrypoint (`--mode ddp|fsdp|pp`, LoRA, H100 defaults).
  - `configs/accelerate_*.yaml` — Accelerate configs.
  - `run_*.sbatch` — SLURM launchers.
  - `scripts/preproc_xlam.py` — dataset mapping (prompt/completion).
- Always assume: TRL 0.21 semantics, tool‑calling completions are **JSON‑only**.

---


