# SFT with LoRA on Llama‑3.2‑3B (DDP & FSDP via Accelerate + SLURM)

This project fine‑tunes a gated model and dataset using [TRL’s `SFTTrainer`](https://github.com/huggingface/trl) with Hugging Face Accelerate on SLURM:

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` (gated)  
- **Dataset:** `Salesforce/xlam-function-calling-60k` (gated)  
- **Modes:**  
  - **DDP** (Distributed Data Parallel) — single or multi‑node  
  - **FSDP** (Fully Sharded Data Parallel) — multi‑node friendly, PEFT‑aware

> ✅ Before running, you must accept the model & dataset terms on Hugging Face and log in locally (see **Gated assets access**).

---

The "download_*" scripts download the data and models respectively. 
The "download_model.py" downloads both Llama 3.2-3B and Qwen3-Coder-30B-A3B. If you're only using one, comment out the other.
Optionally, copy the .env.example to .env and modify appropriately

## 1) Hardware & cluster assumptions

- 2× SLURM nodes; each node has:
  - 16‑core x86_64 CPU, 200 GB RAM
  - **1× NVIDIA H100 80 GB**
- CUDA drivers + compatible PyTorch build installed on nodes
- Nodes can communicate over a fast fabric (IB/RoCE or 10/25/100G TCP)
- Python 3.10+ recommended

---

## 2) Gated assets access (Hugging Face)

These steps are required once per user or machine:

1. **Accept access** on Hugging Face Hub for:
   - `meta-llama/Llama-3.2-3B-Instruct`
   - `Salesforce/xlam-function-calling-60k`
2. **Authenticate locally** on each node (or in the image):
   ```bash
   huggingface-cli login
   ```
   Alternatively, export a token (ephemeral shells):
   ```bash
   export HUGGINGFACE_HUB_TOKEN=hf_xxx_your_token_here
   ```

> If access is not granted, downloads will fail with 403 errors.

---

## 3) Environment setup

Create a clean env (venv, conda, or uv), then install deps:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# Core
pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.41" "accelerate>=0.24" "trl>=0.9.6" "peft>=0.11.1" "datasets>=2.20"

# Optional for QLoRA (DDP/PP only; FSDP uses full/bf16 by default)
pip install bitsandbytes>=0.43

# Telemetry (we’ll wire up later)
pip install mlflow>=2.13 prometheus-client>=0.20 pynvml>=11.5
```

Check GPU visibility:

```bash
python - <<'PY'
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

---

## 4) Files in this repo

```
.
├── train_sft.py                    # main training script (DDP/FSDP via --mode)
├── configs/
│   ├── accelerate_ddp.yaml        # DDP config (1 GPU per node)
│   └── accelerate_fsdp.yaml       # FSDP config (2 nodes × 1 GPU)
├── run_ddp_1node.sbatch           # SLURM batch: single node DDP
└── run_fsdp_2nodes.sbatch         # SLURM batch: two nodes FSDP
```

> If you don’t see these, copy them from this README or your previous messages into the repo.

---

## 5) Accelerate configuration

You can either:
- run the wizard once: `accelerate config` (interactive), **or**
- use the provided YAMLs in `configs/` and pass `--config_file` at launch (recommended for SLURM).

### `configs/accelerate_ddp.yaml` (single GPU per node)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1              # override at launch if multi-node
num_processes: 1             # 1 GPU per node
machine_rank: 0
mixed_precision: bf16
gpu_ids: all
```

### `configs/accelerate_fsdp.yaml` (2 nodes × 1 GPU)
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
num_machines: 2
num_processes: 1
machine_rank: 0
mixed_precision: bf16
gpu_ids: all
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: NO_PREFETCH
  fsdp_sharding_strategy: FULL_SHARD
  offload_params: false
  min_num_params: 0
  state_dict_type: SHARDED_STATE_DICT
  use_orig_params: false
```

---

## 6) SLURM batch scripts

> Adjust `#SBATCH -A`, `-p`, time, and MLflow env as needed.

### A) Single node DDP: `run_ddp_1node.sbatch`
```bash
#!/bin/bash
#SBATCH -J sft-ddp
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load cuda || true
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}
export NUM_MACHINES=1
export MACHINE_RANK=0

mkdir -p logs

srun --cpu-bind=none bash -lc '
source .venv/bin/activate
accelerate launch   --config_file configs/accelerate_ddp.yaml   --num_machines ${NUM_MACHINES}   --machine_rank ${MACHINE_RANK}   --main_process_ip ${MASTER_ADDR}   --main_process_port ${MASTER_PORT}   train_sft.py     --mode ddp     --model-name meta-llama/Llama-3.2-3B-Instruct     --dataset Salesforce/xlam-function-calling-60k     --dataset-split train     --output-dir ./results_ddp_llama3_3b     --max-steps 1000     --per-device-train-batch-size 4     --grad-accum-steps 4     --learning-rate 2e-4     --warmup-ratio 0.1     --logging-steps 10     --save-steps 200     --bf16     --qlora
'
```

### B) Two nodes FSDP: `run_fsdp_2nodes.sbatch`
```bash
#!/bin/bash
#SBATCH -J sft-fsdp
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load cuda || true
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export NUM_MACHINES=$SLURM_NNODES
export MACHINE_RANK=$SLURM_NODEID

mkdir -p logs

srun --cpu-bind=none bash -lc '
source .venv/bin/activate
accelerate launch   --config_file configs/accelerate_fsdp.yaml   --num_machines ${NUM_MACHINES}   --machine_rank ${MACHINE_RANK}   --main_process_ip ${MASTER_ADDR}   --main_process_port ${MASTER_PORT}   train_sft.py     --mode fsdp     --model-name meta-llama/Llama-3.2-3B-Instruct     --dataset Salesforce/xlam-function-calling-60k     --dataset-split train     --output-dir ./results_fsdp_llama3_3b     --max-steps 1000     --per-device-train-batch-size 4     --grad-accum-steps 4     --learning-rate 2e-4     --warmup-ratio 0.1     --logging-steps 10     --save-steps 200     --bf16     --no-qlora
'
```

> **Why `--no-qlora` for FSDP?** FSDP with 4‑bit adapters is often unstable and nullifies sharding benefits. For H100 80GB, BF16 + LoRA under FSDP is typically fine.

---

## 7) Training script usage

The training entrypoint is `train_sft.py`.

Key flags:
- `--mode {ddp,fsdp,pp}` → selects DDP, FSDP, or naive pipeline (device_map='auto')  
- `--model-name` → which HF model repo to load  
- `--dataset` / `--dataset-split`  
- `--qlora` / `--no-qlora` → enable 4‑bit (DDP/PP) or disable (FSDP default)  
- `--bf16` (default on for H100)

Example (manual launch on a single node without SLURM):
```bash
accelerate launch --config_file configs/accelerate_ddp.yaml   train_sft.py   --mode ddp   --model-name meta-llama/Llama-3.2-3B-Instruct   --dataset Salesforce/xlam-function-calling-60k   --output-dir ./results_local_ddp   --max-steps 200 --per-device-train-batch-size 4 --grad-accum-steps 4 --bf16 --qlora
```

Artifacts:
- LoRA adapters saved under `<output-dir>/adapters/`
- (Best‑effort) full model snapshot under `<output-dir>/full-model/`

---

## 8) Troubleshooting

- **403 on model/dataset**: Ensure you’ve accepted the Hugging Face terms and are logged in (`huggingface-cli login` or `HUGGINGFACE_HUB_TOKEN`).
- **NCCL timeouts**: Verify node‑to‑node connectivity and that `MASTER_ADDR`, `MASTER_PORT` are reachable. Consider setting `NCCL_SOCKET_IFNAME` to your fabric (e.g., `ib0` or `eth0`).
- **OOM on H100**:
  - Reduce `--per-device-train-batch-size` or `--max-seq-length`.
  - Increase `--grad-accum-steps` to maintain global batch size.
- **Slow downloads**: Warm caches by running once on a shared filesystem (`HF_HOME`, `TRANSFORMERS_CACHE`).

---

## 9) Next steps (optional)

- Wire **MLflow**, **Prometheus**, and **Grafana** (already scaffolded in `train_sft.py`).  
- Add resume/ckpt strategies and evaluation splits.  
- Experiment with packing, curriculum, or better formatting for function‑calling data.
- Clean up Terraform

Logging to MLFLOW:

```
export MLFLOW_TRACKING_SERVER_CERT_PATH=</path/to/certificate>
export MLFLOW_TRACKING_URI=https://public-tracking-...-mlflow.gw.msp.eu-north1.nebius.cloud
export MLFLOW_TRACKING_USERNAME=mlflow-nebius
export MLFLOW_TRACKING_PASSWORD=<password set at cluster creation>

export MLFLOW_EXPERIMENT_NAME=<default experiment name>
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true
```




## Terraform Files

 - The supplied terraform files are in a rough state
 - They are intended to create the necessary infrastructure for the training (Kubernetes, Grafana/Prometheus, MLFlow, SOperator)
 - They are currently specific to the Nebius platform and rely on its CLI and are based on their Solutions Library
 - Both these TF files and the ones in the Solutions Library need refactoring - expect hand-modifications for the moment.
 
---

**License & credits**  
Built on top of Hugging Face `transformers`, `accelerate`, `trl`, and `peft`. Model and dataset remain subject to their original licenses and usage terms.
