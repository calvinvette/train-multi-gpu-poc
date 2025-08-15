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

# --- NCCL / networking (tune to your fabric) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Choose a stable port and set master to first host in the allocation.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

# HuggingFace/Accelerate multi-node args
export NUM_MACHINES=$SLURM_NNODES
export MACHINE_RANK=$SLURM_NODEID

# Optional: MLflow tracking URI
export MLFLOW_TRACKING_URI="http://public-tracking-e00-mm11asg2freawwc-zecmz1tr6rjc6j9-mlflow.gw.msp.eu-north1.nebius.cloud" # Port 5000?
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

source .venv/bin/activate

# If needed: create logs dir
mkdir -p logs

# Launch: one identical 'srun' command on all nodes; Accelerate uses MACHINE_RANK to coordinate.
srun --cpu-bind=none bash -lc '
accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_machines ${NUM_MACHINES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  train_sft.py \
    --mode fsdp \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dataset Salesforce/xlam-function-calling-60k \
    --dataset-split train \
    --output-dir ./results_fsdp_qwen30b \
    --max-steps 1000 \
    --per-device-train-batch-size 2 \
    --grad-accum-steps 8 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.1 \
    --logging-steps 10 \
    --save-steps 200 \
    --bf16 \
    --no-qlora \
    --gpu-peak-tflops 900 \
    --prometheus-port 8000
'

