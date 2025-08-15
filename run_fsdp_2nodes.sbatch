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

export LAUNCH_DATE=$(date +%Y%m%d@%H.%M.%S)

# --- NCCL / networking (tune to your fabric) ---
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Choose a stable port and set master to first host in the allocation.
# Moved to inside the srun command
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=${MASTER_PORT:-29500}

# HuggingFace/Accelerate multi-node args
# Moved to inside the srun command
# export NUM_MACHINES=$SLURM_NNODES
# export MACHINE_RANK=$SLURM_NODEID

if [ -f .env ]; then
  . ./.env
fi

# Optional: MLflow tracking URI
# Load MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_EXPERIMENT_NAME from .env or use defaults
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://public-tracking-e00-...-mlflow.gw.msp.eu-north1.nebius.cloud/"}
export MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME:-"mlflow-nebius"}
export MLFLOW_TRACKING_PASSWORD=${MLFLOW_TRACKING_PASSWORD:-"password"}
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-"SFTtrain-${LAUNCH_DATE}"}
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=${MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING:-true}

# If needed: create logs dir
mkdir -p logs

# Launch: one identical 'srun' command on all nodes; Accelerate uses MACHINE_RANK to coordinate.
# --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
# --output-dir ./results_fsdp_qwen30b \
srun --cpu-bind=none bash -lc '
source .venv/bin/activate && \
export NUM_MACHINES=$SLURM_NNODES && \
export MACHINE_RANK=$SLURM_NODEID && \
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) && \
export MASTER_PORT=${MASTER_PORT:-29500} && \
accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_machines ${NUM_MACHINES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  train_sft.py \
    --mode fsdp \
    --model-name meta-llama/Meta-Llama-3-8B \
    --dataset Salesforce/xlam-function-calling-60k \
    --dataset-split train \
    --output-dir ./results_fsdp_${LAUNCH_DATE} \
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

