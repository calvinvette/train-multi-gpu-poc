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

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}

export NUM_MACHINES=1
export MACHINE_RANK=0

export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

mkdir -p logs

srun --cpu-bind=none bash -lc '
accelerate launch \
  --config_file configs/accelerate_ddp.yaml \
  --num_machines ${NUM_MACHINES} \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  train_sft.py \
    --mode ddp \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dataset Salesforce/xlam-function-calling-60k \
    --dataset-split train \
    --output-dir ./results_ddp_qwen30b \
    --max-steps 1000 \
    --per-device-train-batch-size 2 \
    --grad-accum-steps 8 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.1 \
    --logging-steps 10 \
    --save-steps 200 \
    --bf16 \
    --qlora \
    --gpu-peak-tflops 900 \
    --prometheus-port 8000
'

