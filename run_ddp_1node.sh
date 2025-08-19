#!/bin/bash

export LAUNCH_DATE=$(date +%Y%m%d@%H.%M.%S)

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-${MASTER_PORT:-29500}}

export NUM_MACHINES=1
export NUM_GPUs=$(nvidia-smi -L | wc | awk '{print $1}')
export MACHINE_RANK=1 #???

# Optional: MLflow tracking URI
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://orin1.drake-ulmer.ts.net:5000"} # Port 5000?
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=${MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING:-true} 
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-"SFTtrain-${LAUNCH_DATE}"}

export SFT_DATASET=${SFT_DATASET:-"Salesforce/xlam-function-calling-60k"}
export SFT_MODEL=${SFT_MODEL:-"meta-llama/Meta-Llama-3-8B"}

source .venv/bin/activate

mkdir -p logs

# --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
# --output-dir ./results_ddp_qwen30b \
### ${MACHINE_RANK} \
# --main_process_ip ${MASTER_ADDR} \
# --gpu-peak-tflops 900 \
# --prometheus-port 8000
accelerate launch \
  --config_file configs/accelerate_ddp.yaml \
  --num_machines ${NUM_MACHINES} \
  --machine_rank 1 \ 
   python train_sft.py \
    --mode ddp \
    --model-name ${SFT_MODEL} \
    --dataset ${SFT_DATASET} \
    --dataset-split train \
    --output-dir ./results_ddp_${SFT_MODEL}_${LAUNCH_DATE} \
    --max-steps 1000 \
    --per-device-train-batch-size 2 \
    --grad-accum-steps 8 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.1 \
    --logging-steps 10 \
    --save-steps 200 \
    --bf16 \
    --qlora \

