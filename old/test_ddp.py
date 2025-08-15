# Based on huggingface/trl
# Naive parallel Pipeline Parallel (aka Model Parallel)
#   change device_map to 'auto'
# Distributed Data Parallel 
#   change device_map to 'DDP'
#   accelerate launch train.py
# Fully Sharded Data Parallel
#   change device_map to 'FSDP'
#   run accelerate config first then
#       choose "this machine"
#       "multi-gpu"
#           2 machines if using 2@ 1GPU-H100 nodes
#           1 machine if using 1@ 8GPU-xH100 nodes
#       Choose FSDP with Full Sharding, TRANSFORMER_BASED_WRAP
#       No splitting of modules (k/v layers)
#       SHARDED_STATE_DICT
#       "no" to use_orig_params
#       # GPUs? 1 for the 2@ 1GPU-H100 or 8 for the 1x8GPU-H100
#           (think #GPUs per node)
#       Choose BF16 for Ampere (A100, Orin) or above
#   accelerate launch train.py
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from accelerate import PartialState

import os

# Enable Metrics for MFU, GPU, mem, etc.
# Using the environment variable
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
# Alternatively use the method:
# mlflow.enable_system_metrics_logging()

# mlflow.set_tracking_uri("http://127.0.0.1:5000")    # TODO - Parameterize this / ENV variable
mlflow.set_tracking_uri("http://public-tracking-e00-mm11asg2freawwc-zecmz1tr6rjc6j9-mlflow.gw.msp.eu-north1.nebius.cloud")


with mlflow.start_run() as run:
    time.sleep(15)

print(mlflow.MlflowClient().get_run(run.info.run_id).data)


# device_map='auto' # for DP and running with test_sft.py
device_map = "DDP"
trust_remote_code = True
attn_implementation="flash_attention_2" # For A100 or Orin (Ampere)
attn_implementation="flash_attention_3" # For HB100+ (Blackwell optimizations in v3)

if device_map == "DDP":                                         # for DDP and running with accelerate
    device_string = PartialState().process_index # Split by GPU
    device_map={'':device_string}
    gradient_checkpointing_kwargs = { "use_reentrant": False},  # DDP is False to save VRAM
elif device_map == 'auto':
    gradient_checkpointing_kwargs = { "use_reentrant": True},   # For Naive Pipeline/Model Parallel
elif device_map == 'FSDP':
    gradient_checkpointing_kwargs = { "use_reentrant": True},   # For Fully Sharded Data Parallel with accelerate
    pass




# Load the dataset
dataset_name = "Salesforce/xlam-function-calling-60k"
dataset = load_dataset(dataset_name, split="train")

# Load model and tokenizer
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

if device_map == "FSDP":                        # accelerate will sort the device map
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=trust_remote_code,
            cache_dir='',
            use_cache = False,
            attn_implementation=attn_implementation,
        )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        cache_dir='',
        use_cache = False,
        attn_implementation=attn_implementation,
        device_map
    )

# PEFT Config
lora_alpha = 8
lora_dropout = 0.1
lora_r = 32
peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
        modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
    )


# Args
max_seq_length = 512
output_dir = "./results"
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
optim = "adamw_torch"
save_steps = 10
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1 # ??? Approx size of guanaco dataset at batch_size 8, ga 2, 2 GPUs
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        bf16=True, # NVIDIA
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
        report_to="wandb"
    )

# PEFT + FSDP Need this to only wrap the PEFT params
if device_map == "FSDP":
    trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# Trainer
trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

# Train
trainer.train()

local_save_path_model = "new-model-local"
local_save_path_adapters = "new-model-adapters-local"

# If we're FSDP, we have to re-assemle the state dictionary from the various nodes
if hasattr(trainer, 'is_fsdp_enabled') and trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.accelerator.print(f"\nModel and tokenizer will be saved {local_save_path_model}\n")
else:
    print(f"\nModel and tokenizer will be saved to {local_save_path_model}\n")

# Save just the lora layers
# if config.get('use_lora', False):
trainer.save_model(local_save_path_adapters)