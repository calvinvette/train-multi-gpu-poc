# Based on huggingface/trl
# Naive parallel Pipeline Parallel (aka Model Parallel)
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from accelerate import PartialState

device_map='auto' # for DP and running with test_sft.py
trust_remote_code=True
attn_implementation="flash_attention_2" # For A100
attn_implementation="flash_attention_3" # For HB100+ (Blackwell optimizations in v3)

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

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        cache_dir='',
        use_cache = False,
        attn_implementation=attn_implementation,
        device_map = device_map,
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
        gradient_checkpointing_kwargs = { "use_reentrant": True}, # For Naive Pipeline/Model Parallel
        report_to="wandb"
    )

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

