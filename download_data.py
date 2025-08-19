#!/usr/bin/env python

# login in to HF first
# $ huggingface-cli login

from datasets import load_dataset
ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v1", split=["code", "math"])
ds = load_dataset("Salesforce/xlam-function-calling-60k", split=["train"])
