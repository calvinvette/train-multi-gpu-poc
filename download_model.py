#!/usr/bin/env python

# login in to HF first
# $ huggingface-cli login

from huggingface_hub import snapshot_download

# download model to local directory
snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", local_dir="llama-3-8b")
snapshot_download(repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct", local_dir="qwen-coder-30b")
