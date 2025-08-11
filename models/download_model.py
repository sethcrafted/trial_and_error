#!/usr/bin/env python

from huggingface_hub import snapshot_download


target_model = 'Qwen/Qwen2.5-0.5B'

model_path = snapshot_download(
   repo_id=target_model,
   local_dir=f"./local/{target_model}"
)