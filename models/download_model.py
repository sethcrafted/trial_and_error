#!/usr/bin/env python

from huggingface_hub import snapshot_download

def download_model(target_model):
    model_path = snapshot_download(
        repo_id=target_model,
        local_dir=f"./local/{target_model}"
    )

def main():
    download_model('Qwen/Qwen2-0.5B')    

if __name__ == "__main__":
    main()


