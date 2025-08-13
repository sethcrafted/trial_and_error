#!/usr/bin/env python

from huggingface_hub import snapshot_download

def download_model(target_model):
    model_path = snapshot_download(
        repo_id=target_model,
        local_dir=f"./local/{target_model}"
    )

def main():
    model_list = [
        'unsloth/llama-2-7b-chat-bnb-4bit',
        'unsloth/llama-3-8b-Instruct-bnb-4bit',
        'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
        'unsloth/Llama-3.2-3B-Instruct-bnb-4bit',
        'unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF'
    ]

    for model in model_list:
        download_model(model)    

if __name__ == "__main__":
    main()


