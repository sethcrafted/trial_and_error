#!/usr/bin/env python

from huggingface_hub import HfApi, list_models, hf_hub_download

import json

def get_model_architecture(model_id):
    try:
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            cache_dir=None  # Don't cache
        )
        with open(config_path) as f:
            config = json.load(f)
        return config.get("architectures", [])
    except:
        return None
    
def get_model_config(api, model):
    # Get config without downloading weights
    try:
        config = api.model_info(model.modelId).config

        if config:
            print(f"{model.modelId}: {config.get('architectures', 'Unknown')}")
    except Exception as e:
        print(f"Failed to fetch model config for {model.modelId}")
    return config

def main():
    api = HfApi()

    print('Getting model configs ...')

    # Get models with specific filters
    models = list_models(
        # filter="pytorch",  # or "tensorflow", "jax", etc.
        sort="downloads",
        direction=-1,
        limit=100000
    )
    
    print("Finding out how many models ...")

    hf_models = []
    for model in models:
        hf_models.append(model)

        if len(hf_models) % 10 == 0:
            print(f"Found {len(hf_models)} models...")
        continue
    
    print(f"Found {len(hf_models)}")



if __name__ == "__main__":
    main()