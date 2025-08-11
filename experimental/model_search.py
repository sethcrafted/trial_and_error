#!/usr/bin/env python

from huggingface_hub import list_models, HfApi
import sys
import csv

models = list_models(
    task="text-generation",
    sort="downloads", 
    direction=-1
)

def simple_model_search():
    from huggingface_hub import list_models

    models = list_models(
        task="text-generation",
        sort="downloads", 
        direction=-1
    )

    count = 0
    for model in models:
        count += 1

        try:
            info = HfApi().model_info(model.modelId)
            # Check config for parameter info
            if hasattr(info, 'safetensors') and info.safetensors:
                total_params = info.safetensors.get('total', 0)
                if total_params <= 20_000_000_000:  # 20B threshold
                    print(f"{model.modelId}: {total_params/1e9:.1f}B params")
        except:
            continue
        if count > 100:
            break
    print("Searched through 100 models.")

def print_model_size(model, info):
    total_size = 0
    model_files = []
    for sibling in info.siblings:
        if sibling.rfilename.endswith(('.bin', '.safetensors', '.gguf')):
            if sibling.size is not None:
                total_size += sibling.size
                model_files.append(f"{sibling.rfilename} ({sibling.size / (1024**3):.2f} GB)")
    
    total_params = info.safetensors.get('total', 0) if hasattr(info, 'safetensors') and info.safetensors else 0

    print(f"{model.modelId}: {total_params/1e9:.1f}B params, {total_size / (1024**3):.2f} GB")

def is_base_model(model_id, model_info):
    """Check if this is a base model (not a fine-tune/derivative)"""
    model_id_lower = model_id.lower()
    
    # Common fine-tuning indicators
    finetune_indicators = [
        'instruct', 'chat', 'it', '-it-', 'sft', 'dpo', 'rlhf', 'orca', 'vicuna', 'alpaca',
        'wizard', 'hermes', 'dolphin', 'airoboros', 'goliath', 'bagel', 'openchat',
        'zephyr', 'starling', 'nous-', 'openorca', 'platypus', 'guanaco', 'samantha',
        'mythomax', 'synthia', 'airoboros', 'manticore', 'superhot', 'uncensored',
        'roleplay', 'storytelling', 'creative', 'fiction', 'novel'
    ]
    
    # Check model name for fine-tune indicators
    for indicator in finetune_indicators:
        if indicator in model_id_lower:
            return False
    
    # Check if model has base_model field (indicating it's derived from another model)
    if hasattr(model_info, 'base_model') and model_info.base_model:
        return False
    
    # Additional heuristics: if author != original model family author, likely fine-tune
    author = model_id.split('/')[0].lower()
    model_name = model_id.split('/')[-1].lower()
    
    # Known base model publishers
    base_publishers = {
        'meta-llama', 'mistralai', 'google', 'microsoft', 'qwen', 'tiiuae',
        'bigscience', 'facebook', 'openai', 'stabilityai', 'huggingface',
        'eleutherai', 'mosaicml'
    }
    
    # If it's from a known base publisher, more likely to be base
    if author in base_publishers:
        return True
    
    # If model name contains version numbers without other indicators, likely base
    import re
    if re.search(r'^[a-z]+-?[0-9]+(\.[0-9]+)?[a-z]?(-[0-9]+[a-z]?)?$', model_name):
        return True
    
    return False

def write_to_csv(csv_data, filename='base_models.csv'):
    """Write current data to CSV file"""
    if csv_data:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['modelId', 'params_B', 'size_GB']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def model_search():
    models_of_interest = []
    total_combined_size = 0
    base_models_found = 0
    derivatives_skipped = 0
    csv_data = []
    csv_filename = 'base_models.csv'

    search = 0
    for model in models:
        search += 1

        if search < 13700:
            if search % 100:
                print(f"Skipped through {search} models...")
            continue

        try:
            info = HfApi().model_info(model.modelId, files_metadata=True)

            # Search Criteria by parameters 
            total_params = 0
            if hasattr(info, 'safetensors') and info.safetensors:
                total_params = info.safetensors.get('total', 0)
                if 1_000_000_000 <= total_params <= 25_000_000_000:  # 1B to 25B threshold
                    
                    # Check if it's a base model
                    if is_base_model(model.modelId, info):
                        model_size = 0
                        for sibling in info.siblings:
                            if sibling.rfilename.endswith(('.bin', '.safetensors', '.gguf')):
                                if sibling.size is not None:
                                    model_size += sibling.size
                        
                        total_combined_size += model_size
                        size_gb = model_size / (1024**3)
                        param_count_b = total_params / 1e9
                        
                        # Add to CSV data
                        csv_data.append({
                            'modelId': model.modelId,
                            'params_B': f"{param_count_b:.1f}",
                            'size_GB': f"{size_gb:.2f}"
                        })
                        
                        print(f"[BASE] ", end="")
                        print_model_size(model, info)
                        models_of_interest.append(model)
                        base_models_found += 1
                    else:
                        derivatives_skipped += 1
        
        except KeyboardInterrupt:
            print("Exiting!")
            sys.exit(0)
        except Exception as e:
            continue
        
        # Report progress and save CSV every 100 models
        if search % 100 == 0:
            print(f"Searched {search} models - Found {base_models_found} base models, skipped {derivatives_skipped} derivatives")
            write_to_csv(csv_data, csv_filename)
            print(f"Progress saved to {csv_filename}")
    
    # Final save to CSV
    write_to_csv(csv_data, csv_filename)
    print(f"\nFinal save: {len(csv_data)} models to {csv_filename}")
    
    print(f"Found {len(models_of_interest)} base models (skipped {derivatives_skipped} derivatives)")
    print(f"Total combined size: {total_combined_size / (1024**3):.2f} GB")

def main():
    print("Try to find models...")
    model_search()


if __name__ == "__main__":
    main()

