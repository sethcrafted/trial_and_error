#!/usr/bin/env python

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Available Qwen models - reusing from the models directory
QWEN_MODELS = [
    ("../models/local/Qwen/Qwen1.5-0.5B/", "Qwen1.5-0.5B"),
    ("../models/local/Qwen/Qwen2-0.5B/", "Qwen2-0.5B"), 
    ("../models/local/Qwen/Qwen2.5-0.5B/", "Qwen2.5-0.5B"),
    ("../models/local/Qwen/Qwen3-0.6B/", "Qwen3-0.6B")
]

class ModelArchitectureExplorer:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self, device_map="cpu"):
        """Load the model and tokenizer"""
        print(f"Loading {self.model_name} from {self.model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map=device_map
            )
            print(f"✅ Successfully loaded {self.model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            return False
    
    def explore_architecture(self) -> Dict:
        """Explore the model architecture using named_modules()"""
        if self.model is None:
            print("Model not loaded. Call load_model() first.")
            return {}
        
        architecture_info = {
            "model_name": self.model_name,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "modules": {},
            "layer_counts": {},
            "module_hierarchy": []
        }
        
        print(f"\n🔍 Exploring architecture of {self.model_name}")
        print("=" * 60)
        
        # Analyze named modules
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            
            # Count parameters for this specific module
            module_params = sum(p.numel() for p in module.parameters())
            direct_params = sum(p.numel() for p in module.parameters(recurse=False))
            
            # Store module information
            architecture_info["modules"][name] = {
                "type": module_type,
                "total_parameters": module_params,
                "direct_parameters": direct_params,
                "has_children": len(list(module.children())) > 0
            }
            
            # Count module types
            if module_type not in architecture_info["layer_counts"]:
                architecture_info["layer_counts"][module_type] = 0
            architecture_info["layer_counts"][module_type] += 1
            
            # Build hierarchy
            architecture_info["module_hierarchy"].append({
                "name": name,
                "type": module_type,
                "depth": name.count('.'),
                "parameters": direct_params
            })
        
        return architecture_info
    
    def print_architecture_summary(self, architecture_info: Dict):
        """Print a formatted summary of the model architecture"""
        print(f"\n📊 ARCHITECTURE SUMMARY: {architecture_info['model_name']}")
        print("=" * 70)
        
        # Parameter counts
        total_params = architecture_info['total_parameters']
        trainable_params = architecture_info['trainable_parameters']
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable:        {total_params - trainable_params:,}")
        print(f"Model Size (approx):  {total_params * 4 / 1024**2:.1f} MB (float32)")
        
        # Module type counts
        print(f"\n🏗️  MODULE TYPE COUNTS:")
        print("-" * 40)
        for module_type, count in sorted(architecture_info['layer_counts'].items()):
            print(f"{module_type:<25} {count:>5}")
        
        print(f"\n🌳 MODULE HIERARCHY (Top-level components):")
        print("-" * 70)
        print(f"{'Module Name':<40} {'Type':<20} {'Parameters':<10}")
        print("-" * 70)
        
        # Show only top-level and important modules
        for module in architecture_info['module_hierarchy']:
            if module['depth'] <= 2 and module['parameters'] > 0:
                name = module['name'] if module['name'] else '(root)'
                print(f"{name:<40} {module['type']:<20} {module['parameters']:>9,}")
    
    def print_detailed_architecture(self, architecture_info: Dict, max_depth=3):
        """Print detailed module hierarchy"""
        print(f"\n🔍 DETAILED MODULE HIERARCHY (max depth: {max_depth}):")
        print("=" * 80)
        print(f"{'Module Name':<50} {'Type':<20} {'Params':<10}")
        print("-" * 80)
        
        for module in architecture_info['module_hierarchy']:
            if module['depth'] <= max_depth:
                indent = "  " * module['depth']
                name = module['name'] if module['name'] else '(root)'
                display_name = f"{indent}{name}"[:50]
                
                print(f"{display_name:<50} {module['type']:<20} {module['parameters']:>9,}")
    
    def save_architecture_info(self, architecture_info: Dict, output_file: str):
        """Save architecture information to JSON file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(architecture_info, f, indent=2)
            print(f"\n💾 Architecture information saved to {output_file}")
        except Exception as e:
            print(f"❌ Failed to save architecture info: {e}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore and visualize Qwen model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_architecture_explorer.py                    # Explore all available models
  python model_architecture_explorer.py --model Qwen2-0.5B # Explore specific model
  python model_architecture_explorer.py --detailed         # Show detailed hierarchy
  python model_architecture_explorer.py --save arch.json   # Save to JSON file
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        help="Specific model to explore (e.g., Qwen2-0.5B)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed module hierarchy"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for detailed view (default: 3)"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        help="Save architecture information to JSON file"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to load model on (default: cpu)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine which models to explore
    models_to_explore = QWEN_MODELS
    if args.model:
        models_to_explore = [(path, name) for path, name in QWEN_MODELS if name == args.model]
        if not models_to_explore:
            print(f"Model '{args.model}' not found.")
            print(f"Available models: {[name for _, name in QWEN_MODELS]}")
            return
    
    device_map = "cpu" if args.device == "cpu" else "auto"
    
    for model_path, model_name in models_to_explore:
        if not os.path.exists(model_path):
            print(f"⚠️  Model {model_name} not found at {model_path}")
            continue
            
        # Create explorer and load model
        explorer = ModelArchitectureExplorer(model_path, model_name)
        
        if not explorer.load_model(device_map):
            continue
        
        # Explore architecture
        arch_info = explorer.explore_architecture()
        
        # Print summary
        explorer.print_architecture_summary(arch_info)
        
        # Print detailed view if requested
        if args.detailed:
            explorer.print_detailed_architecture(arch_info, args.max_depth)
        
        # Save to file if requested
        if args.save:
            output_file = args.save
            if len(models_to_explore) > 1:
                # Add model name to filename for multiple models
                name, ext = os.path.splitext(args.save)
                output_file = f"{name}_{model_name}{ext}"
            explorer.save_architecture_info(arch_info, output_file)
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()