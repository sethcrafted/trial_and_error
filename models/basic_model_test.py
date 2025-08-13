#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
import psutil
import argparse
from dataclasses import dataclass
from typing import List

@dataclass
class PerformanceStats:
    model_name: str
    model_load_time: float
    generation_time: float
    total_time: float
    memory_usage_mb: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    supports_thinking: bool
    success: bool
    device: str
    error_message: str = ""

# Available Qwen models
QWEN_MODELS = [
    ("./local/Qwen/Qwen1.5-0.5B/", "Qwen1.5-0.5B"),
    ("./local/Qwen/Qwen2-0.5B/", "Qwen2-0.5B"),
    ("./local/Qwen/Qwen2.5-0.5B/", "Qwen2.5-0.5B"),
    ("./local/Qwen/Qwen3-0.6B/", "Qwen3-0.6B")
]

def test_qwen(model_path, model_name, device_mode="cpu") -> PerformanceStats:
    stats = PerformanceStats(
        model_name=model_name,
        model_load_time=0,
        generation_time=0,
        total_time=0,
        memory_usage_mb=0,
        input_tokens=0,
        output_tokens=0,
        tokens_per_second=0,
        supports_thinking=False,
        success=False,
        device=device_mode.upper()
    )
    
    start_total_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Measure model loading time
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure device mapping based on mode
        if device_mode.lower() == "cpu":
            device_map = "cpu"
        elif device_mode.lower() == "gpu":
            device_map = "auto"  # Let transformers decide GPU allocation
        else:
            raise ValueError(f"Invalid device_mode: {device_mode}. Use 'cpu' or 'gpu'")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map
        )
        stats.model_load_time = time.time() - load_start

        # prepare the model input
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Check if model supports thinking mode (mainly Qwen3)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )
            stats.supports_thinking = True
        except TypeError:
            # Fall back to standard chat template for older models
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            stats.supports_thinking = False
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        stats.input_tokens = len(model_inputs.input_ids[0])

        # conduct text completion
        generation_start = time.time()
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256  # Reduced for consistent testing
        )
        stats.generation_time = time.time() - generation_start
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        stats.output_tokens = len(output_ids)
        stats.tokens_per_second = stats.output_tokens / stats.generation_time if stats.generation_time > 0 else 0

        if stats.supports_thinking:
            # parsing thinking content for Qwen3
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            print("Thinking content:", thinking_content)
            print("Response:", content)
        else:
            # Standard output for older models
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            print("Response:", content)
        
        # Calculate memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        stats.memory_usage_mb = final_memory - initial_memory
        
        # Calculate total time
        stats.total_time = time.time() - start_total_time
        stats.success = True
        
    except Exception as e:
        stats.error_message = str(e)
        stats.success = False
        stats.total_time = time.time() - start_total_time
    
    return stats

def print_performance_summary(all_stats: List[PerformanceStats]):
    """Print a summary table of performance statistics"""
    print("\n" + "=" * 105)
    print("PERFORMANCE SUMMARY")
    print("=" * 105)
    
    # Header
    header = f"{'Model':<15} {'Device':<6} {'Load(s)':<8} {'Gen(s)':<8} {'Total(s)':<9} {'Mem(MB)':<8} {'In/Out':<10} {'Tok/s':<8} {'Think':<6} {'Status':<8}"
    print(header)
    print("-" * 105)
    
    # Data rows
    for stats in all_stats:
        if stats.success:
            status = "✅ OK"
            memory_str = f"{stats.memory_usage_mb:.1f}"
            tokens_str = f"{stats.input_tokens}/{stats.output_tokens}"
            tok_per_sec = f"{stats.tokens_per_second:.1f}"
            thinking = "Yes" if stats.supports_thinking else "No"
        else:
            status = "❌ FAIL"
            memory_str = "-"
            tokens_str = "-"
            tok_per_sec = "-"
            thinking = "-"
        
        row = f"{stats.model_name:<15} {stats.device:<6} {stats.model_load_time:<8.2f} {stats.generation_time:<8.2f} {stats.total_time:<9.2f} {memory_str:<8} {tokens_str:<10} {tok_per_sec:<8} {thinking:<6} {status:<8}"
        print(row)
        
        if not stats.success:
            print(f"                Error: {stats.error_message[:60]}...")
    
    print("\n" + "=" * 105)
    
    # Best performers
    successful_stats = [s for s in all_stats if s.success]
    if successful_stats:
        fastest_gen = min(successful_stats, key=lambda x: x.generation_time)
        fastest_load = min(successful_stats, key=lambda x: x.model_load_time)
        highest_throughput = max(successful_stats, key=lambda x: x.tokens_per_second)
        
        print("🏆 BEST PERFORMERS:")
        print(f"   Fastest Generation: {fastest_gen.model_name} ({fastest_gen.generation_time:.2f}s)")
        print(f"   Fastest Loading:    {fastest_load.model_name} ({fastest_load.model_load_time:.2f}s)")
        print(f"   Highest Throughput: {highest_throughput.model_name} ({highest_throughput.tokens_per_second:.1f} tok/s)")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test Qwen models with performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basic_model_test.py                 # Test all models on CPU (default)
  python basic_model_test.py --cpu           # Explicitly test on CPU
  python basic_model_test.py --gpu           # Test on GPU (requires CUDA/ROCm)
        """
    )
    
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--cpu", 
        action="store_const", 
        const="cpu", 
        dest="device",
        help="Run models on CPU (default, AMD compatible)"
    )
    device_group.add_argument(
        "--gpu", 
        action="store_const", 
        const="gpu", 
        dest="device",
        help="Run models on GPU (requires CUDA/ROCm setup)"
    )
    
    # Set CPU as default
    parser.set_defaults(device="cpu")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Testing all available Qwen models on {args.device.upper()}...")
    print("=" * 60)
    
    all_stats = []
    
    for model_path, model_name in QWEN_MODELS:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing {model_name} on {args.device.upper()}")
            print("-" * 40)
            stats = test_qwen(model_path, model_name, args.device)
            all_stats.append(stats)
            
            if stats.success:
                print(f"✅ {model_name} completed in {stats.total_time:.2f}s ({stats.tokens_per_second:.1f} tok/s)")
            else:
                print(f"❌ {model_name} failed: {stats.error_message}")
        else:
            print(f"⚠️  Model {model_name} not found at {model_path}")
            failed_stats = PerformanceStats(
                model_name=model_name,
                model_load_time=0, generation_time=0, total_time=0,
                memory_usage_mb=0, input_tokens=0, output_tokens=0,
                tokens_per_second=0, supports_thinking=False,
                success=False, device=args.device.upper(), 
                error_message="Model not found"
            )
            all_stats.append(failed_stats)
        
        print("\n" + "=" * 60)
    
    # Print performance summary
    print_performance_summary(all_stats)

if __name__ == "__main__":
    main()