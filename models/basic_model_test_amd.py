#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Available Qwen models
QWEN_MODELS = [
    ("./local/Qwen/Qwen1.5-0.5B/", "Qwen1.5-0.5B"),
    ("./local/Qwen/Qwen2-0.5B/", "Qwen2-0.5B"),
    ("./local/Qwen/Qwen2.5-0.5B/", "Qwen2.5-0.5B"),
    ("./local/Qwen/Qwen3-0.6B/", "Qwen3-0.6B")
]

def test_qwen_amd(model_path, model_name):
    # AMD-compatible loading (similar to your experimental script)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto"  # No device_map for AMD compatibility
    )

    # Simple test prompt
    prompt = "Give me a short introduction to large language models."
    
    # Try chat template, fall back to direct encoding if not supported
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        text = prompt
    
    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f"Response: {response}")

def main():
    print("Testing Qwen models with AMD-compatible settings...")
    print("=" * 60)
    
    for model_path, model_name in QWEN_MODELS:
        if os.path.exists(model_path):
            print(f"\n🔍 Testing {model_name}")
            print("-" * 40)
            try:
                test_qwen_amd(model_path, model_name)
                print(f"✅ {model_name} test completed successfully")
            except Exception as e:
                print(f"❌ {model_name} test failed: {str(e)}")
        else:
            print(f"⚠️  Model {model_name} not found at {model_path}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()