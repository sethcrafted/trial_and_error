#!/usr/bin/env python

# Load model directly
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-flash-reasoning", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-flash-reasoning", trust_remote_code=True, torch_dtype="auto")

# Test prompt
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        do_sample=True,
        temperature=0.7
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)



