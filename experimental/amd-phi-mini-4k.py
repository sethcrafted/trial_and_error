from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-fp16-onnx-hybrid"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")


