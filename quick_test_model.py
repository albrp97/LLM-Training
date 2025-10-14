#!/usr/bin/env python
"""Quick test to see what the quantized model is actually generating"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

# Import the test module
import importlib.util
test_module_path = Path(__file__).parent / "Testing" / "02_TestModels.py"
spec = importlib.util.spec_from_file_location("test_models", test_module_path)
test_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_models)

def quick_test_model(model_name: str):
    print(f"Testing model: {model_name}")
    
    # Resolve quantization
    quant_context = test_models.resolve_quant_context(model_name, "auto")
    resolved_kv_dtype = test_models.resolve_kv_dtype("auto", quant_context.spec)
    
    print(f"Quantization: {quant_context.method.value if quant_context.method else 'NoQuant'}")
    
    # Load model
    model, tokenizer = test_models.load_model_with_quant(model_name, quant_context, resolved_kv_dtype)
    
    # Test a simple question
    user_prompt = "What is 2 + 2?"
    system_prompt = "Answer the math question. Output your answer inside \\boxed{}, like this: \\boxed{4}"
    
    print(f"Question: {user_prompt}")
    print("Generating response...")
    
    response = test_models.chat(
        model=model,
        tokenizer=tokenizer, 
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=100  # Shorter for testing
    )
    
    print(f"Raw response: '{response}'")
    
    extracted = test_models.extract_boxed(response)
    print(f"Extracted answer: {extracted}")
    
    return response, extracted

if __name__ == "__main__":
    model_path = "Models/Qwen3-0.6B-base_gptq_w4g64"
    quick_test_model(model_path)