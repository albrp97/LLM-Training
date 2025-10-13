#!/usr/bin/env python3
"""Test script to verify QLoRA configuration works correctly."""

import sys
from pathlib import Path

# Add the current directory to path for imports
current_dir = Path.cwd()
sys.path.append(str(current_dir))

import torch
from transformers import BitsAndBytesConfig
from quantization_utils import QuantMethod, QuantizationSpec


def test_qlora_quantization_spec():
    """Test that QLoRA quantization spec is created correctly."""
    
    # Mock the configuration from training script
    qlora_r = 64
    qlora_lora_alpha = 16
    qlora_dropout = 0.05
    keep_lm_head_fp16 = False
    
    method = QuantMethod.QLORA
    
    # Create spec similar to training script
    spec = QuantizationSpec(
        method=method,
        weights_bits=4,
        activations_bits=None,  # QLoRA doesn't quantize activations
        kv_cache_bits=None,     # QLoRA doesn't quantize KV cache by default
        group_size=None,        # NF4 doesn't use explicit group size
        symmetric=False,        # NF4 is asymmetric
        per_channel=None,
        lm_head_dtype="bf16",
        backend="bitsandbytes",
        extras={
            "double_quant": True,
            "base_quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "qlora_r": qlora_r,
            "qlora_lora_alpha": qlora_lora_alpha,
            "qlora_dropout": qlora_dropout,
            "keep_lm_head_fp16": keep_lm_head_fp16,
        },
    )
    
    print("QLoRA QuantizationSpec created successfully!")
    print(f"Method: {spec.method}")
    print(f"Weights bits: {spec.weights_bits}")
    print(f"Activations bits: {spec.activations_bits}")
    print(f"KV cache bits: {spec.kv_cache_bits}")
    print(f"Backend: {spec.backend}")
    print(f"Tag: {spec.tag()}")
    print(f"Extras: {spec.extras}")
    
    # Test metadata generation
    metadata = spec.metadata()
    print(f"\nMetadata generated successfully!")
    print(f"Method: {metadata['method']}")
    print(f"Weights bits: {metadata['weights_bits']}")
    print(f"Backend: {metadata['backend']}")
    
    return spec


def test_bitsandbytes_config():
    """Test that BitsAndBytesConfig can be created correctly."""
    
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        print("\nBitsAndBytesConfig created successfully!")
        print(f"load_in_4bit: {bnb_config.load_in_4bit}")
        print(f"quant_type: {bnb_config.bnb_4bit_quant_type}")
        print(f"double_quant: {bnb_config.bnb_4bit_use_double_quant}")
        print(f"compute_dtype: {bnb_config.bnb_4bit_compute_dtype}")
        
        return bnb_config
        
    except Exception as e:
        print(f"\nError creating BitsAndBytesConfig: {e}")
        return None


def test_quant_method_from_string():
    """Test that QuantMethod.from_any works with different string formats."""
    
    test_cases = ["QLORA", "qlora", "QLoRA", "QLora"]
    
    print("\nTesting QuantMethod.from_any with different string formats:")
    for test_case in test_cases:
        try:
            method = QuantMethod.from_any(test_case)
            print(f"  '{test_case}' -> {method}")
        except Exception as e:
            print(f"  '{test_case}' -> ERROR: {e}")


if __name__ == "__main__":
    print("Testing QLoRA Configuration")
    print("=" * 50)
    
    # Test quantization spec
    spec = test_qlora_quantization_spec()
    
    # Test BitsAndBytes config
    bnb_config = test_bitsandbytes_config()
    
    # Test string parsing
    test_quant_method_from_string()
    
    print("\n" + "=" * 50)
    print("All tests completed!")