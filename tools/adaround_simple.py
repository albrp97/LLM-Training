#!/usr/bin/env python
"""Simplified AdaRound implementation for testing."""

import torch
import torch.nn as nn
import math
from pathlib import Path
from typing import Dict, Tuple


def simple_quantize_weight(weight, bits=4, group_size=128, symmetric=True):
    """Simplified weight quantization with AdaRound-style error minimization."""
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    
    # Pad to group size
    num_groups = math.ceil(weight_flat.numel() / group_size)
    padded_size = num_groups * group_size
    if padded_size > weight_flat.numel():
        padding = torch.zeros(padded_size - weight_flat.numel(), device=weight.device)
        weight_flat = torch.cat([weight_flat, padding])
    
    # Reshape to groups
    weight_groups = weight_flat.view(-1, group_size)
    quantized_groups = []
    
    for group in weight_groups:
        # Compute quantization parameters
        if symmetric:
            abs_max = group.abs().max()
            scale = abs_max / (2**(bits-1) - 1) if abs_max > 0 else 1.0
            zero_point = 0
            qmin, qmax = -(2**(bits-1)), (2**(bits-1) - 1)
        else:
            min_val, max_val = group.min(), group.max()
            scale = (max_val - min_val) / (2**bits - 1) if max_val > min_val else 1.0
            zero_point = -min_val / scale if scale > 0 else 0
            qmin, qmax = 0, 2**bits - 1
        
        # AdaRound: Choose rounding that minimizes reconstruction error
        if scale > 0:
            normalized = group / scale + zero_point
            floor_vals = torch.floor(normalized)
            ceil_vals = floor_vals + 1
            
            # Compute reconstruction errors
            floor_recon = (floor_vals - zero_point) * scale
            ceil_recon = (ceil_vals - zero_point) * scale
            
            floor_error = (group - floor_recon).abs()
            ceil_error = (group - ceil_recon).abs()
            
            # Choose rounding that minimizes error
            use_ceil = ceil_error < floor_error
            quantized = torch.where(use_ceil, ceil_vals, floor_vals)
            quantized = torch.clamp(quantized, qmin, qmax)
            
            dequantized = (quantized - zero_point) * scale
        else:
            dequantized = group
            
        quantized_groups.append(dequantized)
    
    # Reconstruct original shape
    result = torch.cat(quantized_groups).view(original_shape)
    if padded_size > weight_flat.numel():
        # Remove padding
        result = result.flatten()[:weight.numel()].view(original_shape)
    
    return result.to(weight.dtype)


def quantize_with_adaround_simple(
    src: Path, 
    dst: Path, 
    calib_path: Path, 
    bits: int = 4, 
    group_size: int = 128, 
    symmetric: bool = True, 
    seed: int = 13, 
    skip_lm_head: bool = True
) -> Tuple[Path, Dict[str, str]]:
    """Simplified AdaRound implementation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[AdaRound] Loading model from {src}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        src, 
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(src)
    
    # Load calibration data (for metadata only in this simple version)
    with open(calib_path, 'r', encoding='utf-8') as f:
        calibration_prompts = [line.strip() for line in f if line.strip()]
    
    print(f"[AdaRound] Using {len(calibration_prompts)} calibration prompts")
    
    # Apply quantization to Linear layers
    print("[AdaRound] Applying quantization to model layers...")
    model.eval()
    
    quantized_layers = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip LM head if requested
                if skip_lm_head and ('lm_head' in name or 'output' in name or 'head' in name.lower()):
                    print(f"[AdaRound] Skipping LM head: {name}")
                    continue
                    
                print(f"[AdaRound] Quantizing layer: {name}")
                
                # Apply AdaRound quantization to weights
                original_weight = module.weight.data.clone()
                quantized_weight = simple_quantize_weight(
                    original_weight, 
                    bits=bits, 
                    group_size=group_size, 
                    symmetric=symmetric
                )
                module.weight.data = quantized_weight
                quantized_layers.append(name)
    
    print(f"[AdaRound] Quantized {len(quantized_layers)} layers")
    
    # Save quantized model
    print(f"[AdaRound] Saving quantized model to {dst}")
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    
    # Create metadata
    metadata = {
        "method": "AdaRound",
        "weights_bits": bits,
        "activations_bits": None,
        "group_size": group_size,
        "symmetric": symmetric,
        "calibration_samples": len(calibration_prompts),
        "skip_lm_head": skip_lm_head,
        "seed": seed,
        "quantized_layers": len(quantized_layers)
    }
    
    return dst, metadata


if __name__ == "__main__":
    # Test the quantization function
    test_weight = torch.randn(64, 32) * 0.1
    quantized = simple_quantize_weight(test_weight, bits=4, group_size=16)
    print(f"Original weight range: [{test_weight.min():.4f}, {test_weight.max():.4f}]")
    print(f"Quantized weight range: [{quantized.min():.4f}, {quantized.max():.4f}]")
    print(f"Quantization error (MSE): {((test_weight - quantized) ** 2).mean():.6f}")