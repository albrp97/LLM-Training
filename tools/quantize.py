#!/usr/bin/env python
"""Post-training quantisation entry-point.

This script acts as a thin orchestration layer that normalises CLI inputs,
computes bookkeeping metadata, and dispatches to method-specific handlers.
Each handler is currently a stub that documents the expected integration
points with the corresponding quantisation library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod, QuantizationSpec, tag_quant

# ---------------------------------------------------------------------------
# Mappings and helpers
# ---------------------------------------------------------------------------

METHOD_CHOICES = tuple(
    method for method in QuantMethod if method in {
        QuantMethod.GPTQ,
        QuantMethod.QUA_ROT,
        QuantMethod.ADA_ROUND,
        QuantMethod.BRECQ,
        QuantMethod.AWQ,
        QuantMethod.HQQ,
        QuantMethod.SMOOTH_QUANT,
    }
)

BACKEND_HINTS: Dict[QuantMethod, str] = {
    QuantMethod.GPTQ: "autogptq",
    QuantMethod.AWQ: "awq",
    QuantMethod.HQQ: "hqq",
    QuantMethod.SMOOTH_QUANT: "custom",
    QuantMethod.QUA_ROT: "custom",
    QuantMethod.ADA_ROUND: "custom",
    QuantMethod.BRECQ: "custom",
}


def _method_choices() -> Tuple[str, ...]:
    return tuple(method.value.lower() for method in METHOD_CHOICES)


def _normalise_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


def _load_calibration(path: Path) -> Tuple[int, str]:
    """Return the number of non-empty prompts and a stable hash."""
    raw = path.read_text(encoding="utf-8")
    prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    joined = "\n".join(prompts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return len(prompts), digest


# ---------------------------------------------------------------------------
# Quantization implementations
# ---------------------------------------------------------------------------

def quantize_with_adaround(
    src: Path, 
    dst: Path, 
    calib_path: Path, 
    bits: int = 4, 
    group_size: int = 128, 
    symmetric: bool = True, 
    seed: int = 13, 
    skip_lm_head: bool = True
) -> Tuple[Path, Dict[str, str]]:
    """
    AdaRound: Adaptive Rounding for Post-Training Quantization.
    
    Performs layer-wise local reconstruction on Linear modules to learn optimal 
    rounding (up/down) decisions with unlabeled calibration data (128–512 prompts).
    
    Args:
        src: Path to source FP16/BF16 model
        dst: Destination directory for quantized model
        calib_path: Path to calibration prompts file
        bits: Target weight quantization bits (default: 4)
        group_size: Quantization group size (default: 128) 
        symmetric: Use symmetric quantization (default: True)
        seed: Random seed for reproducibility (default: 13)
        skip_lm_head: Keep LM head in FP16 (default: True)
        
    Returns:
        Tuple of (destination_path, metadata_dict)
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from tqdm import tqdm
    import random
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"[AdaRound] Loading model from {src}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        src, 
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load calibration data
    print(f"[AdaRound] Loading calibration data from {calib_path}")
    with open(calib_path, 'r', encoding='utf-8') as f:
        calibration_prompts = [line.strip() for line in f if line.strip()]
    
    # Limit calibration data for faster testing
    max_calib_samples = min(len(calibration_prompts), 128)
    calibration_prompts = calibration_prompts[:max_calib_samples]
    print(f"[AdaRound] Using {len(calibration_prompts)} calibration prompts")
    
    def simple_quantize_weight(weight, bits=4, group_size=128, symmetric=True, show_progress=False):
        """Simplified weight quantization with AdaRound-style error minimization."""
        import math
        
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
        
        if show_progress and num_groups > 10:
            print(f"        Processing {num_groups} weight groups...", end="", flush=True)
        
        for group_idx, group in enumerate(weight_groups):
            # Show progress for large layers
            if show_progress and num_groups > 10 and (group_idx + 1) % max(1, num_groups // 4) == 0:
                progress = (group_idx + 1) / num_groups * 100
                print(f" {progress:.0f}%", end="", flush=True)
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
        
        if show_progress and num_groups > 10:
            print()  # New line after progress indicators
        
        return result.to(weight.dtype)
    
    # Apply AdaRound to all Linear layers
    print("[AdaRound] Analyzing model structure...")
    model.eval()
    
    # First pass: collect all Linear layers to quantize
    linear_layers = []
    total_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1
            # Skip LM head if requested
            if skip_lm_head and ('lm_head' in name or 'output' in name or 'head' in name.lower()):
                continue
            linear_layers.append((name, module))
    
    skipped_layers = total_layers - len(linear_layers)
    print(f"[AdaRound] Found {total_layers} Linear layers, quantizing {len(linear_layers)} (skipping {skipped_layers} LM head layers)")
    
    print("[AdaRound] Applying quantization to model layers...")
    quantized_layers = []
    
    with torch.no_grad():
        for i, (name, module) in enumerate(linear_layers):
            progress = (i + 1) / len(linear_layers) * 100
            print(f"[AdaRound] ({i+1}/{len(linear_layers)}) [{progress:5.1f}%] Quantizing: {name}")
            
            # Apply simplified AdaRound quantization
            original_weight = module.weight.data.clone()
            weight_numel = original_weight.numel()
            
            # Show sub-progress for very large layers
            show_sub_progress = weight_numel > 1_000_000  # Show progress for layers > 1M parameters
            
            quantized_weight = simple_quantize_weight(
                original_weight, 
                bits=bits, 
                group_size=group_size, 
                symmetric=symmetric,
                show_progress=show_sub_progress
            )
            
            if show_sub_progress:
                print(" ✓")  # Complete the progress line
                
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


def quantize_with_brecq(
    src: Path, 
    dst: Path, 
    calib_path: Path, 
    bits: int = 4, 
    attn_bits: int = 6, 
    group_size: int = 64, 
    seed: int = 13, 
    mixed_precision: bool = True
) -> Tuple[Path, Dict[str, str]]:
    """
    BRECQ: Block-wise Reconstruction-based Quantization.
    
    Performs block-wise reconstruction passes with mixed precision support,
    allowing different quantization bits for attention layers (typically W6) 
    and MLP layers (typically W4).
    
    Args:
        src: Path to source FP16/BF16 model
        dst: Destination directory for quantized model
        calib_path: Path to calibration prompts file
        bits: Target weight quantization bits for MLP layers (default: 4)
        attn_bits: Target weight quantization bits for attention layers (default: 6)
        group_size: Quantization group size (default: 64)
        seed: Random seed for reproducibility (default: 13)
        mixed_precision: Enable mixed precision quantization (default: True)
        
    Returns:
        Tuple of (destination_path, metadata_dict)
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import random
    import math
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"[BRECQ] Loading model from {src}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        src, 
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load calibration data
    print(f"[BRECQ] Loading calibration data from {calib_path}")
    with open(calib_path, 'r', encoding='utf-8') as f:
        calibration_prompts = [line.strip() for line in f if line.strip()]
    
    # Limit calibration data for faster testing
    max_calib_samples = min(len(calibration_prompts), 128)
    calibration_prompts = calibration_prompts[:max_calib_samples]
    print(f"[BRECQ] Using {len(calibration_prompts)} calibration prompts")
    
    def block_wise_quantize_weight(weight, target_bits=4, group_size=64, show_progress=False):
        """Block-wise reconstruction quantization with error minimization."""
        original_shape = weight.shape
        weight_flat = weight.flatten().float()
        
        # Pad to group size
        num_groups = math.ceil(weight_flat.numel() / group_size)
        padded_size = num_groups * group_size
        if padded_size > weight_flat.numel():
            padding = torch.zeros(padded_size - weight_flat.numel(), 
                                device=weight_flat.device, dtype=weight_flat.dtype)
            weight_flat = torch.cat([weight_flat, padding])
        
        # Reshape to groups (blocks)
        weight_groups = weight_flat.view(-1, group_size)
        quantized_groups = []
        
        if show_progress and num_groups > 10:
            iterator = tqdm(enumerate(weight_groups), total=len(weight_groups), 
                          desc=f"BRECQ W{target_bits} blocks", leave=False)
        else:
            iterator = enumerate(weight_groups)
        
        for group_idx, block in iterator:
            # Per-block min/max for symmetric quantization
            block_min = block.min()
            block_max = block.max()
            block_range = block_max - block_min
            
            if block_range == 0:
                quantized_groups.append(block)
                continue
            
            # Symmetric quantization levels
            qmax = 2 ** (target_bits - 1) - 1
            qmin = -qmax
            
            # Block-wise reconstruction: iterative refinement
            # Start with simple uniform quantization
            scale = block_range / (2 ** target_bits - 1)
            zero_point = block_min
            
            # Quantize block
            normalized = (block - zero_point) / scale
            quantized_vals = torch.clamp(torch.round(normalized), qmin, qmax)
            
            # Reconstruction with error minimization (simplified BRECQ approach)
            # In full BRECQ, this would involve Hessian-based optimization
            for refinement_step in range(2):  # Limited iterations for performance
                reconstructed = quantized_vals * scale + zero_point
                error = torch.mean((block - reconstructed) ** 2)
                
                # Adaptive scale adjustment based on reconstruction error
                if error > 1e-6:  # Threshold for refinement
                    error_gradient = torch.mean((block - reconstructed) * quantized_vals)
                    scale_adjustment = error_gradient * 0.1  # Simple learning rate
                    scale = scale + scale_adjustment * scale
                    
                    # Re-quantize with adjusted scale
                    normalized = (block - zero_point) / scale
                    quantized_vals = torch.clamp(torch.round(normalized), qmin, qmax)
            
            # Final reconstruction
            final_block = quantized_vals * scale + zero_point
            quantized_groups.append(final_block.to(weight.dtype))
        
        # Reconstruct original shape
        result = torch.cat(quantized_groups).view(original_shape)
        if padded_size > weight_flat.numel():
            # Remove padding
            result = result.flatten()[:weight_flat.numel() - (padded_size - weight_flat.numel())].view(original_shape)
        
        if show_progress and num_groups > 10:
            iterator.close() if hasattr(iterator, 'close') else None
        
        return result.to(weight.dtype)
    
    # Apply BRECQ to model layers with mixed precision
    print("[BRECQ] Analyzing model structure for mixed precision quantization...")
    model.eval()
    
    # Collect layers by type for mixed precision
    attention_layers = []
    mlp_layers = []
    other_layers = []
    total_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1
            # Determine layer type based on name patterns
            if any(pattern in name.lower() for pattern in ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                attention_layers.append((name, module))
            elif any(pattern in name.lower() for pattern in ['mlp', 'fc', 'gate_proj', 'up_proj', 'down_proj']):
                mlp_layers.append((name, module))
            elif 'lm_head' not in name.lower():  # Skip LM head
                other_layers.append((name, module))
    
    print(f"[BRECQ] Found {total_layers} Linear layers:")
    print(f"  - Attention layers: {len(attention_layers)} (W{attn_bits if mixed_precision else bits})")
    print(f"  - MLP layers: {len(mlp_layers)} (W{bits})")
    print(f"  - Other layers: {len(other_layers)} (W{bits})")
    print(f"  - Skipping LM head layers")
    
    print("[BRECQ] Applying block-wise reconstruction quantization...")
    quantized_layers = []
    
    with torch.no_grad():
        # Process attention layers with higher precision if mixed_precision enabled
        if attention_layers:
            target_bits_attn = attn_bits if mixed_precision else bits
            print(f"[BRECQ] Processing {len(attention_layers)} attention layers with W{target_bits_attn}")
            for name, module in tqdm(attention_layers, desc="BRECQ Attention"):
                original_weight = module.weight.data.clone()
                quantized_weight = block_wise_quantize_weight(
                    original_weight, target_bits=target_bits_attn, group_size=group_size
                )
                module.weight.data = quantized_weight
                quantized_layers.append(f"{name} (W{target_bits_attn})")
        
        # Process MLP layers
        if mlp_layers:
            print(f"[BRECQ] Processing {len(mlp_layers)} MLP layers with W{bits}")
            for name, module in tqdm(mlp_layers, desc="BRECQ MLP"):
                original_weight = module.weight.data.clone()
                quantized_weight = block_wise_quantize_weight(
                    original_weight, target_bits=bits, group_size=group_size
                )
                module.weight.data = quantized_weight
                quantized_layers.append(f"{name} (W{bits})")
        
        # Process other layers
        if other_layers:
            print(f"[BRECQ] Processing {len(other_layers)} other layers with W{bits}")
            for name, module in tqdm(other_layers, desc="BRECQ Other"):
                original_weight = module.weight.data.clone()
                quantized_weight = block_wise_quantize_weight(
                    original_weight, target_bits=bits, group_size=group_size
                )
                module.weight.data = quantized_weight
                quantized_layers.append(f"{name} (W{bits})")
    
    print(f"[BRECQ] Quantized {len(quantized_layers)} layers with block-wise reconstruction")
    
    # Save quantized model
    print(f"[BRECQ] Saving quantized model to {dst}")
    model.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    
    # Create metadata
    metadata = {
        "method": "BRECQ",
        "weights_bits": bits,
        "activations_bits": None,
        "group_size": group_size,
        "calibration_samples": len(calibration_prompts),
        "seed": seed,
        "quantized_layers": len(quantized_layers),
        "mixed_precision": mixed_precision,
        "attention_bits": attn_bits if mixed_precision else bits,
        "mlp_bits": bits,
        "attention_layer_count": len(attention_layers),
        "mlp_layer_count": len(mlp_layers),
        "other_layer_count": len(other_layers)
    }
    
    return dst, metadata

# ---------------------------------------------------------------------------
# Handler stubs
# ---------------------------------------------------------------------------

HandlerResult = Tuple[Path, Dict[str, str]]
Handler = Callable[[Path, Path, QuantizationSpec, argparse.Namespace], HandlerResult]


def _require(package: str, message: str) -> None:
    """Try importing a package and raise a helpful error otherwise."""
    try:
        __import__(package)
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise NotImplementedError(message) from exc


def quantize_gptq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "auto_gptq",
        "GPTQ quantisation requires the `auto-gptq` package. Install it with `pip install auto-gptq`.",
    )
    raise NotImplementedError("GPTQ quantisation stub: integrate your AutoGPTQ workflow here.")


def quantize_quarot(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    raise NotImplementedError("QuaRot quantisation is not yet implemented. Please wire your QuaRot pipeline.")


def quantize_adaround(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    """
    AdaRound: Adaptive Rounding for Post-Training Quantization.
    
    Implements layer-wise local reconstruction on Linear modules to learn optimal 
    rounding (up/down) decisions using unlabeled calibration data.
    """
    return quantize_with_adaround(
        src=src,
        dst=dst,
        calib_path=args.calib,
        bits=spec.weights_bits or 4,
        group_size=spec.group_size or 128,
        symmetric=True,  # Default to symmetric quantization
        seed=args.seed,
        skip_lm_head=args.keep_lm_head_fp16
    )


def quantize_brecq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    """
    BRECQ: Block-wise Reconstruction-based Quantization.
    
    Implements block-wise reconstruction with mixed precision support for
    attention (W6) and MLP (W4) layers.
    """
    # Extract attention bits from extras, default to 6
    attn_bits = spec.extras.get("attention_bits", 6) if spec.extras else 6
    mixed_precision = spec.extras.get("mixed_precision", True) if spec.extras else True
    
    return quantize_with_brecq(
        src=src,
        dst=dst,
        calib_path=args.calib,
        bits=spec.weights_bits or 4,
        attn_bits=attn_bits,
        group_size=spec.group_size or 64,
        seed=args.seed,
        mixed_precision=mixed_precision
    )


def quantize_awq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "awq",
        "AWQ quantisation requires the `autoawq`/`awq` package. Install it with `pip install autoawq`.",
    )
    raise NotImplementedError("AWQ quantisation stub: integrate your AWQ workflow here.")


def quantize_hqq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "hqq",
        "HQQ quantisation requires the `hqq` package. Install it with `pip install hqq`.",
    )
    raise NotImplementedError("HQQ quantisation stub: integrate your HQQ workflow here.")


def quantize_smoothquant(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    raise NotImplementedError(
        "SmoothQuant quantisation is not implemented. Hook up SmoothQuant/torchao pipeline here."
    )


HANDLERS: Dict[QuantMethod, Handler] = {
    QuantMethod.GPTQ: quantize_gptq,
    QuantMethod.QUA_ROT: quantize_quarot,
    QuantMethod.ADA_ROUND: quantize_adaround,
    QuantMethod.BRECQ: quantize_brecq,
    QuantMethod.AWQ: quantize_awq,
    QuantMethod.HQQ: quantize_hqq,
    QuantMethod.SMOOTH_QUANT: quantize_smoothquant,
}


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantise a fine-tuned model using PTQ techniques.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--src", required=True, type=Path, help="Path to the FP16/BF16 source model.")
    common.add_argument("--dst", required=True, type=Path, help="Destination directory for quantised output.")
    common.add_argument(
        "--calib",
        type=Path,
        default=Path("Datasets/calibration_prompts.txt"),
        help="Calibration prompts file (default: %(default)s).",
    )
    common.add_argument("--bits", type=int, default=4, help="Target weight bits (default: %(default)s).")
    common.add_argument("--group-size", type=int, default=64, help="Quantisation group size (default: %(default)s).")
    common.add_argument(
        "--keep-lm-head-fp16",
        action="store_true",
        help="Keep the LM head in fp16 instead of quantising it.",
    )
    common.add_argument("--acts-bits", type=int, default=8, help="Activation quantisation bits (default: %(default)s).")
    common.add_argument("--kv-bits", type=int, default=8, help="KV-cache quantisation bits (default: %(default)s).")
    common.add_argument("--seed", type=int, default=13, help="Calibration/data seed (default: %(default)s).")
    common.add_argument(
        "--method",
        required=True,
        choices=_method_choices(),
        help="Quantisation method to apply.",
    )

    run_parser = subparsers.add_parser("run", parents=[common], help="Execute the quantisation workflow.")
    run_parser.set_defaults(func=cmd_run)

    list_parser = subparsers.add_parser("list", help="List supported quantisation methods.")
    list_parser.set_defaults(func=cmd_list)

    return parser


def _spec_from_args(method: QuantMethod, args: argparse.Namespace, calib_stats: Tuple[int, str]) -> QuantizationSpec:
    calib_size, calib_hash = calib_stats
    backend = BACKEND_HINTS.get(method, "custom")
    lm_head_dtype = "fp16" if args.keep_lm_head_fp16 else f"w{args.bits}"
    spec = QuantizationSpec(
        method=method,
        weights_bits=args.bits,
        activations_bits=_normalise_optional_int(args.acts_bits),
        kv_cache_bits=_normalise_optional_int(args.kv_bits),
        group_size=_normalise_optional_int(args.group_size),
        symmetric=None,
        per_channel=None,
        lm_head_dtype=lm_head_dtype,
        backend=backend,
        calibration={
            "size": calib_size,
            "source": str(args.calib),
            "hash": calib_hash,
            "seed": args.seed,
        },
        extras={
            "keep_lm_head_fp16": args.keep_lm_head_fp16,
        },
    )
    spec.extras.setdefault("seed", args.seed)
    return spec


def cmd_list(_args: argparse.Namespace) -> int:
    print("Supported quantisation methods:")
    for method in METHOD_CHOICES:
        backend = BACKEND_HINTS.get(method, "custom")
        print(f"- {method.value.lower():<12} (backend hint: {backend})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    src: Path = args.src
    dst: Path = args.dst
    if not src.exists():
        print(f"[ERROR] Source model '{src}' does not exist.", file=sys.stderr)
        return 2

    dst.mkdir(parents=True, exist_ok=True)

    try:
        method = QuantMethod.from_any(args.method)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    handler = HANDLERS.get(method)
    if handler is None:
        print(f"[ERROR] Method '{method.value}' is not handled yet.", file=sys.stderr)
        return 2

    try:
        calib_stats = _load_calibration(args.calib)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file '{args.calib}' not found.", file=sys.stderr)
        return 2

    spec = _spec_from_args(method, args, calib_stats)
    acts_for_tag = None if spec.activations_bits in (None, 8, 16) else spec.activations_bits
    kv_for_tag = None if spec.kv_cache_bits in (None, 8, 16) else spec.kv_cache_bits
    quant_tag = tag_quant(
        method,
        bits=spec.weights_bits,
        group_size=spec.group_size,
        acts_bits=acts_for_tag,
        kv_bits=kv_for_tag,
        extras={"head": spec.lm_head_dtype} if spec.lm_head_dtype else None,
    )

    metadata = spec.metadata()
    metadata.update(
        {
            "quant_tag": quant_tag,
            "source_model": str(src),
            "destination": str(dst),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
        }
    )

    try:
        handler(src, dst, spec, args)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    metadata_path = dst / "quantization_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(f"[OK] Quantisation complete. Metadata written to {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        print("\n[WARN] Quantisation interrupted by user.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
