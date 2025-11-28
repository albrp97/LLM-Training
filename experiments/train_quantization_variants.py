"""
Train and save quantized model variants for comparison.
This script handles:
1. Training 8B with QLoRA on openmath
2. Creating quantized versions of existing base/nopeft models
"""

import sys
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from utils.quantization_utils import QuantMethod, QuantizationSpec

def safe_serialize(obj):
    """Converts non-serializable objects to serializable formats."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, BitsAndBytesConfig):
        return {k: safe_serialize(v) for k, v in vars(obj).items() if not k.startswith('_')}
    elif hasattr(obj, '__dict__'):
        return {k: safe_serialize(v) for k, v in vars(obj).items() if not k.startswith('_')}
    else:
        return str(obj)

def drop_nulls(obj):
    """Eliminate keys with None values recursively."""
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [drop_nulls(v) for v in obj if v is not None]
    else:
        return obj

def create_quant_spec(method: QuantMethod, bits: int = 4) -> QuantizationSpec:
    """Create quantization spec for a given method and bit width."""
    if method == QuantMethod.QLORA:
        return QuantizationSpec(
            method=method,
            weights_bits=bits,
            activations_bits=16,
            kv_cache_bits=16,
            group_size=None,
            symmetric=None,
            per_channel=None,
            lm_head_dtype="bf16" if torch.cuda.is_available() else "fp16",
            backend="bitsandbytes",
            extras={
                "double_quant": True,
                "base_quant_type": "nf4" if bits == 4 else "int8",
            },
        )
    else:
        # For int8/4bit without QLoRA
        return QuantizationSpec(
            method=QuantMethod.NO_QUANT,  # Loading with BnB config
            weights_bits=bits,
            activations_bits=16,
            kv_cache_bits=16,
            group_size=None,
            symmetric=None,
            per_channel=None,
            lm_head_dtype="bf16" if torch.cuda.is_available() else "fp16",
            backend="bitsandbytes",
            extras={
                "load_in_8bit": bits == 8,
                "load_in_4bit": bits == 4,
                "bnb_4bit_quant_type": "nf4" if bits == 4 else None,
                "bnb_4bit_compute_dtype": "bfloat16" if bits == 4 and torch.cuda.is_available() else None,
            },
        )

def get_directory_size(path: Path) -> int:
    """Calculate total size of all files in a directory in bytes."""
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total

def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def save_quantized_model(source_model_path: str, output_name: str, bits: int = 4, is_qlora: bool = False):
    """
    Load a model from source_model_path and save it quantized.
    
    Args:
        source_model_path: Path to the source model (can be base or nopeft)
        output_name: Name for the output directory
        bits: 4 or 8 for quantization bits
        is_qlora: Whether this is QLoRA (for training) or just load_in_Xbit
    """
    print(f"\n{'='*60}")
    print(f"Creating quantized version: {output_name}")
    print(f"Source: {source_model_path}")
    print(f"Bits: {bits}, QLoRA: {is_qlora}")
    print(f"{'='*60}\n")
    
    # Get original model size
    source_path = Path(source_model_path)
    original_size = get_directory_size(source_path)
    print(f"📊 Original model size: {format_size(original_size)}")
    
    device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    
    # Create quantization config
    if bits == 8:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        quant_tag = "Int8_BnB"
    elif bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        quant_tag = "4bit_BnB" if not is_qlora else "QLoRA_4bit"
    else:
        raise ValueError(f"Unsupported bits: {bits}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source_model_path, trust_remote_code=True)
    
    # Load model with quantization
    print("Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage_trainable = (trainable_params / total_params * 100) if total_params else 0.0
    
    print(f"Total params: {total_params / 1e9:.4f}B")
    print(f"Trainable params: {trainable_params / 1e9:.7f}B")
    print(f"Percentage trainable: {percentage_trainable:.7f}%")
    
    # Create output directory
    output_dir = Path("Models") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    
    # Create quantization spec
    quant_spec = create_quant_spec(QuantMethod.QLORA if is_qlora else QuantMethod.NO_QUANT, bits)
    
    # Create metadata
    metadata = {
        "model_info": {
            "model_name": output_name,
            "base_model": source_model_path,
            "creation_date": datetime.now().isoformat(),
            "model_type": "CausalLM",
            "quantization_tag": quant_tag,
            "has_quantization": True,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable,
            "notes": f"Quantized to {bits}-bit using BitsAndBytes",
        },
        "training_parameters": {
            "train": False,
            "dataset_choice": None,
            "finetuning_strategy": None,
            "quantization_method": "qlora" if is_qlora else f"{bits}bit",
        },
        "peft_config": None,
        "quantization_config": safe_serialize(quant_config),
        "quantization": quant_spec.metadata(),
        "training_stats": {
            "total_steps": 0,
            "epochs_completed": 0,
        },
        "hardware_info": {
            "device": str(model.device),
            "dtype": str(model.dtype),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable,
            "vram_peaks": {
                "overall_max_reserved_gb": 0.0,
                "overall_max_allocated_gb": 0.0,
                "per_gpu": [],
            },
        },
    }
    
    clean_metadata = drop_nulls(metadata)
    
    # Save metadata
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(clean_metadata, f, indent=2, ensure_ascii=False)
    
    # Calculate size reduction
    quantized_size = get_directory_size(output_dir)
    size_reduction = original_size - quantized_size
    reduction_percentage = (size_reduction / original_size * 100) if original_size > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 SIZE COMPARISON")
    print(f"{'='*60}")
    print(f"Original model:   {format_size(original_size)}")
    print(f"Quantized model:  {format_size(quantized_size)}")
    print(f"Size reduction:   {format_size(size_reduction)} ({reduction_percentage:.2f}%)")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    print(f"{'='*60}\n")
    
    print(f"✅ Saved quantized model to: {output_dir}\n")
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return str(output_dir), {
        'original_size': original_size,
        'quantized_size': quantized_size,
        'reduction_percentage': reduction_percentage,
        'compression_ratio': original_size/quantized_size if quantized_size > 0 else 0
    }

def main():
    """Create all quantized variants."""
    
    print("\n" + "="*80)
    print("QUANTIZATION VARIANTS CREATION")
    print("="*80 + "\n")
    
    variants = []
    size_stats = []
    
    # 4B base int8
    print("\n--- Creating 4B base int8 ---")
    path, stats = save_quantized_model(
        source_model_path="Models/Qwen3-4B-base",
        output_name="Qwen3-4B-base_Int8_BnB",
        bits=8,
        is_qlora=False
    )
    variants.append(path)
    size_stats.append(('4B base → Int8', stats))
    
    # 4B nopeft int8
    print("\n--- Creating 4B nopeft int8 ---")
    path, stats = save_quantized_model(
        source_model_path="Models/Qwen3-4B-openmath_SFT_NoPeft_NoQuant",
        output_name="Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB",
        bits=8,
        is_qlora=False
    )
    variants.append(path)
    size_stats.append(('4B nopeft → Int8', stats))
    
    # 4B base 4bit
    print("\n--- Creating 4B base 4bit ---")
    path, stats = save_quantized_model(
        source_model_path="Models/Qwen3-4B-base",
        output_name="Qwen3-4B-base_4bit_BnB",
        bits=4,
        is_qlora=False
    )
    variants.append(path)
    size_stats.append(('4B base → 4bit', stats))
    
    # 4B nopeft 4bit
    print("\n--- Creating 4B nopeft 4bit ---")
    path, stats = save_quantized_model(
        source_model_path="Models/Qwen3-4B-openmath_SFT_NoPeft_NoQuant",
        output_name="Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB",
        bits=4,
        is_qlora=False
    )
    variants.append(path)
    size_stats.append(('4B nopeft → 4bit', stats))
    
    print("\n" + "="*80)
    print("SUMMARY - Created quantized variants:")
    print("="*80)
    for i, v in enumerate(variants, 1):
        print(f"{i}. {v}")
    
    print("\n" + "="*80)
    print("SIZE REDUCTION SUMMARY")
    print("="*80)
    for name, stats in size_stats:
        print(f"\n{name}:")
        print(f"  Original:  {format_size(stats['original_size'])}")
        print(f"  Quantized: {format_size(stats['quantized_size'])}")
        print(f"  Reduction: {stats['reduction_percentage']:.2f}% ({stats['compression_ratio']:.2f}x smaller)")
    
    print("\n✅ All quantized variants created successfully!")
    print("\nNext steps:")
    print("1. Train 8B with QLoRA: python Fine-tuning/01_Train.py")
    print("   (Set MODEL_NAME='Qwen/Qwen3-8B', DATASET_CHOICE='openmath', QUANT_METHOD='QLORA', PEFT_CONFIG='LoRa')")
    print("2. Evaluate all models: python Testing/03_EvaluationOrchestrator.py")
    print("3. Compare metrics: Run Testing/04_CompareMetrics.ipynb")

if __name__ == "__main__":
    main()
