"""
Create 1.7B base model snapshot and optionally quantized variants.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

def create_base_model(model_name: str, output_name: str):
    """Create a base model snapshot."""
    print(f"\n{'='*60}")
    print(f"Creating base model: {output_name}")
    print(f"Source: {model_name}")
    print(f"{'='*60}\n")
    
    device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage_trainable = (trainable_params / total_params * 100) if total_params else 0.0
    
    print(f"Total params: {total_params / 1e9:.4f}B")
    print(f"Trainable params: {trainable_params / 1e9:.7f}B")
    
    # Create output directory
    output_dir = Path("Models") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    
    # Create quantization spec for no quantization
    quant_spec = QuantizationSpec(
        method=QuantMethod.NO_QUANT,
        weights_bits=16,
        activations_bits=16,
        kv_cache_bits=16,
        group_size=None,
        symmetric=None,
        per_channel=None,
        lm_head_dtype="bf16" if torch.cuda.is_available() else "fp16",
        backend="torch",
    )
    
    # Create metadata
    metadata = {
        "model_info": {
            "model_name": output_name,
            "base_model": model_name,
            "fine_tuning_date": datetime.now().isoformat(),
            "model_type": "CausalLM",
            "quantization_tag": quant_spec.tag(),
            "has_quantization": False,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable,
            "notes": "Base model snapshot without fine-tuning.",
        },
        "training_parameters": {
            "train": False,
            "dataset_choice": None,
            "finetuning_strategy": None,
            "quantization_method": "NoQuant",
            "trunc_train": 0,
        },
        "peft_config": None,
        "quantization_config": None,
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
    
    print(f"✅ Saved base model to: {output_dir}\n")
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return str(output_dir)

def main():
    print("\n" + "="*80)
    print("CREATE 1.7B BASE MODEL")
    print("="*80 + "\n")
    
    # Create 1.7B base model
    model_path = create_base_model(
        model_name="Qwen/Qwen2.5-1.5B",
        output_name="Qwen2.5-1.5B-base"
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Created: {model_path}")
    print("\nYou can now:")
    print(f"1. Evaluate it: python Testing/02_TestModels.py {model_path}")
    print("2. Create quantized variants if needed")

if __name__ == "__main__":
    main()
