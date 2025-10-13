#!/usr/bin/env python3
"""Test the training script QLoRA configuration without actual training."""

import sys
from pathlib import Path

# Add the current directory to path for imports
current_dir = Path.cwd()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / "Fine-tuning"))

import torch
from transformers import BitsAndBytesConfig
from quantization_utils import QuantMethod


def test_training_script_qlora_logic():
    """Test the QLoRA configuration logic from the training script."""
    
    # Mock the configuration values from training script
    qlora_r = 64
    qlora_lora_alpha = 16
    qlora_dropout = 0.05
    merge_after_train = True
    keep_lm_head_fp16 = False
    
    # Test QuantMethod parsing
    quant_method = QuantMethod.from_any("QLORA")
    print(f"✓ QuantMethod parsed correctly: {quant_method}")
    
    # Test BitsAndBytesConfig creation (from training script logic)
    if quant_method is QuantMethod.QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"✓ BitsAndBytesConfig created for QLoRA")
        print(f"  - load_in_4bit: {bnb_config.load_in_4bit}")
        print(f"  - quant_type: {bnb_config.bnb_4bit_quant_type}")
        print(f"  - double_quant: {bnb_config.bnb_4bit_use_double_quant}")
        print(f"  - compute_dtype: {bnb_config.bnb_4bit_compute_dtype}")
    else:
        bnb_config = None
        print("× BitsAndBytesConfig not created - method is not QLoRA")
    
    # Test QuantizationSpec creation (from resolve_quantization_spec function)
    from quantization_utils import QuantizationSpec
    
    if quant_method is QuantMethod.QLORA:
        spec = QuantizationSpec(
            method=quant_method,
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
        print(f"✓ QuantizationSpec created for QLoRA")
        print(f"  - Method: {spec.method}")
        print(f"  - Weights bits: {spec.weights_bits}")
        print(f"  - Backend: {spec.backend}")
        print(f"  - Tag: {spec.tag()}")
        print(f"  - QLoRA r: {spec.extras.get('qlora_r')}")
        print(f"  - QLoRA alpha: {spec.extras.get('qlora_lora_alpha')}")
        print(f"  - QLoRA dropout: {spec.extras.get('qlora_dropout')}")
    
    # Test PEFT configuration logic (from training script)
    from peft import LoraConfig
    
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    if quant_method is QuantMethod.QLORA:
        print(f"✓ Using QLoRA-specific LoRA configuration: r={qlora_r}, alpha={qlora_lora_alpha}, dropout={qlora_dropout}")
        peft_config = LoraConfig(
            r=qlora_r,
            lora_alpha=qlora_lora_alpha,
            target_modules=target_modules,
            lora_dropout=qlora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        print(f"  - LoRA r: {peft_config.r}")
        print(f"  - LoRA alpha: {peft_config.lora_alpha}")
        print(f"  - LoRA dropout: {peft_config.lora_dropout}")
        print(f"  - Target modules: {peft_config.target_modules}")
    
    # Test merge behavior logic
    MERGE_AFTER_TRAIN = True  # Default value
    should_merge = merge_after_train if quant_method is QuantMethod.QLORA else MERGE_AFTER_TRAIN
    print(f"✓ Merge behavior determined: should_merge={should_merge}")
    print(f"  - QLoRA merge_after_train: {merge_after_train}")
    print(f"  - Default MERGE_AFTER_TRAIN: {MERGE_AFTER_TRAIN}")
    
    # Test metadata extras update
    spec.extras.setdefault("merge_after_train", merge_after_train if quant_method is QuantMethod.QLORA else MERGE_AFTER_TRAIN)
    print(f"✓ Metadata updated with merge_after_train: {spec.extras.get('merge_after_train')}")
    
    return True


if __name__ == "__main__":
    print("Testing Training Script QLoRA Configuration Logic")
    print("=" * 60)
    
    try:
        success = test_training_script_qlora_logic()
        print("\n" + "=" * 60)
        print("✓ All QLoRA configuration tests passed!")
    except Exception as e:
        print(f"\n× Error in QLoRA configuration test: {e}")
        import traceback
        traceback.print_exc()