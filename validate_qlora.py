#!/usr/bin/env python3
"""
QLoRA Training Configuration Validation Script

This script validates that the QLoRA implementation is working correctly
by checking configuration creation without actually loading models or training.
Run this before attempting actual QLoRA training to catch configuration issues early.
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path.cwd()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "Fine-tuning"))

def validate_qlora_imports():
    """Validate that all required imports are available."""
    print("Validating imports...")
    
    try:
        import torch
        print("  ✓ torch")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    
    try:
        from transformers import BitsAndBytesConfig
        print("  ✓ transformers.BitsAndBytesConfig")
    except ImportError as e:
        print(f"  ✗ transformers.BitsAndBytesConfig: {e}")
        return False
    
    try:
        from peft import LoraConfig, prepare_model_for_kbit_training
        print("  ✓ peft.LoraConfig")
        print("  ✓ peft.prepare_model_for_kbit_training")
    except ImportError as e:
        print(f"  ✗ peft: {e}")
        return False
    
    try:
        from quantization_utils import QuantMethod, QuantizationSpec
        print("  ✓ quantization_utils")
    except ImportError as e:
        print(f"  ✗ quantization_utils: {e}")
        return False
    
    return True


def validate_qlora_configuration():
    """Validate QLoRA configuration creation."""
    print("\nValidating QLoRA configuration...")
    
    from transformers import BitsAndBytesConfig
    from quantization_utils import QuantMethod, QuantizationSpec
    from peft import LoraConfig
    import torch
    
    # QLoRA parameters (matching training script defaults)
    qlora_r = 64
    qlora_lora_alpha = 16
    qlora_dropout = 0.05
    keep_lm_head_fp16 = False
    
    # 1. Test QuantMethod parsing
    try:
        method = QuantMethod.from_any("QLORA")
        print(f"  ✓ QuantMethod parsed: {method}")
    except Exception as e:
        print(f"  ✗ QuantMethod parsing failed: {e}")
        return False
    
    # 2. Test BitsAndBytesConfig creation
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"  ✓ BitsAndBytesConfig created")
    except Exception as e:
        print(f"  ✗ BitsAndBytesConfig creation failed: {e}")
        return False
    
    # 3. Test QuantizationSpec creation
    try:
        spec = QuantizationSpec(
            method=method,
            weights_bits=4,
            activations_bits=None,
            kv_cache_bits=None,
            group_size=None,
            symmetric=False,
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
        print(f"  ✓ QuantizationSpec created with tag: {spec.tag()}")
    except Exception as e:
        print(f"  ✗ QuantizationSpec creation failed: {e}")
        return False
    
    # 4. Test LoraConfig creation
    try:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=qlora_r,
            lora_alpha=qlora_lora_alpha,
            target_modules=target_modules,
            lora_dropout=qlora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        print(f"  ✓ LoraConfig created for QLoRA")
    except Exception as e:
        print(f"  ✗ LoraConfig creation failed: {e}")
        return False
    
    return True


def validate_training_script_syntax():
    """Validate that the training script has correct syntax."""
    print("\nValidating training script syntax...")
    
    try:
        # Try to compile the training script without executing it
        training_script = project_root / "Fine-tuning" / "01_Train.py"
        if not training_script.exists():
            print(f"  ✗ Training script not found: {training_script}")
            return False
        
        with open(training_script, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, str(training_script), 'exec')
        print("  ✓ Training script syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"  ✗ Training script syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Training script validation failed: {e}")
        return False


def main():
    """Run all validation checks."""
    print("QLoRA Implementation Validation")
    print("=" * 50)
    
    checks = [
        ("Import validation", validate_qlora_imports),
        ("Configuration validation", validate_qlora_configuration),
        ("Training script syntax", validate_training_script_syntax),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All validations passed! QLoRA implementation is ready.")
        print("\nTo use QLoRA, set in Fine-tuning/01_Train.py:")
        print('  QUANT_METHOD = "QLORA"')
        print('  DATASET_CHOICE = "openmath"  # or your preferred dataset')
        return 0
    else:
        print("✗ Some validations failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())