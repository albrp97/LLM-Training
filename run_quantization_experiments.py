"""
Master script to train and evaluate quantization variants.

This script:
1. Creates quantized versions of existing models (int8, 4bit)
2. Trains 8B with QLoRA
3. Evaluates all models
4. Generates comparison metrics
"""

import sys
import subprocess
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True

def update_train_config(model_name, dataset, quant_method, peft_config):
    """Update the training configuration in 01_Train.py"""
    train_file = Path("Fine-tuning/01_Train.py")
    
    with open(train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Update configuration lines
    for i, line in enumerate(lines):
        if line.strip().startswith("train = "):
            lines[i] = "train = True\n"
        elif line.strip().startswith("DATASET_CHOICE = "):
            lines[i] = f'DATASET_CHOICE = "{dataset}"\n'
        elif line.strip().startswith("MODEL_NAME = "):
            lines[i] = f'MODEL_NAME = "{model_name}"\n'
        elif line.strip().startswith("PEFT_CONFIG = "):
            lines[i] = f'PEFT_CONFIG = "{peft_config}"\n'
        elif line.strip().startswith("QUANT_METHOD = "):
            lines[i] = f'QUANT_METHOD = "{quant_method}"\n'
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"✓ Updated 01_Train.py: {model_name}, {dataset}, {quant_method}, {peft_config}")

def restore_train_config():
    """Restore default training configuration"""
    update_train_config("Qwen/Qwen3-0.6B", "None", "NoQuant", "NoPeft")

def main():
    print("\n" + "="*80)
    print("QUANTIZATION VARIANTS - COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    steps_completed = []
    
    # Step 1: Create quantized versions of existing models
    if run_command(
        [sys.executable, "train_quantization_variants.py"],
        "Create quantized variants (int8, 4bit) of base and nopeft models"
    ):
        steps_completed.append("Quantized variants created")
    else:
        print("\n❌ Failed to create quantized variants. Stopping.")
        return
    
    # Step 2: Train 8B with QLoRA
    print("\n" + "="*80)
    print("STEP: Train Qwen3-8B with QLoRA on openmath")
    print("="*80 + "\n")
    
    try:
        update_train_config("Qwen/Qwen3-8B", "openmath", "QLORA", "LoRa")
        
        if run_command(
            [sys.executable, "Fine-tuning/01_Train.py"],
            "Training 8B with QLoRA"
        ):
            steps_completed.append("8B QLoRA trained")
        else:
            print("\n⚠️ Training failed, but continuing with evaluation...")
    finally:
        restore_train_config()
    
    # Step 3: Evaluate all models
    print("\n" + "="*80)
    print("STEP: Evaluate all models")
    print("="*80 + "\n")
    
    # List all models to evaluate
    models_dir = Path("Models")
    models_to_evaluate = [
        "Qwen3-4B-base",
        "Qwen3-4B-base_Int8_BnB",
        "Qwen3-4B-base_4bit_BnB",
        "Qwen3-4B-openmath_SFT_NoPeft_NoQuant",
        "Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB",
        "Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB",
        "Qwen3-8B-base",
        "Qwen3-8B-openmath_SFT_LoRa256_QLoRA_4bit",  # Expected output name
    ]
    
    # Filter to only existing models
    existing_models = [m for m in models_to_evaluate if (models_dir / m).exists()]
    
    print(f"Found {len(existing_models)} models to evaluate:")
    for m in existing_models:
        print(f"  - {m}")
    
    # Evaluate each model
    for model in existing_models:
        model_path = f"Models/{model}"
        if run_command(
            [sys.executable, "Testing/02_TestModels.py", model_path],
            f"Evaluating {model}"
        ):
            steps_completed.append(f"Evaluated {model}")
        else:
            print(f"\n⚠️ Failed to evaluate {model}, continuing...")
    
    # Step 4: Generate comparison metrics
    print("\n" + "="*80)
    print("STEP: Generate metrics comparison")
    print("="*80 + "\n")
    
    # We'll need to run the notebook programmatically
    # For now, just inform the user
    print("To generate the comparison CSV, run the notebook:")
    print("  Testing/04_CompareMetrics.ipynb")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"\nCompleted {len(steps_completed)} steps:")
    for i, step in enumerate(steps_completed, 1):
        print(f"  {i}. ✅ {step}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Run the comparison notebook to generate CSV:")
    print("     Testing/04_CompareMetrics.ipynb")
    print("\n2. Review metrics_summary.csv for comparison")
    print("\n3. Models available for testing:")
    for m in existing_models:
        print(f"     - {m}")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
