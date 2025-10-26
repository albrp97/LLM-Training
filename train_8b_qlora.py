"""
Train Qwen3-8B with QLoRA on openmath dataset.
This is a standalone script based on 01_Train.py with configurations for 8B QLoRA training.
"""

import sys
from pathlib import Path

# Temporarily modify 01_Train.py settings
train_script = Path("Fine-tuning/01_Train.py")

# Read the current file
with open(train_script, "r", encoding="utf-8") as f:
    content = f.read()

# Create a modified version
modified_content = content

# Set configurations
replacements = {
    'train = True': 'train = True',
    'DATASET_CHOICE = None': 'DATASET_CHOICE = "openmath"',
    'MODEL_NAME = "Qwen/Qwen3-0.6B"': 'MODEL_NAME = "Qwen/Qwen3-8B"',
    'PEFT_CONFIG = "NoPeft"': 'PEFT_CONFIG = "LoRa"',
    'QUANT_METHOD = "NoQuant"': 'QUANT_METHOD = "QLORA"',
    'lora_r = 256': 'lora_r = 256',
    'num_train_epochs = 1': 'num_train_epochs = 1',
}

for old, new in replacements.items():
    if old in modified_content:
        modified_content = modified_content.replace(old, new, 1)

print("="*80)
print("TRAINING 8B MODEL WITH QLORA")
print("="*80)
print("\nConfiguration:")
print("  Model: Qwen/Qwen3-8B")
print("  Dataset: openmath")
print("  Method: QLoRA (4-bit)")
print("  PEFT: LoRA (rank 256)")
print("  Epochs: 1")
print("\n" + "="*80 + "\n")

# Write to a temporary file and execute
temp_script = Path("Fine-tuning/temp_train_8b_qlora.py")
with open(temp_script, "w", encoding="utf-8") as f:
    f.write(modified_content)

# Execute the script
import subprocess
result = subprocess.run([sys.executable, str(temp_script)], cwd=Path.cwd())

# Clean up
temp_script.unlink(missing_ok=True)

sys.exit(result.returncode)
