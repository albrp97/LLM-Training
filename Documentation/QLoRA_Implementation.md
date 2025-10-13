# QLoRA Implementation Summary

## Overview
This document summarizes the QLoRA (Quantized LoRA) implementation added to the LLM training pipeline. QLoRA enables efficient fine-tuning of large language models by combining 4-bit quantization with LoRA adapters.

## Changes Made

### 1. New Configuration Parameters
Added the following QLoRA-specific configuration parameters in `Fine-tuning/01_Train.py`:

```python
# QLoRA hyperparameters
qlora_r = 64                    # LoRA rank for QLoRA (default: 64)
qlora_lora_alpha = 16          # LoRA alpha for QLoRA (default: 16)  
qlora_dropout = 0.05           # LoRA dropout for QLoRA (default: 0.05)
merge_after_train = True       # Whether to merge adapters after training
keep_lm_head_fp16 = False      # Experimental: keep LM head in fp16 when merging
```

### 2. Enhanced QuantizationSpec for QLoRA
Updated the `resolve_quantization_spec()` function to provide detailed QLoRA metadata:

- **Method**: QLORA
- **Weights bits**: 4 (NF4 quantization)
- **Activations bits**: None (no activation quantization)
- **KV cache bits**: None (no KV cache quantization by default)
- **Group size**: None (NF4 doesn't use explicit group size)
- **Symmetric**: False (NF4 is asymmetric)
- **Backend**: bitsandbytes
- **Extras**: Includes double quantization, NF4 type, compute dtype, and LoRA parameters

### 3. BitsAndBytesConfig Integration
Implemented proper BitsAndBytesConfig creation for QLoRA:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### 4. QLoRA-Specific PEFT Configuration
Added logic to override regular PEFT configuration when using QLoRA:

- Uses QLoRA-specific LoRA parameters (r, alpha, dropout)
- Automatically calls `prepare_model_for_kbit_training()` for QLoRA models
- Updates PEFT_CONFIG string for consistent naming

### 5. Adaptive Merge Behavior
Implemented adaptive merge behavior based on quantization method:

- QLoRA uses `merge_after_train` flag
- Other methods use `MERGE_AFTER_TRAIN` flag
- Includes experimental support for `keep_lm_head_fp16` flag

### 6. Enhanced Metadata
Updated training metadata to include QLoRA-specific information:

- QLoRA LoRA parameters in extras
- Method-specific merge behavior
- Detailed quantization configuration

## Key Features

### 1. Memory Efficiency
QLoRA provides significant memory savings:
- 4-bit weight quantization using NF4
- Double quantization for additional compression
- Keeps computation in bfloat16 for stability

### 2. Training Quality
Maintains training quality through:
- NF4 quantization optimized for neural networks
- LoRA adapters for parameter-efficient fine-tuning
- Proper gradient handling via `prepare_model_for_kbit_training`

### 3. Configuration Flexibility
- Separate configuration parameters for QLoRA vs regular LoRA
- Configurable merge behavior after training
- Experimental LM head dtype preservation

### 4. Metadata Tracking
- Complete quantization metadata for reproducibility
- Tagged model naming for easy identification
- Detailed training parameter logging

## Dependencies
All required dependencies are already included in `pyproject.toml`:

- `bitsandbytes>=0.47.0` - For 4-bit quantization
- `peft>=0.17.1` - For LoRA adapters
- `trl>=0.21.0` - For SFT training
- `torch>=2.8.0` - For base functionality

## Testing
Created comprehensive test scripts to verify:

1. **`test_qlora_config.py`** - Tests quantization spec and config creation
2. **`test_qlora_training_logic.py`** - Tests training script logic without actual training

Both tests pass successfully, confirming proper implementation.

## Usage Example

To use QLoRA for training, set the following in `Fine-tuning/01_Train.py`:

```python
QUANT_METHOD = "QLORA"
DATASET_CHOICE = "openmath"  # or "squad"
PEFT_CONFIG = "LoRa"         # Will be overridden by QLoRA settings

# QLoRA-specific settings
qlora_r = 64
qlora_lora_alpha = 16  
qlora_dropout = 0.05
merge_after_train = True
keep_lm_head_fp16 = False
```

The resulting model will be saved with a tag like: `Qwen3-0.6B-openmath_SFT_LoRa64_QLORA_w4_headbf16`

## Integration with Evaluation
The evaluation script (`Testing/02_TestModels.py`) already includes QLoRA support and will automatically:

- Detect QLoRA models from quantization metadata
- Load models with proper quantization configuration
- Evaluate performance with quantized weights

## Next Steps
This implementation provides a solid foundation for QLoRA training. Future enhancements could include:

1. Advanced LM head dtype handling implementation
2. Additional quantization backends (e.g., GPTQ, AWQ)
3. Dynamic QLoRA parameter selection based on model size
4. Integration with other PEFT methods (DoRA, VeRA) under quantization