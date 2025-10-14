# QLoRA Quantization Method

## Overview

**QLoRA (Quantized Low-Rank Adaptation)** combines 4-bit quantization with LoRA fine-tuning, enabling efficient training of large language models on consumer hardware. Unlike post-training quantization methods, QLoRA quantizes the base model during training while keeping adapter weights in higher precision.

## How QLoRA Works

### Core Innovation
QLoRA addresses the memory bottleneck in fine-tuning by:
1. **4-bit Base Model**: Quantizes pretrained weights to NF4 (Normal Float 4)
2. **FP16 Adapters**: Keeps trainable LoRA adapters in full precision
3. **Double Quantization**: Further compresses quantization constants
4. **Paged Optimizers**: Handles memory spikes during training

### Technical Components

#### NF4 Quantization
- **Normal Float 4**: Optimized 4-bit format for neural network weights
- **Asymmetric**: Better captures weight distributions than symmetric quantization
- **Information-theoretic**: Optimal quantization bins for normal distributions

#### Double Quantization
```python
# First quantization: weights → 4-bit
quantized_weights = quantize_nf4(weights)

# Second quantization: quantization constants → 8-bit  
quantized_constants = quantize_fp8(quantization_constants)
```

## Implementation in Our Codebase

### File Structure
- **Main Integration**: [`Fine-tuning/01_Train.py`](../Fine-tuning/01_Train.py) - Training configuration
- **Quantization Logic**: [`quantization_utils.py`](../quantization_utils.py) - BitsAndBytes config
- **Model Preparation**: Uses `prepare_model_for_kbit_training()`

## Configuration and Usage

### Note: Training-Time Quantization (No CLI)
QLoRA does not use the `tools/quantize.py` CLI interface since it applies quantization during training, not as a post-training step. Configuration is done directly in the training script.

### Training Script Configuration
```python
# In Fine-tuning/01_Train.py
QUANT_METHOD = "QLORA"          # Enable QLoRA quantization
PEFT_CONFIG = "LoRa"            # Auto-overridden to LoRa for QLoRA
lora_r = 256                    # LoRA rank (adapter size)
merge_after_train = True        # Merge adapters after training
```

### QLoRA-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `QUANT_METHOD` | str | Required | Must be "QLORA" |
| `PEFT_CONFIG` | str | Auto-set | Automatically set to "LoRa" |
| `lora_r` | int | 256 | LoRA rank (higher = more parameters, better quality) |
| `merge_after_train` | bool | True | Merge adapters into base model after training |
| `keep_lm_head_fp16` | bool | False | QLoRA typically quantizes LM head too |

### Configuration Examples

#### Standard QLoRA (Recommended)
```python
# Fine-tuning/01_Train.py configuration
DATASET_CHOICE = "openmath"
QUANT_METHOD = "QLORA"
PEFT_CONFIG = "LoRa"       # Auto-overridden
lora_r = 256              # Good balance of quality and efficiency
merge_after_train = True   # Create merged model for inference
```

#### High-Quality QLoRA (More Parameters)
```python
DATASET_CHOICE = "openmath"
QUANT_METHOD = "QLORA"
PEFT_CONFIG = "LoRa"
lora_r = 512              # Higher rank = better quality, more memory
merge_after_train = True
```

#### Efficient QLoRA (Lower Memory)
```python
DATASET_CHOICE = "openmath"
QUANT_METHOD = "QLORA"
PEFT_CONFIG = "LoRa"
lora_r = 128              # Lower rank = less memory, faster training
merge_after_train = True
```

#### Research QLoRA (Keep Adapters Separate)
```python
DATASET_CHOICE = "openmath"
QUANT_METHOD = "QLORA"
PEFT_CONFIG = "LoRa"
lora_r = 256
merge_after_train = False  # Keep adapters separate for analysis
```

### Memory Requirements
| Base Model | FP16 | QLoRA (4-bit) | Memory Reduction |
|------------|------|---------------|------------------|
| 0.5B | ~2GB | ~0.8GB | 60% |
| 1B | ~4GB | ~1.6GB | 60% |
| 3B | ~12GB | ~4.8GB | 60% |
| 7B | ~28GB | ~11.2GB | 60% |

### Model Output Naming
QLoRA creates models with specific naming patterns:
```
{base_model}-{dataset}_SFT_{peft}_QLORA_w4_headbf16/
```

**Examples**:
- `Qwen3-0.6B-openmath_SFT_LoRa256_QLORA_w4_headbf16`
- `Qwen3-0.6B-squad_SFT_LoRa512_QLORA_w4_headbf16`

### Usage Pattern

#### Training Configuration
```python
# In Fine-tuning/01_Train.py
QUANT_METHOD = "QLORA"
PEFT_CONFIG = "LoRa"  # Auto-overridden to LoRa for QLoRA

# QLoRA-specific settings
merge_after_train = True
keep_lm_head_fp16 = False
```

#### Automatic Model Setup
```python
def build_quantization_plan(method: QuantMethod):
    if method is QuantMethod.QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return spec, bnb_config
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `load_in_4bit` | True | Enable 4-bit quantization |
| `bnb_4bit_quant_type` | "nf4" | Quantization format (NF4) |
| `bnb_4bit_use_double_quant` | True | Quantize quantization constants |
| `bnb_4bit_compute_dtype` | bfloat16 | Computation precision for adapters |
| `lora_r` | 256 | LoRA rank (overrides PEFT_CONFIG) |
| `lora_alpha` | 16 | LoRA scaling parameter |

## Model Organization and Naming

### Naming Convention
```
{base_model}-{dataset}_{method}_{peft}_{quantization_tag}
```

**Example**: `Qwen3-0.6B-openmath_SFT_LoRa256_QLORA_w4_headbf16`

### Tag Structure
- **Method**: `QLORA`
- **Weights**: `w4` (4-bit NF4)
- **LM Head**: `headbf16` (compute dtype)

### Special Behavior
```python
# QLoRA overrides PEFT configuration
if quant_method is QuantMethod.QLORA:
    print(f"Using QLoRA with LoRA configuration: r={lora_r}")
    peft_config = LoraConfig(...)  # Forces LoRA regardless of PEFT_CONFIG
    PEFT_CONFIG = f"LoRa{lora_r}"  # Updates naming
```

## Training Pipeline Integration

### Model Preparation
```python
# Load quantized base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config
)

# Prepare for k-bit training
if quant_method is QuantMethod.QLORA:
    model = prepare_model_for_kbit_training(model)

# Add LoRA adapters
model = get_peft_model(model, peft_config)
```

### Memory Tracking
```python
def get_peak_vram_gb():
    # QLoRA training tracks VRAM usage per GPU
    return {
        "overall_max_reserved_gb": max_reserved,
        "overall_max_allocated_gb": max_allocated,
        "per_gpu": [{"gpu": i, "peak_allocated_gb": ...}]
    }
```

### Model Merging
```python
# Post-training: merge adapters back to base model
if peft_config is not None and merge_after_train:
    print("Merging adapters back to base model")
    model = trained_model.merge_and_unload()
    
    # Optional: keep LM head in FP16
    if keep_lm_head_fp16:
        # Architecture-dependent implementation
        pass
```

## Performance Characteristics

### Memory Efficiency
- **Base Model**: ~25% of original size (4-bit vs FP16)
- **Adapters**: Small overhead (~1-5% of base model)
- **Total**: ~65% memory reduction vs full fine-tuning

### Example Memory Usage (0.6B Model)
```
Without QLoRA (FP16):     ~1.2GB base + ~1.2GB gradients = ~2.4GB
With QLoRA (NF4+LoRA):   ~0.3GB base + ~0.05GB adapters + ~0.05GB gradients = ~0.4GB
Memory Reduction:         ~83%
```

### Training Speed
- **Forward Pass**: Slightly slower due to dequantization
- **Backward Pass**: Only through small adapter weights
- **Overall**: 10-20% slower than full precision, but fits in memory

### Accuracy Retention
- **Typical Degradation**: <1% for most tasks with rank 64+
- **Sweet Spot**: Rank 128-256 for good accuracy/efficiency balance
- **Scaling**: Higher ranks → better accuracy but more memory

## Integration with Evaluation Pipeline

### Metadata Preservation
```json
{
  "quantization": {
    "method": "QLORA", 
    "weights_bits": 4,
    "lm_head_dtype": "bf16",
    "backend": "bitsandbytes",
    "extras": {
      "double_quant": true,
      "base_quant_type": "nf4",
      "compute_dtype": "bfloat16",
      "lora_r": 256,
      "merge_after_train": true
    }
  }
}
```

### Automatic Detection
```python
# Testing pipeline recognizes QLoRA models
if "QLORA" in model_name or metadata.get("method") == "QLORA":
    # Handle as quantized model for evaluation
```

## Best Practices

### When to Use QLoRA
- ✅ Fine-tuning large models (7B+) on consumer GPUs
- ✅ Limited VRAM budget (8-24GB)
- ✅ Tasks where slight accuracy trade-off is acceptable
- ✅ Need for efficient storage and deployment

### When to Avoid  
- ❌ Abundant GPU memory available
- ❌ Maximum accuracy is critical
- ❌ Very small models (<1B parameters)
- ❌ Inference-only quantization (use PTQ methods)

### Optimization Guidelines

#### LoRA Rank Selection
```python
# Model size → Recommended rank
lora_r_recommendations = {
    "0.5B-1B":   64,   # 3% trainable parameters  
    "3B-7B":     128,  # 1-2% trainable parameters
    "13B-30B":   256,  # 0.5-1% trainable parameters  
    "70B+":      512,  # 0.2-0.5% trainable parameters
}
```

#### Target Modules
```python
# Attention layers (minimum for effectiveness)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Add MLP for better accuracy (more memory)
target_modules_extended = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

#### Batch Size Tuning
```python
# QLoRA allows larger effective batch sizes
per_device_train_batch_size = 1
gradient_accumulation_steps = 16  # Effective batch size = 16

# Monitor VRAM usage and adjust accordingly
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solutions:
# - Reduce batch size
# - Increase gradient accumulation
# - Reduce LoRA rank
# - Enable gradient checkpointing
```

#### 2. BitsAndBytes Installation
```bash
# Ensure proper CUDA version
pip install bitsandbytes --upgrade
# Check CUDA compatibility
python -c "import bitsandbytes; print(bitsandbytes.cuda_setup.main())"
```

#### 3. Adapter Loading Issues
```python
# Ensure adapters are properly saved
model.save_pretrained(output_dir)  # Saves both base + adapters

# Load for inference
model = AutoModelForCausalLM.from_pretrained(
    output_dir, 
    quantization_config=bnb_config
)
```

### Performance Monitoring
```python
# Track training efficiency
def monitor_qlora_training():
    print(f"Trainable params: {trainable_params:,}")
    print(f"Percentage trainable: {percentage_trainable:.2f}%") 
    print(f"Peak VRAM: {peak_vram_gb:.1f}GB")
    print(f"Memory reduction: {memory_reduction:.1f}%")
```

## Comparison with Other Methods

| Aspect | QLoRA | Full Fine-tuning | PTQ Methods |
|--------|-------|------------------|-------------|
| **Memory** | ~83% reduction | Baseline | ~75% reduction |
| **Accuracy** | ~99% retained | 100% (baseline) | 95-98% retained |
| **Training Time** | 1.1-1.2x slower | 1x (baseline) | N/A (no training) |
| **Use Case** | Training-time efficiency | Maximum accuracy | Inference optimization |
| **Hardware** | Consumer GPUs | High-end GPUs | Any (post-training) |

## Advanced Features

### Gradient Checkpointing
```python
# Enable for further memory savings
gradient_checkpointing = True  # ~20% additional memory reduction
```

### Custom Target Modules  
```python
# Fine-grained control over which layers to adapt
target_modules = [
    "model.layers.*.self_attn.q_proj",  # Query projections only
    "model.layers.*.mlp.down_proj",     # MLP output projections
]
```

### Mixed Precision Training
```python
# QLoRA automatically handles mixed precision
# Base model: 4-bit NF4
# Adapters: BF16/FP16  
# Activations: BF16 (configurable)
```

## Research Background

### Original Paper Contributions
1. **NF4 Data Type**: Information-theoretically optimal 4-bit format
2. **Double Quantization**: Reduces memory overhead of quantization constants
3. **Paged Optimizers**: Handle CUDA memory spikes during training

### Key Innovations
- Maintains full fine-tuning performance with 65% memory reduction
- Enables fine-tuning of 65B models on single consumer GPU
- Preserves gradient flow through quantized weights via straight-through estimation

## Future Improvements

### Planned Enhancements
1. **8-bit Support**: Optional 8-bit quantization for accuracy-critical tasks
2. **Dynamic Quantization**: Runtime adjustment of quantization levels
3. **Multi-GPU Scaling**: Better support for distributed QLoRA training

### Integration Opportunities  
- Combine with structured pruning
- Integration with knowledge distillation
- Hardware-specific optimizations (A100, H100)