# AdaRound Quantization Method

## Overview

**AdaRound (Adaptive Rounding)** is a post-training quantization (PTQ) method that learns optimal rounding decisions for weight quantization to minimize reconstruction error. Unlike traditional quantization that uses simple rounding rules (floor/ceil), AdaRound adaptively chooses the best rounding direction for each weight group based on calibration data.

## How AdaRound Works

### Core Principle
AdaRound addresses the key challenge in weight quantization: **rounding decisions**. When converting FP16/FP32 weights to lower precision (e.g., 4-bit), each weight must be rounded to the nearest quantized value. The choice between rounding up or down significantly impacts model accuracy.

### Algorithm Steps

1. **Group-wise Quantization**: Weights are divided into groups (default: 128 elements)
2. **Scale Calculation**: Each group computes quantization scale and zero-point
3. **Reconstruction Error Minimization**: For each weight, compare reconstruction errors:
   - `floor_error = |original_weight - floor_reconstructed|`
   - `ceil_error = |original_weight - ceil_reconstructed|` 
   - Choose rounding direction that minimizes error
4. **Calibration-Aware**: Uses representative data samples to guide decisions

## Implementation in Our Codebase

### File Structure
- **Main Implementation**: [`tools/quantize.py`](../tools/quantize.py) - `quantize_with_adaround()`
- **Integration**: [`Fine-tuning/01_Train.py`](../Fine-tuning/01_Train.py) - Auto-calibration generation
- **Utilities**: [`quantization_utils.py`](../quantization_utils.py) - Method enum and tagging

### Usage Patterns

#### 1. Automatic via Training Script
```python
# In Fine-tuning/01_Train.py
QUANT_METHOD = "AdaRound"  
PTQ_TARGET_WEIGHTS_BITS = 4
PTQ_TARGET_GROUP_SIZE = 128
```

#### 2. Manual via CLI Tool
```bash
python tools/quantize.py run \
  --src Models/base_model \
  --dst Models/base_model_AdaRound \
  --method adaround \
  --bits 4 --group-size 128 --keep-lm-head-fp16 \
  --calib Datasets/calibration_openmath_5samples.txt
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bits` | 4 | Target weight quantization bits |
| `group_size` | 128 | Number of weights per quantization group |
| `symmetric` | True | Use symmetric quantization range |
| `skip_lm_head` | True | Keep language model head in FP16 |
| `calibration_samples` | 128 | Number of calibration prompts |

### Progress Tracking

AdaRound provides detailed progress feedback:

```
[AdaRound] Found 196 Linear layers, quantizing 196 (skipping 0 LM head layers)
[AdaRound] (1/196) [  0.5%] Quantizing: model.layers.0.self_attn.q_proj
        Processing 16384 weight groups... 25% 50% 75% 100% ✓
[AdaRound] (2/196) [  1.0%] Quantizing: model.layers.0.self_attn.k_proj
        Processing 8192 weight groups... 25% 50% 75% 100% ✓
...
[AdaRound] (196/196) [100.0%] Quantizing: model.layers.27.mlp.down_proj
[AdaRound] Quantized 196 layers
```

## Calibration Data Integration

### Automatic Generation
The training script automatically creates calibration data from the training set:

```python
# Creates calibration_openmath_5samples.txt with 15% of training samples
def create_calibration_data():
    calib_size = min(128, max(8, int(train_size * 0.15)))
    # Sample from training data and format as chat templates
```

### Calibration Requirements
- **Size**: 8-128 prompts (scales with `TRUNC_TRAIN`)
- **Format**: Chat template format matching training data
- **Content**: Representative samples from the same domain
- **Storage**: `Datasets/calibration_{dataset}_{samples}samples.txt`

## Model Naming and Organization

### Naming Convention
```
{base_model}-{dataset}_{method}_{peft}_{quantization_tag}
```

**Example**: `Qwen3-0.6B-openmath_SFT_LoRa256_AdaRound_w4_g128_headfp16`

### Tag Structure
- **Method**: `AdaRound`
- **Weights**: `w4` (4-bit)  
- **Group Size**: `g128` (128 elements)
- **LM Head**: `headfp16` (kept in FP16)

## Performance Characteristics

### Memory Usage
- **Training**: Minimal overhead (PTQ method)
- **Inference**: ~25% VRAM reduction vs FP16
- **Example**: 0.6B model uses ~1.3GB VRAM (vs ~1.7GB FP16)

### Accuracy Trade-offs
- **Strength**: Superior to naive rounding, especially for small models
- **Best Use Cases**: 4-8 bit quantization with group sizes 64-256
- **Limitations**: Requires calibration data representative of target use

### Speed
- **Quantization Time**: ~2-5 minutes for 0.6B model (196 layers)
- **Inference Speed**: Depends on backend support (currently FP16 fallback)

## Integration with Evaluation Pipeline

### Automatic Detection
```python
# Testing/03_EvaluationOrchestrator.py automatically finds AdaRound models
quant_method = detect_method_from_path(model_path)  # → QuantMethod.ADA_ROUND
```

### Metadata Preservation
```json
{
  "quantization": {
    "method": "AdaRound",
    "weights_bits": 4,
    "group_size": 128,
    "calibration_samples": 5,
    "quant_tag": "AdaRound_w4_g128_headfp16"
  }
}
```

## Best Practices

### When to Use AdaRound
- ✅ Post-training quantization of fine-tuned models
- ✅ Need for minimal accuracy degradation 
- ✅ Have representative calibration data available
- ✅ Target 4-8 bit quantization

### When to Avoid
- ❌ During training (use QLoRA instead)
- ❌ Extremely small calibration sets (<10 samples)
- ❌ When inference speed is critical (use simpler methods)

### Optimization Tips
1. **Calibration Quality**: Use diverse, representative samples
2. **Group Size**: Larger groups (256) for better compression, smaller (64) for better accuracy  
3. **LM Head**: Always keep in FP16 for language tasks
4. **Batch Size**: AdaRound works per-layer, so VRAM isn't a major constraint

## Troubleshooting

### Common Issues
1. **"No calibration data found"**: Ensure calibration file exists and is readable
2. **CUDA OOM**: Reduce batch size or use CPU for quantization
3. **Poor accuracy**: Check calibration data quality and representativeness

### Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check quantization metadata
with open("Models/model/quantization_metadata.json") as f:
    metadata = json.load(f)
    print(f"Calibration hash: {metadata['calibration']['hash']}")
```

## Comparison with Other Methods

| Method | Type | Accuracy | Speed | Memory | Use Case |
|--------|------|----------|-------|---------|----------|
| **AdaRound** | PTQ | High | Medium | Low | Post-training quantization |
| QLoRA | Training | High | Fast | Medium | Training-time quantization |
| GPTQ | PTQ | Medium | Fast | Low | Inference-optimized |
| AWQ | PTQ | High | Fast | Low | Activation-aware quantization |

## Future Improvements

### Planned Features
1. **Hardware Acceleration**: Native 4-bit inference kernels
2. **Advanced Calibration**: Task-specific calibration strategies  
3. **Mixed Precision**: Layer-wise bit allocation
4. **Activation Quantization**: Extend to activations and KV-cache

### Research Directions
- Integration with structured pruning
- Learned quantization schedules
- Hardware-specific optimizations