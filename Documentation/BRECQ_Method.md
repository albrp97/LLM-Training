# BRECQ Quantization Method

## Overview

**BRECQ (Block-wise Reconstruction-based Quantization)** is a post-training quantization (PTQ) method that uses block-wise reconstruction with mixed precision support. BRECQ performs iterative refinement on weight blocks to minimize reconstruction error while supporting different precision levels for different layer types (e.g., W6 for attention, W4 for MLP).

## How BRECQ Works

### Core Principle
BRECQ addresses quantization challenges through **block-wise reconstruction** with **mixed precision awareness**. Instead of quantizing all layers uniformly, BRECQ can assign different bit-widths to different layer types based on their sensitivity to quantization.

### Algorithm Steps

1. **Layer Categorization**: Automatically identifies layer types:
   - **Attention layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj` → Higher precision (W6)
   - **MLP layers**: `gate_proj`, `up_proj`, `down_proj`, `fc` → Lower precision (W4)
   - **Other layers**: Remaining linear layers → Configurable precision

2. **Block-wise Reconstruction**: Weights are divided into blocks (default: 64 elements)
   - Each block is quantized independently
   - Iterative refinement with error minimization
   - Adaptive scale adjustment based on reconstruction error

3. **Mixed Precision Quantization**:
   - Attention layers: 6-bit quantization (preserves critical attention patterns)
   - MLP layers: 4-bit quantization (aggressive compression)
   - LM head: Configurable (default: FP16)

4. **Calibration-Aware**: Uses representative calibration data to guide quantization decisions

## Implementation in Our Codebase

### File Structure
- **Main Implementation**: [`tools/quantize.py`](../tools/quantize.py) - `quantize_with_brecq()`
- **Integration**: [`Fine-tuning/01_Train.py`](../Fine-tuning/01_Train.py) - Mixed precision configuration
- **Utilities**: [`quantization_utils.py`](../quantization_utils.py) - BRECQ-specific tagging

### Usage Patterns

#### 1. Automatic via Training Script
```python
# In Fine-tuning/01_Train.py
QUANT_METHOD = "BRECQ"  
PTQ_TARGET_WEIGHTS_BITS = 4  # MLP layers
PTQ_TARGET_GROUP_SIZE = 64   # Block size
# Attention layers automatically set to 6-bit
```

#### 2. Manual via CLI Tool
```bash
python tools/quantize.py run \
  --src Models/base_model \
  --dst Models/base_model_BRECQ \
  --method brecq \
  --bits 4 --group-size 64 \
  --calib Datasets/calibration_openmath_5samples.txt
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bits` | 4 | Target weight bits for MLP layers |
| `attn_bits` | 6 | Target weight bits for attention layers |
| `group_size` | 64 | Number of weights per quantization block |
| `mixed_precision` | True | Enable different bits for different layer types |
| `seed` | 13 | Random seed for reproducible results |
| `calibration_samples` | 128 | Number of calibration prompts |

### Progress Tracking

BRECQ provides detailed mixed precision feedback:

```
[BRECQ] Found 196 Linear layers:
  - Attention layers: 112 (W6)
  - MLP layers: 84 (W4)
  - Other layers: 0 (W4)
  - Skipping LM head layers

[BRECQ] Processing 112 attention layers with W6
BRECQ Attention: 100%|████████████| 112/112 [22:17<00:00, 11.94s/it]

[BRECQ] Processing 84 MLP layers with W4  
BRECQ MLP: 100%|████████████████████| 84/84 [33:38<00:00, 24.03s/it]

[BRECQ] Quantized 196 layers with block-wise reconstruction
```

## Mixed Precision Strategy

### Layer Type Detection
```python
# Automatic layer categorization based on name patterns
attention_patterns = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj']
mlp_patterns = ['mlp', 'fc', 'gate_proj', 'up_proj', 'down_proj']
```

### Precision Assignment
- **Attention (W6)**: Preserves critical attention mechanisms
- **MLP (W4)**: Aggressive compression for feed-forward layers  
- **Mixed Benefits**: Balances accuracy and compression ratio

### Block-wise Reconstruction Process
```python
# Simplified BRECQ algorithm per block
for refinement_step in range(2):
    reconstructed = quantized_vals * scale + zero_point
    error = torch.mean((original_block - reconstructed) ** 2)
    
    if error > threshold:
        # Adaptive scale adjustment
        error_gradient = torch.mean((original_block - reconstructed) * quantized_vals)
        scale_adjustment = error_gradient * learning_rate
        scale = scale + scale_adjustment * scale
```

## Calibration Data Integration

### Automatic Generation
The training script creates calibration data matching the training format:

```python
# Creates calibration prompts with chat template format
def create_calibration_data():
    calib_size = min(128, max(8, int(train_size * 0.15)))
    for _, row in calib_df.iterrows():
        prompt = f"User: {row['question']}\nAssistant:"
        calib_prompts.append(prompt)
```

### Calibration Requirements
- **Size**: 8-128 prompts (15% of training data)
- **Format**: Chat template format matching training
- **Quality**: Representative samples from target domain
- **Usage**: Guides block-wise reconstruction decisions

## Model Naming and Organization

### Naming Convention
```
{base_model}-{dataset}_{method}_{peft}_{quantization_tag}
```

**Example**: `Qwen3-0.6B-openmath_SFT_NoPeft_BRECQ_w4_g64_attn6_headfp16_mix`

### Tag Structure
- **Method**: `BRECQ`
- **MLP Weights**: `w4` (4-bit)
- **Group Size**: `g64` (64 elements)  
- **Attention**: `attn6` (6-bit attention layers)
- **LM Head**: `headfp16` (FP16 preserved)
- **Mixed**: `mix` (mixed precision enabled)

**Full Tag**: `BRECQ_w4_g64_attn6_headfp16_mix`

## Performance Characteristics

### Memory Usage
- **Training**: Minimal overhead (PTQ method)
- **Inference**: ~30-40% VRAM reduction vs FP16
- **Mixed Precision**: Better accuracy/size trade-off than uniform quantization
- **Example**: 0.6B model uses ~1.3GB VRAM (vs ~1.7GB FP16)

### Accuracy Trade-offs
- **Strengths**: 
  - Superior mixed precision handling
  - Preserves attention quality with W6
  - Aggressive MLP compression with W4
- **Best Use Cases**: Models where attention quality is critical
- **Limitations**: Longer quantization time due to iterative refinement

### Speed
- **Quantization Time**: ~55 minutes for 0.6B model (196 layers with refinement)
  - Attention layers: ~22 minutes (112 layers × 11.9s each)  
  - MLP layers: ~34 minutes (84 layers × 24.0s each)
- **Inference Speed**: Standard PyTorch inference (no specialized kernels yet)

## Integration with Evaluation Pipeline

### Automatic Detection
```python
# Supports both training metadata and quantization metadata
quant_context = resolve_quant_context(model_name, "auto")
# → method=BRECQ, source=training_metadata/quantization_metadata
```

### Metadata Preservation
```json
{
  "quantization": {
    "method": "BRECQ",
    "weights_bits": 4,
    "group_size": 64,
    "mixed_precision": true,
    "attention_bits": 6,
    "mlp_bits": 4,
    "attention_layer_count": 112,
    "mlp_layer_count": 84,
    "calibration_samples": 5,
    "quant_tag": "BRECQ_w4_g64_attn6_headfp16_mix"
  }
}
```

## Best Practices

### When to Use BRECQ
- ✅ Models where attention quality is critical
- ✅ Need better accuracy than uniform quantization
- ✅ Have sufficient time for longer quantization process  
- ✅ Target mixed precision deployment
- ✅ Representative calibration data available

### When to Avoid
- ❌ During training (use QLoRA instead)
- ❌ When quantization speed is critical
- ❌ Very small models where mixed precision overhead outweighs benefits
- ❌ Insufficient calibration data (<10 samples)

### Optimization Tips
1. **Layer Balance**: Ensure good attention/MLP layer ratio for mixed precision benefits
2. **Block Size**: Smaller blocks (64) for better accuracy, larger for speed
3. **Calibration Quality**: Use diverse samples covering expected use cases
4. **Precision Tuning**: Adjust `attn_bits` based on model sensitivity analysis
5. **Hardware**: Use GPU acceleration for block-wise operations

## Troubleshooting

### Common Issues
1. **Long quantization time**: Normal for BRECQ due to iterative refinement
2. **CUDA OOM during quantization**: Reduce calibration batch size or use CPU
3. **Poor accuracy**: Check mixed precision settings and calibration quality
4. **Layer detection errors**: Verify layer naming patterns match expectations

### Debugging
```python
# Check layer categorization
attention_layers = []
mlp_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        if any(pattern in name.lower() for pattern in attention_patterns):
            attention_layers.append(name)
        # ... categorize layers

print(f"Attention: {len(attention_layers)}, MLP: {len(mlp_layers)}")
```

### Performance Monitoring
```python
# Track quantization progress per layer type
with torch.no_grad():
    for name, module in attention_layers:
        start_time = time.time()
        quantized_weight = block_wise_quantize_weight(module.weight.data)
        elapsed = time.time() - start_time
        print(f"Attention layer {name}: {elapsed:.1f}s")
```

## Comparison with Other Methods

| Method | Type | Accuracy | Speed | Memory | Mixed Precision | Use Case |
|--------|------|----------|-------|---------|-----------------|----------|
| **BRECQ** | PTQ | Very High | Slow | Low | ✅ Native | Attention-critical models |
| AdaRound | PTQ | High | Medium | Low | ❌ Uniform | General PTQ |
| QLoRA | Training | High | Fast | Medium | ❌ Uniform | Training-time |
| GPTQ | PTQ | Medium | Fast | Low | ❌ Uniform | Inference speed |
| AWQ | PTQ | High | Fast | Low | ⚠️ Limited | Activation-aware |

## Future Improvements

### Planned Features
1. **Hardware Acceleration**: Native mixed precision inference kernels
2. **Dynamic Precision**: Runtime bit-width adjustment based on activation patterns
3. **Advanced Layer Detection**: Transformer architecture-aware layer categorization
4. **Calibration Optimization**: Task-specific calibration sample selection

### Research Directions
- **Learned Block Sizes**: Adaptive block size selection per layer
- **Cross-Layer Optimization**: Global reconstruction across multiple layers
- **Hardware Co-design**: BRECQ-optimized inference engines
- **Automatic Precision Search**: Neural architecture search for optimal bit allocation

## Advanced Configuration

### Custom Layer Categorization
```python
# Override default layer patterns
custom_attention_patterns = ['self_attn', 'cross_attn', 'attention']
custom_mlp_patterns = ['feed_forward', 'ffn', 'linear']

# Custom precision assignment
layer_precision_map = {
    'attention': 8,  # W8 for critical attention
    'mlp': 3,        # W3 for aggressive MLP compression
    'other': 4       # W4 for everything else
}
```

### Calibration Strategies
```python
# Dataset-specific calibration
if dataset == 'code':
    calib_samples = 256  # More samples for code complexity
elif dataset == 'math':  
    calib_samples = 64   # Fewer high-quality math samples
    
# Domain adaptation
calib_prompts = filter_by_difficulty(calib_prompts, min_difficulty=0.7)
```