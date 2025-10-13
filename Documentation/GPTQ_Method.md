# GPTQ Quantization Method

## Overview

**GPTQ (Gradient-based Post-Training Quantization)** is an advanced post-training quantization method that uses second-order information (Hessian) to minimize quantization error. It processes layers sequentially and optimizes quantization parameters to maintain model accuracy while achieving significant compression.

## How GPTQ Works

### Core Algorithm
GPTQ treats quantization as an optimization problem:

1. **Layer-wise Processing**: Quantizes one layer at a time while keeping others fixed
2. **Hessian Approximation**: Uses gradient information to understand weight importance
3. **Optimal Brain Compression**: Applies OBC algorithm to minimize reconstruction error
4. **Sequential Updates**: Propagates quantization errors through subsequent layers

### Mathematical Foundation
```
Minimize: ||W - Ŵ||²_H
Where: H = Hessian matrix (curvature information)
       W = original weights  
       Ŵ = quantized weights
```

### Key Advantages
- **Accuracy Preservation**: Better than naive quantization methods
- **Hardware Efficient**: Optimized for GPU inference
- **Mature Ecosystem**: Well-supported by inference frameworks

## Implementation Status in Our Codebase

### Current Implementation
```python
# File: tools/quantize.py
def quantize_gptq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "auto_gptq",
        "GPTQ quantisation requires the `auto-gptq` package. Install it with `pip install auto-gptq`.",
    )
    raise NotImplementedError("GPTQ quantisation stub: integrate your AutoGPTQ workflow here.")
```

**Status**: 🚧 **Placeholder Implementation** - Ready for integration

### Planned Implementation Architecture

#### AutoGPTQ Integration
```python
def quantize_gptq_full(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace):
    """Full GPTQ implementation using AutoGPTQ library."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=spec.weights_bits,
        group_size=spec.group_size,
        damp_percent=0.01,
        desc_act=False,
        static_groups=False,
        sym=spec.symmetric or True,
        true_sequential=True,
        model_name_or_path=None,
        model_file_base_name="model"
    )
    
    # Load model and calibration data
    model = AutoGPTQForCausalLM.from_pretrained(src, quantize_config)
    calibration_data = load_calibration_dataset(args.calib)
    
    # Perform quantization
    model.quantize(calibration_data)
    
    # Save quantized model
    model.save_quantized(dst, use_safetensors=True)
```

## Configuration and Usage

### Planned Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bits` | 4 | Weight quantization bits (4, 8) |
| `group_size` | 128 | Quantization group size (-1 for per-channel) |
| `damp_percent` | 0.01 | Damping factor for Hessian regularization |
| `desc_act` | False | Use descending activation order |
| `static_groups` | False | Use static quantization groups |
| `sym` | True | Symmetric vs asymmetric quantization |
| `true_sequential` | True | Sequential layer processing |

### Usage Patterns

#### CLI Interface
```bash
python tools/quantize.py run \
  --src Models/base_model \
  --dst Models/base_model_GPTQ \
  --method gptq \
  --bits 4 --group-size 128 \
  --calib Datasets/calibration_data.txt
```

#### Training Script Integration
```python
# In Fine-tuning/01_Train.py
QUANT_METHOD = "GPTQ"
PTQ_TARGET_WEIGHTS_BITS = 4
PTQ_TARGET_GROUP_SIZE = 128
```

### Model Naming Convention
```
{base_model}-{dataset}_{method}_{peft}_{quantization_tag}
```

**Example**: `Qwen3-0.6B-openmath_SFT_LoRa256_GPTQ_w4_g128_headfp16`

## Integration Requirements

### Dependencies
```bash
# Required packages
pip install auto-gptq
pip install transformers>=4.32.0
pip install accelerate>=0.20.0

# Optional: for better performance
pip install triton  # GPU kernel acceleration
```

### Hardware Requirements
- **GPU Memory**: 2-4x model size during quantization
- **CUDA**: Version 11.8+ recommended
- **Time**: ~10-30 minutes for 7B models depending on calibration size

## Performance Characteristics

### Expected Results

#### Memory Reduction
- **4-bit**: ~75% memory reduction vs FP16
- **8-bit**: ~50% memory reduction vs FP16
- **Inference**: Significant speedup on supported hardware

#### Accuracy Retention
| Model Size | 4-bit GPTQ | 8-bit GPTQ |
|------------|------------|------------|
| 0.5-1B | 95-97% | 98-99% |
| 3-7B | 97-98% | 99%+ |
| 13B+ | 98-99% | 99.5%+ |

### Calibration Requirements
- **Dataset Size**: 128-512 samples recommended
- **Quality**: Should represent target distribution
- **Format**: Text samples (automatically tokenized)

## Integration Roadmap

### Phase 1: Basic Implementation ✅
- [x] CLI interface placeholder
- [x] Configuration structure  
- [x] Metadata handling
- [x] Error checking and validation

### Phase 2: AutoGPTQ Integration 🚧
- [ ] AutoGPTQ library integration
- [ ] Calibration data preprocessing
- [ ] Model loading and quantization
- [ ] Safetensors saving support
- [ ] Progress tracking integration

### Phase 3: Advanced Features 📋
- [ ] Dynamic group size selection
- [ ] Mixed-precision support (per-layer bits)
- [ ] Hardware-specific optimizations
- [ ] Batch quantization support

### Phase 4: Evaluation Integration 📋  
- [ ] Automatic model detection
- [ ] Inference speed benchmarking
- [ ] Accuracy comparison with other methods
- [ ] Memory profiling

## Comparison with Other Methods

| Aspect | GPTQ | AdaRound | QLoRA | AWQ |
|--------|------|----------|-------|-----|
| **Type** | PTQ | PTQ | Training-time | PTQ |
| **Accuracy** | High | Medium-High | High | High |
| **Speed** | Fast inference | Medium | Training focus | Very fast |
| **Memory** | Low | Low | Medium | Low |
| **Calibration** | Required | Required | N/A | Required |
| **Hardware Support** | Good | Limited | Excellent | Good |

## Best Practices (Planned)

### When to Use GPTQ
- ✅ Production inference optimization
- ✅ Large models (7B+) requiring compression
- ✅ Hardware with GPTQ kernel support
- ✅ Accuracy is important but some degradation acceptable

### When to Avoid
- ❌ Training-time quantization needs (use QLoRA)
- ❌ Extreme accuracy requirements
- ❌ Very small models (<1B parameters)
- ❌ Frequent model updates (quantization overhead)

### Optimization Guidelines

#### Calibration Data Selection
```python
# Recommended calibration approach
calibration_strategies = {
    "general": "Use diverse text from target domain",
    "math": "Include mathematical problems and reasoning",
    "code": "Use code samples and programming tasks",
    "chat": "Include conversational examples"
}
```

#### Group Size Selection
```python
group_size_recommendations = {
    128: "Good balance of accuracy and compression",
    64: "Better accuracy, slightly less compression", 
    -1: "Per-channel (best accuracy, less compression)"
}
```

## Troubleshooting Guide

### Common Issues (Anticipated)

#### Installation Problems
```bash
# CUDA compatibility issues
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# Triton compilation issues  
pip install triton --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu118/
```

#### Memory Issues
```python
# Reduce memory usage during quantization
quantize_config.damp_percent = 0.1  # Higher damping
quantize_config.batch_size = 1      # Smaller batches
```

#### Accuracy Degradation
```python
# Improve accuracy
quantize_config.bits = 8            # Use 8-bit instead of 4-bit
quantize_config.group_size = 64     # Smaller groups
quantize_config.desc_act = True     # Better activation handling
```

## Research Background

### Original Contributions
- **Optimal Brain Compression**: Applies pruning insights to quantization
- **Second-order Optimization**: Uses Hessian information for better quantization
- **Sequential Processing**: Layer-by-layer approach prevents error accumulation

### Key Publications
- Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Follows work on Optimal Brain Damage/Surgeon for neural network compression

## Implementation Examples

### Basic Usage (Planned)
```python
# Simple quantization
python tools/quantize.py run \
  --method gptq \
  --src Models/Qwen3-0.6B-base \
  --dst Models/Qwen3-0.6B-base-gptq \
  --bits 4 --group-size 128

# With custom calibration
python tools/quantize.py run \
  --method gptq \
  --src Models/Qwen3-0.6B-openmath \
  --dst Models/Qwen3-0.6B-openmath-gptq \
  --bits 4 --group-size 64 \
  --calib Datasets/calibration_openmath_128samples.txt
```

### Advanced Configuration (Planned)
```python
# High accuracy settings
python tools/quantize.py run \
  --method gptq \
  --bits 8 --group-size 64 \
  --symmetric False \
  --desc-act True
  
# Fast inference settings  
python tools/quantize.py run \
  --method gptq \
  --bits 4 --group-size 128 \
  --static-groups True
```

## Future Enhancements

### Near-term Goals
1. **Complete AutoGPTQ Integration**: Full implementation with progress tracking
2. **Kernel Optimization**: Leverage optimized inference kernels
3. **Batch Processing**: Support quantizing multiple models
4. **Validation Suite**: Automated accuracy testing

### Long-term Vision
1. **Mixed Precision**: Per-layer bit allocation
2. **Dynamic Quantization**: Runtime quantization adjustment
3. **Custom Kernels**: Optimized kernels for specific hardware
4. **Integration**: Seamless integration with deployment frameworks

## Contributing

To implement GPTQ support:

1. **Install Dependencies**: `pip install auto-gptq`
2. **Implement Core Function**: Replace stub in `tools/quantize.py`
3. **Add Progress Tracking**: Follow AdaRound pattern
4. **Test Integration**: Ensure evaluation pipeline compatibility
5. **Documentation**: Update this file with actual implementation details

## Status Summary

- **Infrastructure**: ✅ Complete (CLI, config, metadata)
- **Core Implementation**: 🚧 In Progress (requires AutoGPTQ integration)  
- **Testing**: ❌ Pending implementation
- **Documentation**: ✅ Complete (this document)
- **Evaluation Integration**: ✅ Ready (automatic detection implemented)