# AutoWeightQuantization (AWQ) Method

## Overview

**AWQ (Activation-aware Weight Quantization)** is a sophisticated 4-bit post-training quantization method that preserves model accuracy by identifying and protecting salient weights during quantization. Unlike uniform quantization approaches, AWQ analyzes activation patterns to determine which weights are most critical for maintaining model performance.

## Purpose and Use Cases

### Core Innovation
AWQ's key insight is that not all weights contribute equally to model performance. By analyzing activation magnitudes across calibration data, AWQ identifies "salient" weights that should remain at higher precision while aggressively quantizing less important weights.

### When to Use AWQ
- ✅ **Production Inference**: Optimized for deployment scenarios
- ✅ **Accuracy-Critical Applications**: Best balance of compression and accuracy
- ✅ **GPU Inference**: Designed for CUDA optimization
- ✅ **Large Model Deployment**: Particularly effective on 7B+ models
- ✅ **Batch Processing**: Excellent throughput with proper kernels

## Implementation Status in Our Codebase

### Current Status: **PLANNED IMPLEMENTATION**

Our AWQ implementation is currently in the planning phase. The infrastructure is prepared in [`quantization_utils.py`](../quantization_utils.py) with placeholder implementation in [`tools/quantize.py`](../tools/quantize.py).

### File Structure
- **Configuration Support**: [`quantization_utils.py`](../quantization_utils.py) - AWQ enum and metadata
- **Implementation Target**: [`tools/quantize.py`](../tools/quantize.py) - `quantize_with_awq()` function
- **Training Integration**: [`Fine-tuning/01_Train.py`](../Fine-tuning/01_Train.py) - PTQ configuration
- **Evaluation**: [`Testing/02_TestModels.py`](../Testing/02_TestModels.py) - Automatic detection

### Planned Architecture

#### Target Implementation
```python
def quantize_with_awq(
    model_path: str,
    output_path: str,
    calibration_dataset: str,
    bits: int = 4,
    group_size: int = 128,
    salient_ratio: float = 0.01,
    calib_samples: int = 128
) -> str:
    """
    AWQ quantization implementation
    
    Args:
        model_path: Path to model to quantize
        output_path: Where to save quantized model
        calibration_dataset: Calibration data path
        bits: Weight quantization bits (4)
        group_size: Quantization group size (128)
        salient_ratio: Fraction of weights to keep salient
        calib_samples: Calibration samples for analysis
    """
    # Implementation planned
```

## AWQ Algorithm Principles

### Activation Analysis Phase
1. **Calibration Forward Passes**: Run model on calibration data
2. **Activation Recording**: Capture activation magnitudes per layer
3. **Salience Scoring**: Compute weight importance based on activations
4. **Threshold Selection**: Identify top percentile of salient weights

### Quantization Strategy
```python
# Planned AWQ workflow
def awq_quantization_workflow():
    # 1. Analyze activation patterns
    salience_scores = compute_activation_salience(model, calibration_data)
    
    # 2. Identify salient weights (typically top 1%)
    salient_mask = select_salient_weights(salience_scores, ratio=0.01)
    
    # 3. Mixed precision quantization
    for layer in model.layers:
        if layer.name in salient_mask:
            # Keep salient weights at higher precision
            layer.weights = quantize_weights(layer.weights, bits=8)  # or keep FP16
        else:
            # Aggressive quantization for non-salient weights
            layer.weights = quantize_weights(layer.weights, bits=4)
```

### Key Innovation: Per-Channel Scaling
AWQ uses per-channel scaling factors that are optimized based on activation patterns rather than just weight statistics.

## Integration with Our Training Pipeline

### Configuration Setup
```python
# In Fine-tuning/01_Train.py
QUANT_METHOD = "AWQ"
AWQ_BITS = 4
AWQ_GROUP_SIZE = 128
AWQ_SALIENT_RATIO = 0.01

# Automatic calibration data generation
if QUANT_METHOD == "AWQ":
    create_calibration_data(
        model_name=MODEL_NAME,
        dataset_name=DATASET_CHOICE,
        output_path=f"Datasets/calibration_prompts.txt",
        num_samples=128
    )
```

### Quantization Specification
```python
def resolve_quantization_spec(method: QuantMethod) -> QuantizationSpec:
    if method is QuantMethod.AWQ:
        return QuantizationSpec(
            method=method,
            weights_bits=4,              # Primary quantization
            activations_bits=16,         # Full precision activations
            kv_cache_bits=16,           # Full precision cache
            group_size=128,             # Standard AWQ group size
            lm_head_dtype="bf16",       # Keep head at full precision
            backend="autoawq",          # Planned backend
            salient_ratio=0.01,         # 1% salient weights
            mixed_precision=True        # Enable mixed precision
        )
```

## Model Organization and Naming

### Naming Convention
```
{base_model}-{dataset}_{method}_{peft}_{quantization_tag}
```

**Example**: `Qwen3-0.6B-openmath_SFT_LoRa256_AWQ_w4g128_s1pct_headbf16`

### Tag Structure Breakdown
- **Method**: `AWQ`
- **Weights**: `w4` (4-bit weights)
- **Group Size**: `g128` (128 group size)
- **Salient**: `s1pct` (1% salient ratio)
- **Head**: `headbf16` (language model head in bfloat16)

### Directory Structure (Planned)
```
Models/
├── Qwen3-0.6B-openmath_SFT_LoRa256_AWQ_w4g128_s1pct_headbf16/
│   ├── model.safetensors              # Quantized weights
│   ├── config.json                    # AWQ-specific config
│   ├── quantization_config.json       # Salience maps, scales
│   ├── training_metadata.json         # Full quantization metadata
│   └── tokenizer files
```

## Performance Characteristics

### Expected Performance Profile

#### Memory Usage
- **Weights**: ~4 bits per parameter (with salient weight overhead)
- **Activations**: Full precision (FP16/BF16)
- **KV Cache**: Full precision
- **Overhead**: Scaling factors and salience metadata (~5% additional)

#### Accuracy Retention
Based on AWQ paper results:
| Model Size | Baseline Accuracy | AWQ Accuracy | Retention Rate |
|------------|-------------------|--------------|----------------|
| 7B LLaMA | 100% | ~98.5% | 98.5% |
| 13B LLaMA | 100% | ~99.0% | 99.0% |
| 30B+ | 100% | ~99.2% | 99.2% |

#### Inference Speed
- **Standard Implementation**: 1.2-1.5x faster than FP16
- **Optimized Kernels**: Up to 3x faster with AWQ-optimized CUDA kernels
- **Memory Bandwidth**: ~75% reduction in weight transfer

### Hardware Requirements
```python
# Minimum requirements for AWQ quantization
min_requirements = {
    "vram_gb": 8,           # For quantization process
    "cuda_compute": "7.5",  # For optimized kernels
    "disk_space_gb": 10,    # Temporary storage during quantization
    "calibration_time": "30min"  # For 7B model
}
```

## Implementation Dependencies

### Required Libraries (Planned)
```python
# Planned dependency stack
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0", 
    "autoawq>=0.1.0",       # Main AWQ library
    "awq-inference",        # Optimized inference kernels
    "accelerate>=0.20.0"    # Multi-GPU support
]
```

### Installation Commands
```bash
# Planned installation (when implemented)
pip install autoawq
pip install awq-inference  # For optimized kernels
```

## Calibration Data Requirements

### Optimal Calibration Strategy
AWQ requires high-quality calibration data that represents the target use case:

```python
# Calibration data generation (already implemented)
def create_awq_calibration_data():
    """Generate calibration data optimized for AWQ analysis"""
    
    # Target characteristics for AWQ:
    samples = {
        "diversity": "High diversity in prompts and responses",
        "length": "Variable length (128-1024 tokens)",
        "domain": "Representative of inference workload", 
        "size": "128-512 samples (diminishing returns beyond)"
    }
    
    return generate_calibration_prompts(
        dataset=DATASET_CHOICE,
        num_samples=128,
        max_length=512,
        diversity_sampling=True
    )
```

### Data Quality Impact
| Calibration Quality | Expected Accuracy | Notes |
|-------------------|-------------------|-------|
| **High Quality** | 98.5%+ retention | Domain-matched, diverse |
| **Generic** | 97%+ retention | Standard datasets |
| **Poor Quality** | 95%+ retention | Limited diversity |

## Algorithm Details

### Salient Weight Detection
```python
# Planned salient weight analysis
def compute_activation_salience(model, calibration_data):
    """
    Compute per-weight salience scores based on activation patterns
    """
    salience_scores = {}
    
    for batch in calibration_data:
        # Forward pass with activation hooks
        activations = record_activations(model, batch)
        
        for layer_name, activation in activations.items():
            # Compute salience metric (simplified)
            weight = model.get_parameter(layer_name + '.weight')
            
            # AWQ uses magnitude-based salience
            salience = torch.abs(activation).mean(dim=0)
            
            if layer_name not in salience_scores:
                salience_scores[layer_name] = salience
            else:
                salience_scores[layer_name] += salience
    
    # Normalize and return top percentile
    return select_top_salient_weights(salience_scores, ratio=0.01)
```

### Mixed Precision Strategy
```python
def apply_mixed_precision_quantization(model, salience_mask):
    """
    Apply different quantization levels based on salience
    """
    for layer_name, layer in model.named_modules():
        if hasattr(layer, 'weight'):
            if layer_name in salience_mask:
                # Salient weights: lighter quantization or keep FP16
                layer.weight = quantize_tensor(layer.weight, bits=8, method="symmetric")
            else:
                # Non-salient weights: aggressive 4-bit quantization
                layer.weight = quantize_tensor(layer.weight, bits=4, method="asymmetric")
```

## Integration with Evaluation Pipeline

### Automatic Detection
```python
# Testing/02_TestModels.py
def detect_quantization_method(model_path):
    """Detect AWQ quantization from model path"""
    if "AWQ" in model_path:
        return QuantMethod.AWQ
    
    # Also check config files for AWQ metadata
    config_path = os.path.join(model_path, "quantization_config.json")
    if os.path.exists(config_path):
        config = json.load(open(config_path))
        if config.get("quantization_method") == "awq":
            return QuantMethod.AWQ
```

### Performance Benchmarking
```python
# Planned AWQ evaluation metrics
def evaluate_awq_performance(model_path):
    """Comprehensive AWQ evaluation"""
    
    # Load quantized model
    model = load_awq_model(model_path)
    
    # Benchmark suite
    results = {
        "accuracy": evaluate_accuracy(model),
        "inference_speed": benchmark_inference_speed(model),
        "memory_usage": measure_memory_usage(model),
        "quantization_ratio": compute_compression_ratio(model),
        "kernel_optimization": test_optimized_kernels(model)
    }
    
    return results
```

## Comparison with Other Methods

### AWQ vs AdaRound vs GPTQ
| Method | **AWQ** | AdaRound | GPTQ |
|--------|---------|----------|------|
| **Approach** | Activation-aware | Rounding optimization | Hessian-based |
| **Accuracy** | 98.5%+ | 97%+ | 98%+ |
| **Speed** | 3x (optimized) | 1.1x | 1.5x |
| **Memory** | 75% reduction | 75% reduction | 75% reduction |
| **Setup Time** | Medium (30min) | Fast (10min) | Slow (60min) |
| **Hardware Req** | CUDA 7.5+ | Any | Any |

### Use Case Recommendations
| Scenario | Best Method | Rationale |
|----------|-------------|-----------|
| **Production Inference** | **AWQ** | Optimal speed/accuracy balance |
| **Quick Experimentation** | AdaRound | Fastest quantization |
| **Research/Analysis** | GPTQ | More thorough optimization |
| **Edge Deployment** | **AWQ** | Best kernel optimization |
| **Training Integration** | QLoRA | Only training-time option |

## Troubleshooting (Planned)

### Common Issues and Solutions

#### Quantization Failures
```python
# Planned error handling
def handle_awq_errors():
    common_errors = {
        "CUDA_OUT_OF_MEMORY": "Reduce calibration batch size",
        "SALIENT_DETECTION_FAILED": "Increase calibration sample diversity",
        "KERNEL_NOT_FOUND": "Install awq-inference package",
        "ACCURACY_DEGRADATION": "Increase salient_ratio or group_size"
    }
```

#### Performance Issues
```python
# Optimization troubleshooting
def optimize_awq_performance():
    optimizations = {
        "slow_inference": "Install AWQ CUDA kernels",
        "high_memory": "Enable gradient checkpointing during quantization",
        "accuracy_loss": "Improve calibration data quality",
        "quantization_slow": "Use smaller calibration dataset"
    }
```

### Debugging Tools
```python
# Planned debugging utilities
def debug_awq_quantization():
    """Debug AWQ quantization process"""
    
    # Salience analysis
    plot_salience_distribution(model, calibration_data)
    
    # Quantization impact per layer
    analyze_layer_accuracy_impact(model, test_data)
    
    # Memory and speed profiling
    profile_quantized_inference(model)
```

## Development Roadmap

### Phase 1: Basic Implementation (Planned)
- [ ] Integrate AutoAWQ library
- [ ] Implement `quantize_with_awq()` function
- [ ] Add progress tracking and error handling
- [ ] Test with small models (0.6B parameters)

### Phase 2: Optimization (Planned)
- [ ] CUDA kernel integration for inference speedup
- [ ] Multi-GPU quantization support
- [ ] Advanced calibration strategies
- [ ] Batch quantization for multiple models

### Phase 3: Advanced Features (Future)
- [ ] Dynamic salient ratio based on layer analysis
- [ ] Mixed-bit quantization (2/4/8 bit combinations)
- [ ] Online calibration during inference
- [ ] AWQ-specific PEFT integration

## Research Integration

### Experimental Design
```python
# Planned AWQ experiments
awq_experiments = {
    "salient_ratio_sweep": [0.005, 0.01, 0.02, 0.05],  # Test different ratios
    "group_size_analysis": [32, 64, 128, 256],         # Optimal group sizes
    "calibration_size": [64, 128, 256, 512],           # Sample efficiency
    "mixed_precision": ["4bit", "4+8bit", "4+16bit"]   # Precision combinations
}
```

### Metrics Collection
```python
def comprehensive_awq_analysis():
    """Collect comprehensive AWQ performance data"""
    
    metrics = {
        "accuracy_retention": measure_accuracy_retention(),
        "inference_latency": benchmark_inference_speed(),
        "memory_reduction": calculate_memory_savings(),
        "throughput_improvement": measure_batch_throughput(),
        "salience_effectiveness": analyze_salience_correlation(),
        "calibration_efficiency": test_calibration_sample_sizes()
    }
    
    return generate_awq_research_report(metrics)
```

## Best Practices (When Implemented)

### Optimal Configuration
```python
# Recommended AWQ settings
optimal_awq_config = {
    "bits": 4,                    # Standard AWQ quantization
    "group_size": 128,           # Balanced accuracy/efficiency
    "salient_ratio": 0.01,       # 1% salient weights (standard)
    "calibration_samples": 128,  # Sufficient for good salience detection
    "batch_size": 1,            # During quantization (memory constraint)
    "max_length": 512,          # For calibration sequences
}
```

### Quality Assurance
```python
# AWQ validation pipeline
def validate_awq_quantization(original_model, quantized_model):
    """Comprehensive validation of AWQ quantization"""
    
    checks = {
        "accuracy_threshold": 0.97,      # Minimum 97% accuracy retention
        "memory_reduction": 0.7,         # At least 70% memory reduction  
        "inference_speedup": 1.2,        # Minimum 20% speed improvement
        "salience_preservation": 0.99,   # 99% salient weight preservation
    }
    
    return run_validation_suite(original_model, quantized_model, checks)
```

AWQ represents the cutting edge of activation-aware quantization and will provide our platform with state-of-the-art efficiency while maintaining high accuracy. The implementation is prioritized for production deployment scenarios where optimal inference performance is required.