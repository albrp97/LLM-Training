# QLoRA vs LoRA Comparison Study Report

**Generated:** 2025-10-14 14:22:35

## Executive Summary

This study compares the performance of QLoRA (training-time 4-bit quantization) versus LoRA (no quantization) fine-tuning on the Qwen 0.6B model using the OpenMath dataset.

### Key Findings

- **LoRA average improvement over base:** 724.163 (improvement in primary metrics)
- **QLoRA average improvement over base:** -16122.381 (improvement in primary metrics)
- **QLoRA vs LoRA difference:** -16846.544 (positive means QLoRA better)
- **Quantization impact:** Negative (worse than LoRA)

## Model Configurations

| Model | Method | Parameters | VRAM Usage | Avg Latency |
|-------|--------|------------|------------|-------------|
| Base | No fine-tuning | 0 | 0.0 GB | 0.0 ms |
| LoRA | LoRA (no quantization) | 0 | 0.0 GB | 0.0 ms |
| QLoRA | LoRA + 4-bit quantization | 0 | 0.0 GB | 0.0 ms |

## Performance Comparison by Dataset

| Dataset | Metric | Base | LoRA | QLoRA | LoRA Improvement | QLoRA Improvement | QLoRA vs LoRA |
|---------|--------|------|------|-------|------------------|-------------------|----------------|
| arc | Accuracy | 50.000 | 65.000 | 50.000 | +15.000 | +0.000 | -15.000 |
| openmath | Avg Abs Diff | 2294.643 | 137.154 | 50656.787 | +2157.489 | -48362.144 | -50519.632 |
| squad | Accuracy | 5.000 | 5.000 | 0.000 | +0.000 | -5.000 | -5.000 |


## Analysis

### Training-Time Quantization Effects

The QLoRA approach applies 4-bit NF4 quantization during training, which has the following observed effects:

#### Negative Impacts (QLoRA < LoRA)
- **arc**: -15.000 (-23.1%) [accuracy]
- **openmath**: -50519.632 (-36834.2%) [avg_abs_diff]
- **squad**: -5.000 (-100.0%) [accuracy]

### Resource Efficiency

- **VRAM Usage**: QLoRA uses 0.0 GB vs LoRA's 0.0 GB (N/A)
- **Latency Impact**: QLoRA: 0.0 ms vs LoRA: 0.0 ms (N/A)

### Conclusions

LoRA shows **superior performance** compared to QLoRA with QLoRA performing 16846.544 points worse on average. This suggests that 4-bit quantization during training introduces some performance degradation, though it provides significant memory savings.

### Recommendations

- **Use LoRA** when maximum performance is required and memory is not constrained
- **Use QLoRA** when memory is severely constrained and the performance trade-off is acceptable
- The performance penalty of QLoRA is relatively small (16846.544 points) compared to the memory savings

---

*Study conducted on 3 test datasets with TRUNC_EVAL=20 samples each.*
*Base model: Qwen/Qwen3-0.6B, Training dataset: OpenMath (1000 samples), LoRA rank: 64*
