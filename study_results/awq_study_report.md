# AWQ Post-Training Quantization Study Report

**Generated:** 2025-10-14 17:34:09

## Executive Summary

This study analyzes the effects of AWQ (Activation-aware Weight Quantization) post-training quantization on the Qwen 0.6B model, comparing both base models and fine-tuned models across multiple test datasets.

### Key Findings

**Base Model Impact:**
- **Average AWQ improvement:** -2.500 (across all metrics)
- **Positive impacts:** 1/2 datasets
- **Overall impact:** Negative

**Fine-tuned Model Impact:**
- **Average AWQ improvement:** 15.000 (across all metrics)
- **Positive impacts:** 1/2 datasets
- **Overall impact:** Positive

## Model Configurations

| Model | Quantization | Parameters | VRAM Usage | Description |
|-------|--------------|------------|------------|-------------|
| Base | None | 0 | 1.4 GB | Original base model |
| Base AWQ | 4-bit W4G128 | 0 | 1.4 GB | Base + AWQ quantization |
| NoPeft | None | 0 | 1.3 GB | OpenMath fine-tuned |
| NoPeft AWQ | 4-bit W4G128 | 0 | 1.4 GB | Fine-tuned + AWQ quantization |

## Performance Analysis

### Base Model Quantization Effects

| Dataset | Metric | No Quant | AWQ | Improvement | Impact |
|---------|--------|----------|-----|-------------|---------|
| arc | Accuracy | 50.000 | 0.000 | -50.000 | ❌ |
| squad | Accuracy | 5.000 | 50.000 | +45.000 | ✅ |

### Fine-tuned Model Quantization Effects

| Dataset | Metric | No Quant | AWQ | Improvement | Impact |
|---------|--------|----------|-----|-------------|---------|
| arc | Accuracy | 15.000 | 0.000 | -15.000 | ❌ |
| squad | Accuracy | 5.000 | 50.000 | +45.000 | ✅ |

## Analysis

### AWQ Algorithm Performance

AWQ (Activation-aware Weight Quantization) uses calibration data to compute optimal per-channel scaling factors that preserve important activations while quantizing weights to 4-bit precision.

**Observed Effects:**

#### Base Model Results

**Positive Impacts:**
- **squad**: +45.000 (+900.0%) [accuracy]

**Negative Impacts:**
- **arc**: -50.000 (-100.0%) [accuracy]

#### Fine-tuned Model Results

**Positive Impacts:**
- **squad**: +45.000 (+900.0%) [accuracy]

**Negative Impacts:**
- **arc**: -15.000 (-100.0%) [accuracy]

### Resource Efficiency

**VRAM Usage:**
- **Base Model**: 1.4 GB (AWQ) vs 1.4 GB (No Quant) - -0.1 GB saved
- **Fine-tuned Model**: 1.4 GB (AWQ) vs 1.3 GB (No Quant) - -0.2 GB saved

### Conclusions

⚖️ **AWQ shows mixed results**, performing better on some model types. Consider model-specific evaluation for deployment decisions.

### Recommendations

- **Use AWQ** for fine-tuned models where it shows benefits
- **Consider alternatives** for base models

---

*Study conducted on 2 test datasets with TRUNC_EVAL=20 samples each.*
*Base model: Qwen/Qwen3-0.6B, Fine-tuning: OpenMath (1000 samples), AWQ config: 4-bit weights, group size 128*
*Calibration: N/A samples from OpenMath dataset*
