# GPTQ Quantization Study Report

**Generated:** 2025-10-14 13:56:51
**Models Evaluated:** 4 variants of Qwen3-0.6B
**Evaluation Samples:** 60 per model (TRUNC_EVAL=20)
**Training Dataset:** OpenMathInstruct-2 (1000 samples)
**Quantization Method:** GPTQ W4G64 (4-bit weights, group size 64)

## Executive Summary
❌ **Training Impact**: -15.00% accuracy change
📊 **Base Model GPTQ**: 76.9% accuracy retention, 0.10x speed improvement
📊 **Trained Model GPTQ**: 249.9% accuracy retention, 0.00x speed improvement
🏆 **Best Model**: Base Original (21.67% accuracy)

## Model Comparison
| Model | Accuracy (%) | Avg Latency (s) | Tokens/Response | VRAM (GB) | Method |
|-------|-------------|-----------------|-----------------|-----------|---------|
| Base Original | 21.67 | 3.12 | 102.8 | 1.28 | NoQuant W16 |
| Base GPTQ | 16.67 | 29.69 | 967.6 | 1.32 | GPTQ W4 |
| Trained Original | 6.67 | 0.13 | 2.8 | 1.22 | NoQuant W16 |
| Trained GPTQ | 16.67 | 29.74 | 935.8 | 1.32 | GPTQ W4 |

## Performance Analysis

### Training Effectiveness
⚠️ Training showed minimal improvement (-15.00% change)
💡 Consider: Base model quantization may be sufficient

### GPTQ Quantization Impact

**Base Model Quantization:**
- Accuracy retention: 76.9%
- Speed improvement: 0.10x faster
- VRAM reduction: -0.04 GB
⚠️ Significant accuracy degradation

**Trained Model Quantization:**
- Accuracy retention: 249.9%
- Speed improvement: 0.00x faster
- VRAM reduction: -0.10 GB
✅ Excellent quantization quality

## Per-Dataset Performance
| Dataset | Base Original | Base GPTQ | Trained Original | Trained GPTQ |
|---------|---------------|-----------|------------------|--------------|
| ai2_arc | 50.00% | 0.00% | 15.00% | 0.00% |
| OpenMathInstruct-2 | 10.00% | 0.00% | 0.00% | 0.00% |
| squad_v2 | 5.00% | 50.00% | 5.00% | 50.00% |

## Recommendations
🎯 **Primary Recommendation**: Base model is sufficient (training provides minimal benefits)
🎯 **Quantization Recommendation**: Consider whether speed/memory benefits justify accuracy loss

## Technical Details
- **Base Model**: Qwen3-0.6B
- **Training Method**: Supervised Fine-Tuning (SFT) with no PEFT
- **Training Data**: OpenMathInstruct-2 dataset (1000 samples)
- **Quantization**: GPTQ with 4-bit weights, group size 64
- **Calibration**: 100 samples from training data
- **Evaluation**: 20 samples per test dataset (4 datasets total)
- **Hardware**: CUDA-enabled GPU