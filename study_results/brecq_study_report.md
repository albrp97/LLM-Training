# BRECQ (Block-wise Reconstruction) Post-Training Quantization Study

**Generated**: 2025-10-15 11:50:11

## Executive Summary

This report analyzes the effectiveness of BRECQ (Block-wise Reconstruction-based Quantization) on the Qwen3-0.6B model across both base and fine-tuned variants. BRECQ uses mixed precision quantization: W4 for MLP layers and W6 for attention layers with block-wise reconstruction to minimize quantization error.

### Key Findings

- **Overall BRECQ Win Rate**: Base model 50.0%, Fine-tuned model 50.0%
- **Average Performance Impact**: Base +400.000%, Fine-tuned +400.000%
- **Speed Improvement**: Base +0.0%, Fine-tuned +0.0%
- **VRAM Reduction**: Base -4.3%, Fine-tuned -14.5%

## Model Configurations

### Base Models
- **No Quantization**: `Models/Qwen3-0.6B-base`
- **BRECQ W4/W6**: `Models/Qwen3-0.6B-base_brecq_w4g64_mix_attn6`

### Fine-tuned Models  
- **No Quantization**: `Models/Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant`
- **BRECQ W4/W6**: `Models/Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_brecq_w4g64_mix_attn6`

## Performance Analysis

### Base Model Results

| Dataset | No Quant | BRECQ W4/W6 | Change | Metric |
|---------|----------|-------------|---------|---------|
| test-ai2_arc.parquet | 50.000 | 0.000 | ❌ -100.00% | accuracy |
| test-squad_v2.parquet | 5.000 | 50.000 | ✅ +900.00% | accuracy |

**Base Model Summary**: 1/2 datasets improved (50.0% win rate)

### Fine-tuned Model Results

| Dataset | No Quant | BRECQ W4/W6 | Change | Metric |
|---------|----------|-------------|---------|---------|
| test-ai2_arc.parquet | 15.000 | 0.000 | ❌ -100.00% | accuracy |
| test-squad_v2.parquet | 5.000 | 50.000 | ✅ +900.00% | accuracy |

**Fine-tuned Model Summary**: 1/2 datasets improved (50.0% win rate)

## Resource Efficiency

### VRAM Usage
- **Base Model**: 1.38 GB → 1.43 GB (-4.3%)
- **Fine-tuned Model**: 1.25 GB → 1.43 GB (-14.5%)

### Inference Speed
- **Base Model Latency**: 0.00 ms → 0.00 ms (+0.0%)
- **Fine-tuned Model Latency**: 0.00 ms → 0.00 ms (+0.0%)

### Model Size
- **Parameters**: 0 → 0 (✅ Preserved)

## Technical Details

### BRECQ Configuration
- **MLP Layer Quantization**: W4 (4-bit weights)
- **Attention Layer Quantization**: W6 (6-bit weights) 
- **Mixed Precision**: Enabled
- **Group Size**: 64
- **Reconstruction Method**: Block-wise error minimization

### Quantization Method
BRECQ performs block-wise reconstruction with mixed precision support:
1. **Block-wise Processing**: Quantizes weights in blocks with local reconstruction
2. **Mixed Precision**: Uses W6 for attention layers (Q,K,V,O projections) and W4 for MLP layers
3. **Error Minimization**: Iterative refinement to reduce quantization error per block
4. **Calibration**: Uses sample data to optimize quantization parameters

## Conclusions

### Overall Effectiveness
✅ **BRECQ shows positive results** on both base and fine-tuned models with consistent accuracy improvements.

### Recommendations

⚠️ BRECQ shows **mixed results for base models** - evaluate per use case.

⚠️ BRECQ shows **mixed results for fine-tuned models** - evaluate per use case.

### Mixed Precision Impact

The mixed precision approach (W6 attention + W4 MLP) balances:
- **Attention Preservation**: Higher precision for critical attention computations
- **Efficiency**: Aggressive quantization for compute-heavy MLP layers
- **Quality**: Block-wise reconstruction minimizes quantization artifacts

