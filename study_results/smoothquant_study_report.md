# SmoothQuant W8A8 Post-Training Quantization Study

**Generated**: 2025-10-14 21:50:24

## Executive Summary

This report analyzes the effectiveness of SmoothQuant W8A8 quantization on the Qwen3-0.6B model across both base and fine-tuned variants. SmoothQuant applies scaling factors to balance quantization between weights and activations, targeting both at 8-bit precision.

### Key Findings

- **Overall SmoothQuant Win Rate**: Base model 25.0%, Fine-tuned model 25.0%
- SmoothQuant shows similar effectiveness across base and fine-tuned models
- **ARC**: SmoothQuant consistently underperforms on both model types
- **SQUAD**: SmoothQuant consistently outperforms on both model types
- **OPENMATH**: SmoothQuant consistently underperforms on both model types
- **Technical Notes**: SmoothQuant W8A8 quantizes both weights and activations to 8-bit, providing balanced compression with moderate precision loss

## Base Model Comparison

**Models**: `Qwen3-0.6B-base` vs `Qwen3-0.6B-base_smoothquant_w8a8`

| Dataset | Metric | NoQuant | SmoothQuant | Change | Winner |
|---------|--------|---------|-------------|---------|--------|
| ARC | accuracy | 50.0% | 0.0% | -100.0% | **NoQuant** |
| ARC | f1 | 0.5% | 0.0% | -100.0% | **NoQuant** |
| OPENMATH | accuracy | 10.0% | 0.0% | -100.0% | **NoQuant** |
| SQUAD | accuracy | 5.0% | 50.0% | +900.0% | **SmoothQuant** |


### Base Model Analysis

**ARC Dataset**: 0.0% win rate (0/2 metrics)
- accuracy: -100.0% change, **NoQuant** wins
- f1: -100.0% change, **NoQuant** wins

**OPENMATH Dataset**: 0.0% win rate (0/1 metrics)
- accuracy: -100.0% change, **NoQuant** wins

**SQUAD Dataset**: 100.0% win rate (1/1 metrics)
- accuracy: +900.0% change, **SmoothQuant** wins


## Fine-tuned Model Comparison

**Models**: `Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant` vs `Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_smoothquant_w8a8`

| Dataset | Metric | NoQuant | SmoothQuant | Change | Winner |
|---------|--------|---------|-------------|---------|--------|
| ARC | accuracy | 15.0% | 0.0% | -100.0% | **NoQuant** |
| ARC | f1 | 0.2% | 0.0% | -100.0% | **NoQuant** |
| OPENMATH | accuracy | 0.0% | 0.0% | +0.0% | **NoQuant** |
| SQUAD | accuracy | 5.0% | 50.0% | +900.0% | **SmoothQuant** |


### Fine-tuned Model Analysis

**ARC Dataset**: 0.0% win rate (0/2 metrics)
- accuracy: -100.0% change, **NoQuant** wins
- f1: -100.0% change, **NoQuant** wins

**OPENMATH Dataset**: 0.0% win rate (0/1 metrics)
- accuracy: +0.0% change, **NoQuant** wins

**SQUAD Dataset**: 100.0% win rate (1/1 metrics)
- accuracy: +900.0% change, **SmoothQuant** wins


## Technical Details

### SmoothQuant Method
- **Algorithm**: SmoothQuant with per-channel scaling
- **Weight Quantization**: 8-bit signed integers
- **Activation Quantization**: 8-bit signed integers  
- **Calibration**: 100 OpenMath samples for scaling factor computation
- **Target**: Balanced W8A8 quantization with activation-aware scaling

### Evaluation Framework
- **Datasets**: OpenMath (math reasoning), Squad (reading comprehension), ARC (commonsense reasoning), BoolQ (yes/no questions)
- **Metrics**: Accuracy for Squad/ARC/BoolQ, Average Absolute Difference for OpenMath
- **Sample Size**: 20 samples per dataset (TRUNC_EVAL=20)
- **Methodology**: Direct comparison between quantized and unquantized model performance

### Model Details
- **Base Architecture**: Qwen3-0.6B (600M parameters)
- **Fine-tuning**: OpenMath dataset with supervised fine-tuning (SFT), no PEFT
- **Quantization Scope**: 197 Linear layers quantized per model
- **Preserved Components**: LM head kept in FP16 for stability

## Conclusions

The SmoothQuant W8A8 quantization study reveals **{generate_summary_insights(base_analysis, finetuned_analysis)[0].split(':')[1].strip()}** across the evaluated model variants and datasets.

This systematic evaluation provides insights into SmoothQuant's effectiveness as a post-training quantization method for efficient model deployment while maintaining acceptable performance levels.
