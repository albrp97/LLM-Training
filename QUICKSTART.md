# Quick Start: Quantization Experiments

## TL;DR - Run This Command

```powershell
python run_quantization_experiments.py
```

This will automatically:
1. ✅ Create int8 and 4-bit quantized variants of 4B base and nopeft models
2. ✅ Train 8B model with QLoRA on openmath dataset  
3. ✅ Evaluate all models on test benchmarks
4. ✅ Generate comparison metrics

## Manual Quick Steps

If you prefer to run steps individually:

### 1. Create Quantized Variants (~5-10 min)
```powershell
python train_quantization_variants.py
```

### 2. Train 8B QLoRA (~30-60 min depending on GPU)
```powershell
# Edit Fine-tuning/01_Train.py first:
# - MODEL_NAME = "Qwen/Qwen3-8B"
# - DATASET_CHOICE = "openmath"
# - QUANT_METHOD = "QLORA"
# - PEFT_CONFIG = "LoRa"

python Fine-tuning/01_Train.py
```

### 3. Evaluate All Models (~10-20 min per model)
```powershell
python Testing/03_EvaluationOrchestrator.py
```

### 4. Generate Comparison CSV
Open and run: `Testing/04_CompareMetrics.ipynb`

## Models That Will Be Created/Tested

| Model | Size | Type | Quantization |
|-------|------|------|--------------|
| Qwen3-4B-base | 4B | Base | None |
| Qwen3-4B-base_Int8_BnB | 4B | Base | Int8 |
| Qwen3-4B-base_4bit_BnB | 4B | Base | 4-bit |
| Qwen3-4B-openmath_SFT_NoPeft_NoQuant | 4B | Fine-tuned | None |
| Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB | 4B | Fine-tuned | Int8 |
| Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB | 4B | Fine-tuned | 4-bit |
| Qwen3-8B-base | 8B | Base | None |
| Qwen3-8B-openmath_SFT_LoRa256_QLoRA_4bit | 8B | QLoRA Trained | 4-bit |

## Expected Results Location

- Models: `Models/` directory
- Evaluation metrics: `Testing/metrics/` directory  
- Summary CSV: `Testing/metrics_summary.csv`

## Time Estimates

- Creating quantized variants: ~10 minutes
- Training 8B QLoRA: 30-60 minutes (depends on GPU)
- Evaluating each model: ~10-20 minutes
- **Total: 2-4 hours** (mostly automated)
