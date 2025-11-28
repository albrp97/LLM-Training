# Quantization Experiments Guide

This guide walks you through testing different quantization methods on Qwen models.

## Overview

We'll compare the following models:
1. **Qwen2.5-1.5B-base** - Smaller base model (1.5B parameters)
2. **Qwen3-4B-base** - Base 4B model (no quantization)
3. **Qwen3-4B-base_Int8_BnB** - 4B base quantized to int8
4. **Qwen3-4B-base_4bit_BnB** - 4B base quantized to 4-bit
5. **Qwen3-4B-openmath_SFT_NoPeft_NoQuant** - 4B fine-tuned without PEFT
6. **Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB** - 4B fine-tuned quantized to int8
7. **Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB** - 4B fine-tuned quantized to 4-bit
8. **Qwen3-8B-base** - Base 8B model
9. **Qwen3-8B-openmath_SFT_LoRa256_QLoRA_4bit** - 8B trained with QLoRA

## Step-by-Step Instructions

### Option A: Run Everything Automatically

Run the master orchestration script:

```powershell
python run_quantization_experiments.py
```

This will:
1. Create all quantized variants
2. Train the 8B model with QLoRA
3. Evaluate all models
4. Generate comparison metrics

### Option B: Run Steps Manually

#### Step 1: Create 1.5B Base Model (Optional)

```powershell
python create_1.5b_base.py
```

#### Step 2: Create Quantized Variants

```powershell
python train_quantization_variants.py
```

This creates:
- `Qwen3-4B-base_Int8_BnB`
- `Qwen3-4B-base_4bit_BnB`
- `Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB`
- `Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB`

#### Step 3: Train 8B with QLoRA

Edit `Fine-tuning/01_Train.py` and set:
```python
train = True
DATASET_CHOICE = "openmath"
MODEL_NAME = "Qwen/Qwen3-8B"
PEFT_CONFIG = "LoRa"
QUANT_METHOD = "QLORA"
lora_r = 256
```

Then run:
```powershell
python Fine-tuning/01_Train.py
```

#### Step 4: Evaluate All Models

Evaluate each model individually:

```powershell
# 1.5B base
python Testing/02_TestModels.py Models/Qwen2.5-1.5B-base

# 4B variants
python Testing/02_TestModels.py Models/Qwen3-4B-base
python Testing/02_TestModels.py Models/Qwen3-4B-base_Int8_BnB
python Testing/02_TestModels.py Models/Qwen3-4B-base_4bit_BnB
python Testing/02_TestModels.py Models/Qwen3-4B-openmath_SFT_NoPeft_NoQuant
python Testing/02_TestModels.py Models/Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB
python Testing/02_TestModels.py Models/Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB

# 8B variants
python Testing/02_TestModels.py Models/Qwen3-8B-base
python Testing/02_TestModels.py Models/Qwen3-8B-openmath_SFT_LoRa256_QLoRA_4bit
```

Or use the orchestrator:
```powershell
python Testing/03_EvaluationOrchestrator.py
```

#### Step 5: Generate Comparison Metrics

Run the notebook:
```powershell
# Open in VS Code or Jupyter
Testing/04_CompareMetrics.ipynb
```

Or run it programmatically (if you have nbconvert):
```powershell
jupyter nbconvert --to notebook --execute Testing/04_CompareMetrics.ipynb
```

#### Step 6: Review Results

Check the generated CSV:
```powershell
cat Testing/metrics_summary.csv
```

## Expected Outputs

After running all steps, you should have:

### Models Directory Structure:
```
Models/
├── Qwen2.5-1.5B-base/
├── Qwen3-4B-base/
├── Qwen3-4B-base_Int8_BnB/
├── Qwen3-4B-base_4bit_BnB/
├── Qwen3-4B-openmath_SFT_NoPeft_NoQuant/
├── Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB/
├── Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB/
├── Qwen3-8B-base/
└── Qwen3-8B-openmath_SFT_LoRa256_QLoRA_4bit/
```

### Testing Directory Structure:
```
Testing/
├── metrics/
│   ├── Qwen2.5-1.5B-base.json
│   ├── Qwen3-4B-base.json
│   ├── Qwen3-4B-base_Int8_BnB.json
│   ├── Qwen3-4B-base_4bit_BnB.json
│   ├── Qwen3-4B-openmath_SFT_NoPeft_NoQuant.json
│   ├── Qwen3-4B-openmath_SFT_NoPeft_Int8_BnB.json
│   ├── Qwen3-4B-openmath_SFT_NoPeft_4bit_BnB.json
│   ├── Qwen3-8B-base.json
│   └── Qwen3-8B-openmath_SFT_LoRa256_QLoRA_4bit.json
└── metrics_summary.csv
```

## Key Metrics to Compare

The `metrics_summary.csv` will include:

1. **VRAM Usage:**
   - `train_peak_vram_reserved_gb`
   - `train_peak_vram_allocated_gb`
   - `eval_peak_vram_reserved_gb`
   - `eval_peak_vram_allocated_gb`

2. **Performance Metrics:**
   - `ai2_arc__macro_f1` - Reasoning capability
   - `OpenMathInstruct-2__avg_abs_diff` - Math problem solving (lower is better)
   - `squad_v2__F1` - Question answering capability

3. **Latency:**
   - `ai2_arc__latency_mean_s`
   - `OpenMathInstruct-2__latency_mean_s`
   - `squad_v2__latency_mean_s`

## What to Look For

1. **Memory Efficiency:** How much do int8 and 4-bit quantization reduce VRAM?
2. **Performance Impact:** Does quantization affect model accuracy?
3. **Training Efficiency:** How does QLoRA compare to full fine-tuning?
4. **Inference Speed:** Do quantized models run faster?

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in `01_Train.py`
- Use smaller models first
- Enable gradient checkpointing

### Model Not Found
- Ensure model was created/downloaded successfully
- Check Models directory exists
- Verify model name matches exactly

### Evaluation Fails
- Check that model has proper metadata files
- Ensure test datasets are in `Datasets/` folder
- Verify CUDA is available if using GPU

## Notes

- QLoRA (Quantized LoRA) allows training large models efficiently using 4-bit quantization
- BitsAndBytes provides both int8 and 4-bit quantization support
- Base models should already exist; we're just creating quantized variants
- The 8B QLoRA training may take significant time depending on your hardware
