# Quick Start Guide

This guide helps you reproduce our quantization and scaling experiments or run your own.

## Prerequisites

- Python 3.12+
- CUDA 12.9+ compatible GPU
- Sufficient VRAM for your target model (see Hardware Requirements below)

## Installation

```bash
# Clone the repository
git clone https://github.com/albrp97/LLM-Training.git
cd LLM-Training

# Install dependencies using uv
uv sync
```

## Hardware Requirements by Model

| Model Config | Training VRAM | Inference VRAM | Suitable GPUs |
|--------------|---------------|----------------|---------------|
| 0.6B-1.7B | 4-8 GB | 3-4 GB | Most modern GPUs |
| 4B Full FT | 32 GB | 8 GB | RTX 4090/5090, A100 |
| 8B LoRA/DoRA | 20 GB | 16 GB | RTX 4090/5090, A6000 |
| 8B QLoRA ✨ | **8 GB** | 6 GB | RTX 4060 Ti, 4070+ |
| 8B VeRA | 16 GB | 16 GB | RTX 4090/5090 |

**Key Insight**: QLoRA enables 8B training on mid-range consumer GPUs!

## Quick Experiments

### Option 1: Automated Full Pipeline

Run all experiments from Documentation/03 (4B and 8B with quantization):

```bash
uv run python experiments/run_quantization_experiments.py
```

This automatically:
1. ✅ Creates Int8 and 4-bit quantized variants of 4B models
2. ✅ Trains 8B with QLoRA on OpenMath dataset
3. ✅ Evaluates all models on test benchmarks
4. ✅ Generates comparison metrics

**Time**: 2-4 hours (mostly automated)
**Output**: `Testing/metrics_summary.csv` with all results

### Option 2: Train a Specific Configuration

Edit `Fine-tuning/01_Train.py` to configure:

Edit `Fine-tuning/01_Train.py` to configure:

```python
# Example: 4B Full Fine-Tuning (Best In-Domain Performance)
train = True
DATASET_CHOICE = "openmath"
MODEL_NAME = "Qwen/Qwen3-4B"
PEFT_CONFIG = "NoPeft"      # Full fine-tuning
QUANT_METHOD = "NoQuant"    # No quantization
num_train_epochs = 1
```

Or for consumer GPU with 8B model:

```python
# Example: 8B QLoRA (Best for <12GB VRAM)
train = True
DATASET_CHOICE = "openmath"
MODEL_NAME = "Qwen/Qwen3-8B"
PEFT_CONFIG = "LoRa"        # Parameter-efficient
QUANT_METHOD = "QLORA"      # 4-bit quantization
lora_r = 256
num_train_epochs = 1
```

Then run:
```bash
uv run python Fine-tuning/01_Train.py
```

### Option 3: Evaluate Existing Models

```bash
# Evaluate all untested models
uv run python Testing/03_EvaluationOrchestrator.py

# Evaluate a specific model
uv run python Testing/02_TestModels.py --model-path "Models/Qwen3-4B-openmath_SFT_NoPeft_NoQuant"

# Compare all results
# Open and run: Testing/04_CompareMetrics.ipynb
```

## Recommended Configurations by Goal

### Goal: Best Overall Performance (In-Domain + OOD)
**→ 4B Full FT (NoPeft, NoQuant)**
- Training VRAM: 32 GB (requires RTX 4090+ or workstation GPU)
- Math accuracy: ★★★★★ (AbsDiff 9.0)
- Reasoning (ARC): ★★★★☆ (F1 0.826)
- QA (SQuAD): ★★★☆☆ (F1 32.9)
- Inference: Fast (0.61s on math), low VRAM (7.9 GB)

### Goal: Best OOD Reasoning on Consumer GPU
**→ 8B DoRA (NoQuant) or 8B QLoRA**
- 8B DoRA: 20 GB training VRAM
- 8B QLoRA: **8 GB training VRAM** ✨
- Math accuracy: ★★☆☆☆ (poor, AbsDiff 117-136)
- Reasoning (ARC): ★★★★★ (F1 0.88-0.90)
- QA (SQuAD): ★★★★☆ (F1 40-49)

### Goal: Maximum Training Efficiency
**→ 8B VeRA**
- Training VRAM: 16 GB (30-41% less than LoRA/DoRA)
- Quality: Slight drop vs LoRA/DoRA
- Use when: Training VRAM is constrained but you need 8B

### Goal: Extreme Inference Memory Constraints
**→ 4B-4bit PTQ (Post-Training Quantization)**
- Inference VRAM: 2.8 GB (63% reduction)
- **Warning**: 5× slower, major accuracy loss on fine-tuned models
- Only use with base (non-fine-tuned) models for extreme edge deployment

## Step-by-Step: Reproducing Key Results

### Experiment 1: Compare Full FT vs LoRA (Phase 1)
```bash
# Train 0.6B with Full FT
# Edit Fine-tuning/01_Train.py:
# MODEL_NAME = "Qwen/Qwen3-0.6B", PEFT_CONFIG = "NoPeft"
uv run python Fine-tuning/01_Train.py

# Train 0.6B with LoRA r=256
# Edit: PEFT_CONFIG = "LoRa", lora_r = 256
uv run python Fine-tuning/01_Train.py

# Evaluate both
uv run python Testing/03_EvaluationOrchestrator.py
```

### Experiment 2: Compare PEFT Methods (Phase 2)
```bash
# Train with LoRA, DoRA, VeRA on same dataset
# Edit Fine-tuning/01_Train.py for each:
# PEFT_CONFIG = "LoRa"  (r=256)
# PEFT_CONFIG = "DoRa"  (r=256)
# PEFT_CONFIG = "VeRa"  (r=512)

# Run training for each configuration
uv run python Fine-tuning/01_Train.py

# Evaluate all
uv run python Testing/03_EvaluationOrchestrator.py
```

### Experiment 3: Scale to 4B and 8B with Quantization (Phase 3)
```bash
# Automated pipeline for all 4B/8B configurations
uv run python experiments/run_quantization_experiments.py
```

## Understanding the Results

### Metrics Files

After evaluation, find results in:
- `Testing/metrics/<model-name>.json` - Detailed per-model results
- `Testing/metrics_summary.csv` - Comparison table across all models

### Key Metrics Explained

| Metric | Description | Lower is Better? |
|--------|-------------|------------------|
| **ai2_arc__macro_f1** | Reasoning accuracy (ARC dataset) | Higher ✓ |
| **OpenMathInstruct-2__avg_abs_diff** | Math error (AbsDiff) | Lower ✓ |
| **squad_v2__F1** | Extractive QA accuracy | Higher ✓ |
| **latency_mean_s** | Inference speed per sample | Lower ✓ |
| **train_peak_vram_allocated_gb** | Training memory | Lower ✓ |
| **eval_peak_vram_allocated_gb** | Inference memory | Lower ✓ |

### Expected Results (from our experiments)

**Best In-Domain (OpenMath)**:
- 4B-SFT: AbsDiff **9.0** (winner 🏆)
- 8B models: AbsDiff 117-146 (worse despite larger size)

**Best OOD Reasoning (ARC)**:
- 8B-DoRA/LoRA: F1 **0.8995** (winner 🏆)
- 4B-SFT: F1 0.8259 (good)

**Best Training Efficiency**:
- 8B-QLoRA: **7.6 GB** training VRAM (winner 🏆)
- 8B-VeRA: 16.2 GB
- 8B-LoRA: 20.2 GB

## Troubleshooting

### CUDA Out of Memory (OOM)

1. **During Training**:
   - Use QLoRA instead of LoRA (`QUANT_METHOD = "QLORA"`)
   - Use VeRA instead of LoRA (`PEFT_CONFIG = "VeRa"`)
   - Reduce batch size or model size

2. **During Evaluation**:
   - Load model with quantization (edit `Testing/02_TestModels.py`)
   - Evaluate one dataset at a time
   - Close other GPU applications

### Slow Training

- Expected: 30-60 minutes for 1 epoch on 1,000 samples (8B QLoRA)
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is properly configured

### Import Errors

```bash
# If you see: ModuleNotFoundError: No module named 'utils'
# The imports were updated - make sure you have latest code
# Or manually add to PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
```

## Next Steps

1. **Read the research papers** in `Documentation/` for detailed analysis
2. **Experiment with configurations** - try different PEFT methods and datasets
3. **Analyze your results** using `Testing/04_CompareMetrics.ipynb`
4. **Check the tools/** for advanced quantization experiments

## Citation

If you use this work, please reference:
- Documentation/03_SFT-Quantization-Adapters.tex (latest, comprehensive)
- Documentation/02_Adapters-Lora_Vera_Dora.tex (PEFT methods)
- Documentation/01_Finetuning_ModelSize_LoraR.md (initial findings)

---

**Questions?** Check the main [README.md](README.md) or open an issue.
