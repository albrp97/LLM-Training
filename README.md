# LLM Training

A comprehensive research repository exploring Large Language Model fine-tuning, parameter-efficient methods, and quantization techniques. This project systematically investigates the trade-offs between model size, training methods, quantization strategies, and hardware constraints to provide actionable guidance for deploying LLMs on consumer-grade hardware.

## Research Overview

This repository documents three major experimental phases:

1. **Phase 1: Model Size & LoRA Rank** - Comparing 0.6B and 1.7B models with various LoRA configurations vs full fine-tuning
2. **Phase 2: PEFT Methods** - Comprehensive comparison of LoRA, DoRA, and VeRA across tasks
3. **Phase 3: Scaling & Quantization** - Scaling to 4B and 8B models using quantization (QLoRA, PTQ) to overcome hardware limitations

### Key Findings

- **Full fine-tuning on smaller models** often outperforms PEFT on larger models for in-domain tasks
- **QLoRA enables 8B training** on consumer GPUs (<12GB VRAM) with 60%+ memory reduction
- **Post-training quantization** is a last resort—severe latency penalties and accuracy degradation on fine-tuned models
- **DoRA** offers the best OOD (out-of-distribution) performance among PEFT methods
- **Task type matters**: PEFT works better for numeracy/MCQ; full FT excels at extractive QA

## Core Techniques

### PEFT (Parameter-Efficient Fine-Tuning)
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with low-rank matrices injected into attention layers
- **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Separates magnitude and direction for better OOD generalization
- **VeRA (Vector-based Random Matrix Adaptation)**: Shared random bases for minimal trainable parameters
- **NoPeft**: Full supervised fine-tuning baseline for maximum in-domain accuracy

### Quantization Methods
- **QLoRA**: 4-bit NormalFloat (NF4) quantization with double quantization and paged optimizers
- **BitsAndBytes PTQ**: Post-training quantization in 4-bit and 8-bit (INT8) formats
- **AWQ/GPTQ/SmoothQuant**: Advanced quantization techniques (implementation placeholders in tools/)

### Training Methodology
- **SFT (Supervised Fine-Tuning)**: Structured training with chat-style templates
- **Multi-task Evaluation**: ARC (reasoning), OpenMath (numeracy), SQuAD v2 (extractive QA)
- **Comprehensive Metrics**: Accuracy, latency, VRAM (training & inference), F1 scores

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.9+ (for GPU support)
- 16GB+ RAM (recommended)


## Project Structure

### Core Directories

- **`Fine-tuning/`** - Production training scripts
  - `01_Train.py` - Main configurable training script with PEFT and quantization support

- **`Testing/`** - Model evaluation pipeline
  - `02_TestModels.py` - Comprehensive evaluation on ARC, SQuAD, OpenMath
  - `03_EvaluationOrchestrator.py` - Batch evaluation automation
  - `04_CompareMetrics.ipynb` - Cross-model comparison and visualization
  - `metrics/` - JSON evaluation results for all tested models

- **`experiments/`** - Experimental scripts and playground
  - `00_Playground.ipynb/py` - Sandbox for quick tests
  - `create_1.5b_base.py` - Base model snapshot creation with quantization variants
  - `run_quantization_experiments.py` - Master orchestration for quantization experiments
  - `train_8b_qlora.py` - Standalone QLoRA training script
  - `train_quantization_variants.py` - Create and compare quantized model variants
  - `QUANTIZATION_EXPERIMENTS.md` - Quantization experiments documentation

- **`utils/`** - Shared utility modules
  - `quantization_utils.py` - Unified quantization methods, specs, and metadata handling

- **`tools/`** - Command-line tools for quantization and analysis
  - `quantize.py` - Post-training quantization orchestration
  - `apply_smoothquant.py`, `compare_smoothquant.py` - SmoothQuant experiments
  - `base_model_analysis.py`, `detailed_comparison.py` - Model analysis utilities

- **`DataPreparation/`** - Data preprocessing notebooks
  - `1_PreprocessData.ipynb` - Raw data cleaning and formatting
  - `2_ReduceTestData.ipynb`, `3_ReduceTrainData.ipynb` - Dataset size optimization

- **`Documentation/`** - Research papers and analysis
  - `01_Finetuning_ModelSize_LoraR.md/.tex` - Phase 1: Model size and LoRA rank study
  - `02_Adapters-Lora_Vera_Dora.tex` - Phase 2: PEFT methods comparison
  - `03_SFT-Quantization-Adapters.tex` - Phase 3: Scaling with quantization

- **`docs/`** - Additional technical documentation
  - Quantization implementation summaries and method overviews

- **`Datasets/`** - Training and evaluation datasets (parquet format)
  - ARC, SQuAD v2, OpenMathInstruct-2

- **`Models/`** - Trained model outputs (organized by configuration)

## Tested Model Configurations

Our experiments cover **12 major configurations** across model sizes, PEFT methods, and quantization:

| Model | Size | Training | Quantization | Use Case |
|-------|------|----------|--------------|----------|
| Qwen3-1.7B-base | 1.7B | None | None | Baseline |
| Qwen3-4B-base | 4B | None | None/Int8/4bit | Baseline + PTQ variants |
| Qwen3-4B-SFT | 4B | Full FT | None/Int8/4bit | In-domain champion |
| Qwen3-8B-base | 8B | None | None | Quality baseline |
| Qwen3-8B-LoRA | 8B | LoRA (r=256) | None/QLoRA | OOD reasoning |
| Qwen3-8B-DoRA | 8B | DoRA (r=256) | None | Best OOD efficiency |
| Qwen3-8B-VeRA | 8B | VeRA (r=512) | None | Minimum training VRAM |
## Quick Start

See **[QUICKSTART.md](QUICKSTART.md)** for running the full quantization experiment pipeline.

### Basic Usage

1. **Train a model** (example: 4B with full fine-tuning):
```bash
# Edit Fine-tuning/01_Train.py configuration:
# - MODEL_NAME = "Qwen/Qwen3-4B"
# - DATASET_CHOICE = "openmath"
# - PEFT_CONFIG = "NoPeft"
# - QUANT_METHOD = "NoQuant"

uv run python Fine-tuning/01_Train.py
```

2. **Evaluate models**:
```bash
# Automatically evaluate all untested models
uv run python Testing/03_EvaluationOrchestrator.py

# Or evaluate a specific model
uv run python Testing/02_TestModels.py --model-path "Models/Qwen3-4B-openmath_SFT_NoPeft_NoQuant"
```

3. **Compare results**:
```bash
# Open and run Testing/04_CompareMetrics.ipynb
# Generates metrics_summary.csv with all model comparisons
```

## Hardware Requirements

Our experiments demonstrate fine-tuning at different VRAM scales:

- **0.6B-1.7B models**: ~4-8GB VRAM (entry-level GPUs)
- **4B models (full FT)**: ~32GB VRAM (high-end consumer/workstation)
- **8B models (LoRA/DoRA)**: ~20GB VRAM (RTX 4090/5090 class)
- **8B models (QLoRA)**: ~8GB VRAM (mid-range consumer GPUs) ✨

**Key insight**: QLoRA enables 8B training on <12GB GPUs with 62% less VRAM than standard LoRA.

## Research Documentation

Detailed analysis and findings are available in the **Documentation/** folder:

1. **[01_Finetuning_ModelSize_LoraR.md](Documentation/01_Finetuning_ModelSize_LoraR.md)**
   - Comparison of 0.6B vs 1.7B models
   - LoRA rank analysis (r=32 to r=1024)
   - Full FT vs PEFT trade-offs
   - **Key finding**: Full FT on small models beats LoRA on larger models for in-domain tasks

2. **[02_Adapters-Lora_Vera_Dora.tex](Documentation/02_Adapters-Lora_Vera_Dora.tex)**
   - Comprehensive PEFT methods comparison
   - LoRA vs DoRA vs VeRA analysis
   - Training VRAM and efficiency trade-offs
   - **Key finding**: VeRA for strict VRAM caps; LoRA/DoRA when feasible; Full FT when possible

3. **[03_SFT-Quantization-Adapters.tex](Documentation/03_SFT-Quantization-Adapters.tex)**
   - Scaling to 4B and 8B models
   - QLoRA implementation and analysis
   - Post-training quantization (PTQ) evaluation
   - **Key finding**: 4B-SFT is the best balanced choice; 8B-DoRA for OOD tasks; PTQ is last resort

## Performance Highlights

Based on comprehensive testing across 12 configurations:

### Best Configurations by Use Case

| Use Case | Recommended Config | Why |
|----------|-------------------|-----|
| **Balanced Performance** | 4B-SFT (NoQuant) | Best multi-task quality, low latency, moderate VRAM |
| **In-Domain Math/Numeric** | 4B-SFT (NoQuant) | 99.3% error reduction, 95.6% latency reduction vs base |
| **OOD Reasoning (ARC)** | 8B-DoRA (NoQuant) | Highest F1 (0.8995), best latency among 8B variants |
| **OOD Extractive QA (SQuAD)** | 8B-LoRA (NoQuant) | Highest F1 (49.31), 49.8% improvement over 4B-SFT |
| **Consumer GPU (<12GB)** | 8B-QLoRA | 62% less training VRAM, only 2.1% F1 drop on ARC |
| **Minimum Training VRAM** | 8B-VeRA | 30-41% less VRAM than LoRA/DoRA |
| **Extreme Inference VRAM** | 4B-4bit PTQ | 63.8% less VRAM, but 5× slower, avoid for fine-tuned models |

### Key Metrics Summary

**Training VRAM Efficiency**:
- 4B Full FT: 32.3GB allocated
- 8B LoRA: 20.2GB allocated
- 8B QLoRA: **7.6GB allocated** (-62.3% vs LoRA) ✨
- 8B VeRA: 16.2GB allocated (-20% vs LoRA)

**Inference Performance (OpenMath task)**:
- 4B-SFT: AbsDiff **9.0**, latency **0.61s**, eval VRAM **7.87GB** 🏆
- 8B-DoRA: AbsDiff 136.4, latency 6.25s, eval VRAM 15.63GB
- 8B-QLoRA: AbsDiff 117.0, latency 5.40s, eval VRAM **6.02GB**

**Post-Training Quantization (PTQ) Trade-offs**:
- 4B-4bit: -63.8% VRAM, but +409% latency, +561% error on fine-tuned OpenMath
- 4B-Int8: -42.9% VRAM, but +372% latency, +797% error on fine-tuned OpenMath
- **Conclusion**: PTQ only for extreme VRAM constraints on base models; catastrophic for fine-tuned numeracy

## Engineering Principles from Research

Based on empirical testing across 12 configurations:

1. **SFT on smaller models beats LoRA on larger models for in-domain tasks**
   - 4B-SFT achieved 16× better math accuracy than 8B-LoRA
   - Full parameter updates enable tighter calibration than low-rank adapters

2. **Post-training quantization is a last resort**
   - Use only when VRAM is absolute hard constraint
   - Severe latency penalties (3-6× slower) and accuracy regressions on fine-tuned models

3. **QLoRA is an enabler, not a free lunch**
   - 62% training VRAM reduction enables 8B on consumer GPUs
   - Trade-off: -19.4% F1 on precision-sensitive tasks (SQuAD)

4. **DoRA offers best OOD efficiency**
   - Matches/exceeds LoRA efficacy with better inference latency
   - Optimal for scaling logical capabilities on OOD tasks

5. **Task dictates method choice**
   - Numeracy/MCQ: PEFT acceptable under constraints
   - Extractive QA: Full FT or larger base model required

6. **Method selection ladder**
   - Strictest VRAM cap → VeRA
   - Moderate constraints → LoRA/DoRA
   - Quality-first → Full FT on appropriate size

## Contributing & Citation

This is a research repository. If you use these findings or code in your work, please reference the corresponding documentation papers in the `Documentation/` folder.

## License

See repository license file for details.

## Acknowledgments

Built with:
- **Hugging Face Transformers, TRL, PEFT, BitsAndBytes**
- **Qwen3 model family** (Alibaba Cloud)
- **Datasets**: ARC (AI2), SQuAD v2 (Stanford), OpenMathInstruct-2 (NVIDIA)

---

**Authors**: Alberto Rodero & Pablo Lobato  
**Last Updated**: November 2025
