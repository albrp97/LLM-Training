# Project Structure

This document describes the organization of the LLM-Training research repository after cleanup and restructuring (November 2025).

## Overview

This repository contains a systematic investigation of LLM fine-tuning across **three experimental phases**:

1. **Phase 1** (Doc 01): Model size (0.6B vs 1.7B) and LoRA rank analysis
2. **Phase 2** (Doc 02): PEFT methods comparison (LoRA, DoRA, VeRA)
3. **Phase 3** (Doc 03): Scaling to 4B and 8B with quantization strategies

Total tested configurations: **12 major model variants** across size, PEFT method, and quantization.

## Root Directory

The root now contains only essential project files:

```
.
├── pyproject.toml              # Python project configuration and dependencies
├── uv.lock                     # UV package manager lock file
├── .gitignore                  # Git ignore configuration
├── README.md                   # Main project documentation with research findings
├── QUICKSTART.md               # Quick start guide for experiments
└── PROJECT_STRUCTURE.md        # This file - project organization guide
```

## Main Directories

### `experiments/` - Experimental Scripts

Experimental scripts and playground files for testing quantization methods and model configurations.

**Contents:**
- `00_Playground.ipynb/py` - Sandbox for quick experiments and testing
- `create_1.5b_base.py` - Create 1.7B base model snapshots with optional quantized variants
- `run_quantization_experiments.py` - **Master orchestration script** for full experiment pipeline (Phase 3)
- `train_8b_qlora.py` - Standalone script for training Qwen3-8B with QLoRA
- `train_quantization_variants.py` - Script to train and save quantized model variants
- `QUANTIZATION_EXPERIMENTS.md` - Documentation on quantization experiments
- `README.md` - Guide to experimental scripts

**When to use:**
- Testing new configurations before committing to production
- Running full quantization experiment pipeline (`run_quantization_experiments.py`)
- Quick prototyping and validation

### `utils/` - Shared Utility Modules

Centralized utility modules used across the project.

**Contents:**
- `quantization_utils.py` - **Core utilities**: `QuantMethod` enum, `QuantizationSpec` dataclass, quantization tagging/metadata
- `__init__.py` - Package initialization exposing main utilities

**Key exports:**
- `QuantMethod`: Unified enum for all quantization approaches (NoQuant, INT8, BNB4BIT, QLORA, etc.)
- `QuantizationSpec`: Standardized metadata container for quantization configurations
- `tag_quant()`: Generate consistent quantization identifiers

**Import pattern** (after restructuring):
```python
from utils.quantization_utils import QuantMethod, QuantizationSpec
```

### `Fine-tuning/` - Production Training Scripts

Main training pipeline for production model fine-tuning.

**Contents:**
- `01_Train.py` - **Main configurable training script**
  - Supports all PEFT methods (LoRA, DoRA, VeRA, NoPeft)
  - Supports all quantization methods (QLoRA, NoQuant)
  - Configurable hyperparameters (epochs, batch size, learning rate)
  - Automatic metadata tracking and model saving
- `tmp/` - Temporary notebooks for training exploration

**Usage:**
Edit configuration variables in `01_Train.py` and run:
```python
train = True
DATASET_CHOICE = "openmath"  # or "squad", "arc"
MODEL_NAME = "Qwen/Qwen3-4B"
PEFT_CONFIG = "LoRa"  # or "DoRa", "VeRa", "NoPeft"
QUANT_METHOD = "QLORA"  # or "NoQuant", "INT8", etc.
lora_r = 256
num_train_epochs = 1
```

### `Testing/` - Model Evaluation Pipeline

Comprehensive evaluation system for trained models.

**Contents:**
- `02_TestModels.py` - **Core evaluation script**
  - Multi-dataset evaluation (ARC, SQuAD, OpenMath)
  - Task-specific metrics (F1, AbsDiff, accuracy)
  - Performance tracking (latency, VRAM, tokens/sec)
  - Supports batch evaluation with progress tracking
- `03_EvaluationOrchestrator.py` - **Batch automation**
  - Automatically finds and evaluates untested models
  - Manages evaluation pipeline with error recovery
- `04_CompareMetrics.ipynb` - **Results analysis**
  - Cross-model comparison and visualization
  - Generates `metrics_summary.csv`
- `metrics/` - JSON files with detailed per-model results
  - 12 model configurations tested
  - Example: `Qwen3-8B-openmath_SFT_LoRa256_QLORA_w4_headbf16.json`
- `metrics_summary.csv` - **Comparison table** across all models

**Metrics tracked:**
- Training VRAM (reserved, allocated)
- Evaluation VRAM (reserved, allocated)
- Task scores (ARC F1, OpenMath AbsDiff, SQuAD F1)
- Latency per task
- Token generation stats

### `tools/` - Command-Line Quantization Tools

Advanced quantization and analysis utilities.

**Contents:**
- `quantize.py` - Post-training quantization orchestration
- `test_quant_tag.py` - Quantization tagging tests
- `apply_smoothquant.py` - SmoothQuant application
- `compare_smoothquant.py` - SmoothQuant comparison utilities
- `compare_three_way.py`, `detailed_comparison.py` - Multi-model analysis
- `base_model_analysis.py` - Base model profiling
- `smoothquant_*.py` - SmoothQuant experimental tools

**Purpose:**
CLI tools for advanced users to apply and compare quantization methods outside the main training loop.

### `DataPreparation/` - Data Preprocessing

Notebooks for dataset preparation and optimization.

**Contents:**
- `1_PreprocessData.ipynb` - Raw data cleaning and chat-template formatting
- `2_ReduceTestData.ipynb` - Test set size optimization (200 samples per task)
- `3_ReduceTrainData.ipynb` - Training set optimization (1,000 samples)

**Datasets prepared:**
- **ARC** (AI2): Multiple-choice reasoning
- **SQuAD v2**: Extractive question answering
- **OpenMathInstruct-2** (NVIDIA): Numerical problem solving

### `Documentation/` - Research Papers

Comprehensive documentation of experimental findings.

**Contents:**
- `01_Finetuning_ModelSize_LoraR.md/.tex` - **Phase 1 research**
  - 0.6B vs 1.7B model comparison
  - LoRA rank analysis (r=32 to r=1024)
  - Full FT vs LoRA trade-offs
  - **Key finding**: Full FT on small models > LoRA on large models for in-domain tasks

- `02_Adapters-Lora_Vera_Dora.tex` - **Phase 2 research**
  - Comprehensive PEFT methods comparison
  - LoRA vs DoRA vs VeRA empirical analysis
  - Training VRAM efficiency analysis
  - **Key finding**: VeRA for strict caps; LoRA/DoRA when feasible; Full FT when possible

- `03_SFT-Quantization-Adapters.tex` - **Phase 3 research** (latest)
  - Scaling to 4B and 8B models
  - QLoRA implementation and 62% VRAM reduction
  - Post-training quantization (PTQ) evaluation
  - **Key findings**: 
    - 4B-SFT is best balanced choice
    - 8B-DoRA for OOD tasks
    - PTQ is last resort (catastrophic for fine-tuned numeracy)

**Format:** Academic LaTeX papers with full experimental details, metrics tables, and analysis.

### `docs/` - Additional Technical Documentation

Supplementary technical notes and summaries.

**Contents:**
- `Final_Summary.md` - Overall project summary
- `Quantization_Methods_Overview.md` - Quantization techniques reference
- `SmoothQuant_*.md` - SmoothQuant implementation notes and results

### `Datasets/` - Training & Evaluation Data

Preprocessed datasets in Parquet format for efficient loading.

**Structure:**
```
Datasets/
├── arc-train.parquet          # ARC training set (1,000 samples)
├── arc-test.parquet           # ARC test set (200 samples)
├── openmath-train.parquet     # OpenMath training set (1,000 samples)
├── openmath-test.parquet      # OpenMath test set (200 samples)
├── squad-train.parquet        # SQuAD training set (1,000 samples)
├── squad-test.parquet         # SQuAD test set (200 samples)
└── Copy/                      # Backup copies
```

### `Models/` - Trained Model Outputs

Organized storage for trained models with metadata.

**Naming convention:**
```
{model_size}-{dataset}_{training_method}_{peft_config}_{quant_method}/
```

**Examples:**
- `Qwen3-4B-openmath_SFT_NoPeft_NoQuant/` - 4B full fine-tuning, no quantization
- `Qwen3-8B-openmath_SFT_LoRa256_QLORA_w4_headbf16/` - 8B LoRA with QLoRA

**Contents per model:**
- Model weights and configuration
- Training metadata JSON
- Adapter weights (for PEFT models)
- Tokenizer files

## Import Changes After Restructuring

Since `quantization_utils.py` moved from root to `utils/`, all imports were updated:

**Old (before cleanup):**
```python
from quantization_utils import QuantMethod, QuantizationSpec
```

**New (after cleanup):**
```python
sys.path.append(str(Path(__file__).parent.parent))
from utils.quantization_utils import QuantMethod, QuantizationSpec
```

**Affected files (all updated):**
- `Fine-tuning/01_Train.py`
- `Testing/02_TestModels.py`
- `tools/quantize.py`
- `tools/test_quant_tag.py`
- `experiments/create_1.5b_base.py`
- `experiments/train_quantization_variants.py`

## Tested Configurations Summary

**12 major configurations** evaluated across 3 benchmarks:

| Model | Size | Training | Quantization | Primary Use Case |
|-------|------|----------|--------------|------------------|
| Qwen3-1.7B-base | 1.7B | None | None | Entry-level baseline |
| Qwen3-4B-base | 4B | None | None/Int8/4bit | Mid-tier baseline + PTQ |
| Qwen3-4B-SFT | 4B | Full FT | None/Int8/4bit | **Best in-domain (math)** |
| Qwen3-8B-base | 8B | None | None | Quality baseline |
| Qwen3-8B-LoRA | 8B | LoRA (r=256) | None/QLoRA | OOD reasoning/QA |
| Qwen3-8B-DoRA | 8B | DoRA (r=256) | None | **Best OOD efficiency** |
| Qwen3-8B-VeRA | 8B | VeRA (r=512) | None | Minimum training VRAM |

## Workflow

### Typical Research Workflow

1. **Data Preparation** (`DataPreparation/`)
   - Preprocess and reduce datasets to manageable sizes

2. **Training** (`Fine-tuning/01_Train.py` or `experiments/`)
   - Configure and train models with chosen PEFT/quantization

3. **Evaluation** (`Testing/`)
   - Run `03_EvaluationOrchestrator.py` for batch evaluation
   - Or `02_TestModels.py` for individual models

4. **Analysis** (`Testing/04_CompareMetrics.ipynb`)
   - Generate comparison tables
   - Visualize trade-offs

5. **Documentation** (`Documentation/`)
   - Write up findings in LaTeX papers

### Quick Start Workflow

For reproducing key experiments:

```bash
# Full automated pipeline (Phase 3)
uv run python experiments/run_quantization_experiments.py

# Manual training
uv run python Fine-tuning/01_Train.py

# Batch evaluation
uv run python Testing/03_EvaluationOrchestrator.py

# Compare results
# Open Testing/04_CompareMetrics.ipynb
```

## Key Design Decisions

1. **Centralized Utilities** - `utils/quantization_utils.py` provides single source of truth for quantization specs
2. **Experiments Isolation** - Experimental scripts separate from production training
3. **Metadata Tracking** - All models save comprehensive training metadata
4. **Reproducibility** - Configuration-based training enables exact reproduction
5. **Comprehensive Evaluation** - Multi-task testing reveals method trade-offs

## Further Reading

- **Quick start**: [QUICKSTART.md](QUICKSTART.md)
- **Main documentation**: [README.md](README.md)
- **Research papers**: `Documentation/` folder
- **Experiment details**: `experiments/README.md`

---

**Last Updated**: November 2025  
**Authors**: Alberto Rodero & Pablo Lobato
