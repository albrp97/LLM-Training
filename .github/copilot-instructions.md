# LLM Training Project - AI Agent Instructions

This repository implements a comprehensive LLM fine-tuning and quantization research platform with systematic evaluation workflows.

## Architecture Overview

The codebase follows a config-driven experimental pipeline:
- **Training**: `Fine-tuning/01_Train.py` - Central training script with inline configuration
- **Evaluation**: `Testing/02_TestModels.py` + `Testing/03_EvaluationOrchestrator.py` - Automated batch evaluation
- **Quantization**: `quantization_utils.py` - Unified quantization abstraction layer
- **Data**: Preprocessed parquet files in `Datasets/`, structured metadata in `Models/`, results in `Testing/metrics/`

## Key Patterns & Conventions

### Configuration-First Design
All experiments are configured by editing variables at the top of `Fine-tuning/01_Train.py`:
- `DATASET_CHOICE`: "openmath", "squad", "arc", "boolq" or `None` for base model snapshots
- `PEFT_CONFIG`: "LoRa", "VeRa", "DoRa", "NoPeft" with rank parameters (e.g., `lora_r = 256`)
- `QUANT_METHOD`: "NoQuant", "QLORA", "GPTQ", "AWQ", etc. with PTQ targets for post-training quantization
- `TRUNC_TRAIN`/`TRUNC_EVAL`: Dataset size limits for rapid experimentation

### Quantization System
The `QuantizationSpec` class in `quantization_utils.py` provides method-agnostic metadata:
- **Training-time**: Only QLoRA (4-bit NF4 with double quantization)
- **Post-training**: GPTQ, AWQ, HQQ via `tools/quantize.py` (planned implementation)
- **Naming**: Auto-generated tags like `QLORA_w4_headbf16` encode all quantization settings

### Model Organization
Models follow the naming pattern: `{base}-{dataset}_{method}_{peft}_{quant_tag}`:
```
Models/Qwen3-0.6B-openmath_SFT_LoRa256_QLORA_w4_headbf16/
├── model.safetensors
├── config.json
├── training_metadata.json  # Comprehensive experiment record
└── tokenizer files
```

### Evaluation Workflow
1. `03_EvaluationOrchestrator.py` discovers untested models (missing `Testing/metrics/{model}.json`)
2. Calls `02_TestModels.py` for each model across all test datasets
3. Results stored as structured JSON with task-specific metrics (accuracy, F1, MCC, latency)

### Chat Template System
Models use a consistent chat template for SFT:
```
System: {system_content}
User: {user_content}
Assistant: {assistant_content} <|endoftext|>
```
Answers must be wrapped in `\boxed{}` for automated parsing.

## Development Workflows

### Running Experiments
```bash
# Edit configuration in Fine-tuning/01_Train.py
# Run training
python Fine-tuning/01_Train.py

# Evaluate all untested models
python Testing/03_EvaluationOrchestrator.py

# Manual evaluation of specific model
python Testing/02_TestModels.py Models/model-name/
```

### Debugging & Monitoring
- VRAM tracking: Training script captures peak memory usage per GPU
- Progress: tqdm progress bars for dataset processing and evaluation
- Metadata: All experiment parameters preserved in `training_metadata.json`
- Error handling: Orchestrator continues on single model failures

### Dataset Integration
Add new datasets by:
1. Creating parquet files with `question`, `answer`, `context` (optional) columns
2. Adding dataset info to `datasets_info` dict in `02_TestModels.py`
3. Updating dataset loading logic in `01_Train.py`

## Critical Dependencies
- Uses `uv` package manager (mentioned in README but no pyproject.toml found)
- HuggingFace ecosystem: transformers, peft, trl, datasets
- Quantization: bitsandbytes (QLoRA), autogptq/awq (planned PTQ)
- Hardware: CUDA required for meaningful experiments

## Common Gotchas
- QLoRA overrides PEFT_CONFIG to use LoRA regardless of setting
- `merge_after_train` behavior differs between QLoRA and other methods
- PTQ methods in `tools/quantize.py` are implementation placeholders
- Dataset truncation (`TRUNC_TRAIN=5`) is often left on for development
- Base model snapshots require `DATASET_CHOICE = None`