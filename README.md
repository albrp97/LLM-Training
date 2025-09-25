# LLM Training

An investigation repository for exploring Large Language Model fine-tuning, evaluation and quantization techniques. This repository focuses on parameter-efficient fine-tuning methods, quantization and comprehensive performance analysis across different model configurations.

## Core Techniques

### PEFT (Parameter-Efficient Fine-Tuning)
- **LoRa (Low-Rank Adaptation)**: Efficient fine-tuning with low-rank matrices
- **VeRa (Vector-based Random Matrix Adaptation)**: Vector-based parameter-efficient adaptation
- **DoRa (Weight-Decomposed Low-Rank Adaptation)**: Magnitude and direction adaptation
- **NoPeft**: Full fine-tuning baseline

### Quantization Methods
- **QLoRA**: 4-bit quantization with double quantization
- **AWQ/GPTQ**: Advanced quantization techniques (implementation placeholders)

### Training Methodology
- **SFT (Supervised Fine-Tuning)**: Structured training with chat templates
- **Configurable Hyperparameters**: Flexible training setup

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.9+ (for GPU support)
- 16GB+ RAM (recommended)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/albrp97/LLM-Training.git
cd LLM-Training

# Install dependencies using uv
uv sync
```

### Verify Installation
```bash
# Test basic functionality
uv run python 00_Playground.py

# Check GPU availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Files

### Training Scripts
- **`Fine-tuning/1_Train.py`**: Main training script
  - Handles model loading and preparation
  - Implements PEFT configurations (LoRa, VeRa, DoRa)
  - Manages quantization setup
  - Trains models with configurable hyperparameters
  - Saves models with metadata tracking

### Evaluation Scripts
- **`Testing/02_TestModels.py`**: Comprehensive model evaluation
  - Evaluates models on multiple datasets (ARC, SQuAD, OpenMath)
  - Implements task-specific metrics (accuracy, F1, MCC, etc.)
  - Tracks performance metrics (latency, token generation, VRAM)
  - Supports batch evaluation and progress tracking

- **`Testing/04_EvaluationOrchestrator.py`**: Batch evaluation orchestrator
  - Automatically finds and evaluates untested models
  - Manages evaluation pipeline
  - Handles error recovery and logging

### Data Processing
- **`DataPreparation/1_PreprocessData.ipynb`**: Data preprocessing notebook
  - Cleans and preprocesses raw datasets
  - Handles data formatting and validation

- **`DataPreparation/2_ReduceTestData.ipynb`**: Test dataset reduction
  - Reduces test set sizes for faster evaluation
  - Maintains dataset representativeness

- **`DataPreparation/3_ReduceTrainData.ipynb`**: Training dataset reduction
  - Optimizes training dataset size
  - Balances dataset efficiency and model performance

### Data Organization
- **`Datasets/`**: Training and test datasets
  - Parquet files for efficient storage
  - ARC, SQuAD v2, OpenMathInstruct-2 datasets
  - Separate train/test splits

- **`Models/`**: Trained model outputs
  - Organized by configuration (model-dataset-method-quantization)
  - Includes training metadata and model weights
  - Checkpoint directories for training resumption

- **`Testing/metrics/`**: Evaluation results
  - JSON files with detailed performance metrics
  - Task-specific scores and performance analytics
  - Cross-model comparison data

- **`Documentation/`**: Analysis outputs
  - Performance comparisons and visualizations
  - Experimental results and findings

## Key Features

### Configurable Training Pipeline
- **Parameter Flexibility**: Easily modify PEFT configurations, quantization methods and training hyperparameters
- **Multiple Experiment Types**: Train with different combinations of LoRa, VeRa, DoRa or NoPeft methods
- **Quantization Options**: Experiment with QLoRA, AWQ or no quantization to compare performance vs efficiency

### Automated Evaluation System
- **Batch Processing**: Automatically evaluate multiple models across all test datasets
- **Comprehensive Metrics**: Task-specific evaluation including accuracy, F1 scores and latency
- **Result Organization**: Automatic saving of evaluation results with detailed performance analytics

### Experimental Workflow
- **Configuration-Based Training**: Modify parameters in `1_Train.py` to experiment with different setups
- **Evaluation**: Use `04_EvaluationOrchestrator.py` to automatically test all untrained models
- **Performance Comparison**: View results in `Testing/metrics/` to compare different configurations

## Usage Flow

### 1. Configure Training Parameters
Edit `Fine-tuning/1_Train.py` to set your desired configuration:

### 2. Train the Model
```bash
# Run training with your configuration
uv run python Fine-tuning/1_Train.py
```

### 3. Automated Evaluation
```bash
# Automatically evaluate all untrained models
uv run python Testing/04_EvaluationOrchestrator.py
```
