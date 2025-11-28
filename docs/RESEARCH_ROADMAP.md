# Research Roadmap: Advanced Quantization Methods Comparison

**Experiment Phase 4**: Comprehensive comparison of advanced quantization techniques on Llama-3.2-1B-Instruct

---

## Overview

This phase focuses on comparing **6 advanced quantization methods** beyond the basic BitsAndBytes approach used in previous phases. We will systematically evaluate PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training) methods on a unified baseline.

### Key Constraints

- **Model**: `meta-llama/Llama-3.2-1B-Instruct` (fixed for all experiments)
- **Platform**: WSL (Ubuntu/Linux) - **ALWAYS** use Linux environment
- **Branch**: `advanced-quantization-llama`
- **Precision targets**: 4-bit and 8-bit (INT8/INT4) where applicable

---

## Quantization Methods to Compare

### 1. **AWQ (Activation-aware Weight Quantization)** - PTQ
- **Type**: Post-Training Quantization
- **Paper**: AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (2023)
- **Key Idea**: Protects salient weights based on activation magnitudes
- **Library**: `AutoAWQ` or `transformers` integration
- **Precision**: 4-bit (W4A16)
- **Advantages**: Fast inference, preserves accuracy on important weights
- **Use Case**: Production deployment with minimal accuracy loss

### 2. **GPTQ (Generative Pre-trained Transformer Quantization)** - PTQ
- **Type**: Post-Training Quantization
- **Paper**: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)
- **Key Idea**: Layer-wise quantization with approximate second-order information (Hessian)
- **Library**: `auto-gptq` or `optimum`
- **Precision**: 4-bit, 8-bit (INT4/INT8)
- **Advantages**: Strong accuracy retention, well-tested
- **Use Case**: High-quality compression for inference

### 3. **HQQ (Half-Quadratic Quantization)** - PTQ
- **Type**: Post-Training Quantization
- **Paper**: HQQ: Half-Quadratic Quantization of Large Machine Learning Models (2024)
- **Key Idea**: Optimize quantization parameters via half-quadratic splitting
- **Library**: `hqq` (already partially integrated in transformers)
- **Precision**: 4-bit, 8-bit
- **Advantages**: Fast quantization, no calibration data needed
- **Use Case**: Quick deployment without calibration datasets

### 4. **SmoothQuant** - PTQ
- **Type**: Post-Training Quantization
- **Paper**: SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (2023)
- **Key Idea**: Migrate difficulty from activations to weights via channel-wise scaling
- **Library**: Custom implementation or `smoothquant` package
- **Precision**: 8-bit (W8A8 - weights and activations)
- **Advantages**: Enables full INT8 inference (weights + activations)
- **Use Case**: Hardware with INT8 acceleration (e.g., NVIDIA Tensor Cores)

### 5. **AdaRound (Adaptive Rounding)** - PTQ
- **Type**: Post-Training Quantization
- **Paper**: Up or Down? Adaptive Rounding for Post-Training Quantization (2020)
- **Key Idea**: Learn optimal rounding (up/down) for each weight
- **Library**: Custom implementation or `neural-compressor`
- **Precision**: 4-bit, 8-bit
- **Advantages**: Better than naive rounding, layer-wise optimization
- **Use Case**: When standard PTQ degrades too much

### 6. **QuaRot (Quantization with Rotation)** - PTQ/Hybrid
- **Type**: Post-Training Quantization (with preprocessing)
- **Paper**: QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (2024)
- **Key Idea**: Apply rotation matrices to eliminate outliers before quantization
- **Library**: Custom implementation (research code available)
- **Precision**: 4-bit
- **Advantages**: Handles outliers effectively, uniform quantization
- **Use Case**: Extreme compression scenarios

---

## Experimental Design

### Baseline

- **Model**: `meta-llama/Llama-3.2-1B-Instruct` (full precision BF16/FP32)
- **Datasets**: 
  - ARC (reasoning)
  - SQuAD v2 (extractive QA)
  - OpenMathInstruct-2 (numeracy)
- **Sample size**: 200 samples per dataset for evaluation

### Quantization Variants

For each method, we will create:
1. **4-bit variant** (where supported)
2. **8-bit variant** (where supported)

Total configurations: **~12-15 quantized models** + 1 baseline = **13-16 total**

### Metrics

**Accuracy Metrics** (per dataset):
- ARC: Macro F1, accuracy
- SQuAD: F1, Exact Match
- OpenMath: Mean Absolute Difference

**Efficiency Metrics**:
- Inference latency (per sample)
- VRAM usage (inference)
- Model size on disk (MB/GB)
- Quantization time (seconds)

**Trade-off Analysis**:
- Accuracy vs Model Size
- Accuracy vs Latency
- Quantization Time vs Final Quality

---

## Project Structure

```
LLM-Training/
├── experiments/
│   └── advanced-quantization/          # New folder for this phase
│       ├── quantize_awq.py            # AWQ quantization script
│       ├── quantize_gptq.py           # GPTQ quantization script
│       ├── quantize_hqq.py            # HQQ quantization script
│       ├── quantize_smoothquant.py    # SmoothQuant quantization script
│       ├── quantize_adaround.py       # AdaRound quantization script
│       ├── quantize_quarot.py         # QuaRot quantization script
│       ├── run_all_quantization.sh    # Master orchestration script (bash)
│       ├── evaluate_quantized.py      # Unified evaluation script
│       └── README.md                  # Experiment-specific documentation
│
├── Models/
│   └── Llama-3.2-1B-Instruct/         # All quantized variants stored here
│       ├── baseline/                  # Full precision baseline
│       ├── awq-4bit/
│       ├── awq-8bit/
│       ├── gptq-4bit/
│       ├── gptq-8bit/
│       ├── hqq-4bit/
│       ├── hqq-8bit/
│       ├── smoothquant-8bit/
│       ├── adaround-4bit/
│       ├── adaround-8bit/
│       ├── quarot-4bit/
│       └── metadata/                  # JSON files with quantization configs
│
├── Testing/
│   └── metrics/
│       └── advanced-quantization/     # Results for this experiment
│           ├── llama-3.2-1b-baseline.json
│           ├── llama-3.2-1b-awq-4bit.json
│           ├── llama-3.2-1b-gptq-4bit.json
│           └── ... (one per variant)
│
├── docs/
│   ├── RESEARCH_ROADMAP.md           # This file - overall plan
│   ├── Quantization_Methods_Deep_Dive.md  # Technical details per method
│   └── Phase4_Advanced_Quantization.md    # Final results and analysis
│
└── .agent_instructions.md            # Agent guidelines for this experiment
```

---

## Implementation Roadmap

### Phase 1: Setup & Baseline (Week 1)

**Tasks:**
1. ✅ Create new branch `advanced-quantization-llama`
2. ✅ Create project structure and documentation
3. ✅ Set up WSL environment verification
4. ⬜ Install required libraries:
   - `auto-gptq`
   - `autoawq`
   - `hqq`
   - `neural-compressor` (for AdaRound)
   - `smoothquant` or custom implementation
   - `quarot` (research code)
5. ⬜ Download and test baseline `meta-llama/Llama-3.2-1B-Instruct`
6. ⬜ Evaluate baseline on all 3 datasets
7. ⬜ Document baseline metrics

**Deliverables:**
- Working WSL environment with all dependencies
- Baseline evaluation results (JSON + summary)

### Phase 2: PTQ Methods Implementation (Week 2-3)

**Priority Order** (based on maturity and ease of integration):

#### Week 2: Mature Libraries
1. **GPTQ** (most mature)
   - Use `auto-gptq` library
   - Create 4-bit and 8-bit variants
   - Evaluate and document
   
2. **AWQ** (well-documented)
   - Use `autoawq` library
   - Create 4-bit variant
   - Evaluate and document

3. **HQQ** (already in transformers)
   - Use `HQQConfig` from transformers
   - Create 4-bit and 8-bit variants
   - Evaluate and document

#### Week 3: Advanced/Custom Methods
4. **SmoothQuant**
   - Implement or adapt from research code
   - Create 8-bit (W8A8) variant
   - Evaluate and document

5. **AdaRound**
   - Use `neural-compressor` or custom implementation
   - Create 4-bit and 8-bit variants
   - Evaluate and document

6. **QuaRot**
   - Adapt research code
   - Create 4-bit variant
   - Evaluate and document

**Testing Protocol** (for each method):
```bash
# 1. Quantize model
python experiments/advanced-quantization/quantize_<method>.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --bits 4 \
  --output Models/Llama-3.2-1B-Instruct/<method>-4bit

# 2. Verify quantization (quick sanity check)
python experiments/advanced-quantization/verify_model.py \
  --model-path Models/Llama-3.2-1B-Instruct/<method>-4bit

# 3. Evaluate on all datasets
python Testing/02_TestModels.py \
  --model-path Models/Llama-3.2-1B-Instruct/<method>-4bit \
  --output Testing/metrics/advanced-quantization/

# 4. Compare with baseline
python experiments/advanced-quantization/compare_to_baseline.py \
  --baseline Testing/metrics/advanced-quantization/llama-3.2-1b-baseline.json \
  --quantized Testing/metrics/advanced-quantization/llama-3.2-1b-<method>-4bit.json
```

**Deliverables (per method):**
- Quantized model saved in `Models/`
- Evaluation metrics JSON in `Testing/metrics/advanced-quantization/`
- Metadata JSON with quantization config
- Quick comparison report

### Phase 3: Analysis & Documentation (Week 4)

**Tasks:**
1. ⬜ Generate comprehensive comparison tables
2. ⬜ Create visualizations:
   - Accuracy vs Model Size scatter plots
   - Latency comparison bar charts
   - Pareto frontiers (accuracy vs efficiency)
3. ⬜ Write `Phase4_Advanced_Quantization.md` with:
   - Methodology
   - Results tables
   - Method recommendations
   - Trade-off analysis
4. ⬜ Update main README with Phase 4 summary
5. ⬜ Merge branch to main

**Deliverables:**
- Final research document
- Visualization plots
- Updated README
- Merged PR

---

## Evaluation Protocol

### Standard Testing Procedure

**For every quantized model**, we MUST:

1. **Sanity Check** (quick verification):
   ```python
   # Test basic generation
   prompt = "What is 2+2?"
   output = model.generate(...)
   assert len(output) > 0
   assert output is not None
   ```

2. **Full Evaluation** (3 datasets):
   ```bash
   python Testing/02_TestModels.py \
     --model-path <path-to-quantized-model> \
     --datasets arc squad openmath \
     --num-samples 200 \
     --output Testing/metrics/advanced-quantization/
   ```

3. **Metadata Recording**:
   - Quantization method
   - Precision (4-bit, 8-bit)
   - Quantization time
   - Model size before/after
   - Library version
   - Quantization config (JSON)

4. **Comparison**:
   - Compare to baseline (% degradation)
   - Compare to other methods at same bit-width
   - Note any failures or issues

### Success Criteria

A quantized model is considered **successful** if:
- ✅ Loads without errors
- ✅ Generates coherent text
- ✅ Completes evaluation on all 3 datasets
- ✅ Model size reduced by expected amount
- ✅ Accuracy degradation < 15% vs baseline (acceptable range)

A quantized model is **flagged for investigation** if:
- ⚠️ Accuracy degradation > 15%
- ⚠️ Generates gibberish or fails to respond
- ⚠️ Latency is worse than baseline
- ⚠️ Quantization process fails or takes >1 hour

---

## WSL/Linux Requirements

### Always Use WSL

**Why Linux?**
- Better support for quantization libraries (especially GPTQ, AWQ)
- Consistent behavior across implementations
- Better CUDA compatibility
- Avoid Windows path and environment issues

**Verification**:
```bash
# Check we're in WSL
uname -a  # Should show "Linux"
echo $WSL_DISTRO_NAME  # Should show Ubuntu or similar

# Check CUDA availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Environment Setup**:
```bash
# Activate WSL
wsl

# Navigate to project
cd /mnt/c/Users/AlbertoTC/Documents/code/LLM-Training

# Activate environment (if using venv/conda)
# source venv/bin/activate

# Verify dependencies
python -c "import auto_gptq, awq, transformers"
```

### WSL-Specific Notes

- **Path format**: Use Linux paths (`/mnt/c/...`) not Windows paths (`C:\...`)
- **Line endings**: Ensure scripts use LF not CRLF (`dos2unix` if needed)
- **Permissions**: May need `chmod +x` for bash scripts
- **CUDA**: Verify CUDA drivers work in WSL2

---

## Dependencies

### Core Libraries

```bash
# Install in this order
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0
pip install accelerate
pip install datasets
pip install optimum

# Quantization libraries
pip install auto-gptq  # GPTQ
pip install autoawq    # AWQ
pip install hqq        # HQQ (if not in transformers)
pip install neural-compressor  # AdaRound

# For SmoothQuant and QuaRot - may need custom install from GitHub
# git clone <repo> && cd <repo> && pip install -e .
```

### Optional Tools

```bash
pip install pandas matplotlib seaborn  # For analysis
pip install jupyter notebook           # For interactive analysis
pip install tensorboard                # For logging (optional)
```

---

## Risk Mitigation

### Common Issues & Solutions

1. **CUDA OOM during quantization**
   - Solution: Use CPU quantization (slower but works)
   - Fallback: Use smaller calibration dataset
   - Last resort: Use colab/cloud with more VRAM

2. **Library compatibility issues**
   - Solution: Use docker container with pinned versions
   - Fallback: Create separate conda envs per method

3. **Method implementation not available**
   - Solution: Adapt research code (mark as "research prototype")
   - Fallback: Skip method, document why

4. **Quantized model gibberish output**
   - Solution: Check calibration data quality
   - Fallback: Try different quantization config
   - Document: Mark method as "failed" for this model

---

## Timeline

- **Week 1**: Setup, baseline, GPTQ, AWQ
- **Week 2**: HQQ, SmoothQuant
- **Week 3**: AdaRound, QuaRot
- **Week 4**: Analysis, documentation, merge

**Total**: ~4 weeks for complete experiment

---

## Success Metrics

This experiment is successful if:

1. ✅ At least **4 out of 6 methods** successfully quantize the model
2. ✅ All quantized models evaluated on all 3 datasets
3. ✅ Comprehensive comparison table generated
4. ✅ Clear recommendations provided for each use case
5. ✅ Results reproducible by following documentation

---

## Notes

- **Always verify you're in WSL** before running scripts
- **Always use `meta-llama/Llama-3.2-1B-Instruct`** - no other models
- **Always test each quantized model** before moving to next
- **Always save metadata** with each quantized model
- **Always compare to baseline** after each quantization

---

**Last Updated**: November 28, 2025  
**Branch**: `advanced-quantization-llama`  
**Status**: Planning Phase
