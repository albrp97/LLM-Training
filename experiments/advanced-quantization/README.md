# Advanced Quantization Methods Experiment

**Phase 4**: Comprehensive comparison of 6 advanced quantization techniques

**Model**: `meta-llama/Llama-3.2-1B-Instruct` (fixed baseline)  
**Platform**: WSL/Linux (required)  
**Branch**: `advanced-quantization-llama`

---

## Quick Start

### Prerequisites

1. **Must be in WSL/Linux**:
   ```bash
   uname -a  # Should show "Linux"
   ```

2. **Install dependencies**:
   ```bash
   pip install auto-gptq autoawq hqq optimum neural-compressor
   ```

3. **Verify CUDA**:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Run Experiments

```bash
# 1. Establish baseline
python common/baseline.py

# 2. Run quantization methods (one at a time)
python quantize_gptq.py --bits 4 --output ../../Models/Llama-3.2-1B-Instruct/gptq-4bit
python quantize_awq.py --bits 4 --output ../../Models/Llama-3.2-1B-Instruct/awq-4bit
python quantize_hqq.py --bits 4 --output ../../Models/Llama-3.2-1B-Instruct/hqq-4bit
# ... etc

# 3. Evaluate each
python ../../Testing/02_TestModels.py --model-path ../../Models/Llama-3.2-1B-Instruct/gptq-4bit

# 4. Run all at once (when scripts ready)
bash run_all_quantization.sh

# 5. Analyze results
python analyze_results.py
```

---

## Quantization Methods

### Status Tracker

| Method | Type | 4-bit | 8-bit | Status | Notes |
|--------|------|-------|-------|--------|-------|
| **GPTQ** | PTQ | ⬜ | ⬜ | Not started | Mature, start here |
| **AWQ** | PTQ | ⬜ | - | Not started | 4-bit only |
| **HQQ** | PTQ | ⬜ | ⬜ | Not started | No calibration |
| **SmoothQuant** | PTQ | - | ⬜ | Not started | W8A8 only |
| **AdaRound** | PTQ | ⬜ | ⬜ | Not started | Slow but accurate |
| **QuaRot** | PTQ | ⬜ | - | Not started | Research code |

Legend:
- ⬜ Not started
- 🔄 In progress
- ✅ Complete
- ❌ Failed
- ⚠️ Partial (issues noted)

---

## Results Summary

### Model Sizes

| Model | Precision | Size (MB) | Compression | Status |
|-------|-----------|-----------|-------------|--------|
| Baseline | FP16/BF16 | ~2,400 | 1.00× | ⬜ |
| GPTQ-4bit | INT4 | - | - | ⬜ |
| GPTQ-8bit | INT8 | - | - | ⬜ |
| AWQ-4bit | INT4 | - | - | ⬜ |
| HQQ-4bit | INT4 | - | - | ⬜ |
| HQQ-8bit | INT8 | - | - | ⬜ |
| SmoothQuant-8bit | INT8 | - | - | ⬜ |
| AdaRound-4bit | INT4 | - | - | ⬜ |
| AdaRound-8bit | INT8 | - | - | ⬜ |
| QuaRot-4bit | INT4 | - | - | ⬜ |

### Accuracy (ARC F1)

| Model | F1 | Δ vs Baseline | Status |
|-------|-----|---------------|--------|
| Baseline | - | 0.0% | ⬜ |
| GPTQ-4bit | - | - | ⬜ |
| AWQ-4bit | - | - | ⬜ |
| HQQ-4bit | - | - | ⬜ |
| SmoothQuant-8bit | - | - | ⬜ |
| AdaRound-4bit | - | - | ⬜ |
| QuaRot-4bit | - | - | ⬜ |

*(Tables will be filled as experiments complete)*

---

## Evaluation Protocol

### For Each Quantized Model

1. **Quantization** (save to `Models/`):
   ```bash
   python quantize_<method>.py --bits 4 --output <path> --save-metadata
   ```

2. **Verification** (sanity check):
   ```bash
   python verify_model.py --model-path <path>
   ```

3. **Evaluation** (full benchmarks):
   ```bash
   python ../../Testing/02_TestModels.py --model-path <path>
   ```

4. **Comparison** (vs baseline):
   ```bash
   python compare_to_baseline.py --quantized <path>
   ```

5. **Documentation** (update this README)

---

## Metrics Collected

### Quantization Metrics
- Quantization time (seconds)
- Model size before/after (MB)
- Compression ratio
- Peak memory usage (GB)

### Accuracy Metrics
- **ARC**: Macro F1, accuracy
- **SQuAD v2**: F1, Exact Match
- **OpenMath**: Mean Absolute Difference

### Efficiency Metrics
- Inference latency per sample (s)
- Tokens/second generation rate
- VRAM usage during inference (GB)

---

## Files in This Directory

### Quantization Scripts
- `quantize_gptq.py` - GPTQ quantization
- `quantize_awq.py` - AWQ quantization
- `quantize_hqq.py` - HQQ quantization
- `quantize_smoothquant.py` - SmoothQuant quantization
- `quantize_adaround.py` - AdaRound quantization
- `quantize_quarot.py` - QuaRot quantization

### Utility Scripts
- `common/baseline.py` - Baseline model evaluation
- `common/evaluation.py` - Shared evaluation functions
- `common/metadata.py` - Metadata handling
- `common/verification.py` - Model verification
- `verify_model.py` - Quick model verification
- `compare_to_baseline.py` - Comparison utility
- `analyze_results.py` - Results analysis and visualization

### Orchestration
- `run_all_quantization.sh` - Master script to run all methods
- `requirements.txt` - Python dependencies for this experiment

---

## Common Issues

### CUDA Out of Memory
```bash
# Try CPU quantization
python quantize_<method>.py --device cpu ...

# Or reduce calibration samples
python quantize_<method>.py --calib-samples 128 ...
```

### Library Not Found
```bash
# Install missing library
pip install <library>

# Check installed version
pip show <library>
```

### Model Loading Error
```bash
# Verify model path
ls -lh ../../Models/Llama-3.2-1B-Instruct/<method>-4bit/

# Test loading
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('<path>')"
```

---

## Next Steps

1. ⬜ Install all dependencies
2. ⬜ Evaluate baseline model
3. ⬜ Implement GPTQ (start with most mature)
4. ⬜ Implement AWQ
5. ⬜ Implement HQQ
6. ⬜ Implement SmoothQuant
7. ⬜ Implement AdaRound
8. ⬜ Implement QuaRot
9. ⬜ Generate comparison analysis
10. ⬜ Write Phase 4 documentation

---

## References

- See `docs/Quantization_Methods_Deep_Dive.md` for technical details
- See `docs/RESEARCH_ROADMAP.md` for overall experiment plan
- See `.agent_instructions.md` for workflow guidelines

---

**Last Updated**: November 28, 2025  
**Status**: Setup complete, ready to begin experiments
