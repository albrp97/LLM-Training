# Quantization Testing Priority List

**Goal:** Prove that larger models (>1B params) can survive W4 quantization, unlike 0.6B which failed catastrophically with 0% accuracy.

**Test Configuration:** TRUNC_EVAL=2 (quick validation with 2 samples per dataset)

**Success Criteria:**
- Accuracy >60% (vs 0% on 0.6B)
- Format compliance >80%
- Quantization completes without OOM errors

---

## Model Priority (Largest to Smallest)

### 1. Qwen3-4B (4.0B params, 36 layers)
   - **Status:** ❌ FAILED - OOM during quantization (37.5GB allocated on 24GB GPU)
   - **Skip Reason:** Too large for available VRAM during quantization process
   - **Base Model:** Not tested (deleted)

### 2. Llama-3.2-3B-Instruct (3.21B params, 28 layers)
   - **Status:** ❌ FAILED - OOM during quantization (37GB allocated on 24GB GPU)
   - **Skip Reason:** Too large for available VRAM during quantization process
   - **Base Model:** Not tested (deleted)

### 3. Qwen3-1.7B (1.7B params, 28 layers) ⭐ **CURRENT**
   - **Status:** 🟡 IN PROGRESS
   - **Base Model:** ✅ Created at `Models/Qwen3-1.7B-base`
   - **Baseline Results:** 33.33% accuracy (2/6 correct), 100% format compliance on ARC, 50% on OpenMath
   - **Parameters per Layer:** ~60.7M (2.8x more than 0.6B's 21.4M)
   - **Expected:** Should survive quantization better than 0.6B

---

## Quantization Method Priority (Fastest to Slowest)

### Method 1: AWQ (Activation-Aware Weight Quantization)
   - **Speed:** <1 second per layer (~28s total for 28 layers)
   - **Memory:** Requires full FP16 model + activation statistics
   - **Qwen3-1.7B Status:** ❌ FAILED - Shape mismatch error during quantization
   - **Error:** `shape '[2097152, 1]' is invalid for input of size 4294967296`
   - **Conclusion:** AWQ implementation bug with Qwen3 architecture

### Method 2: HQQ (Half-Quadratic Quantization)
   - **Speed:** ~1-2 seconds per layer (~50s total)
   - **Memory:** Most efficient (calibration-free)
   - **Qwen3-1.7B Status:** ❌ FAILED - Model loading error during inference
   - **Error:** `AssertionError` in bitsandbytes weight loading
   - **Conclusion:** HQQ weights not properly saved or incompatible format

### Method 3: GPTQ (Generalized Post-Training Quantization) ⭐ **NEXT**
   - **Speed:** ~2-5 minutes total
   - **Memory:** Moderate (sequential layer quantization)
   - **Qwen3-1.7B Status:** 🔲 NOT TESTED
   - **Known Results:** Works reliably on 0.6B (0% accuracy but no technical errors)

### Method 4: SmoothQuant
   - **Speed:** ~6-7 seconds per layer (~3-4 minutes total)
   - **Memory:** Requires activation statistics
   - **Qwen3-1.7B Status:** 🔲 NOT TESTED
   - **Known Results:** Works on 0.6B (0% accuracy but no technical errors)

---

## Testing Workflow

```bash
# For each model (in order: 4B → 3B → 1.7B):
# 1. Create base model
python Fine-tuning/01_Train.py  # with DATASET_CHOICE = None

# 2. Test base model (establish baseline)
python Testing/02_TestModels.py Models/{MODEL}-base --trunc-eval 2

# 3. For each method (AWQ → HQQ → GPTQ → SmoothQuant):
python tools/quantize.py run --method {METHOD} \
    --src Models/{MODEL}-base \
    --dst Models/{MODEL}-base_{method}_w4g{G} \
    --calib Datasets/calibration_openmath_10samples.txt \
    --bits 4 --group-size {64|128} --keep-lm-head-fp16

# 4. Test quantized model
python Testing/02_TestModels.py Models/{MODEL}-base_{method}_w4g{G} --trunc-eval 2

# 5. If OOM: Delete model, move to next smaller model
# 6. If quantization error: Delete model, try next method
# 7. If inference success: Compare accuracy to baseline
```

---

## Current Status Summary

| Model | Size | Base Created | Base Accuracy | AWQ | HQQ | GPTQ | SmoothQuant |
|-------|------|--------------|---------------|-----|-----|------|-------------|
| Qwen3-4B | 4.0B | ❌ | - | ❌ OOM | - | - | - |
| Llama-3.2-3B | 3.21B | ❌ | - | ❌ OOM | - | - | - |
| Qwen3-1.7B | 1.7B | ✅ | 33.33% | ❌ Bug | ❌ Load Error | 🔲 | 🔲 |

---

## Next Steps

1. **Try GPTQ on Qwen3-1.7B** (most reliable method from 0.6B testing)
2. **Try SmoothQuant on Qwen3-1.7B** (second most reliable)
3. **If both fail:** Document that 1.7B is minimum viable size and quantization method matters
4. **If both succeed:** Compare accuracy vs baseline (33.33%) to validate hypothesis

---

## Hypothesis Validation

**Original Hypothesis:** 0.6B failed because 21.4M params/layer is insufficient for W4 quantization (75% precision loss).

**Validation Criteria:**
- ✅ 1.7B (60.7M params/layer) should achieve >20% accuracy (vs 0% on 0.6B)
- ✅ At least one quantization method should work without technical errors
- ✅ Model should maintain reasonable format compliance (>50%)

**Current Evidence:**
- ✅ 1.7B base model shows 33.33% accuracy (2.8x more params/layer than 0.6B)
- 🟡 AWQ and HQQ failed due to implementation bugs, not model size
- 🔲 GPTQ/SmoothQuant pending - should work based on 0.6B experience
