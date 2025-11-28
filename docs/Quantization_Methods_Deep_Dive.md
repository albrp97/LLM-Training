# Advanced Quantization Methods: Technical Deep Dive

Detailed technical overview of the 6 quantization methods being compared in Phase 4.

---

## Summary Comparison Table

| Method | Type | Precision | Calibration Needed | Speed | Accuracy | Complexity | Library Maturity |
|--------|------|-----------|-------------------|-------|----------|------------|------------------|
| **GPTQ** | PTQ | 4/8-bit | Yes (small) | Fast | High | Medium | ⭐⭐⭐⭐⭐ |
| **AWQ** | PTQ | 4-bit | Yes (small) | Very Fast | High | Low | ⭐⭐⭐⭐⭐ |
| **HQQ** | PTQ | 4/8-bit | No | Very Fast | Medium | Low | ⭐⭐⭐⭐ |
| **SmoothQuant** | PTQ | 8-bit (W8A8) | Yes | Fast | High | Medium | ⭐⭐⭐ |
| **AdaRound** | PTQ | 4/8-bit | Yes | Slow | High | High | ⭐⭐⭐ |
| **QuaRot** | PTQ/Hybrid | 4-bit | No | Medium | High | Very High | ⭐⭐ (Research) |

---

## 1. GPTQ (Generative Pre-trained Transformer Quantization)

### Type
**Post-Training Quantization (PTQ)**

### Paper
- **Title**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- **Authors**: Elias Frantar et al.
- **Year**: 2022
- **Link**: https://arxiv.org/abs/2210.17323

### Core Concept

GPTQ quantizes weights using **layer-wise optimization** with approximate second-order information (Hessian matrix). It solves the quantization problem as:

```
min ||WX - Q(W)X||²
```

Where:
- `W` = original full-precision weights
- `Q(W)` = quantized weights
- `X` = calibration activations
- Goal: minimize reconstruction error

### Algorithm

1. **Input**: Pre-trained model, calibration dataset
2. **For each layer**:
   - Collect activations from calibration data
   - Compute Hessian (or approximation)
   - Quantize weights row-by-row using optimal quantization policy
   - Update remaining weights to compensate for errors
3. **Output**: Fully quantized model

### Key Features

- **Hessian-aware**: Uses second-order information (more accurate than naive rounding)
- **Layer-wise**: Processes one layer at a time (memory efficient)
- **Fast**: Quantizes large models in minutes to hours
- **Flexible**: Supports 2-bit, 3-bit, 4-bit, 8-bit

### Implementation

**Library**: `auto-gptq`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load model
model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,                    # 4-bit quantization
    group_size=128,            # Group size for quantization
    damp_percent=0.01,         # Damping for Hessian
    desc_act=False,            # Activation order
    sym=True,                  # Symmetric quantization
    true_sequential=True       # Sequential layer processing
)

# Load and quantize
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config
)

# Quantize with calibration data
model.quantize(calibration_dataset)

# Save
model.save_quantized("./llama-gptq-4bit")
```

### Pros
- ✅ High accuracy retention
- ✅ Fast quantization
- ✅ Well-tested and documented
- ✅ Hardware-accelerated inference available
- ✅ Supports various precisions

### Cons
- ❌ Requires calibration data
- ❌ Larger quantization overhead than HQQ
- ❌ Group-based quantization (not per-tensor)

### Expected Performance
- **Model size**: ~25% of original (4-bit) or ~50% (8-bit)
- **Accuracy drop**: 1-5% typical
- **Quantization time**: 5-30 minutes
- **Inference speedup**: 2-4× (with optimized kernels)

---

## 2. AWQ (Activation-aware Weight Quantization)

### Type
**Post-Training Quantization (PTQ)**

### Paper
- **Title**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
- **Authors**: Ji Lin et al. (MIT)
- **Year**: 2023
- **Link**: https://arxiv.org/abs/2306.00978

### Core Concept

AWQ identifies and **protects salient weights** based on activation magnitudes. Key insight:

> "Not all weights are equal; some are more important based on the activations they interact with."

The method:
1. Analyzes activation distributions
2. Identifies salient (important) channels
3. Scales weights channel-wise before quantization
4. Quantizes with minimal loss on critical paths

### Algorithm

1. **Activation Analysis**:
   - Run calibration data through model
   - Compute per-channel activation magnitudes: `s_i = ||X_i||`
   
2. **Channel Scaling**:
   - Scale weights: `W' = W · diag(s)^α`
   - Scale activations: `X' = X · diag(s)^(-α)`
   - Typically `α = 0.5` for balance

3. **Quantization**:
   - Quantize scaled weights `W'`
   - Store scale factors with model

4. **Inference**:
   - Apply quantized weights
   - Compensate with stored scales

### Key Features

- **Activation-aware**: Protects important channels
- **Per-channel scaling**: Fine-grained protection
- **No retraining**: Pure PTQ
- **Fast quantization**: Faster than GPTQ

### Implementation

**Library**: `autoawq`

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_name)

# Configure quantization
quant_config = {
    "zero_point": True,        # Use zero-point quantization
    "q_group_size": 128,       # Group size
    "w_bit": 4,                # 4-bit weights
    "version": "GEMM"          # Kernel version
}

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_dataset
)

# Save
model.save_quantized("./llama-awq-4bit")
```

### Pros
- ✅ Excellent accuracy preservation
- ✅ Very fast quantization (~minutes)
- ✅ Fast inference (optimized CUDA kernels)
- ✅ Simple to use
- ✅ Good documentation

### Cons
- ❌ Primarily 4-bit (limited precision options)
- ❌ Requires calibration data
- ❌ Less flexible than GPTQ

### Expected Performance
- **Model size**: ~25% of original
- **Accuracy drop**: 0.5-3% typical (better than GPTQ)
- **Quantization time**: 2-10 minutes
- **Inference speedup**: 3-4× (excellent kernel support)

---

## 3. HQQ (Half-Quadratic Quantization)

### Type
**Post-Training Quantization (PTQ)**

### Paper
- **Title**: "HQQ: Half-Quadratic Quantization of Large Machine Learning Models"
- **Authors**: Hicham Badri et al.
- **Year**: 2024
- **Link**: https://arxiv.org/abs/2406.09904

### Core Concept

HQQ reformulates quantization as a **half-quadratic optimization problem**:

```
min ||W - Q(W)||² + λ·R(Q)
```

Uses **iterative optimization** with alternating minimization:
1. Fix quantization, optimize scale/zero-point
2. Fix scale/zero-point, optimize quantization
3. Repeat until convergence

Key advantage: **No calibration data needed!**

### Algorithm

1. **Initialize**: Random or uniform quantization
2. **Iterate**:
   - Update quantization parameters (scale, zero-point)
   - Requantize weights
   - Check convergence
3. **Output**: Quantized weights + parameters

### Key Features

- **Calibration-free**: No data needed
- **Fast**: Very quick quantization
- **Flexible**: Supports various bit-widths
- **Simple**: Easy to implement

### Implementation

**Library**: `transformers` (HQQConfig) or standalone `hqq`

```python
from transformers import AutoModelForCausalLM, HQQConfig

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Configure HQQ
hqq_config = HQQConfig(
    nbits=4,                   # 4-bit
    group_size=64,             # Quantization group size
    quant_zero=True,           # Use zero-point
    quant_scale=False,         # Fixed scale
    axis=1                     # Quantization axis
)

# Load with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=hqq_config,
    device_map="auto"
)

# Already quantized on load!
```

### Pros
- ✅ **No calibration data needed**
- ✅ Very fast quantization
- ✅ Simple implementation
- ✅ Memory efficient
- ✅ Good for quick prototyping

### Cons
- ❌ Slightly lower accuracy than GPTQ/AWQ
- ❌ Less mature ecosystem
- ❌ Fewer optimization options

### Expected Performance
- **Model size**: ~25% (4-bit) or ~50% (8-bit)
- **Accuracy drop**: 3-7% typical
- **Quantization time**: 1-5 minutes (fastest)
- **Inference speedup**: 2-3× (depends on kernel support)

---

## 4. SmoothQuant

### Type
**Post-Training Quantization (PTQ)** - with activation quantization

### Paper
- **Title**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
- **Authors**: Guangxuan Xiao et al. (MIT)
- **Year**: 2023
- **Link**: https://arxiv.org/abs/2211.10438

### Core Concept

Most PTQ methods only quantize **weights** (W4A16 or W8A16). SmoothQuant enables **full INT8** (W8A8) by:

**Problem**: Activations have outliers that are hard to quantize.

**Solution**: **Migrate difficulty from activations to weights** via channel-wise scaling.

Formula:
```
Y = (Xdiag(s)^(-1)) · (diag(s)W) = X' · W'
```

Where:
- `s` = per-channel smoothing factors
- `X'` = smoothed activations (easier to quantize)
- `W'` = modified weights (absorb difficulty)

### Algorithm

1. **Analyze activations**: Find per-channel outlier magnitudes
2. **Compute smoothing factors**: `s_i = max(|X_i|)^α / max(|W_i|)^(1-α)`
3. **Apply smoothing**: 
   - Scale activations: `X' = X · diag(s)^(-1)`
   - Scale weights: `W' = diag(s) · W`
4. **Quantize both**: Use INT8 for weights and activations
5. **Fuse scales**: Integrate into layer parameters

### Key Features

- **W8A8**: Full INT8 (weights + activations)
- **Hardware-friendly**: Exploits INT8 tensor cores
- **Outlier handling**: Smooths activation distributions
- **Efficient**: Fast inference on modern GPUs

### Implementation

**Library**: Custom implementation or `smoothquant` package

```python
# Pseudo-code (implementation varies)
from smoothquant import smooth_lm

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply SmoothQuant
smoothed_model = smooth_lm(
    model,
    calibration_dataset,
    alpha=0.5,              # Balance factor
    smooth_method="channel" # Channel-wise smoothing
)

# Quantize to INT8
quantized_model = quantize_to_int8(smoothed_model)
```

### Pros
- ✅ Full INT8 (weights + activations)
- ✅ Hardware accelerated (Tensor Cores)
- ✅ High throughput
- ✅ Good accuracy retention

### Cons
- ❌ Requires calibration data
- ❌ More complex setup
- ❌ Less mature tooling
- ❌ Limited to 8-bit (not 4-bit)

### Expected Performance
- **Model size**: ~50% of original (8-bit)
- **Accuracy drop**: 2-5% typical
- **Quantization time**: 10-30 minutes
- **Inference speedup**: 2-3× (with INT8 Tensor Cores)

---

## 5. AdaRound (Adaptive Rounding)

### Type
**Post-Training Quantization (PTQ)**

### Paper
- **Title**: "Up or Down? Adaptive Rounding for Post-Training Quantization"
- **Authors**: Markus Nagel et al.
- **Year**: 2020 (ICML)
- **Link**: https://arxiv.org/abs/2004.10568

### Core Concept

Standard quantization **rounds weights naively** (floor/round). AdaRound **learns optimal rounding** via optimization.

For each weight `w`:
- Option 1: Round down (floor)
- Option 2: Round up (ceil)

AdaRound learns a **rounding decision** for each weight:
```
w_q = floor(w) + h(w)
```

Where `h(w) ∈ [0,1]` is learned via gradient descent.

### Algorithm

1. **Initialize**: Start with standard rounding
2. **Define loss**: 
   ```
   L = ||WX - W_qX||² + λ·Reg(h)
   ```
   Where `Reg(h)` encourages binary decisions
3. **Optimize**: Use gradient descent on `h`
4. **Finalize**: Binarize `h` (round to 0 or 1)
5. **Output**: Optimally rounded quantized weights

### Key Features

- **Learned rounding**: Better than naive floor/ceil
- **Layer-wise**: Processes each layer independently
- **Flexible**: Works with any bit-width
- **Accurate**: Better than standard rounding

### Implementation

**Library**: `neural-compressor` (Intel) or custom

```python
from neural_compressor import PostTrainingQuantConfig, quantization

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Configure AdaRound
config = PostTrainingQuantConfig(
    approach="post_training_static_quant",
    backend="default",
    recipes={
        "adaround": {
            "num_iterations": 1000,
            "alpha": 0.01
        }
    }
)

# Quantize
q_model = quantization.fit(
    model,
    config,
    calib_dataloader=calibration_data
)

# Save
q_model.save("./llama-adaround-4bit")
```

### Pros
- ✅ Better accuracy than naive rounding
- ✅ Works with any PTQ method
- ✅ Flexible bit-widths
- ✅ Theoretically grounded

### Cons
- ❌ Slower quantization (requires optimization)
- ❌ Requires calibration data
- ❌ More complex than basic PTQ
- ❌ Less mature tooling

### Expected Performance
- **Model size**: ~25% (4-bit) or ~50% (8-bit)
- **Accuracy drop**: 2-4% typical (better than naive)
- **Quantization time**: 30-60 minutes (slow)
- **Inference speedup**: 2-3× (standard kernels)

---

## 6. QuaRot (Quantization with Rotation)

### Type
**Hybrid PTQ** (with preprocessing)

### Paper
- **Title**: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
- **Authors**: Saleh Ashkboos et al. (IST Austria)
- **Year**: 2024
- **Link**: https://arxiv.org/abs/2404.00456

### Core Concept

**Problem**: Weight/activation outliers make quantization difficult.

**Solution**: **Apply rotation matrices** to eliminate outliers before quantization.

Key insight: Hadamard or random orthogonal rotations can **distribute outliers** uniformly:

```
Y = XW  →  Y = (XQ)(Q^T W)
```

Where:
- `Q` = orthogonal rotation matrix (e.g., Hadamard)
- `XQ` = rotated activations (no outliers)
- `Q^T W` = rotated weights (no outliers)
- Result is mathematically equivalent!

### Algorithm

1. **Choose rotation**: Hadamard or random orthogonal `Q`
2. **Rotate model**:
   - For each layer: `W_new = Q^T · W · Q`
   - Activations implicitly rotated during inference
3. **Quantize**: Now outlier-free, can use simple quantization
4. **Inference**: 
   - Apply rotation before quantized matmul
   - Rotate back after

### Key Features

- **Outlier elimination**: Distributes values uniformly
- **Simple quantization**: No complex schemes needed
- **4-bit friendly**: Enables aggressive compression
- **Novel approach**: Research-stage method

### Implementation

**Library**: Research code (custom implementation needed)

```python
# Pseudo-code (simplified)
from quarot import apply_rotation, quantize_rotated

model_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply Hadamard rotation
rotated_model = apply_rotation(
    model,
    rotation_type="hadamard",  # Or "random"
    online=False               # Pre-compute rotations
)

# Quantize (now outlier-free)
quantized_model = quantize_rotated(
    rotated_model,
    bits=4,
    method="uniform"  # Simple uniform quantization works!
)

# Save with rotation metadata
save_quarot_model(quantized_model, "./llama-quarot-4bit")
```

### Pros
- ✅ Handles outliers elegantly
- ✅ Enables simple quantization
- ✅ Good 4-bit accuracy
- ✅ Novel approach

### Cons
- ❌ Research-stage (limited tooling)
- ❌ Requires rotation overhead at inference
- ❌ More complex implementation
- ❌ Less tested than GPTQ/AWQ

### Expected Performance
- **Model size**: ~25% of original (4-bit)
- **Accuracy drop**: 2-5% typical
- **Quantization time**: 15-45 minutes
- **Inference speedup**: 2-3× (rotation overhead reduces gains)

---

## Method Selection Guide

### Choose GPTQ if:
- ✅ Need high accuracy
- ✅ Have calibration data
- ✅ Want mature, tested solution
- ✅ Need flexibility (2/3/4/8-bit)

### Choose AWQ if:
- ✅ Need best accuracy at 4-bit
- ✅ Want fast quantization
- ✅ Have optimized inference kernels
- ✅ Prefer simple workflow

### Choose HQQ if:
- ✅ Don't have calibration data
- ✅ Need very fast quantization
- ✅ Prototyping/experimenting
- ✅ Memory is constrained

### Choose SmoothQuant if:
- ✅ Need W8A8 (full INT8)
- ✅ Have INT8 tensor core hardware
- ✅ Want maximum throughput
- ✅ Can tolerate 8-bit only

### Choose AdaRound if:
- ✅ Standard PTQ accuracy too low
- ✅ Can afford longer quantization time
- ✅ Need best possible rounding
- ✅ Experimenting with novel approaches

### Choose QuaRot if:
- ✅ Model has severe outlier issues
- ✅ Interested in research methods
- ✅ Willing to implement custom code
- ✅ Need outlier-free 4-bit

---

## Compatibility Matrix

| Method | Llama 3.2 | GPU Needed | CPU Fallback | Calibration Data | Min Memory (1B model) |
|--------|-----------|------------|--------------|------------------|----------------------|
| GPTQ | ✅ Yes | Preferred | ✅ Yes | ✅ Required | ~2GB |
| AWQ | ✅ Yes | Preferred | ⚠️ Limited | ✅ Required | ~2GB |
| HQQ | ✅ Yes | Optional | ✅ Yes | ❌ Not needed | ~1.5GB |
| SmoothQuant | ✅ Yes | ✅ Required | ❌ No | ✅ Required | ~2GB |
| AdaRound | ✅ Yes | Preferred | ✅ Yes | ✅ Required | ~3GB |
| QuaRot | ⚠️ Experimental | ✅ Required | ❌ No | ❌ Not needed | ~2.5GB |

---

## References

1. **GPTQ**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for GPTs" (2022)
2. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression" (2023)
3. **HQQ**: Badri et al., "HQQ: Half-Quadratic Quantization of Large Machine Learning Models" (2024)
4. **SmoothQuant**: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization" (2023)
5. **AdaRound**: Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization" (2020)
6. **QuaRot**: Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" (2024)

---

**Last Updated**: November 28, 2025  
**Phase**: 4 - Advanced Quantization Methods
