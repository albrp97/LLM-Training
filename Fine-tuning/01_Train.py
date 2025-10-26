from datetime import datetime
import json
from typing import Optional, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, VeraConfig
from datasets import Dataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import (
    PTQ_METHODS,
    QuantMethod,
    QuantizationSpec,
    tag_quant,
)


# -----------------------------------------------------------
# User configuration
# -----------------------------------------------------------

train = True

DATASET_CHOICE = "openmath"
# options: None (saves base model), "openmath", "squad"
# arc results have no variation, so we will not test arc for now
# we will not test boolq for now

# Optional truncation switches (set to a positive int to limit samples).
TRUNC_TRAIN = None  # Number of training samples to keep; None/0 keeps all.

FINETUNING = "SFT"

MODEL_NAME = "Qwen/Qwen3-8B"

device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}


PEFT_CONFIG = "LoRa" 
# options: "NoPeft", "LoRa", "VeRa", "DoRa"
# -----------------------------------------------------------
# LoRa hyperparameters
# -----------------------------------------------------------

lora_r = 256
# 32 is 1.5% --
# 64 is 3%
# 128 is 5.8%
# 256 is 11% --
# 512 is 20%
# 1024 is 33% --

target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_alpha = 16
lora_dropout = 0.1

# -----------------------------------------------------------
# VeRa hyperparameters
# -----------------------------------------------------------

# VeRA parameter dimension (“rank”). Choose higher values than LoRA ranks here, since VeRA uses far fewer parameters than LoRA
vera_r = 512
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
vera_dropout = 0.1
# Initial init value for vera_lambda_d vector used when initializing the VeRA parameters. Small values (<=0.1) are recommended
vera_d_initial = 0.1

QUANT_METHOD = "QLORA"  
# options: "NoQuant", "QLORA", "GPTQ", "QuaRot", "AdaRound", "BRECQ", "AWQ", "HQQ", "SmoothQuant"

# Target settings used for PTQ pipelines (applied post-training via tools/quantize.py)
PTQ_TARGET_WEIGHTS_BITS = 4
PTQ_TARGET_GROUP_SIZE = 64
PTQ_TARGET_ACTS_BITS = 8
PTQ_TARGET_KV_BITS = 8

# Controls whether adapters are merged back into the base model on save.
MERGE_AFTER_TRAIN = True

# -----------------------------------------------------------
# Training hyperparameters
# -----------------------------------------------------------

# Batch size per GPU during training (effective batch size grows with gradient accumulation)
per_device_train_batch_size = 1

# Batch size per GPU during evaluation
per_device_eval_batch_size = 8

# Number of steps to accumulate gradients before performing a backward/update pass.
# Useful when GPU memory is limited and batch size is very small.
gradient_accumulation_steps = 8

# Learning rate for optimizer. Common values for fine-tuning range between 1e-5 and 1e-4.
learning_rate = 1e-5

# Weight decay for regularization (helps prevent overfitting)
weight_decay = 0.0

# Maximum allowed gradient norm (gradient clipping). Prevents exploding gradients.
max_grad_norm = 1.0

# Number of full passes (epochs) over the training dataset
num_train_epochs = 1

# Maximum training steps. If set, this overrides `num_train_epochs`. Otherwise, set to -1.
max_steps = -1

# Warmup ratio for the learning rate schedule (fraction of training used to "warm up")
warmup_ratio = 0.06

# Frequency (in steps) at which training logs are reported
logging_steps = 10

# If True, groups training samples by similar length for efficiency
group_by_length = False

# Enables gradient checkpointing to reduce memory usage.
# Slows training slightly but allows fine-tuning larger models.
gradient_checkpointing = True

# Concatenates multiple training examples into a single sequence to improve efficiency
packing = False

# If True, loss is only computed on assistant responses (ignores user/system messages)
assistant_only_loss = False

# Maximum sequence length (in tokens) for tokenized inputs
max_length = 1024


# -----------------------------------------------------------
# Aux functions
# -----------------------------------------------------------

def load_model_and_tokenizer(model_name, quantization_config, device_map):
    """Load the model/tokenizer pair with the appropriate quantisation plan."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {"device_map": device_map, "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        # BitsAndBytes handles dtype internally; avoid overriding.
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if not tokenizer.chat_template:
        tokenizer.chat_template = """{% for message in messages %}
                {% if message['role'] == 'system' %}System: {{ message['content'] }}\n
                {% elif message['role'] == 'user' %}User: {{ message['content'] }}\n
                {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }} <|endoftext|>
                {% endif %}
                {% endfor %}"""

    # Tokenizer config
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def preprocess_function(df, context: bool):
    if context:
        processed_data = df.apply(lambda row: {
            "prompt": [{"role": "user", "content": row["question"] + row["context"]}],
            "completion": [{"role": "assistant", "content": row["answer"]}]
        }, axis=1)
    else:
        processed_data = df.apply(lambda row: {
            "prompt": [{"role": "user", "content": row["question"]}],
            "completion": [{"role": "assistant", "content": row["answer"]}]
        }, axis=1)

    # Convert the resulta to HuggingFace Dataset
    return Dataset.from_list(processed_data.tolist())


def safe_serialize(obj):
    """Converts no serializable objects to serializable formats."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, SFTConfig):
        return {k: safe_serialize(v) for k, v in obj.to_dict().items()}
    elif isinstance(obj, BitsAndBytesConfig):
        return {k: safe_serialize(v) for k, v in vars(obj).items() if not k.startswith('_')}
    elif hasattr(obj, '__dict__'):
        return {k: safe_serialize(v) for k, v in vars(obj).items() if not k.startswith('_')}
    else:
        return str(obj)


def drop_nulls(obj):
    """Eliminate keys with None values recursively."""
    if isinstance(obj, dict):
        return {k: drop_nulls(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [drop_nulls(v) for v in obj if v is not None]
    else:
        return obj


def get_lm_head_dtype() -> str:
    """Return a friendly dtype label for LM head weights."""
    if torch.cuda.is_available():
        return "bf16"
    return "fp16"


def resolve_quantization_spec(method: QuantMethod) -> QuantizationSpec:
    """Map the configured method to a QuantizationSpec instance."""
    lm_head_dtype = get_lm_head_dtype()

    if method is QuantMethod.NO_QUANT:
        return QuantizationSpec(
            method=method,
            weights_bits=16,
            activations_bits=16,
            kv_cache_bits=16,
            group_size=None,
            symmetric=None,
            per_channel=None,
            lm_head_dtype=lm_head_dtype,
            backend="torch",
        )

    if method is QuantMethod.QLORA:
        return QuantizationSpec(
            method=method,
            weights_bits=4,
            activations_bits=16,
            kv_cache_bits=16,
            group_size=None,
            symmetric=None,
            per_channel=None,
            lm_head_dtype="bf16",
            backend="bitsandbytes",
            extras={
                "double_quant": True,
                "base_quant_type": "nf4",
            },
        )

    if method in PTQ_METHODS:
        ptq_backend_map = {
            QuantMethod.GPTQ: "autogptq",
            QuantMethod.AWQ: "awq",
            QuantMethod.HQQ: "hqq",
            QuantMethod.SMOOTH_QUANT: "custom",
            QuantMethod.QUA_ROT: "custom",
            QuantMethod.ADA_ROUND: "custom",
            QuantMethod.BRECQ: "custom",
        }
        backend = ptq_backend_map.get(method, "custom")
        extras = {
            "ptq_planned": True,
            "trained_weights_bits": 16,
            "trained_activations_bits": 16,
        }
        return QuantizationSpec(
            method=method,
            weights_bits=PTQ_TARGET_WEIGHTS_BITS,
            activations_bits=PTQ_TARGET_ACTS_BITS,
            kv_cache_bits=PTQ_TARGET_KV_BITS,
            group_size=PTQ_TARGET_GROUP_SIZE,
            symmetric=None,
            per_channel=None,
            lm_head_dtype="fp16",
            backend=backend,
            extras=extras,
        )

    raise ValueError(f"Unsupported quantization method '{method.value}'")


def build_quantization_plan(method: QuantMethod) -> Tuple[QuantizationSpec, Optional[BitsAndBytesConfig]]:
    """Create both metadata and runtime configs for the selected method."""
    spec = resolve_quantization_spec(method)

    if method is QuantMethod.QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        return spec, bnb_config

    return spec, None

# ============================================================
# Dataset selection
# ============================================================


if DATASET_CHOICE is None:
    quant_method = QuantMethod.from_any(QUANT_METHOD)
    quant_spec, quantization_config = build_quantization_plan(quant_method)
    quant_spec.extras.setdefault("merge_after_train", MERGE_AFTER_TRAIN)
    quant_tag = quant_spec.tag()
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, quantization_config, device_map)
    base_model_name = MODEL_NAME.split("/")[-1]
    new_model_name = f"{base_model_name}-base"
    output_dir = f"Models/{new_model_name}"

    total_params = 0
    trainable_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage_trainable = (trainable_params / total_params * 100) if total_params else 0.0

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    base_metadata = {
        "model_info": {
            "model_name": new_model_name,
            "base_model": MODEL_NAME,
            "fine_tuning_date": datetime.now().isoformat(),
            "model_type": "CausalLM",
            "quantization_tag": quant_tag,
            "has_quantization": quant_method is QuantMethod.QLORA,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable,
            "notes": "Base model snapshot without fine-tuning.",
        },
        "training_parameters": {
            "train": False,
            "dataset_choice": DATASET_CHOICE,
            "finetuning_strategy": None,
            "quantization_method": quant_method.value,
            "trunc_train": TRUNC_TRAIN or 0,
        },
        "peft_config": None,
        "quantization_config": safe_serialize(quantization_config) if quantization_config else None,
        "quantization": quant_spec.metadata(),
        "training_stats": {
            "total_steps": 0,
            "epochs_completed": 0,
        },
        "hardware_info": {
            "device": str(model.device),
            "dtype": str(model.dtype),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable,
            "vram_peaks": {
                "overall_max_reserved_gb": 0.0,
                "overall_max_allocated_gb": 0.0,
                "per_gpu": [],
            },
        },
    }

    clean_metadata = drop_nulls(base_metadata)

    with open(f"{output_dir}/training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(clean_metadata, f, indent=4, ensure_ascii=False)

    print(f"\nNo dataset selected. Saved base model and metadata to: {output_dir}\n")
    raise SystemExit(0)
elif DATASET_CHOICE == "arc":
    df = pd.read_parquet("Datasets/train-ai2_arc.parquet")
elif DATASET_CHOICE == "openmath":
    df = pd.read_parquet("Datasets/train-OpenMathInstruct-2.parquet")
elif DATASET_CHOICE == "boolq":
    df = pd.read_parquet("Datasets/train-boolq.parquet")
elif DATASET_CHOICE == "squad":
    df = pd.read_parquet("Datasets/train-squad_v2.parquet")
else:
    raise ValueError("Invalid DATASET_CHOICE")

if TRUNC_TRAIN and TRUNC_TRAIN > 0:
    df = df.head(TRUNC_TRAIN)

context = DATASET_CHOICE != "arc" and DATASET_CHOICE != "openmath"

print(f"\nLoaded dataset: {DATASET_CHOICE}\nNumber of samples: {len(df)}\n")

# --------------------------------------------
# QUANTIZATION METHOD
# --------------------------------------------

quant_method = QuantMethod.from_any(QUANT_METHOD)
quant_spec, quantization_config = build_quantization_plan(quant_method)
quant_spec.extras.setdefault("merge_after_train", MERGE_AFTER_TRAIN)

model, tokenizer = load_model_and_tokenizer(MODEL_NAME, quantization_config, device_map)

match PEFT_CONFIG:
    case "LoRa":
        PEFT_CONFIG += f"{lora_r}"
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    case "DoRa":
        PEFT_CONFIG += f"{lora_r}"
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True
        )
    case "VeRa":
        PEFT_CONFIG += f"{vera_r}"
        peft_config = VeraConfig(
            r=vera_r,
            target_modules=target_modules,
            vera_dropout=vera_dropout,
            d_initial=vera_d_initial,
            bias="none",
            task_type="CAUSAL_LM"
        )
    case _:
        peft_config = None

if peft_config is not None:
    if quant_method is QuantMethod.QLORA:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)


# Output folder where both checkpoints and the final fine-tuned model will be saved
model_name = MODEL_NAME.split("/")[-1]
quant_tag = quant_spec.tag()
new_model_name = f"{model_name}-{DATASET_CHOICE}_{FINETUNING}_{PEFT_CONFIG}_{quant_tag}"
output_dir = f"Models/{new_model_name}"

# Preprocess the dataset into tokenized format
train_dataset = preprocess_function(df, context)

# ===============================
# Training configuration
# ===============================
sft_config = SFTConfig(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    logging_steps=logging_steps,
    packing=packing,
    assistant_only_loss=assistant_only_loss,
    max_length=max_length,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
)


# ===============================
# Trainer initialization
# ===============================
sft_trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# Calculate total and trainable parameters
trainable_params = 0
total_params = 0
for _, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
percentage_trainable = trainable_params / total_params * 100

# Print a summary of the model
print("\nModel Summary:")
print(f"Model name: {MODEL_NAME}")
print(f"Number of parameters: {total_params / 1e9:.4f} billion")
print(f"Number of trainable parameters: {trainable_params / 1e9:.7f} billion")
print(f"Percentage of trainable parameters: {percentage_trainable:.7f}%")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Output directory: {output_dir}")
print(f"Quantization plan: {quant_tag} ({quant_method.value})")
print('\n========================================')

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")

    device = torch.device("cuda")
    current_gpu_name = torch.cuda.get_device_name(device.index)
    print(f"Using GPU: {current_gpu_name}")

    memory_stats = torch.cuda.memory_stats(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = memory_stats['allocated_bytes.all.current']
    reserved_memory = memory_stats['reserved_bytes.all.current']
    free_memory = total_memory - reserved_memory

    print(f"Total VRAM: {total_memory / (1024 ** 3):.2f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")

print('========================================\n')


# ---- VRAM helpers -----------------------------------------------------------
def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 3)

def reset_peak_vram():
    """Reset CUDA peak memory counters and clear cache on all GPUs."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

def get_peak_vram_gb():
    """
    Returns:
        {
          "overall_max_reserved_gb": float,
          "overall_max_allocated_gb": float,
          "per_gpu": [{"gpu": i, "peak_allocated_gb": .., "peak_reserved_gb": ..}, ...]
        }
    """
    if not torch.cuda.is_available():
        return {
            "overall_max_reserved_gb": 0.0,
            "overall_max_allocated_gb": 0.0,
            "per_gpu": []
        }
    per_gpu = []
    max_res = 0
    max_all = 0
    for i in range(torch.cuda.device_count()):
        peak_alloc = torch.cuda.max_memory_allocated(i)
        peak_res   = torch.cuda.max_memory_reserved(i)
        per_gpu.append({
            "gpu": i,
            "peak_allocated_gb": _bytes_to_gb(peak_alloc),
            "peak_reserved_gb":  _bytes_to_gb(peak_res),
        })
        max_res = max(max_res, peak_res)
        max_all = max(max_all, peak_alloc)
    return {
        "overall_max_reserved_gb": _bytes_to_gb(max_res),
        "overall_max_allocated_gb": _bytes_to_gb(max_all),
        "per_gpu": per_gpu
    }



# ===============================
# Training
# ===============================
if train:
    reset_peak_vram()
    
    sft_trainer.train()
    
    train_vram_peaks = get_peak_vram_gb()

    # ===============================
    # Save the fine-tuned model, tokenizer, and metadata
    # ===============================
    trained_model = sft_trainer.model
    if peft_config is not None and MERGE_AFTER_TRAIN:
        model = trained_model.merge_and_unload()
    else:
        model = trained_model

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    quant_spec.extras.setdefault("merge_after_train", MERGE_AFTER_TRAIN)
    training_params = safe_serialize(sft_trainer.args)
    if isinstance(training_params, dict):
        training_params["quantization_method"] = quant_method.value
        training_params["merge_after_train"] = MERGE_AFTER_TRAIN
        training_params["trunc_train"] = TRUNC_TRAIN or 0

    # Create complete metadata report
    training_metadata = {
        "model_info": {
            "model_name": new_model_name,
            "base_model": MODEL_NAME,
            "fine_tuning_date": datetime.now().isoformat(),
            "model_type": "CausalLM",
            "quantization_tag": quant_tag,
            "has_quantization": quant_method is QuantMethod.QLORA,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable
        },
        "training_parameters": training_params,
        "peft_config": safe_serialize(peft_config),
        "quantization_config": safe_serialize(quantization_config) if quantization_config else None,
        "quantization": quant_spec.metadata(),
        "training_stats": {
            "total_steps": sft_trainer.state.max_steps,
            "epochs_completed": sft_trainer.state.epoch,
        },
        "hardware_info": {
            "device": str(model.device),
            "dtype": str(model.dtype),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage_trainable": percentage_trainable,
            # New: VRAM peaks measured during training
            "vram_peaks": {
                "overall_max_reserved_gb": train_vram_peaks["overall_max_reserved_gb"],
                "overall_max_allocated_gb": train_vram_peaks["overall_max_allocated_gb"],
                "per_gpu": train_vram_peaks["per_gpu"],
            },
        },
    }

    clean_metadata = drop_nulls(training_metadata)

    # Save metadata to JSON file
    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(clean_metadata, f, indent=4, ensure_ascii=False)

    # Save model config
    model.config.save_pretrained(output_dir)

    print(f"\nModel saved in: {output_dir}")
    print("File structure:")
    print("  - model.safetensors")
    print("  - config.json")
    print("  - training_metadata.json")
    print("  - tokenizer files")
