from datetime import datetime
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig, GPTQConfig, BitsAndBytesConfig, pipeline
from trl import SFTTrainer, SFTConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset


# -----------------------------------------------------------
# User configuration
# -----------------------------------------------------------

DATASET_CHOICE = "openmath"       # options: "arc", "boolq", "squad", "openmath"   
FINETUNING = "SFT"

MODEL_NAME = "Qwen/Qwen3-0.6B"

QUANT_METHOD = "None_Lora32"  # options: "None", "QLORA", "AWQ", "GPTQ"

device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

# -----------------------------------------------------------
# LoRA hyperparameters
# -----------------------------------------------------------

lora_r = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_alpha = 16
lora_dropout = 0.1


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

def load_model_and_tokenizer(model_name, quantization_config, device_map):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    
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

# ============================================================
# Dataset selection
# ============================================================

if DATASET_CHOICE == "arc":
    df = pd.read_parquet("Datasets/train-ai2_arc.parquet")
elif DATASET_CHOICE == "openmath":
    df = pd.read_parquet("Datasets/train-OpenMathInstruct-2.parquet")
elif DATASET_CHOICE == "boolq":
    df = pd.read_parquet("Datasets/train-boolq.parquet")
elif DATASET_CHOICE == "squad":
    df = pd.read_parquet("Datasets/train-squad_v2.parquet")
else:
    raise ValueError("Invalid DATASET_CHOICE")

context = DATASET_CHOICE != "arc" and DATASET_CHOICE != "openmath"

print(f"\nLoaded dataset: {DATASET_CHOICE}\nNumber of samples: {len(df)}\n")

# --------------------------------------------
# QUANTIZATION METHOD
# --------------------------------------------

match QUANT_METHOD:
    case "QLORA":
        load_in_4bit = True # Load the model in 4-bit
        bnb_4bit_quant_type = "nf4"
        bnb_4bit_use_double_quant = True # Saves more memory at no additional performance
        bnb_4bit_compute_dtype = torch.bfloat16 


        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )

    case "AWQ":
        bits = 4
        fuse_max_seq_len = 512
        do_fuse = True

        quantization_config = AwqConfig(
            bits=bits,
            fuse_max_seq_len=fuse_max_seq_len,
            do_fuse=do_fuse,
        )

    case "GPTQ":
        bits = 4
        v2 = True # Activate GPTQ v2
        
        quantization_config = GPTQConfig(
            bits=bits, 
            dataset=dataset, 
            tokenizer=tokenizer,
            v2=v2
        )


    case "adaround":
        raise NotImplementedError("Implement AdaRound here")

    case "brecq":
        raise NotImplementedError("Implement BRECQ here")

    case "quarot":
        raise NotImplementedError("Implement QuaRot here")

    case _:
        quantization_config = None
        
model, tokenizer = load_model_and_tokenizer(MODEL_NAME, quantization_config, device_map)
model = prepare_model_for_kbit_training(model)



config = LoraConfig(
    r=lora_r, 
    lora_alpha=lora_alpha, 
    target_modules=target_modules, 
    lora_dropout=lora_dropout, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# Output folder where both checkpoints and the final fine-tuned model will be saved
model_name = MODEL_NAME.split("/")[-1]
new_model_name = f"{model_name}-{DATASET_CHOICE}_{FINETUNING}_{QUANT_METHOD}" 
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
    max_steps=max_steps
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
print(f"Hidden size: {model.config.hidden_size}\n")

print('========================================')

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


# ===============================
# Training
# ===============================
sft_trainer.train()

# ===============================
# Save the fine-tuned model, tokenizer, and metadata
# ===============================

model = sft_trainer.model.merge_and_unload()
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

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


# Create complete metadata report
training_metadata = {
    "model_info": {
        "model_name": new_model_name,
        "base_model": MODEL_NAME,
        "fine_tuning_date": datetime.now().isoformat(),
        "model_type": "CausalLM",
        "has_quantization": quantization_config is not None,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "percentage_trainable": percentage_trainable
    },
    "training_parameters": safe_serialize(sft_trainer.args),
    "lora_config": safe_serialize(config),
    "quantization_config": safe_serialize(quantization_config) if quantization_config else None,
    "training_stats": {
        "total_steps": sft_trainer.state.max_steps,
        "epochs_completed": sft_trainer.state.epoch,
    },
    "hardware_info": {
        "device": str(model.device),
        "dtype": str(model.dtype),
    }
}

clean_metadata = drop_nulls(training_metadata)

# Save metadata to JSON file
with open(f"{output_dir}/training_metadata.json", "w") as f:
    json.dump(clean_metadata, f, indent=4, ensure_ascii=False)

# Save model config
model.config.save_pretrained(output_dir)

print(f"\nModel saved in: {output_dir}")
print("File structure:")
print(f"  - model.safetensors")
print(f"  - config.json")
print(f"  - training_metadata.json")
print(f"  - tokenizer files")