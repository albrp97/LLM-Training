import re
import json
import math
import argparse
import gc
import torch
from datetime import datetime
from pathlib import Path
import pandas as pd
import string
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm.auto import tqdm
import statistics as stats
import time
from collections import Counter


datasetTrunc = None
VERBOSE = False  # set to False to hide per-dataset logs and summary prints

datasets_info = {
    "test-ai2_arc.parquet": {
        "system_prompt": (
            "You are taking a multiple-choice test.\n"
            "Each question will have exactly 4 options: A, B, C or D.\n"
            "Read the question and choose the correct answer.\n"
            "Output only the letter of the correct answer inside \\boxed{}, like this: \\boxed{C}"
        ),
        "context": False,
        "task": "mcq4",
    },
    "test-boolq.parquet": {
        "system_prompt": (
            "You are answering a True/False question.\n"
            "The question will be accompanied by a short passage of context.\n"
            "Your answer must be either False or True.\n"
            "Output your answer inside \\boxed{}, like this: \\boxed{True}"
        ),
        "context": True,
        "task": "boolq",
    },
    "test-squad_v2.parquet": {
        "system_prompt": (
            "You are answering a question based on a passage.\n"
            "Read the context carefully and provide the exact answer span from the passage.\n"
            "Do not add extra words or explanations.\n"
            "Output your answer inside \\boxed{}, like this: \\boxed{Einstein}"
        ),
        "context": True,
        "task": "squad_v2",
    },
    "test-OpenMathInstruct-2.parquet": {
        "system_prompt": (
            "You are solving math problems.\n"
            "Read each problem carefully and provide only the final numeric answer.\n"
            "Output your answer inside \\boxed{}, like this: \\boxed{42}\n"
            "Do not include any explanation or intermediate steps."
        ),
        "context": False,
        "task": "math_numeric",
    },
}

device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}


# Define custom load function
def load_custom_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer


# Define chat function
def chat(model, tokenizer, user_prompt, system_prompt, max_new_tokens=1000):
    messages = []

    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "system", "content": system_prompt})
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.max_position_embeddings,
        enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_k=None,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
        )

    model_response = tokenizer.decode(
        outputs[0][len(inputs[0]) :], skip_special_tokens=True
    )

    return model_response


def evaluate_answer(model_output: str, correct_answer: str) -> bool:
    match = re.search(r"\\boxed\{(.+?)\}", model_output)
    if not match:
        return False  # No valid boxed answer found

    extracted = match.group(1).upper()
    is_correct = extracted == correct_answer.upper()

    return is_correct


def vlog(msg: str):
    if VERBOSE:
        tqdm.write(msg)


def _safe_filename(name: str) -> str:
    # If it's a HuggingFace repo id like "org/model" (optionally with a revision "@rev"),
    # keep both parts but replace "/" with "__".
    if re.match(r"^[\w\-]+/[\w\.\-]+(@[\w\.\-]+)?$", name):
        base = name.replace("/", "__")
    else:
        # Treat as local path → keep only the last component (folder/file name)
        base = Path(name).name

    # Sanitize anything that Windows wouldn't like or that could create dirs
    base = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", base)
    return base


def _mode(values):
    if not values:
        return None
    try:
        return stats.mode(values)  # unique mode
    except stats.StatisticsError:
        mm = stats.multimode(values)
        return mm[0] if mm else None


# METRICS
# ---------------------------
# Parsing + normalization helpers
# ---------------------------
_BOXED_RE = re.compile(r"\\boxed\{\s*(.*?)\s*\}", flags=re.DOTALL)


def extract_boxed(text: str):
    if not isinstance(text, str):
        return None
    m = _BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def normalize_arc_option(s: str):
    if s is None:
        return None
    t = s.strip().upper()
    if len(t) > 1:
        for ch in t:
            if ch in {"A", "B", "C", "D"}:
                return ch
        return None
    return t if t in {"A", "B", "C", "D"} else None


def normalize_bool(s: str):
    if s is None:
        return None
    t = s.strip().lower()
    if t in {"true", "t", "yes", "y", "1"}:
        return True
    if t in {"false", "f", "no", "n", "0"}:
        return False
    return None


_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_EMPTY_ANSWERS = {"", "unanswerable", "unknown", "no answer", "n a", "none", "null"}


def squad_normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    tokens = [w for w in s.split() if w not in _ARTICLES]
    return " ".join(tokens)


def squad_em(pred: str, golds):
    pn = squad_normalize(pred)
    for g in golds if isinstance(golds, list) else [golds]:
        if pn == squad_normalize(str(g)):
            return 1
    return 0


def squad_f1(pred: str, golds):
    pn = squad_normalize(pred).split()
    best = 0.0
    for g in golds if isinstance(golds, list) else [golds]:
        gn = squad_normalize(str(g)).split()
        if not pn and not gn:
            best = max(best, 1.0)
            continue
        if not pn or not gn:
            best = max(best, 0.0)
            continue
        common = Counter(pn) & Counter(gn)
        num_same = sum(common.values())
        if num_same == 0:
            best = max(best, 0.0)
            continue
        precision = num_same / len(pn)
        recall = num_same / len(gn)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def is_empty_like(s: str) -> bool:
    return squad_normalize(s) in _EMPTY_ANSWERS


# ---------------------------
# Unified metrics dispatcher
# ---------------------------
def compute_metrics(kind: str, golds, preds):
    """
    kind: "mcq4" | "boolq" | "squad_v2" | "generic"
    golds/preds: lists aligned with dataset kind
    Returns: {
        correct, incorrect, accuracy,
        metrics: {...}   # task-specific fields
    }
    """
    if kind == "mcq4":
        classes = ["A", "B", "C", "D"]
        idx = {c: i for i, c in enumerate(classes)}
        cm = [[0] * 4 for _ in range(4)]
        correct = total = 0
        for t, p in zip(golds, preds):
            total += 1
            if p is not None and t == p:
                correct += 1
            if (t in idx) and (p in idx):
                cm[idx[t]][idx[p]] += 1
        per_class = {}
        f1s = []
        for i, c in enumerate(classes):
            tp = cm[i][i]
            fp = sum(cm[r][i] for r in range(4)) - tp
            fn = sum(cm[i][r] for r in range(4)) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
            f1s.append(f1)
        acc = (correct / total * 100) if total else 0.0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        return {
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(acc, 2),
            "metrics": {
                "task": "ARC (4-way MCQ)",
                "macro_f1": round(macro_f1, 4),
                "per_class": per_class,
                "confusion_matrix": {
                    classes[i]: {classes[j]: cm[i][j] for j in range(4)}
                    for i in range(4)
                },
                "support": total,
            },
        }

    if kind == "boolq":
        TP = TN = FP = FN = 0
        for t, p in zip(golds, preds):
            if p is None:
                if t:
                    FN += 1
                else:
                    FP += 1
            elif t and p:
                TP += 1
            elif (not t) and (not p):
                TN += 1
            elif (not t) and p:
                FP += 1
            elif t and (not p):
                FN += 1
        total = TP + TN + FP + FN
        acc = (TP + TN) / total * 100 if total else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        prec_n = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        rec_n = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0.0
        macro_f1 = (f1 + f1_n) / 2
        tpr = rec
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        bal_acc = (tpr + tnr) / 2
        denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = ((TP * TN - FP * FN) / denom) if denom > 0 else 0.0
        return {
            "correct": TP + TN,
            "incorrect": FP + FN,
            "accuracy": round(acc, 2),
            "metrics": {
                "task": "BoolQ (binary)",
                "f1_pos": round(f1, 4),
                "macro_f1": round(macro_f1, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "MCC": round(mcc, 4),
                "confusion_matrix": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
                "support": total,
            },
        }

    if kind == "squad_v2":
        ems, f1s, has_g, has_p = [], [], [], []
        for gold, pred in zip(golds, preds):
            ems.append(squad_em(pred, gold))
            f1s.append(squad_f1(pred, gold))
            g_has = not all(
                is_empty_like(str(g))
                for g in (gold if isinstance(gold, list) else [gold])
            )
            p_has = not is_empty_like(pred)
            has_g.append(g_has)
            has_p.append(p_has)
        total = len(golds)

        def _avg(x):
            return sum(x) / len(x) if x else 0.0

        EM = _avg(ems) * 100
        F1 = _avg(f1s) * 100
        has_idx = [i for i, g in enumerate(has_g) if g]
        no_idx = [i for i, g in enumerate(has_g) if not g]
        HasAns_EM = _avg([ems[i] for i in has_idx]) * 100 if has_idx else 0.0
        HasAns_F1 = _avg([f1s[i] for i in has_idx]) * 100 if has_idx else 0.0
        NoAns_Acc = (
            _avg([1.0 if not has_p[i] else 0.0 for i in no_idx]) * 100
            if no_idx
            else 0.0
        )
        # AvNA (answer vs no-answer)
        TP = TN = FP = FN = 0
        for g, p in zip(has_g, has_p):
            if g and p:
                TP += 1
            elif (not g) and (not p):
                TN += 1
            elif (not g) and p:
                FP += 1
            elif g and (not p):
                FN += 1
        AvNA = (TP + TN) / total * 100 if total else 0.0
        # define "correct" as exact matches
        correct = int(sum(ems))
        return {
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(EM, 2),  # EM as accuracy
            "metrics": {
                "task": "SQuAD v2",
                "EM": round(EM, 2),
                "F1": round(F1, 2),
                "HasAns_EM": round(HasAns_EM, 2),
                "HasAns_F1": round(HasAns_F1, 2),
                "NoAns_Accuracy": round(NoAns_Acc, 2),
                "AvNA_Accuracy": round(AvNA, 2),
                "support": total,
            },
        }
    if kind == "math_numeric":
        # Comparar respuestas numéricas con tolerancia
        correct = 0
        total = len(golds)
        incorrect = 0
        tolerance = 1e-4  # puedes ajustar la tolerancia si lo deseas
        diffs = []
        for gold, pred in zip(golds, preds):
            try:
                gold_num = float(gold)
                pred_num = float(pred) if pred is not None and pred != "" else None
                if pred_num is not None and abs(gold_num - pred_num) < tolerance:
                    correct += 1
                else:
                    incorrect += 1
                if pred_num is not None:
                    diffs.append(abs(gold_num - pred_num))
            except Exception:
                incorrect += 1
                diffs.append(None)
        acc = correct / total * 100 if total else 0.0
        avg_diff = (
            sum([d for d in diffs if d is not None])
            / len([d for d in diffs if d is not None])
            if diffs
            else None
        )
        return {
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(acc, 2),
            "metrics": {
                "task": "OpenMathInstruct-2 (numeric)",
                "avg_abs_diff": round(avg_diff, 6) if avg_diff is not None else None,
                "support": total,
            },
        }

    # generic fallback: string equality
    correct = sum(1 for g, p in zip(golds, preds) if (p is not None and p == g))
    total = len(golds)
    acc = correct / total * 100 if total else 0.0
    return {
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(acc, 2),
        "metrics": {"task": "Generic", "support": total},
    }


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

def _bytes_to_gb(n: int) -> float:
    return round(n / (1024 ** 3), 3)

def print_vram_report(title: str = ""):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    if title:
        print(f"\n=== VRAM Report: {title} @ {datetime.now().strftime('%H:%M:%S')} ===")
    else:
        print(f"\n=== VRAM Report @ {datetime.now().strftime('%H:%M:%S')} ===")

    torch.cuda.synchronize()
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)  # bytes
        used_total = total - free  # everything used on the GPU
        alloc = torch.cuda.memory_allocated(i)  # tensors by *this* process
        reserved = torch.cuda.memory_reserved(i)  # cached by PyTorch for reuse

        print(f"[GPU {i}] {name}")
        print(f"  Total:  {_fmt_bytes(total)}")
        print(f"  Used*:  {_fmt_bytes(used_total)}   (*overall, all processes)")
        print(f"  Free:   {_fmt_bytes(free)}")
        print(f"  PyTorch Allocated: {_fmt_bytes(alloc)}   (your tensors)")
        print(f"  PyTorch Reserved:  {_fmt_bytes(reserved)} (cache for reuse)")
        # Peak stats since last reset (see helpers below)
        try:
            peak_alloc = torch.cuda.max_memory_allocated(i)
            peak_reserved = torch.cuda.max_memory_reserved(i)
            print(f"  Peak Allocated:    {_fmt_bytes(peak_alloc)}")
            print(f"  Peak Reserved:     {_fmt_bytes(peak_reserved)}")
        except Exception:
            pass
    print()


def reset_vram():
    gc.collect()

    # actually release CUDA caches on every GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on benchmark datasets"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Path to the model directory or model name to evaluate",
    )
    return parser.parse_args()


def evaluate_model(model_name: str, datasetTrunc: int = None, verbose: bool = False):
    """Evaluate a model on all test datasets"""
    metrics_dir = Path("Testing/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = metrics_dir / f"{_safe_filename(model_name)}.json"
    if metrics_path.exists():
        print(f"Metrics for model '{model_name}' already exist at {metrics_path}. Skipping evaluation.")
        return

    print("------------\n")
    print(f"Evaluating model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map=device_map
    )
    
    # Start a fresh peak measurement for this model's evaluation
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.reset_peak_memory_stats()

    # Load datasets
    datasets = list(Path("Datasets").glob("test-*.parquet"))
    loaded = [(ds, pd.read_parquet(ds)) for ds in datasets]
    if datasetTrunc is not None:
        loaded = [(ds, df.head(datasetTrunc)) for ds, df in loaded]

    total_samples = sum(len(df) for _, df in loaded)
    per_dataset = {}
    all_latencies, all_tokens = [], []

    if verbose:
        tqdm.write(f"Model: {model_name}")

    with tqdm(
        total=total_samples,
        desc=Path(model_name).name,
        dynamic_ncols=True,
        bar_format="{desc} ({n_fmt}/{total_fmt}) |{bar}| {percentage:3.0f}% {rate_fmt} {elapsed}<{remaining}",
        leave=True,
    ) as pbar:
        per_dataset = evaluate_datasets(model, tokenizer, loaded, all_latencies, all_tokens, pbar, verbose)
        # Capture peak VRAM during evaluation (for this process)
        hardware = {}
        if torch.cuda.is_available():
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
            hardware = {
                "peak_vram_reserved_gb": _bytes_to_gb(max_res),
                "peak_vram_allocated_gb": _bytes_to_gb(max_all),
                "per_gpu": per_gpu,
            }


    # Generate final report
    model_report = generate_model_report(model_name, per_dataset, all_latencies, all_tokens)
    
    if hardware:
        model_report["hardware"] = hardware
        # Convenience copy for tools expecting summary-only fields
        model_report.setdefault("summary", {})["peak_vram_reserved_gb"]  = hardware["peak_vram_reserved_gb"]
        model_report["summary"]["peak_vram_allocated_gb"] = hardware["peak_vram_allocated_gb"]


    # Save results
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(model_report, f, indent=4, ensure_ascii=False)

    if verbose:
        print(f"💾 Métricas guardadas en {metrics_path}")
    else:
        print(f"saved at {metrics_path}")


def evaluate_datasets(
    model, tokenizer, loaded_datasets, all_latencies, all_tokens, pbar, verbose
):
    """Evaluate model on multiple datasets"""
    per_dataset = {}

    for dataset_path, df in loaded_datasets:
        dataset_name = dataset_path.name
        info = datasets_info.get(
            dataset_name, {"system_prompt": "", "context": False, "task": "generic"}
        )
        kind = info.get("task", "generic")

        dataset_results = evaluate_single_dataset(
            model, tokenizer, df, info, kind, all_latencies, all_tokens, pbar
        )
        per_dataset[dataset_name] = dataset_results

        log_dataset_results(dataset_name, kind, dataset_results, verbose)

    return per_dataset


def evaluate_single_dataset(
    model, tokenizer, df, info, kind, all_latencies, all_tokens, pbar
):
    """Evaluate model on a single dataset"""
    latencies, tokens_out = [], []
    format_ok = 0
    golds, preds = [], []

    for _, row in df.iterrows():
        user_prompt = row["question"]
        if info["context"]:
            user_prompt += row["context"]

        # Get model response and metrics
        t0 = time.perf_counter()
        response = chat(model, tokenizer, user_prompt, info["system_prompt"])
        dt = time.perf_counter() - t0
        latencies.append(dt)

        try:
            tok_count = len(tokenizer.encode(response, add_special_tokens=False))
        except Exception:
            tok_count = 0
        tokens_out.append(tok_count)

        # Process prediction
        pred_boxed = extract_boxed(response)
        if pred_boxed is not None:
            format_ok += 1

        # Collect gold/pred by kind
        process_prediction(kind, row, pred_boxed, golds, preds)

        pbar.update(1)

    # Aggregate metrics
    all_latencies.extend(latencies)
    all_tokens.extend(tokens_out)

    return calculate_dataset_metrics(
        kind, golds, preds, latencies, tokens_out, format_ok
    )


def process_prediction(kind, row, pred_boxed, golds, preds):
    """Process a single prediction based on task type"""
    if kind == "mcq4":
        golds.append(normalize_arc_option(str(row["answer"])))
        preds.append(normalize_arc_option(pred_boxed))
    elif kind == "boolq":
        golds.append(True if str(row["answer"]).strip().lower() == "true" else False)
        preds.append(normalize_bool(pred_boxed))
    elif kind == "squad_v2":
        golds.append(row["answer"])
        preds.append((pred_boxed or "").strip())
    else:
        golds.append(str(row["answer"]).strip())
        preds.append((pred_boxed or "").strip())


def calculate_dataset_metrics(kind, golds, preds, latencies, tokens_out, format_ok):
    """Calculate metrics for a dataset"""
    lat_metrics = calculate_latency_metrics(latencies)
    tok_metrics = calculate_token_metrics(tokens_out)

    res = compute_metrics(kind, golds, preds)
    total = len(golds)
    format_rate = (format_ok / total * 100) if total else 0.0

    return {
        "type": kind,
        "total_samples": total,
        "format_success_rate": round(format_rate, 2),
        "correct": int(res["correct"]),
        "incorrect": int(res["incorrect"]),
        "accuracy": round(res["accuracy"], 2),
        "metrics": res["metrics"],
        "latency_seconds": lat_metrics,
        "tokens_generated": tok_metrics,
    }


def generate_model_report(model_name, per_dataset, all_latencies, all_tokens):
    """Generate the final model report"""
    grand_total = sum(v["total_samples"] for v in per_dataset.values())
    grand_correct = sum(v["correct"] for v in per_dataset.values())
    grand_incorrect = sum(v["incorrect"] for v in per_dataset.values())
    overall_acc = (grand_correct / grand_total * 100) if grand_total else 0.0

    return {
        "model_name": model_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "num_datasets": len(per_dataset),
            "total_samples": grand_total,
            "total_correct": grand_correct,
            "total_incorrect": grand_incorrect,
            "overall_accuracy": round(overall_acc, 2),
            "latency_seconds": calculate_latency_metrics(all_latencies),
            "tokens_generated": calculate_token_metrics(all_tokens),
        },
        "datasets": per_dataset,
    }


def calculate_latency_metrics(latencies):
    """Calculate latency metrics"""
    if not latencies:
        return {"sum": 0, "mean": 0, "median": 0, "mode": None, "std": 0}

    lat_sum = float(sum(latencies))
    return {
        "per_prompt": [round(x, 4) for x in latencies],
        "sum": round(lat_sum, 4),
        "mean": round(lat_sum / len(latencies), 4),
        "median": round(stats.median(latencies), 4),
        "mode": _mode([round(x, 3) for x in latencies]),
        "std": round(stats.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
    }


def calculate_token_metrics(tokens):
    """Calculate token metrics"""
    if not tokens:
        return {"sum": 0, "mean": 0, "median": 0, "mode": None, "std": 0}

    tok_sum = int(sum(tokens))
    return {
        "per_prompt": tokens,
        "sum": tok_sum,
        "mean": round(tok_sum / len(tokens), 2),
        "median": round(stats.median(tokens), 2),
        "mode": _mode(tokens),
        "std": round(stats.stdev(tokens), 2) if len(tokens) > 1 else 0.0,
    }


def log_dataset_results(dataset_name, kind, results, verbose):
    """Log results for a dataset if verbose mode is enabled"""
    if not verbose:
        return

    vlog("*" * 30)
    vlog(f"Dataset: {dataset_name} [{kind}]")

    if kind == "mcq4":
        vlog(
            f"🎯 Acc {results['accuracy']:.2f}% | Macro-F1 {results['metrics']['macro_f1']:.4f}"
        )
    elif kind == "boolq":
        vlog(
            f"🎯 Acc {results['accuracy']:.2f}% | F1(pos) {results['metrics']['f1_pos']:.4f} | MCC {results['metrics']['MCC']:.4f}"
        )
    elif kind == "squad_v2":
        vlog(
            f"🎯 EM {results['metrics']['EM']:.2f}% | F1 {results['metrics']['F1']:.2f}% | AvNA {results['metrics']['AvNA_Accuracy']:.2f}%"
        )
    else:
        vlog(f"🎯 Acc {results['accuracy']:.2f}%")


def main():
    args = parse_args()
    evaluate_model(args.model_name, datasetTrunc, VERBOSE)


if __name__ == "__main__":
    main()
