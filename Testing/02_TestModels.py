import re
import json
import math
from pathlib import Path
import pandas as pd
import torch
import string
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils.logging import set_verbosity_error
from tqdm.auto import tqdm
from datetime import datetime
import statistics as stats
import time
from collections import Counter, defaultdict
from typing import List, Union, Dict, Any

datasetTrunc=None

device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
# Define custom load function
def load_custom_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tokenizer

# Define chat function
def chat(model, tokenizer, user_prompt, system_prompt, max_new_tokens=1000):
    messages = []
    
    messages.append({'role': 'user', 'content': user_prompt})
    messages.append({'role': 'system', 'content': system_prompt})
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings, enable_thinking=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, top_k=None, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, temperature=None, top_p=None)

    model_response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    

    return model_response

def evaluate_answer(model_output: str, correct_answer: str) -> bool:
    match = re.search(r"\\boxed\{(.+?)\}", model_output)
    if not match:
        return False  # No valid boxed answer found

    extracted = match.group(1).upper()
    is_correct = extracted == correct_answer.upper()

    return is_correct

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
    }

}


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
    if s is None: return None
    t = s.strip().lower()
    if t in {"true","t","yes","y","1"}: return True
    if t in {"false","f","no","n","0"}: return False
    return None

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_EMPTY_ANSWERS = {"", "unanswerable", "unknown", "no answer", "n a", "none", "null"}

def squad_normalize(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    tokens = [w for w in s.split() if w not in _ARTICLES]
    return " ".join(tokens)

def squad_em(pred: str, golds):
    pn = squad_normalize(pred)
    for g in (golds if isinstance(golds, list) else [golds]):
        if pn == squad_normalize(str(g)):
            return 1
    return 0

def squad_f1(pred: str, golds):
    pn = squad_normalize(pred).split()
    best = 0.0
    for g in (golds if isinstance(golds, list) else [golds]):
        gn = squad_normalize(str(g)).split()
        if not pn and not gn:
            best = max(best, 1.0); continue
        if not pn or not gn:
            best = max(best, 0.0); continue
        common = Counter(pn) & Counter(gn)
        num_same = sum(common.values())
        if num_same == 0:
            best = max(best, 0.0); continue
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
        classes = ["A","B","C","D"]; idx = {c:i for i,c in enumerate(classes)}
        cm = [[0]*4 for _ in range(4)]
        correct = total = 0
        for t, p in zip(golds, preds):
            total += 1
            if p is not None and t == p:
                correct += 1
            if (t in idx) and (p in idx):
                cm[idx[t]][idx[p]] += 1
        per_class = {}
        f1s = []
        for i,c in enumerate(classes):
            tp = cm[i][i]
            fp = sum(cm[r][i] for r in range(4)) - tp
            fn = sum(cm[i][r] for r in range(4)) - tp
            prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
            per_class[c] = {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4)}
            f1s.append(f1)
        acc = (correct/total*100) if total else 0.0
        macro_f1 = sum(f1s)/len(f1s) if f1s else 0.0
        return {
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(acc, 2),
            "metrics": {
                "task": "ARC (4-way MCQ)",
                "macro_f1": round(macro_f1,4),
                "per_class": per_class,
                "confusion_matrix": {classes[i]: {classes[j]: cm[i][j] for j in range(4)} for i in range(4)},
                "support": total,
            }
        }

    if kind == "boolq":
        TP=TN=FP=FN=0
        for t,p in zip(golds, preds):
            if p is None:
                if t: FN += 1
                else: FP += 1
            elif t and p: TP += 1
            elif (not t) and (not p): TN += 1
            elif (not t) and p: FP += 1
            elif t and (not p): FN += 1
        total = TP+TN+FP+FN
        acc = (TP+TN)/total*100 if total else 0.0
        prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
        rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        prec_n = TN/(TN+FN) if (TN+FN)>0 else 0.0
        rec_n  = TN/(TN+FP) if (TN+FP)>0 else 0.0
        f1_n   = 2*prec_n*rec_n/(prec_n+rec_n) if (prec_n+rec_n)>0 else 0.0
        macro_f1 = (f1 + f1_n)/2
        tpr = rec
        tnr = TN/(TN+FP) if (TN+FP)>0 else 0.0
        bal_acc = (tpr+tnr)/2
        denom = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc = ((TP*TN - FP*FN)/denom) if denom>0 else 0.0
        return {
            "correct": TP+TN,
            "incorrect": FP+FN,
            "accuracy": round(acc, 2),
            "metrics": {
                "task": "BoolQ (binary)",
                "f1_pos": round(f1,4),
                "macro_f1": round(macro_f1,4),
                "balanced_accuracy": round(bal_acc,4),
                "MCC": round(mcc,4),
                "confusion_matrix": {"TP":TP,"TN":TN,"FP":FP,"FN":FN},
                "support": total,
            }
        }

    if kind == "squad_v2":
        ems, f1s, has_g, has_p = [], [], [], []
        for gold, pred in zip(golds, preds):
            ems.append(squad_em(pred, gold))
            f1s.append(squad_f1(pred, gold))
            g_has = not all(is_empty_like(str(g)) for g in (gold if isinstance(gold, list) else [gold]))
            p_has = not is_empty_like(pred)
            has_g.append(g_has); has_p.append(p_has)
        total = len(golds)
        def _avg(x): return sum(x)/len(x) if x else 0.0
        EM = _avg(ems)*100; F1 = _avg(f1s)*100
        has_idx = [i for i,g in enumerate(has_g) if g]
        no_idx  = [i for i,g in enumerate(has_g) if not g]
        HasAns_EM = _avg([ems[i] for i in has_idx])*100 if has_idx else 0.0
        HasAns_F1 = _avg([f1s[i] for i in has_idx])*100 if has_idx else 0.0
        NoAns_Acc = _avg([1.0 if not has_p[i] else 0.0 for i in no_idx])*100 if no_idx else 0.0
        # AvNA (answer vs no-answer)
        TP=TN=FP=FN=0
        for g,p in zip(has_g,has_p):
            if g and p: TP+=1
            elif (not g) and (not p): TN+=1
            elif (not g) and p: FP+=1
            elif g and (not p): FN+=1
        AvNA = (TP+TN)/total*100 if total else 0.0
        # define "correct" as exact matches
        correct = int(sum(ems))
        return {
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": round(EM, 2),  # EM as accuracy
            "metrics": {
                "task": "SQuAD v2",
                "EM": round(EM,2),
                "F1": round(F1,2),
                "HasAns_EM": round(HasAns_EM,2),
                "HasAns_F1": round(HasAns_F1,2),
                "NoAns_Accuracy": round(NoAns_Acc,2),
                "AvNA_Accuracy": round(AvNA,2),
                "support": total,
            }
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
        avg_diff = sum([d for d in diffs if d is not None]) / len([d for d in diffs if d is not None]) if diffs else None
        return {
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": round(acc, 2),
            "metrics": {
                "task": "OpenMathInstruct-2 (numeric)",
                "avg_abs_diff": round(avg_diff, 6) if avg_diff is not None else None,
                "support": total,
            }
        }

    # generic fallback: string equality
    correct = sum(1 for g,p in zip(golds,preds) if (p is not None and p == g))
    total = len(golds)
    acc = correct/total*100 if total else 0.0
    return {
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(acc, 2),
        "metrics": {"task": "Generic", "support": total}
    }


import os, gc
import torch
from datetime import datetime

def _fmt_bytes(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024

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
        used_total = total - free                 # everything used on the GPU
        alloc = torch.cuda.memory_allocated(i)    # tensors by *this* process
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

# Directory with Fine Tuned models
models_dir = Path("Models")

# Make sure metrics dir exists BEFORE building the model list (so we can check for existing JSONs)
metrics_dir = Path("Testing/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)

models = []

for model_dir in models_dir.iterdir():
    if not model_dir.is_dir():
        continue

    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        continue

    # Expected metrics filename for local folders matches Path(...).name + ".json"
    metrics_path = metrics_dir / f"{model_dir.name}.json"
    if metrics_path.exists():
        # Already tested → skip
        continue

    # Add model (use the folder path string, as in your current code)
    models.append(str(model_dir))


datasets = list(Path("Datasets").glob('test-*.parquet'))

metrics_dir = Path("Testing/metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)

results_file = Path("Testing/benchmark_results.json")
if results_file.exists():
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}



VERBOSE = False  # set to False to hide per-dataset logs and summary prints

def vlog(msg: str):
    if VERBOSE:
        tqdm.write(msg)

def _safe_filename(name: str) -> str:
    # If it's a HuggingFace repo id like "org/model" (optionally with a revision "@rev"),
    # keep both parts but replace "/" with "__".
    if re.match(r'^[\w\-]+/[\w\.\-]+(@[\w\.\-]+)?$', name):
        base = name.replace('/', '__')
    else:
        # Treat as local path → keep only the last component (folder/file name)
        base = Path(name).name

    # Sanitize anything that Windows wouldn't like or that could create dirs
    base = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', base)
    return base

def _mode(values):
    if not values:
        return None
    try:
        return stats.mode(values)  # unique mode
    except stats.StatisticsError:
        mm = stats.multimode(values)
        return mm[0] if mm else None

if models == []:
    print("No more models left to test")
    reset_vram()
    exit()
else:
    print(f"Models left to test: {len(models)}")
    print(models)

for model_name in models:
    print('------------\n')
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device_map)

    # preload
    # if truncation is none then we load the entire dataset
    if datasetTrunc is None:
        loaded = [(ds, pd.read_parquet(ds)) for ds in datasets]
    else:
        loaded = [(ds, pd.read_parquet(ds).head(datasetTrunc)) for ds in datasets]
    total_samples = sum(len(df) for _, df in loaded)

    per_dataset = {}
    all_latencies, all_tokens = [], []

    if VERBOSE:
        tqdm.write(f"Model: {model_name}")

    with tqdm(
        total=total_samples,
        desc=model_name.split("/")[-1],
        dynamic_ncols=True,
        bar_format="{desc} ({n_fmt}/{total_fmt}) |{bar}| {percentage:3.0f}% {rate_fmt} {elapsed}<{remaining}",
        leave=True
    ) as pbar:

        for dataset_path, df in loaded:
            dataset_name = dataset_path.name
            info = datasets_info.get(dataset_name, {"system_prompt":"", "context": False, "task":"generic"})
            kind = info.get("task", "generic")

            latencies, tokens_out = [], []
            format_ok = 0

            # golds/preds for dispatcher
            golds, preds = [], []

            for _, row in df.iterrows():
                user_prompt = row["question"]
                if info["context"]:
                    user_prompt += row["context"]

                t0 = time.perf_counter()
                response = chat(model, tokenizer, user_prompt, info["system_prompt"])
                dt = time.perf_counter() - t0
                latencies.append(dt)

                try:
                    tok_count = len(tokenizer.encode(response, add_special_tokens=False))
                except Exception:
                    tok_count = 0
                tokens_out.append(tok_count)

                pred_boxed = extract_boxed(response)
                if pred_boxed is not None:
                    format_ok += 1

                # collect gold/pred by kind
                if kind == "mcq4":
                    golds.append(normalize_arc_option(str(row["answer"])))
                    preds.append(normalize_arc_option(pred_boxed))
                elif kind == "boolq":
                    golds.append(True if str(row["answer"]).strip().lower() == "true" else False)
                    preds.append(normalize_bool(pred_boxed))
                elif kind == "squad_v2":
                    golds.append(row["answer"])  # string or list of strings
                    preds.append((pred_boxed or "").strip())
                else:
                    golds.append(str(row["answer"]).strip())
                    preds.append((pred_boxed or "").strip())

                pbar.update(1)

            # latency/token aggregates
            all_latencies.extend(latencies)
            all_tokens.extend(tokens_out)

            lat_sum = float(sum(latencies))
            lat_mean = (lat_sum / len(latencies)) if latencies else 0.0
            lat_median = stats.median(latencies) if latencies else 0.0
            lat_mode = _mode([round(x, 3) for x in latencies])
            lat_std = stats.stdev(latencies) if len(latencies) > 1 else 0.0

            tok_sum = int(sum(tokens_out))
            tok_mean = (tok_sum / len(tokens_out)) if tokens_out else 0.0
            tok_median = stats.median(tokens_out) if tokens_out else 0.0
            tok_mode = _mode(tokens_out)
            tok_std = stats.stdev(tokens_out) if len(tokens_out) > 1 else 0.0

            # compute dataset metrics via dispatcher
            res = compute_metrics(kind, golds, preds)
            total = len(golds)
            format_rate = (format_ok / total * 100) if total else 0.0

            per_dataset[dataset_name] = {
                "type": kind,
                "total_samples": total,
                "format_success_rate": round(format_rate, 2),
                "correct": int(res["correct"]),
                "incorrect": int(res["incorrect"]),
                "accuracy": round(res["accuracy"], 2),
                "metrics": res["metrics"],
                "latency_seconds": {
                    "per_prompt": [round(x, 4) for x in latencies],
                    "sum": round(lat_sum, 4),
                    "mean": round(lat_mean, 4),
                    "median": round(lat_median, 4),
                    "mode": lat_mode,
                    "std": round(lat_std, 4),
                },
                "tokens_generated": {
                    "per_prompt": tokens_out,
                    "sum": tok_sum,
                    "mean": round(tok_mean, 2),
                    "median": round(tok_median, 2),
                    "mode": tok_mode,
                    "std": round(tok_std, 2),
                },
            }

            # optional logs
            vlog("*"*30)
            vlog(f"Dataset: {dataset_name} [{kind}]")
            if kind == "mcq4":
                vlog(f"🎯 Acc {res['accuracy']:.2f}% | Macro-F1 {res['metrics']['macro_f1']:.4f}")
            elif kind == "boolq":
                vlog(f"🎯 Acc {res['accuracy']:.2f}% | F1(pos) {res['metrics']['f1_pos']:.4f} | MCC {res['metrics']['MCC']:.4f}")
            elif kind == "squad_v2":
                vlog(f"🎯 EM {res['metrics']['EM']:.2f}% | F1 {res['metrics']['F1']:.2f}% | AvNA {res['metrics']['AvNA_Accuracy']:.2f}%")
            else:
                vlog(f"🎯 Acc {res['accuracy']:.2f}%")
            vlog(f"⏱️ Latency sum/mean/median/mode/std -> {lat_sum:.3f} / {lat_mean:.3f} / {lat_median:.3f} / {lat_mode} / {lat_std:.3f}")
            vlog(f"🔢 Tokens  sum/mean/median/mode/std -> {tok_sum} / {tok_mean:.2f} / {tok_median:.2f} / {tok_mode} / {tok_std:.2f}")

    # ----- model summary -----
    grand_total = sum(v["total_samples"] for v in per_dataset.values())
    grand_correct = sum(v["correct"] for v in per_dataset.values())
    grand_incorrect = sum(v["incorrect"] for v in per_dataset.values())
    overall_acc = (grand_correct / grand_total * 100) if grand_total else 0.0

    g_lat_sum = float(sum(all_latencies))
    g_lat_mean = (g_lat_sum / len(all_latencies)) if all_latencies else 0.0
    g_lat_median = stats.median(all_latencies) if all_latencies else 0.0
    g_lat_mode = _mode([round(x, 3) for x in all_latencies]) if all_latencies else None
    g_lat_std = stats.stdev(all_latencies) if len(all_latencies) > 1 else 0.0

    g_tok_sum = int(sum(all_tokens))
    g_tok_mean = (g_tok_sum / len(all_tokens)) if all_tokens else 0.0
    g_tok_median = stats.median(all_tokens) if all_tokens else 0.0
    g_tok_mode = _mode(all_tokens) if all_tokens else None
    g_tok_std = stats.stdev(all_tokens) if len(all_tokens) > 1 else 0.0

    model_report = {
        "model_name": model_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "num_datasets": len(per_dataset),
            "total_samples": grand_total,
            "total_correct": grand_correct,
            "total_incorrect": grand_incorrect,
            "overall_accuracy": round(overall_acc, 2),
            "latency_seconds": {
                "sum": round(g_lat_sum, 4),
                "mean": round(g_lat_mean, 4),
                "median": round(g_lat_median, 4),
                "mode": g_lat_mode,
                "std": round(g_lat_std, 4),
            },
            "tokens_generated": {
                "sum": g_tok_sum,
                "mean": round(g_tok_mean, 2),
                "median": round(g_tok_median, 2),
                "mode": g_tok_mode,
                "std": round(g_tok_std, 2),
            },
        },
        "datasets": per_dataset
    }

    out_path = metrics_dir / f"{_safe_filename(model_name)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model_report, f, indent=4, ensure_ascii=False)

    if VERBOSE:
        print(f"💾 Métricas guardadas en {out_path}")
    else:
        print(f"saved at {out_path}")
    
    # only process the first model and it breaks the loop
    break
    
