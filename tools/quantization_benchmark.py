#!/usr/bin/env python
"""Comprehensive quantization benchmarking with enhanced metrics.

This module provides comprehensive metrics collection for quantization methods,
including model size, memory usage, quality degradation, and inference speed.
"""

from __future__ import annotations

import gc
import json
import psutil
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod, QuantizationSpec


def measure_model_size_metrics(model_path: Path, original_model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Comprehensive model size and memory metrics."""
    metrics = {}
    
    # Disk storage metrics
    def get_directory_size(path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total += file_path.stat().st_size
        return total
    
    # Model size on disk
    model_size_bytes = get_directory_size(model_path)
    metrics['model_size_mb'] = model_size_bytes / (1024 * 1024)
    metrics['model_size_gb'] = model_size_bytes / (1024 * 1024 * 1024)
    
    # Compression ratio vs original
    if original_model_path and original_model_path.exists():
        original_size = get_directory_size(original_model_path)
        metrics['original_size_mb'] = original_size / (1024 * 1024)
        metrics['compression_ratio'] = original_size / model_size_bytes if model_size_bytes > 0 else 1.0
        metrics['size_reduction_percent'] = (1 - model_size_bytes / original_size) * 100 if original_size > 0 else 0.0
    
    # Memory footprint when loaded
    try:
        # Measure loading memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="cpu"  # Load on CPU to measure RAM usage
        )
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        metrics['ram_usage_mb'] = memory_after - memory_before
        
        # Model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics['total_parameters'] = total_params
        metrics['trainable_parameters'] = trainable_params
        metrics['parameters_millions'] = total_params / 1_000_000
        
        # GPU memory usage (if available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated_before = torch.cuda.memory_allocated()
            
            model = model.to('cuda')
            torch.cuda.synchronize()
            
            memory_allocated_after = torch.cuda.memory_allocated()
            gpu_memory_mb = (memory_allocated_after - memory_allocated_before) / (1024 * 1024)
            metrics['gpu_memory_mb'] = gpu_memory_mb
            
            # GPU memory efficiency
            if 'compression_ratio' in metrics:
                metrics['gpu_memory_efficiency'] = metrics['compression_ratio'] / (gpu_memory_mb / 1024) if gpu_memory_mb > 0 else 0
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Warning: Could not measure memory metrics: {e}")
        metrics['memory_measurement_error'] = str(e)
    
    return metrics


def measure_quality_degradation(
    original_model_path: Path, 
    quantized_model_path: Path, 
    test_prompts: Optional[List[str]] = None
) -> Dict[str, float]:
    """Measure quality degradation between original and quantized models."""
    
    if test_prompts is None:
        # Default test prompts for quality assessment
        test_prompts = [
            "What is the capital of France?",
            "Solve for x: 2x + 5 = 13",
            "Write a short story about a robot.",
            "Explain quantum computing in simple terms.",
            "Translate 'hello world' to Spanish."
        ]
    
    metrics = {}
    
    try:
        # Load both models
        print("[Quality] Loading original model...")
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        
        print("[Quality] Loading quantized model...")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        original_model.eval()
        quantized_model.eval()
        
        # Collect outputs for comparison
        original_outputs = []
        quantized_outputs = []
        embedding_similarities = []
        
        print(f"[Quality] Evaluating {len(test_prompts)} prompts...")
        
        with torch.no_grad():
            for prompt in tqdm(test_prompts, desc="Quality evaluation"):
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                try:
                    # Generate responses
                    original_output = original_model.generate(
                        **inputs, max_new_tokens=50, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id
                    )
                    quantized_output = quantized_model.generate(
                        **inputs, max_new_tokens=50, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode responses
                    original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
                    quantized_text = tokenizer.decode(quantized_output[0], skip_special_tokens=True)
                    
                    original_outputs.append(original_text)
                    quantized_outputs.append(quantized_text)
                    
                    # Compare hidden states (embedding similarity)
                    try:
                        orig_hidden = original_model(**inputs, output_hidden_states=True).hidden_states[-1]
                        quant_hidden = quantized_model(**inputs, output_hidden_states=True).hidden_states[-1]
                        
                        # Average over sequence length and batch
                        orig_embedding = orig_hidden.mean(dim=1).cpu().numpy()
                        quant_embedding = quant_hidden.mean(dim=1).cpu().numpy()
                        
                        similarity = cosine_similarity(orig_embedding, quant_embedding)[0, 0]
                        embedding_similarities.append(similarity)
                        
                    except Exception as e:
                        print(f"Warning: Could not compute embedding similarity for prompt: {e}")
                        
                except Exception as e:
                    print(f"Warning: Could not generate for prompt '{prompt[:50]}...': {e}")
                    continue
        
        # Compute quality metrics
        if embedding_similarities:
            metrics['avg_embedding_similarity'] = np.mean(embedding_similarities)
            metrics['min_embedding_similarity'] = np.min(embedding_similarities)
            metrics['embedding_similarity_std'] = np.std(embedding_similarities)
        
        # Text similarity metrics
        text_similarities = []
        for orig, quant in zip(original_outputs, quantized_outputs):
            similarity = SequenceMatcher(None, orig, quant).ratio()
            text_similarities.append(similarity)
        
        if text_similarities:
            metrics['avg_text_similarity'] = np.mean(text_similarities)
            metrics['min_text_similarity'] = np.min(text_similarities)
        
        # Response length consistency
        orig_lengths = [len(text.split()) for text in original_outputs]
        quant_lengths = [len(text.split()) for text in quantized_outputs]
        
        if orig_lengths and quant_lengths:
            metrics['avg_length_ratio'] = np.mean([q/o if o > 0 else 1.0 for o, q in zip(orig_lengths, quant_lengths)])
            metrics['length_variation'] = np.std([abs(o-q) for o, q in zip(orig_lengths, quant_lengths)])
        
        del original_model, quantized_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Warning: Could not measure quality degradation: {e}")
        metrics['quality_measurement_error'] = str(e)
    
    return metrics


def benchmark_inference_speed(model_path: Path, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark inference speed with various metrics."""
    try:
        print(f"[Inference] Loading model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        # Test prompts of different lengths
        test_prompts = [
            "Hello",  # Short
            "What is the capital of France? Please explain.",  # Medium
            "Write a detailed explanation of quantum computing and its applications in modern technology.",  # Long
        ]
        
        metrics = {}
        
        print(f"[Inference] Running {num_runs} inference runs per prompt type...")
        
        for prompt_type, prompt in zip(["short", "medium", "long"], test_prompts):
            times = []
            token_counts = []
            
            for run in range(num_runs):
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=50, 
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Count generated tokens
                input_length = inputs['input_ids'].shape[1]
                output_length = outputs.shape[1]
                new_tokens = output_length - input_length
                token_counts.append(new_tokens)
            
            # Compute metrics for this prompt type
            avg_time = sum(times) / len(times)
            avg_tokens = sum(token_counts) / len(token_counts)
            tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
            
            metrics[f"{prompt_type}_avg_time_ms"] = avg_time * 1000
            metrics[f"{prompt_type}_tokens_per_second"] = tokens_per_second
            metrics[f"{prompt_type}_time_per_token_ms"] = (avg_time / avg_tokens * 1000) if avg_tokens > 0 else 0
        
        # Overall metrics
        all_times = []
        for prompt_type in ["short", "medium", "long"]:
            all_times.append(metrics[f"{prompt_type}_avg_time_ms"] / 1000)
        
        metrics["overall_avg_latency_ms"] = (sum(all_times) / len(all_times)) * 1000
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return metrics
        
    except Exception as e:
        return {"inference_benchmark_error": str(e)}


def measure_quantization_performance(
    src: Path, 
    method_config: Dict, 
    config_name: str, 
    method_name: str
) -> Dict:
    """Enhanced quantization with performance timing and resource usage."""
    
    # Resource monitoring setup
    performance_metrics = {
        'peak_ram_mb': 0,
        'peak_gpu_mb': 0,
        'cpu_usage_percent': [],
        'gpu_utilization_percent': [],
        'quantization_phases': {},
        'throughput_samples_per_second': 0
    }
    
    monitoring_active = True
    
    def monitor_resources():
        """Background thread to monitor resource usage."""
        process = psutil.Process()
        
        while monitoring_active:
            try:
                # CPU and RAM monitoring
                cpu_percent = process.cpu_percent()
                ram_mb = process.memory_info().rss / (1024 * 1024)
                performance_metrics['peak_ram_mb'] = max(performance_metrics['peak_ram_mb'], ram_mb)
                performance_metrics['cpu_usage_percent'].append(cpu_percent)
                
                # GPU monitoring
                if torch.cuda.is_available():
                    try:
                        gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        performance_metrics['peak_gpu_mb'] = max(performance_metrics['peak_gpu_mb'], gpu_mb)
                        
                        # GPU utilization via GPUtil if available
                        try:
                            import GPUtil
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                performance_metrics['gpu_utilization_percent'].append(gpus[0].load * 100)
                        except ImportError:
                            pass
                    except Exception:
                        pass
                
                time.sleep(1)  # Monitor every second
                
            except Exception:
                break
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Phase timing
    phase_timers = {}
    
    def start_phase(phase_name: str):
        phase_timers[phase_name] = time.time()
    
    def end_phase(phase_name: str):
        if phase_name in phase_timers:
            elapsed = time.time() - phase_timers[phase_name]
            performance_metrics['quantization_phases'][phase_name] = elapsed
            return elapsed
        return 0
    
    dst = f"{src}_{config_name}_{method_name}"
    
    # Build command with phase tracking
    cmd = [
        "python", "tools/quantize.py", "run",
        "--src", str(src),
        "--dst", dst,
        "--method", method_config.get("method", method_name.lower().split('_')[0]),
        "--bits", str(method_config["bits"]),
        "--keep-lm-head-fp16",
        "--seed", "13"
    ]
    
    # Add method-specific parameters
    if "group_size" in method_config:
        cmd.extend(["--group-size", str(method_config["group_size"])])
    if "acts_bits" in method_config:
        cmd.extend(["--acts-bits", str(method_config["acts_bits"])])
    if "kv_bits" in method_config:
        cmd.extend(["--kv-bits", str(method_config["kv_bits"])])
    if method_config.get("calib", True):
        cmd.extend(["--calib", "Datasets/calibration_openmath_5samples.txt"])
    
    print(f"[Study] Running: {' '.join(cmd)}")
    
    start_phase("total_quantization")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        end_phase("total_quantization")
        
        # Stop monitoring
        monitoring_active = False
        monitor_thread.join(timeout=2)
        
        # Calculate throughput (samples per second)
        if method_config.get("calib", True):
            try:
                with open("Datasets/calibration_openmath_5samples.txt", 'r') as f:
                    num_samples = len([line for line in f if line.strip()])
                performance_metrics['throughput_samples_per_second'] = num_samples / elapsed if elapsed > 0 else 0
            except:
                pass
        
        # Compute average resource usage
        if performance_metrics['cpu_usage_percent']:
            performance_metrics['avg_cpu_percent'] = sum(performance_metrics['cpu_usage_percent']) / len(performance_metrics['cpu_usage_percent'])
        if performance_metrics['gpu_utilization_percent']:
            performance_metrics['avg_gpu_utilization'] = sum(performance_metrics['gpu_utilization_percent']) / len(performance_metrics['gpu_utilization_percent'])
        
        # Model size metrics
        if result.returncode == 0:
            size_metrics = measure_model_size_metrics(Path(dst), src)
            performance_metrics.update(size_metrics)
        
        if result.returncode == 0:
            return {
                "status": "success",
                "elapsed_time": elapsed,
                "output_path": dst,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "performance": performance_metrics
            }
        else:
            return {
                "status": "failed",
                "elapsed_time": elapsed,
                "error": result.stderr,
                "returncode": result.returncode,
                "performance": performance_metrics
            }
            
    except subprocess.TimeoutExpired:
        monitoring_active = False
        return {
            "status": "timeout",
            "elapsed_time": 3600,
            "error": "Quantization timed out after 1 hour",
            "performance": performance_metrics
        }
    except Exception as e:
        monitoring_active = False
        return {
            "status": "error",
            "elapsed_time": time.time() - start_time,
            "error": str(e),
            "performance": performance_metrics
        }


def generate_quantization_analysis(study_results: Dict) -> Dict:
    """Generate comprehensive analysis of quantization results."""
    
    analysis = {
        "performance_ranking": {},
        "efficiency_analysis": {},
        "quality_vs_compression": {},
        "recommendations": {}
    }
    
    # Extract data for analysis
    all_results = []
    for config_name, methods in study_results.get("benchmarks", {}).items():
        for method_name, metrics in methods.items():
            model_size = metrics.get("model_size", {})
            quality = metrics.get("quality_degradation", {})
            inference = metrics.get("inference_speed", {})
            quant_perf = metrics.get("quantization_performance", {})
            
            result = {
                "config": config_name,
                "method": method_name,
                "compression_ratio": model_size.get("compression_ratio", 1.0),
                "size_mb": model_size.get("model_size_mb", 0),
                "quantization_time": quant_perf.get("elapsed_time", 0),
                "embedding_similarity": quality.get("avg_embedding_similarity", 0),
                "inference_speed": inference.get("overall_avg_latency_ms", float('inf'))
            }
            all_results.append(result)
    
    if not all_results:
        return analysis
    
    # Performance ranking
    for metric in ["compression_ratio", "quantization_time", "embedding_similarity", "inference_speed"]:
        sorted_results = sorted(
            all_results, 
            key=lambda x: x[metric], 
            reverse=(metric in ["compression_ratio", "embedding_similarity"])
        )
        analysis["performance_ranking"][metric] = [
            {"method": f"{r['config']}_{r['method']}", "value": r[metric]} 
            for r in sorted_results[:5]
        ]
    
    # Efficiency analysis (compression ratio vs quality)
    efficiency_scores = []
    for result in all_results:
        if result["embedding_similarity"] > 0 and result["compression_ratio"] > 1:
            efficiency = result["compression_ratio"] * result["embedding_similarity"]
            efficiency_scores.append({
                "method": f"{result['config']}_{result['method']}",
                "efficiency_score": efficiency,
                "compression_ratio": result["compression_ratio"],
                "quality_score": result["embedding_similarity"]
            })
    
    analysis["efficiency_analysis"]["top_efficient_methods"] = sorted(
        efficiency_scores, key=lambda x: x["efficiency_score"], reverse=True
    )[:5]
    
    # Recommendations
    if all_results:
        best_compression = max(all_results, key=lambda x: x["compression_ratio"])
        best_quality = max(all_results, key=lambda x: x["embedding_similarity"])
        fastest_quantization = min(all_results, key=lambda x: x["quantization_time"])
        fastest_inference = min(all_results, key=lambda x: x["inference_speed"])
        
        analysis["recommendations"] = {
            "best_compression": f"{best_compression['config']}_{best_compression['method']} ({best_compression['compression_ratio']:.2f}x)",
            "best_quality": f"{best_quality['config']}_{best_quality['method']} (similarity: {best_quality['embedding_similarity']:.3f})",
            "fastest_quantization": f"{fastest_quantization['config']}_{fastest_quantization['method']} ({fastest_quantization['quantization_time']:.1f}s)",
            "fastest_inference": f"{fastest_inference['config']}_{fastest_inference['method']} ({fastest_inference['inference_speed']:.1f}ms)",
            "balanced_choice": analysis["efficiency_analysis"]["top_efficient_methods"][0]["method"] if efficiency_scores else "N/A"
        }
    
    return analysis


def save_comprehensive_report(study_results: Dict, output_path: Path):
    """Save comprehensive study results with formatted report."""
    
    # Save raw results
    with open(output_path / "comprehensive_study_results.json", "w") as f:
        json.dump(study_results, f, indent=2)
    
    # Generate markdown report
    report_lines = []
    report_lines.append("# Quantization Study Results")
    report_lines.append(f"\nGenerated: {study_results.get('start_time', 'Unknown')}")
    report_lines.append(f"Base Model: {study_results.get('base_model', 'Unknown')}")
    
    # Summary table
    if "analysis" in study_results and "recommendations" in study_results["analysis"]:
        recs = study_results["analysis"]["recommendations"]
        report_lines.append("\n## Recommendations")
        report_lines.append("| Category | Method | Value |")
        report_lines.append("|----------|--------|-------|")
        for category, recommendation in recs.items():
            report_lines.append(f"| {category.replace('_', ' ').title()} | {recommendation} | - |")
    
    # Detailed results
    if "benchmarks" in study_results:
        report_lines.append("\n## Detailed Results")
        for config_name, methods in study_results["benchmarks"].items():
            report_lines.append(f"\n### {config_name.title()} Configuration")
            for method_name, metrics in methods.items():
                report_lines.append(f"\n#### {method_name}")
                
                # Model size metrics
                size = metrics.get("model_size", {})
                if size:
                    report_lines.append("**Model Size:**")
                    report_lines.append(f"- Size: {size.get('model_size_mb', 0):.1f} MB")
                    report_lines.append(f"- Compression: {size.get('compression_ratio', 1.0):.2f}x")
                    report_lines.append(f"- Parameters: {size.get('parameters_millions', 0):.1f}M")
                
                # Quality metrics
                quality = metrics.get("quality_degradation", {})
                if quality:
                    report_lines.append("\n**Quality:**")
                    report_lines.append(f"- Embedding similarity: {quality.get('avg_embedding_similarity', 0):.3f}")
                    report_lines.append(f"- Text similarity: {quality.get('avg_text_similarity', 0):.3f}")
                
                # Performance metrics
                inference = metrics.get("inference_speed", {})
                if inference:
                    report_lines.append("\n**Inference:**")
                    report_lines.append(f"- Avg latency: {inference.get('overall_avg_latency_ms', 0):.1f} ms")
                    report_lines.append(f"- Short prompt: {inference.get('short_tokens_per_second', 0):.1f} tokens/s")
    
    # Save markdown report
    with open(output_path / "quantization_study_report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"Comprehensive study results saved to {output_path}/")


if __name__ == "__main__":
    print("Quantization benchmark module - use via quantization_study.py")
