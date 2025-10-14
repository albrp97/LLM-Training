#!/usr/bin/env python
"""Comprehensive quantization study script for comparing methods."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod
from tools.quantization_benchmark import (
    benchmark_inference_speed,
    generate_quantization_analysis,
    measure_model_size_metrics,
    measure_quality_degradation,
    save_comprehensive_report,
)


def get_available_models() -> Dict[str, Path]:
    """Discover available models for quantization study."""
    models_dir = Path("Models")
    available_models = {}
    
    if not models_dir.exists():
        print(f"Warning: Models directory {models_dir} does not exist")
        return available_models
    
    # Look for specific model patterns
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            model_name = model_path.name
            
            # Base model (0.6B base)
            if "Qwen3-0.6B-base" in model_name:
                available_models["base"] = model_path
            
            # Trained model (0.6B openmath trained)
            elif "Qwen3-0.6B-openmath" in model_name and "NoQuant" in model_name:
                available_models["trained"] = model_path
    
    return available_models


def run_quantization_method(
    src_model: Path, 
    method: str, 
    bits: int = 4, 
    group_size: int = 64
) -> Optional[Path]:
    """Run a single quantization method and return the output path."""
    
    # Generate output path
    method_lower = method.lower()
    dst_name = f"{src_model.name}_{method}_w{bits}g{group_size}"
    dst_path = src_model.parent / dst_name
    
    # Check if already exists
    if dst_path.exists():
        print(f"Quantized model already exists: {dst_path}")
        return dst_path
    
    # Build quantization command
    cmd = [
        sys.executable, "tools/quantize.py", "run",
        "--src", str(src_model),
        "--dst", str(dst_path),
        "--method", method_lower,
        "--bits", str(bits),
        "--group-size", str(group_size),
        "--keep-lm-head-fp16",
        "--seed", "13"
    ]
    
    # Add calibration data for methods that need it
    if method_lower not in ["hqq"]:
        calib_file = "Datasets/calibration_openmath_100samples.txt"
        if Path(calib_file).exists():
            cmd.extend(["--calib", calib_file])
        else:
            print(f"Warning: Calibration file {calib_file} not found")
            return None
    
    print(f"Running quantization: {' '.join(cmd)}")
    print(f"Progress will be shown in real-time...\n")
    
    try:
        # Run without capture_output to show progress in real-time
        result = subprocess.run(cmd, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"\n✓ Quantization successful: {dst_path}")
            return dst_path
        else:
            print(f"\n✗ Quantization failed with return code: {result.returncode}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"✗ Quantization timed out after 1 hour")
        return None
    except Exception as e:
        print(f"✗ Quantization error: {e}")
        return None


def evaluate_model(model_path: Path) -> Optional[Dict]:
    """Run evaluation on a model using 02_TestModels.py."""
    
    cmd = [
        sys.executable, "Testing/02_TestModels.py", 
        str(model_path),
        "--trunc-eval", "50"
    ]
    
    print(f"Evaluating model: {model_path.name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            # Load the generated metrics file
            metrics_file = Path("Testing/metrics") / f"{model_path.name}.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: Metrics file not found for {model_path.name}")
                return None
        else:
            print(f"Evaluation failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Evaluation timed out for {model_path.name}")
        return None
    except Exception as e:
        print(f"Evaluation error: {e}")
        return None


def run_single_method_study(
    base_model: Path,
    trained_model: Path, 
    method: str,
    output_dir: Path
) -> Dict:
    """Run comprehensive study for a single quantization method."""
    
    print(f"\n{'='*60}")
    print(f"QUANTIZATION METHOD: {method.upper()}")
    print(f"{'='*60}")
    
    study_results = {
        "method": method,
        "base_model": str(base_model),
        "trained_model": str(trained_model),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "comparative_analysis": {},
    }
    
    # Define all model variants to test
    models_to_test = {
        "base_original": base_model,
        "trained_original": trained_model,
    }
    
    # Apply quantization to both models
    print(f"\n--- Applying {method} quantization ---")
    
    base_quantized = run_quantization_method(base_model, method)
    if base_quantized:
        models_to_test["base_quantized"] = base_quantized
    
    trained_quantized = run_quantization_method(trained_model, method)
    if trained_quantized:
        models_to_test["trained_quantized"] = trained_quantized
    
    # Comprehensive evaluation of all models
    print(f"\n--- Comprehensive evaluation ---")
    
    for model_type, model_path in models_to_test.items():
        print(f"\nEvaluating {model_type}: {model_path.name}")
        
        model_results = {
            "path": str(model_path),
            "evaluation_metrics": None,
            "size_metrics": None,
            "inference_speed": None,
            "quality_degradation": None
        }
        
        # 1. Task evaluation (accuracy metrics)
        eval_results = evaluate_model(model_path)
        if eval_results:
            model_results["evaluation_metrics"] = eval_results
        
        # 2. Model size metrics
        original_model = base_model if "base" in model_type else trained_model
        size_metrics = measure_model_size_metrics(model_path, original_model)
        model_results["size_metrics"] = size_metrics
        
        # 3. Inference speed benchmarking
        inference_metrics = benchmark_inference_speed(model_path, num_runs=3)
        model_results["inference_speed"] = inference_metrics
        
        # 4. Quality degradation (only for quantized models)
        if "quantized" in model_type:
            original = base_model if "base" in model_type else trained_model
            quality_metrics = measure_quality_degradation(original, model_path)
            model_results["quality_degradation"] = quality_metrics
        
        study_results["models"][model_type] = model_results
    
    # Generate comparative analysis
    print(f"\n--- Generating comparative analysis ---")
    study_results["comparative_analysis"] = generate_comparative_analysis(study_results["models"])
    
    study_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save method-specific results
    method_file = output_dir / f"{method.lower()}_study_results.json"
    with open(method_file, "w") as f:
        json.dump(study_results, f, indent=2)
    
    # Generate method-specific report
    generate_method_report(study_results, output_dir / f"{method.lower()}_report.md")
    
    print(f"\n✓ {method} study completed successfully!")
    print(f"Results saved to: {method_file}")
    
    return study_results


def generate_comparative_analysis(models: Dict) -> Dict:
    """Generate comparative analysis between base, trained, and quantized models."""
    
    analysis = {
        "accuracy_comparison": {},
        "size_comparison": {},
        "speed_comparison": {},
        "quality_impact": {},
        "summary": {}
    }
    
    # Extract metrics for easier comparison
    base_orig = models.get("base_original", {})
    trained_orig = models.get("trained_original", {})
    base_quant = models.get("base_quantized", {})
    trained_quant = models.get("trained_quantized", {})
    
    # Accuracy comparison
    def get_accuracy(model_data):
        eval_metrics = model_data.get("evaluation_metrics", {})
        if eval_metrics and "summary" in eval_metrics:
            return eval_metrics["summary"].get("overall_accuracy", 0)
        return 0
    
    accuracies = {
        "base_original": get_accuracy(base_orig),
        "trained_original": get_accuracy(trained_orig),
        "base_quantized": get_accuracy(base_quant),
        "trained_quantized": get_accuracy(trained_quant)
    }
    
    analysis["accuracy_comparison"] = accuracies
    
    # Training benefit analysis
    if accuracies["base_original"] > 0 and accuracies["trained_original"] > 0:
        training_benefit = accuracies["trained_original"] - accuracies["base_original"]
        analysis["accuracy_comparison"]["training_benefit"] = training_benefit
    
    # Size comparison
    def get_size_mb(model_data):
        size_metrics = model_data.get("size_metrics", {})
        return size_metrics.get("model_size_mb", 0)
    
    sizes = {
        "base_original": get_size_mb(base_orig),
        "trained_original": get_size_mb(trained_orig),
        "base_quantized": get_size_mb(base_quant),
        "trained_quantized": get_size_mb(trained_quant)
    }
    
    analysis["size_comparison"] = sizes
    
    # Speed comparison
    def get_latency(model_data):
        inference = model_data.get("inference_speed", {})
        return inference.get("overall_avg_latency_ms", 0)
    
    latencies = {
        "base_original": get_latency(base_orig),
        "trained_original": get_latency(trained_orig),
        "base_quantized": get_latency(base_quant),
        "trained_quantized": get_latency(trained_quant)
    }
    
    analysis["speed_comparison"] = latencies
    
    # Quality impact analysis
    def get_quality_score(model_data):
        quality = model_data.get("quality_degradation", {})
        return quality.get("avg_embedding_similarity", 1.0)  # 1.0 for original models
    
    quality_scores = {
        "base_quantized": get_quality_score(base_quant),
        "trained_quantized": get_quality_score(trained_quant)
    }
    
    analysis["quality_impact"] = quality_scores
    
    # Summary insights
    summary = {}
    
    # Training effectiveness
    if accuracies["trained_original"] > accuracies["base_original"]:
        summary["training_effective"] = True
        summary["training_improvement"] = accuracies["trained_original"] - accuracies["base_original"]
    else:
        summary["training_effective"] = False
    
    # Quantization impact on base model
    if accuracies["base_quantized"] > 0:
        summary["base_quant_accuracy_retention"] = accuracies["base_quantized"] / accuracies["base_original"] * 100
    
    # Quantization impact on trained model  
    if accuracies["trained_quantized"] > 0:
        summary["trained_quant_accuracy_retention"] = accuracies["trained_quantized"] / accuracies["trained_original"] * 100
    
    # Size reduction
    if sizes["base_quantized"] > 0:
        summary["base_size_reduction"] = (1 - sizes["base_quantized"] / sizes["base_original"]) * 100
    if sizes["trained_quantized"] > 0:
        summary["trained_size_reduction"] = (1 - sizes["trained_quantized"] / sizes["trained_original"]) * 100
    
    # Speed improvement
    if latencies["base_quantized"] > 0:
        summary["base_speed_improvement"] = (latencies["base_original"] / latencies["base_quantized"] - 1) * 100
    if latencies["trained_quantized"] > 0:
        summary["trained_speed_improvement"] = (latencies["trained_original"] / latencies["trained_quantized"] - 1) * 100
    
    analysis["summary"] = summary
    
    return analysis


def generate_method_report(study_results: Dict, report_path: Path):
    """Generate a markdown report for a single method study."""
    
    method = study_results["method"]
    models = study_results["models"]
    analysis = study_results["comparative_analysis"]
    
    lines = []
    lines.append(f"# {method.upper()} Quantization Study Report")
    lines.append(f"\nGenerated: {study_results['start_time']}")
    lines.append(f"Base Model: {Path(study_results['base_model']).name}")
    lines.append(f"Trained Model: {Path(study_results['trained_model']).name}")
    
    # Executive summary
    lines.append("\n## Executive Summary")
    summary = analysis.get("summary", {})
    
    if summary.get("training_effective", False):
        improvement = summary.get("training_improvement", 0)
        lines.append(f"✓ Training improved accuracy by {improvement:.2f} percentage points")
    else:
        lines.append("⚠️ Training did not improve accuracy significantly")
    
    base_retention = summary.get("base_quant_accuracy_retention", 0)
    trained_retention = summary.get("trained_quant_accuracy_retention", 0)
    
    if base_retention > 0:
        lines.append(f"📊 Base model quantization retained {base_retention:.1f}% of accuracy")
    if trained_retention > 0:
        lines.append(f"📊 Trained model quantization retained {trained_retention:.1f}% of accuracy")
    
    base_size_reduction = summary.get("base_size_reduction", 0)
    trained_size_reduction = summary.get("trained_size_reduction", 0)
    
    if base_size_reduction > 0:
        lines.append(f"💾 Base model size reduced by {base_size_reduction:.1f}%")
    if trained_size_reduction > 0:
        lines.append(f"💾 Trained model size reduced by {trained_size_reduction:.1f}%")
    
    # Detailed comparison table
    lines.append("\n## Model Comparison")
    lines.append("| Model | Accuracy (%) | Size (MB) | Latency (ms) | Quality Score |")
    lines.append("|-------|-------------|-----------|--------------|---------------|")
    
    model_order = ["base_original", "base_quantized", "trained_original", "trained_quantized"]
    model_names = {
        "base_original": "Base Original",
        "base_quantized": f"Base {method}",
        "trained_original": "Trained Original", 
        "trained_quantized": f"Trained {method}"
    }
    
    for model_type in model_order:
        if model_type in models:
            model_data = models[model_type]
            
            # Extract metrics
            accuracy = 0
            eval_metrics = model_data.get("evaluation_metrics", {})
            if eval_metrics and "summary" in eval_metrics:
                accuracy = eval_metrics["summary"].get("overall_accuracy", 0)
            
            size = 0
            size_metrics = model_data.get("size_metrics", {})
            if size_metrics:
                size = size_metrics.get("model_size_mb", 0)
            
            latency = 0
            inference = model_data.get("inference_speed", {})
            if inference:
                latency = inference.get("overall_avg_latency_ms", 0)
            
            quality = 1.0  # Default for original models
            if "quantized" in model_type:
                quality_metrics = model_data.get("quality_degradation", {})
                if quality_metrics:
                    quality = quality_metrics.get("avg_embedding_similarity", 0)
            
            lines.append(f"| {model_names[model_type]} | {accuracy:.2f} | {size:.1f} | {latency:.1f} | {quality:.3f} |")
    
    # Task-specific results
    lines.append("\n## Task-Specific Results")
    
    for model_type in model_order:
        if model_type in models:
            model_data = models[model_type]
            eval_metrics = model_data.get("evaluation_metrics", {})
            
            if eval_metrics and "per_dataset" in eval_metrics:
                lines.append(f"\n### {model_names[model_type]}")
                
                for dataset, results in eval_metrics["per_dataset"].items():
                    accuracy = results.get("accuracy", 0)
                    task = results.get("metrics", {}).get("task", "Unknown")
                    lines.append(f"- **{dataset}** ({task}): {accuracy:.2f}% accuracy")
    
    # Quality analysis (for quantized models)
    lines.append("\n## Quality Degradation Analysis")
    
    for model_type in ["base_quantized", "trained_quantized"]:
        if model_type in models:
            model_data = models[model_type]
            quality_metrics = model_data.get("quality_degradation", {})
            
            if quality_metrics:
                model_name = model_names[model_type]
                lines.append(f"\n### {model_name}")
                
                embedding_sim = quality_metrics.get("avg_embedding_similarity", 0)
                text_sim = quality_metrics.get("avg_text_similarity", 0)
                length_ratio = quality_metrics.get("avg_length_ratio", 1.0)
                
                lines.append(f"- Embedding similarity: {embedding_sim:.3f}")
                lines.append(f"- Text similarity: {text_sim:.3f}")
                lines.append(f"- Response length ratio: {length_ratio:.3f}")
    
    # Recommendations
    lines.append("\n## Recommendations")
    
    if base_retention > 90 and trained_retention > 90:
        lines.append(f"✅ {method} quantization is highly recommended for both models")
    elif base_retention > 85 and trained_retention > 85:
        lines.append(f"✅ {method} quantization provides good balance of size and accuracy")
    else:
        lines.append(f"⚠️ {method} quantization causes significant accuracy degradation")
    
    if summary.get("training_effective", False):
        lines.append("✅ Training before quantization is beneficial")
    else:
        lines.append("❓ Training benefit is minimal - consider base model quantization only")
    
    # Save report
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive quantization study")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["gptq", "awq", "hqq", "smoothquant", "quarot", "adaround", "brecq"],
        help="Quantization method to test (if not specified, discovers available models)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="study_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Path to base model (auto-discovered if not specified)"
    )
    parser.add_argument(
        "--trained-model", 
        type=str,
        help="Path to trained model (auto-discovered if not specified)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover or validate models
    if args.base_model and args.trained_model:
        base_model = Path(args.base_model)
        trained_model = Path(args.trained_model)
        
        if not base_model.exists():
            print(f"Error: Base model not found: {base_model}")
            return 1
        if not trained_model.exists():
            print(f"Error: Trained model not found: {trained_model}")
            return 1
    else:
        print("Discovering available models...")
        available_models = get_available_models()
        
        if "base" not in available_models:
            print("Error: No base model found. Expected pattern: Qwen3-0.6B-base")
            return 1
        if "trained" not in available_models:
            print("Error: No trained model found. Expected pattern: Qwen3-0.6B-openmath*NoQuant")
            return 1
        
        base_model = available_models["base"]
        trained_model = available_models["trained"]
        
        print(f"Found base model: {base_model}")
        print(f"Found trained model: {trained_model}")
    
    # Run study for specified method
    if args.method:
        study_results = run_single_method_study(
            base_model, trained_model, args.method, output_dir
        )
        
        print(f"\n🎉 Study completed successfully!")
        print(f"Results available in: {output_dir}")
        
    else:
        print("No method specified. Available models:")
        print(f"Base: {base_model}")
        print(f"Trained: {trained_model}")
        print("\nTo run a study, specify --method (e.g., --method gptq)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())