#!/usr/bin/env python
"""Generate comprehensive GPTQ comparison report from evaluation metrics."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

def load_metrics(metrics_dir: Path) -> Dict[str, Dict]:
    """Load all metrics files for the GPTQ study."""
    
    model_files = {
        "base_original": "Models__Qwen3-0.6B-base.json",
        "base_gptq": "Models__Qwen3-0.6B-base_gptq_w4g64.json", 
        "trained_original": "Models__Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant.json",
        "trained_gptq": "Models__Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_gptq_w4g64.json"
    }
    
    metrics = {}
    
    for model_type, filename in model_files.items():
        file_path = metrics_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                metrics[model_type] = json.load(f)
            print(f"✓ Loaded metrics for {model_type}")
        else:
            print(f"✗ Missing metrics file: {filename}")
            return None
    
    return metrics

def extract_key_metrics(metrics: Dict[str, Dict]) -> Dict[str, Dict]:
    """Extract key comparison metrics from all models."""
    
    comparison = {}
    
    for model_type, data in metrics.items():
        summary = data.get("summary", {})
        
        # Overall metrics
        overall_accuracy = summary.get("overall_accuracy", 0)
        total_samples = summary.get("total_samples", 0)
        
        # Latency metrics
        latency = summary.get("latency_seconds", {})
        avg_latency = latency.get("mean", 0)
        total_time = latency.get("sum", 0)
        
        # Token metrics
        tokens = summary.get("tokens_generated", {})
        avg_tokens = tokens.get("mean", 0)
        total_tokens = tokens.get("sum", 0)
        
        # Hardware metrics
        peak_vram_gb = summary.get("peak_vram_allocated_gb", 0)
        
        # Quantization info
        quant_info = summary.get("quantization", {})
        method = quant_info.get("method", "NoQuant")
        weights_bits = quant_info.get("weights_bits")
        group_size = quant_info.get("group_size")
        
        # Per-dataset breakdown
        datasets = data.get("datasets", {})
        dataset_accuracies = {}
        for dataset_name, dataset_data in datasets.items():
            dataset_accuracies[dataset_name] = dataset_data.get("accuracy", 0)
        
        comparison[model_type] = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "avg_latency_sec": avg_latency,
            "total_time_sec": total_time,
            "avg_tokens_per_response": avg_tokens,
            "total_tokens": total_tokens,
            "peak_vram_gb": peak_vram_gb,
            "quantization_method": method,
            "weights_bits": weights_bits,
            "group_size": group_size,
            "dataset_accuracies": dataset_accuracies
        }
    
    return comparison

def calculate_improvements(comparison: Dict[str, Dict]) -> Dict[str, Any]:
    """Calculate training and quantization impacts."""
    
    base_orig = comparison["base_original"]
    base_gptq = comparison["base_gptq"]
    trained_orig = comparison["trained_original"]
    trained_gptq = comparison["trained_gptq"]
    
    analysis = {}
    
    # Training impact (original models)
    training_accuracy_gain = trained_orig["overall_accuracy"] - base_orig["overall_accuracy"]
    analysis["training_improvement"] = {
        "accuracy_gain_pct": training_accuracy_gain,
        "is_beneficial": training_accuracy_gain > 0
    }
    
    # Quantization impact on base model
    base_quant_accuracy_loss = base_orig["overall_accuracy"] - base_gptq["overall_accuracy"]
    base_quant_accuracy_retention = (base_gptq["overall_accuracy"] / base_orig["overall_accuracy"]) * 100 if base_orig["overall_accuracy"] > 0 else 0
    base_speed_improvement = (base_orig["avg_latency_sec"] / base_gptq["avg_latency_sec"]) if base_gptq["avg_latency_sec"] > 0 else 1
    base_vram_reduction = base_orig["peak_vram_gb"] - base_gptq["peak_vram_gb"]
    
    analysis["base_quantization_impact"] = {
        "accuracy_loss_pct": base_quant_accuracy_loss,
        "accuracy_retention_pct": base_quant_accuracy_retention,
        "speed_improvement_factor": base_speed_improvement,
        "vram_reduction_gb": base_vram_reduction
    }
    
    # Quantization impact on trained model
    trained_quant_accuracy_loss = trained_orig["overall_accuracy"] - trained_gptq["overall_accuracy"]
    trained_quant_accuracy_retention = (trained_gptq["overall_accuracy"] / trained_orig["overall_accuracy"]) * 100 if trained_orig["overall_accuracy"] > 0 else 0
    trained_speed_improvement = (trained_orig["avg_latency_sec"] / trained_gptq["avg_latency_sec"]) if trained_gptq["avg_latency_sec"] > 0 else 1
    trained_vram_reduction = trained_orig["peak_vram_gb"] - trained_gptq["peak_vram_gb"]
    
    analysis["trained_quantization_impact"] = {
        "accuracy_loss_pct": trained_quant_accuracy_loss,
        "accuracy_retention_pct": trained_quant_accuracy_retention,
        "speed_improvement_factor": trained_speed_improvement,
        "vram_reduction_gb": trained_vram_reduction
    }
    
    # Overall best performance
    all_accuracies = {
        "Base Original": base_orig["overall_accuracy"],
        "Base GPTQ": base_gptq["overall_accuracy"],
        "Trained Original": trained_orig["overall_accuracy"],
        "Trained GPTQ": trained_gptq["overall_accuracy"]
    }
    
    best_model = max(all_accuracies, key=all_accuracies.get)
    analysis["best_performance"] = {
        "model": best_model,
        "accuracy": all_accuracies[best_model]
    }
    
    return analysis

def generate_markdown_report(comparison: Dict[str, Dict], analysis: Dict[str, Any]) -> str:
    """Generate comprehensive markdown report."""
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    lines = []
    lines.append("# GPTQ Quantization Study Report")
    lines.append(f"\n**Generated:** {timestamp}")
    lines.append(f"**Models Evaluated:** 4 variants of Qwen3-0.6B")
    lines.append(f"**Evaluation Samples:** {comparison['base_original']['total_samples']} per model (TRUNC_EVAL=20)")
    lines.append(f"**Training Dataset:** OpenMathInstruct-2 (1000 samples)")
    lines.append(f"**Quantization Method:** GPTQ W4G64 (4-bit weights, group size 64)")
    
    # Executive Summary
    lines.append("\n## Executive Summary")
    
    training_benefit = analysis["training_improvement"]
    if training_benefit["is_beneficial"]:
        lines.append(f"✅ **Training Impact**: +{training_benefit['accuracy_gain_pct']:.2f}% accuracy improvement")
    else:
        lines.append(f"❌ **Training Impact**: {training_benefit['accuracy_gain_pct']:.2f}% accuracy change")
    
    base_quant = analysis["base_quantization_impact"]
    lines.append(f"📊 **Base Model GPTQ**: {base_quant['accuracy_retention_pct']:.1f}% accuracy retention, {base_quant['speed_improvement_factor']:.2f}x speed improvement")
    
    trained_quant = analysis["trained_quantization_impact"]
    lines.append(f"📊 **Trained Model GPTQ**: {trained_quant['accuracy_retention_pct']:.1f}% accuracy retention, {trained_quant['speed_improvement_factor']:.2f}x speed improvement")
    
    best = analysis["best_performance"]
    lines.append(f"🏆 **Best Model**: {best['model']} ({best['accuracy']:.2f}% accuracy)")
    
    # Detailed Comparison Table
    lines.append("\n## Model Comparison")
    lines.append("| Model | Accuracy (%) | Avg Latency (s) | Tokens/Response | VRAM (GB) | Method |")
    lines.append("|-------|-------------|-----------------|-----------------|-----------|---------|")
    
    model_names = {
        "base_original": "Base Original",
        "base_gptq": "Base GPTQ",
        "trained_original": "Trained Original", 
        "trained_gptq": "Trained GPTQ"
    }
    
    for model_key in ["base_original", "base_gptq", "trained_original", "trained_gptq"]:
        data = comparison[model_key]
        name = model_names[model_key]
        accuracy = data["overall_accuracy"]
        latency = data["avg_latency_sec"]
        tokens = data["avg_tokens_per_response"]
        vram = data["peak_vram_gb"]
        method = data["quantization_method"]
        if data["weights_bits"]:
            method += f" W{data['weights_bits']}"
        
        lines.append(f"| {name} | {accuracy:.2f} | {latency:.2f} | {tokens:.1f} | {vram:.2f} | {method} |")
    
    # Performance Analysis
    lines.append("\n## Performance Analysis")
    
    lines.append("\n### Training Effectiveness")
    training_data = analysis["training_improvement"]
    if training_data["is_beneficial"]:
        lines.append(f"✅ Training on OpenMathInstruct-2 improved accuracy by **{training_data['accuracy_gain_pct']:.2f} percentage points**")
        lines.append("📈 Recommendation: Training before quantization is beneficial")
    else:
        lines.append(f"⚠️ Training showed minimal improvement ({training_data['accuracy_gain_pct']:.2f}% change)")
        lines.append("💡 Consider: Base model quantization may be sufficient")
    
    lines.append("\n### GPTQ Quantization Impact")
    
    # Base model quantization
    base_data = analysis["base_quantization_impact"]
    lines.append(f"\n**Base Model Quantization:**")
    lines.append(f"- Accuracy retention: {base_data['accuracy_retention_pct']:.1f}%")
    lines.append(f"- Speed improvement: {base_data['speed_improvement_factor']:.2f}x faster")
    lines.append(f"- VRAM reduction: {base_data['vram_reduction_gb']:.2f} GB")
    
    if base_data["accuracy_retention_pct"] > 95:
        lines.append("✅ Excellent quantization quality")
    elif base_data["accuracy_retention_pct"] > 90:
        lines.append("✅ Good quantization quality")
    else:
        lines.append("⚠️ Significant accuracy degradation")
    
    # Trained model quantization
    trained_data = analysis["trained_quantization_impact"]
    lines.append(f"\n**Trained Model Quantization:**")
    lines.append(f"- Accuracy retention: {trained_data['accuracy_retention_pct']:.1f}%")
    lines.append(f"- Speed improvement: {trained_data['speed_improvement_factor']:.2f}x faster")
    lines.append(f"- VRAM reduction: {trained_data['vram_reduction_gb']:.2f} GB")
    
    if trained_data["accuracy_retention_pct"] > 95:
        lines.append("✅ Excellent quantization quality")
    elif trained_data["accuracy_retention_pct"] > 90:
        lines.append("✅ Good quantization quality")
    else:
        lines.append("⚠️ Significant accuracy degradation")
    
    # Per-Dataset Performance
    lines.append("\n## Per-Dataset Performance")
    
    datasets = list(comparison["base_original"]["dataset_accuracies"].keys())
    
    lines.append("| Dataset | Base Original | Base GPTQ | Trained Original | Trained GPTQ |")
    lines.append("|---------|---------------|-----------|------------------|--------------|")
    
    for dataset in datasets:
        base_orig_acc = comparison["base_original"]["dataset_accuracies"][dataset]
        base_gptq_acc = comparison["base_gptq"]["dataset_accuracies"][dataset]
        trained_orig_acc = comparison["trained_original"]["dataset_accuracies"][dataset]
        trained_gptq_acc = comparison["trained_gptq"]["dataset_accuracies"][dataset]
        
        # Clean up dataset name
        clean_name = dataset.replace("test-", "").replace(".parquet", "")
        
        lines.append(f"| {clean_name} | {base_orig_acc:.2f}% | {base_gptq_acc:.2f}% | {trained_orig_acc:.2f}% | {trained_gptq_acc:.2f}% |")
    
    # Recommendations
    lines.append("\n## Recommendations")
    
    best_model = analysis["best_performance"]["model"]
    
    if "Trained" in best_model:
        lines.append("🎯 **Primary Recommendation**: Use trained model (fine-tuning provides clear benefits)")
    else:
        lines.append("🎯 **Primary Recommendation**: Base model is sufficient (training provides minimal benefits)")
    
    if "GPTQ" in best_model:
        lines.append("⚡ **Quantization Recommendation**: GPTQ quantization is recommended (good accuracy retention with speed/memory benefits)")
    else:
        lines.append("🎯 **Quantization Recommendation**: Consider whether speed/memory benefits justify accuracy loss")
    
    # Technical Details
    lines.append("\n## Technical Details")
    lines.append("- **Base Model**: Qwen3-0.6B")
    lines.append("- **Training Method**: Supervised Fine-Tuning (SFT) with no PEFT")
    lines.append("- **Training Data**: OpenMathInstruct-2 dataset (1000 samples)")
    lines.append("- **Quantization**: GPTQ with 4-bit weights, group size 64")
    lines.append("- **Calibration**: 100 samples from training data")
    lines.append("- **Evaluation**: 20 samples per test dataset (4 datasets total)")
    lines.append("- **Hardware**: CUDA-enabled GPU")
    
    return "\n".join(lines)

def save_json_summary(comparison: Dict[str, Dict], analysis: Dict[str, Any], output_path: Path):
    """Save comprehensive JSON summary."""
    
    summary = {
        "study_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "GPTQ",
            "base_model": "Qwen3-0.6B",
            "training_dataset": "OpenMathInstruct-2",
            "training_samples": 1000,
            "evaluation_samples_per_dataset": 20,
            "quantization_config": {
                "method": "GPTQ",
                "weights_bits": 4,
                "group_size": 64,
                "calibration_samples": 100
            }
        },
        "model_comparison": comparison,
        "analysis": analysis,
        "summary": {
            "best_model": analysis["best_performance"]["model"],
            "best_accuracy": analysis["best_performance"]["accuracy"],
            "training_beneficial": analysis["training_improvement"]["is_beneficial"],
            "training_gain": analysis["training_improvement"]["accuracy_gain_pct"],
            "base_quantization_retention": analysis["base_quantization_impact"]["accuracy_retention_pct"],
            "trained_quantization_retention": analysis["trained_quantization_impact"]["accuracy_retention_pct"]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 JSON summary saved: {output_path}")

def main():
    """Generate GPTQ comparison report."""
    
    print("🔍 GPTQ Quantization Study Report Generator")
    print("=" * 50)
    
    # Load metrics
    metrics_dir = Path("Testing/metrics")
    print(f"📂 Loading metrics from {metrics_dir}")
    
    metrics = load_metrics(metrics_dir)
    if metrics is None:
        print("❌ Failed to load all required metrics files")
        return 1
    
    # Extract and analyze
    print("📊 Analyzing performance metrics...")
    comparison = extract_key_metrics(metrics)
    analysis = calculate_improvements(comparison)
    
    # Generate outputs
    output_dir = Path("study_results")
    output_dir.mkdir(exist_ok=True)
    
    print("📝 Generating markdown report...")
    markdown_content = generate_markdown_report(comparison, analysis)
    markdown_path = output_dir / "gptq_study_report.md"
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"📄 Markdown report saved: {markdown_path}")
    
    print("💾 Generating JSON summary...")
    json_path = output_dir / "gptq_study_summary.json"
    save_json_summary(comparison, analysis, json_path)
    
    # Print key findings
    print("\n🎉 Study Complete! Key Findings:")
    print(f"🏆 Best Model: {analysis['best_performance']['model']} ({analysis['best_performance']['accuracy']:.2f}% accuracy)")
    print(f"📈 Training Impact: {'+' if analysis['training_improvement']['accuracy_gain_pct'] > 0 else ''}{analysis['training_improvement']['accuracy_gain_pct']:.2f}% accuracy change")
    print(f"⚡ Base Quantization: {analysis['base_quantization_impact']['accuracy_retention_pct']:.1f}% retention, {analysis['base_quantization_impact']['speed_improvement_factor']:.2f}x speed")
    print(f"⚡ Trained Quantization: {analysis['trained_quantization_impact']['accuracy_retention_pct']:.1f}% retention, {analysis['trained_quantization_impact']['speed_improvement_factor']:.2f}x speed")
    
    print(f"\n📁 Reports available in: {output_dir}/")
    print("   - gptq_study_report.md (detailed analysis)")
    print("   - gptq_study_summary.json (structured data)")
    
    return 0

if __name__ == "__main__":
    exit(main())