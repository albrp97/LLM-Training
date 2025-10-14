#!/usr/bin/env python3
"""
AWQ Post-Training Quantization Comparison Report Generator

Analyzes the effects of AWQ (Activation-aware Weight Quantization) on both
base models and fine-tuned models by comparing performance across test datasets.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_metrics_file(filepath: Path) -> Dict[str, Any]:
    """Load and return metrics from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def extract_key_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key performance metrics from the full metrics data."""
    key_metrics = {
        'model_name': metrics.get('model_name', 'Unknown'),
        'quantization_method': metrics.get('quantization_method', 'Unknown'),
        'total_params': metrics.get('total_params', 0),
        'vram_usage_gb': metrics.get('hardware', {}).get('peak_vram_reserved_gb', 0),
        'avg_latency_ms': 0,
        'total_eval_time_s': metrics.get('total_eval_time_s', 0),
        'datasets': {}
    }
    
    # Extract dataset-specific metrics from the datasets section
    datasets = metrics.get('datasets', {})
    for dataset_name, dataset_metrics in datasets.items():
        if isinstance(dataset_metrics, dict):
            # Get the main performance metric based on dataset type
            main_metric = None
            main_value = None
            
            # Check for different metric types
            if 'avg_abs_diff' in dataset_metrics.get('metrics', {}):
                # OpenMath uses avg_abs_diff (lower is better)
                main_metric = 'avg_abs_diff'
                main_value = dataset_metrics['metrics']['avg_abs_diff']
            elif 'accuracy' in dataset_metrics:
                # Most other datasets use accuracy
                main_metric = 'accuracy'  
                main_value = dataset_metrics['accuracy']
            elif 'F1' in dataset_metrics.get('metrics', {}):
                # Some datasets use F1 score
                main_metric = 'F1'
                main_value = dataset_metrics['metrics']['F1']
            
            if main_metric and main_value is not None:
                # Clean up dataset name for display
                clean_name = dataset_name.replace('test-', '').replace('.parquet', '')
                if clean_name == 'OpenMathInstruct-2':
                    clean_name = 'openmath'
                elif clean_name == 'ai2_arc':
                    clean_name = 'arc'
                elif clean_name == 'squad_v2':
                    clean_name = 'squad'
                
                key_metrics['datasets'][clean_name] = {
                    'metric_name': main_metric,
                    'metric_value': main_value,
                    'accuracy': dataset_metrics.get('accuracy', 0),
                    'f1_macro': dataset_metrics.get('metrics', {}).get('macro_f1', 0),
                    'mcc': dataset_metrics.get('metrics', {}).get('MCC', 0),
                    'total_samples': dataset_metrics.get('total_samples', 0),
                    'avg_latency_ms': dataset_metrics.get('latency_seconds', {}).get('mean', 0) * 1000 if dataset_metrics.get('latency_seconds', {}).get('mean') else 0
                }
    
    return key_metrics


def calculate_quantization_impact(no_quant_metrics: Dict, awq_metrics: Dict) -> Dict[str, Any]:
    """Calculate the impact of AWQ quantization on model performance."""
    impacts = {
        'per_dataset': {},
        'summary': {}
    }
    
    # Calculate impact for each dataset
    for dataset in no_quant_metrics['datasets']:
        if dataset in awq_metrics['datasets']:
            no_quant_data = no_quant_metrics['datasets'][dataset]
            awq_data = awq_metrics['datasets'][dataset]
            
            # Get the main metric values
            no_quant_val = no_quant_data['metric_value']
            awq_val = awq_data['metric_value']
            metric_name = no_quant_data['metric_name']
            
            # For avg_abs_diff, lower is better, so we need to invert the improvement calculation
            is_lower_better = (metric_name == 'avg_abs_diff')
            
            if is_lower_better:
                # For metrics where lower is better (like avg_abs_diff)
                improvement = no_quant_val - awq_val  # positive = AWQ better
                improvement_pct = ((no_quant_val - awq_val) / no_quant_val * 100) if no_quant_val > 0 else 0
            else:
                # For metrics where higher is better (like accuracy, F1)
                improvement = awq_val - no_quant_val  # positive = AWQ better
                improvement_pct = ((awq_val - no_quant_val) / no_quant_val * 100) if no_quant_val > 0 else 0
            
            impacts['per_dataset'][dataset] = {
                'metric_improvement': improvement,
                'metric_improvement_pct': improvement_pct,
                'no_quant_value': no_quant_val,
                'awq_value': awq_val,
                'metric_name': metric_name,
                'quantization_impact': 'positive' if improvement > 0 else 'negative'
            }
    
    # Calculate summary statistics
    improvements = [d['metric_improvement'] for d in impacts['per_dataset'].values()]
    impacts['summary'] = {
        'avg_improvement': sum(improvements) / len(improvements) if improvements else 0,
        'total_datasets': len(improvements),
        'positive_impacts': len([i for i in improvements if i > 0]),
        'negative_impacts': len([i for i in improvements if i < 0]),
        'overall_impact': 'positive' if sum(improvements) > 0 else 'negative'
    }
    
    return impacts


def generate_markdown_report(
    base_no_quant: Dict,
    base_awq: Dict,
    nopeft_no_quant: Dict, 
    nopeft_awq: Dict,
    base_impact: Dict,
    nopeft_impact: Dict
) -> str:
    """Generate a comprehensive markdown report."""
    
    report = f"""# AWQ Post-Training Quantization Study Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This study analyzes the effects of AWQ (Activation-aware Weight Quantization) post-training quantization on the Qwen 0.6B model, comparing both base models and fine-tuned models across multiple test datasets.

### Key Findings

**Base Model Impact:**
- **Average AWQ improvement:** {base_impact['summary']['avg_improvement']:.3f} (across all metrics)
- **Positive impacts:** {base_impact['summary']['positive_impacts']}/{base_impact['summary']['total_datasets']} datasets
- **Overall impact:** {base_impact['summary']['overall_impact'].title()}

**Fine-tuned Model Impact:**
- **Average AWQ improvement:** {nopeft_impact['summary']['avg_improvement']:.3f} (across all metrics)
- **Positive impacts:** {nopeft_impact['summary']['positive_impacts']}/{nopeft_impact['summary']['total_datasets']} datasets
- **Overall impact:** {nopeft_impact['summary']['overall_impact'].title()}

## Model Configurations

| Model | Quantization | Parameters | VRAM Usage | Description |
|-------|--------------|------------|------------|-------------|
| Base | None | {base_no_quant['total_params']:,} | {base_no_quant['vram_usage_gb']:.1f} GB | Original base model |
| Base AWQ | 4-bit W4G128 | {base_awq['total_params']:,} | {base_awq['vram_usage_gb']:.1f} GB | Base + AWQ quantization |
| NoPeft | None | {nopeft_no_quant['total_params']:,} | {nopeft_no_quant['vram_usage_gb']:.1f} GB | OpenMath fine-tuned |
| NoPeft AWQ | 4-bit W4G128 | {nopeft_awq['total_params']:,} | {nopeft_awq['vram_usage_gb']:.1f} GB | Fine-tuned + AWQ quantization |

## Performance Analysis

### Base Model Quantization Effects

| Dataset | Metric | No Quant | AWQ | Improvement | Impact |
|---------|--------|----------|-----|-------------|---------|"""

    # Base model comparison table
    for dataset in sorted(base_no_quant['datasets'].keys()):
        if dataset in base_awq['datasets']:
            no_quant_val = base_no_quant['datasets'][dataset]['metric_value']
            awq_val = base_awq['datasets'][dataset]['metric_value']
            metric_name = base_no_quant['datasets'][dataset]['metric_name']
            improvement = base_impact['per_dataset'][dataset]['metric_improvement']
            impact = "✅" if improvement > 0 else "❌"
            
            metric_display = metric_name.replace('_', ' ').title()
            if metric_name == 'avg_abs_diff':
                metric_display = "Avg Abs Diff"
            
            report += f"\n| {dataset} | {metric_display} | {no_quant_val:.3f} | {awq_val:.3f} | {improvement:+.3f} | {impact} |"

    report += f"""

### Fine-tuned Model Quantization Effects

| Dataset | Metric | No Quant | AWQ | Improvement | Impact |
|---------|--------|----------|-----|-------------|---------|"""

    # Fine-tuned model comparison table
    for dataset in sorted(nopeft_no_quant['datasets'].keys()):
        if dataset in nopeft_awq['datasets']:
            no_quant_val = nopeft_no_quant['datasets'][dataset]['metric_value']
            awq_val = nopeft_awq['datasets'][dataset]['metric_value']
            metric_name = nopeft_no_quant['datasets'][dataset]['metric_name']
            improvement = nopeft_impact['per_dataset'][dataset]['metric_improvement']
            impact = "✅" if improvement > 0 else "❌"
            
            metric_display = metric_name.replace('_', ' ').title()
            if metric_name == 'avg_abs_diff':
                metric_display = "Avg Abs Diff"
            
            report += f"\n| {dataset} | {metric_display} | {no_quant_val:.3f} | {awq_val:.3f} | {improvement:+.3f} | {impact} |"

    report += f"""

## Analysis

### AWQ Algorithm Performance

AWQ (Activation-aware Weight Quantization) uses calibration data to compute optimal per-channel scaling factors that preserve important activations while quantizing weights to 4-bit precision.

**Observed Effects:**

#### Base Model Results
"""
    
    # Analyze base model impacts
    base_positive = []
    base_negative = []
    
    for dataset, data in base_impact['per_dataset'].items():
        if data['quantization_impact'] == 'positive':
            base_positive.append(f"- **{dataset}**: +{data['metric_improvement']:.3f} ({data['metric_improvement_pct']:+.1f}%) [{data['metric_name']}]")
        else:
            base_negative.append(f"- **{dataset}**: {data['metric_improvement']:.3f} ({data['metric_improvement_pct']:+.1f}%) [{data['metric_name']}]")
    
    if base_positive:
        report += "\n**Positive Impacts:**\n" + "\n".join(base_positive) + "\n"
    if base_negative:
        report += "\n**Negative Impacts:**\n" + "\n".join(base_negative) + "\n"

    report += f"""
#### Fine-tuned Model Results
"""
    
    # Analyze fine-tuned model impacts
    nopeft_positive = []
    nopeft_negative = []
    
    for dataset, data in nopeft_impact['per_dataset'].items():
        if data['quantization_impact'] == 'positive':
            nopeft_positive.append(f"- **{dataset}**: +{data['metric_improvement']:.3f} ({data['metric_improvement_pct']:+.1f}%) [{data['metric_name']}]")
        else:
            nopeft_negative.append(f"- **{dataset}**: {data['metric_improvement']:.3f} ({data['metric_improvement_pct']:+.1f}%) [{data['metric_name']}]")
    
    if nopeft_positive:
        report += "\n**Positive Impacts:**\n" + "\n".join(nopeft_positive) + "\n"
    if nopeft_negative:
        report += "\n**Negative Impacts:**\n" + "\n".join(nopeft_negative) + "\n"

    # Calculate VRAM differences
    base_vram_diff = base_no_quant['vram_usage_gb'] - base_awq['vram_usage_gb'] if base_no_quant['vram_usage_gb'] > 0 else 0
    nopeft_vram_diff = nopeft_no_quant['vram_usage_gb'] - nopeft_awq['vram_usage_gb'] if nopeft_no_quant['vram_usage_gb'] > 0 else 0

    report += f"""
### Resource Efficiency

**VRAM Usage:**
- **Base Model**: {base_awq['vram_usage_gb']:.1f} GB (AWQ) vs {base_no_quant['vram_usage_gb']:.1f} GB (No Quant) - {base_vram_diff:.1f} GB saved
- **Fine-tuned Model**: {nopeft_awq['vram_usage_gb']:.1f} GB (AWQ) vs {nopeft_no_quant['vram_usage_gb']:.1f} GB (No Quant) - {nopeft_vram_diff:.1f} GB saved

### Conclusions

"""
    
    if base_impact['summary']['overall_impact'] == 'positive' and nopeft_impact['summary']['overall_impact'] == 'positive':
        report += "✅ **AWQ shows positive impact** on both base and fine-tuned models, providing good memory efficiency with maintained or improved performance.\n\n"
    elif base_impact['summary']['overall_impact'] == 'positive' or nopeft_impact['summary']['overall_impact'] == 'positive':
        report += "⚖️ **AWQ shows mixed results**, performing better on some model types. Consider model-specific evaluation for deployment decisions.\n\n"
    else:
        report += "❌ **AWQ shows negative impact** on both model types, though it provides memory savings. Consider the performance trade-off carefully.\n\n"

    report += f"""### Recommendations

"""
    
    if base_impact['summary']['avg_improvement'] > 0 and nopeft_impact['summary']['avg_improvement'] > 0:
        report += "- **Use AWQ** for both base and fine-tuned models as it provides memory savings with performance benefits\n"
        report += "- AWQ's activation-aware scaling effectively preserves model performance under 4-bit quantization\n"
    elif base_impact['summary']['avg_improvement'] > 0:
        report += "- **Use AWQ** primarily for base models where it shows clear benefits\n"
        report += "- **Careful evaluation needed** for fine-tuned models before deployment\n"
    elif nopeft_impact['summary']['avg_improvement'] > 0:
        report += "- **Use AWQ** for fine-tuned models where it shows benefits\n"
        report += "- **Consider alternatives** for base models\n"
    else:
        report += "- **Consider alternatives** to AWQ for this model family and dataset combination\n"
        report += "- **Evaluate other quantization methods** like GPTQ, HQQ, or QLoRA for better performance preservation\n"
        report += f"- Current performance penalty: Base {abs(base_impact['summary']['avg_improvement']):.3f}, Fine-tuned {abs(nopeft_impact['summary']['avg_improvement']):.3f}\n"

    report += f"""
---

*Study conducted on {base_impact['summary']['total_datasets']} test datasets with TRUNC_EVAL=20 samples each.*
*Base model: Qwen/Qwen3-0.6B, Fine-tuning: OpenMath (1000 samples), AWQ config: 4-bit weights, group size 128*
*Calibration: {base_awq.get('calibration_samples', 'N/A')} samples from OpenMath dataset*
"""
    
    return report


def main():
    """Main function to generate the AWQ comparison report."""
    metrics_dir = Path("Testing/metrics")
    study_results_dir = Path("study_results")
    study_results_dir.mkdir(exist_ok=True)
    
    # Define the models for our AWQ study
    target_models = {
        'base_no_quant': 'Models__Qwen3-0.6B-base.json',
        'base_awq': 'Qwen3-0.6B-base_awq_w4g128.json',
        'nopeft_no_quant': 'Models__Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant.json',
        'nopeft_awq': 'Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_awq_w4g128.json'
    }
    
    # Load metrics for each model
    print("Loading metrics files...")
    metrics = {}
    for model_type, filename in target_models.items():
        filepath = metrics_dir / filename
        if filepath.exists():
            metrics[model_type] = load_metrics_file(filepath)
            print(f"✅ Loaded {model_type}: {filename}")
        else:
            print(f"❌ Missing {model_type}: {filename}")
            return
    
    # Extract key metrics
    print("\nExtracting key metrics...")
    key_metrics = {}
    for model_type, data in metrics.items():
        key_metrics[model_type] = extract_key_metrics(data)
    
    # Calculate quantization impacts
    print("Calculating quantization impacts...")
    base_impact = calculate_quantization_impact(
        key_metrics['base_no_quant'], 
        key_metrics['base_awq']
    )
    
    nopeft_impact = calculate_quantization_impact(
        key_metrics['nopeft_no_quant'], 
        key_metrics['nopeft_awq']
    )
    
    # Generate markdown report
    print("Generating markdown report...")
    markdown_report = generate_markdown_report(
        key_metrics['base_no_quant'],
        key_metrics['base_awq'],
        key_metrics['nopeft_no_quant'],
        key_metrics['nopeft_awq'],
        base_impact,
        nopeft_impact
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary JSON
    summary_data = {
        'study_type': 'awq_post_training_quantization',
        'generated_at': datetime.now().isoformat(),
        'models_compared': list(target_models.keys()),
        'key_metrics': key_metrics,
        'base_model_impact': base_impact,
        'nopeft_model_impact': nopeft_impact,
        'conclusions': {
            'base_model_winner': 'awq' if base_impact['summary']['avg_improvement'] > 0 else 'no_quant',
            'nopeft_model_winner': 'awq' if nopeft_impact['summary']['avg_improvement'] > 0 else 'no_quant',
            'overall_awq_effectiveness': 'positive' if (base_impact['summary']['avg_improvement'] + nopeft_impact['summary']['avg_improvement']) > 0 else 'negative'
        }
    }
    
    summary_path = study_results_dir / f"awq_study_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Save markdown report
    report_path = study_results_dir / f"awq_study_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"\n✅ Reports generated:")
    print(f"📊 Summary JSON: {summary_path}")
    print(f"📄 Markdown report: {report_path}")
    
    # Quick summary
    base_winner = "AWQ" if base_impact['summary']['avg_improvement'] > 0 else "No Quantization"
    nopeft_winner = "AWQ" if nopeft_impact['summary']['avg_improvement'] > 0 else "No Quantization"
    
    print(f"\n🎯 Study Results:")
    print(f"📈 Base Model: {base_winner} performs better by {abs(base_impact['summary']['avg_improvement']):.3f} points on average")
    print(f"📈 Fine-tuned Model: {nopeft_winner} performs better by {abs(nopeft_impact['summary']['avg_improvement']):.3f} points on average")


if __name__ == "__main__":
    main()