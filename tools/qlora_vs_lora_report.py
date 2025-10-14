#!/usr/bin/env python3
"""
QLoRA vs LoRA Comparison Report Generator

Analyzes the differences between QLoRA (training-time quantization) and LoRA (no quantization)
by comparing performance metrics across test datasets.
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
        'vram_usage_gb': metrics.get('vram_usage_gb', 0),
        'avg_latency_ms': metrics.get('avg_latency_ms', 0),
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


def calculate_performance_differences(base_metrics: Dict, lora_metrics: Dict, qlora_metrics: Dict) -> Dict[str, Any]:
    """Calculate performance differences between methods."""
    differences = {
        'lora_vs_base': {},
        'qlora_vs_base': {},
        'qlora_vs_lora': {},
        'summary': {}
    }
    
    # Calculate improvements for each dataset
    for dataset in lora_metrics['datasets']:
        if dataset in base_metrics['datasets'] and dataset in qlora_metrics['datasets']:
            base_data = base_metrics['datasets'][dataset]
            lora_data = lora_metrics['datasets'][dataset]
            qlora_data = qlora_metrics['datasets'][dataset]
            
            # Get the main metric values
            base_val = base_data['metric_value']
            lora_val = lora_data['metric_value']
            qlora_val = qlora_data['metric_value']
            metric_name = lora_data['metric_name']
            
            # For avg_abs_diff, lower is better, so we need to invert the improvement calculation
            is_lower_better = (metric_name == 'avg_abs_diff')
            
            if is_lower_better:
                # For metrics where lower is better (like avg_abs_diff)
                lora_improvement = base_val - lora_val  # positive = improvement
                qlora_improvement = base_val - qlora_val  # positive = improvement
                qlora_vs_lora_diff = lora_val - qlora_val  # positive = QLoRA better
                
                lora_improvement_pct = ((base_val - lora_val) / base_val * 100) if base_val > 0 else 0
                qlora_improvement_pct = ((base_val - qlora_val) / base_val * 100) if base_val > 0 else 0
                qlora_vs_lora_pct = ((lora_val - qlora_val) / lora_val * 100) if lora_val > 0 else 0
            else:
                # For metrics where higher is better (like accuracy, F1)
                lora_improvement = lora_val - base_val  # positive = improvement
                qlora_improvement = qlora_val - base_val  # positive = improvement
                qlora_vs_lora_diff = qlora_val - lora_val  # positive = QLoRA better
                
                lora_improvement_pct = ((lora_val - base_val) / base_val * 100) if base_val > 0 else 0
                qlora_improvement_pct = ((qlora_val - base_val) / base_val * 100) if base_val > 0 else 0
                qlora_vs_lora_pct = ((qlora_val - lora_val) / lora_val * 100) if lora_val > 0 else 0
            
            differences['lora_vs_base'][dataset] = {
                'metric_improvement': lora_improvement,
                'metric_improvement_pct': lora_improvement_pct,
                'base_value': base_val,
                'lora_value': lora_val,
                'metric_name': metric_name
            }
            
            differences['qlora_vs_base'][dataset] = {
                'metric_improvement': qlora_improvement,
                'metric_improvement_pct': qlora_improvement_pct,
                'base_value': base_val,
                'qlora_value': qlora_val,
                'metric_name': metric_name
            }
            
            differences['qlora_vs_lora'][dataset] = {
                'metric_difference': qlora_vs_lora_diff,
                'metric_difference_pct': qlora_vs_lora_pct,
                'lora_value': lora_val,
                'qlora_value': qlora_val,
                'metric_name': metric_name,
                'quantization_impact': 'positive' if qlora_vs_lora_diff > 0 else 'negative'
            }
    
    # Calculate average improvements
    lora_improvements = [d['metric_improvement'] for d in differences['lora_vs_base'].values()]
    qlora_improvements = [d['metric_improvement'] for d in differences['qlora_vs_base'].values()]
    qlora_vs_lora_diffs = [d['metric_difference'] for d in differences['qlora_vs_lora'].values()]
    
    differences['summary'] = {
        'avg_lora_improvement': sum(lora_improvements) / len(lora_improvements) if lora_improvements else 0,
        'avg_qlora_improvement': sum(qlora_improvements) / len(qlora_improvements) if qlora_improvements else 0,
        'avg_qlora_vs_lora_diff': sum(qlora_vs_lora_diffs) / len(qlora_vs_lora_diffs) if qlora_vs_lora_diffs else 0,
        'quantization_impact_summary': 'positive' if sum(qlora_vs_lora_diffs) > 0 else 'negative',
        'total_datasets_compared': len(qlora_vs_lora_diffs)
    }
    
    return differences


def generate_markdown_report(
    base_metrics: Dict, 
    lora_metrics: Dict, 
    qlora_metrics: Dict, 
    differences: Dict
) -> str:
    """Generate a comprehensive markdown report."""
    
    report = f"""# QLoRA vs LoRA Comparison Study Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This study compares the performance of QLoRA (training-time 4-bit quantization) versus LoRA (no quantization) fine-tuning on the Qwen 0.6B model using the OpenMath dataset.

### Key Findings

- **LoRA average improvement over base:** {differences['summary']['avg_lora_improvement']:.3f} (improvement in primary metrics)
- **QLoRA average improvement over base:** {differences['summary']['avg_qlora_improvement']:.3f} (improvement in primary metrics)
- **QLoRA vs LoRA difference:** {differences['summary']['avg_qlora_vs_lora_diff']:.3f} (positive means QLoRA better)
- **Quantization impact:** {differences['summary']['quantization_impact_summary'].title()} ({'better' if differences['summary']['quantization_impact_summary'] == 'positive' else 'worse'} than LoRA)

## Model Configurations

| Model | Method | Parameters | VRAM Usage | Avg Latency |
|-------|--------|------------|------------|-------------|
| Base | No fine-tuning | {base_metrics['total_params']:,} | {base_metrics['vram_usage_gb']:.1f} GB | {base_metrics['avg_latency_ms']:.1f} ms |
| LoRA | LoRA (no quantization) | {lora_metrics['total_params']:,} | {lora_metrics['vram_usage_gb']:.1f} GB | {lora_metrics['avg_latency_ms']:.1f} ms |
| QLoRA | LoRA + 4-bit quantization | {qlora_metrics['total_params']:,} | {qlora_metrics['vram_usage_gb']:.1f} GB | {qlora_metrics['avg_latency_ms']:.1f} ms |

## Performance Comparison by Dataset

"""
    
    # Dataset comparison table
    report += "| Dataset | Metric | Base | LoRA | QLoRA | LoRA Improvement | QLoRA Improvement | QLoRA vs LoRA |\n"
    report += "|---------|--------|------|------|-------|------------------|-------------------|----------------|\n"
    
    for dataset in sorted(lora_metrics['datasets'].keys()):
        if dataset in base_metrics['datasets'] and dataset in qlora_metrics['datasets']:
            base_val = base_metrics['datasets'][dataset]['metric_value']
            lora_val = lora_metrics['datasets'][dataset]['metric_value']
            qlora_val = qlora_metrics['datasets'][dataset]['metric_value']
            metric_name = lora_metrics['datasets'][dataset]['metric_name']
            
            lora_imp = differences['lora_vs_base'][dataset]['metric_improvement']
            qlora_imp = differences['qlora_vs_base'][dataset]['metric_improvement']
            qlora_vs_lora = differences['qlora_vs_lora'][dataset]['metric_difference']
            
            # Format metric name for display
            metric_display = metric_name.replace('_', ' ').title()
            if metric_name == 'avg_abs_diff':
                metric_display = "Avg Abs Diff"
            
            report += f"| {dataset} | {metric_display} | {base_val:.3f} | {lora_val:.3f} | {qlora_val:.3f} | "
            report += f"{lora_imp:+.3f} | {qlora_imp:+.3f} | {qlora_vs_lora:+.3f} |\n"
    
    report += f"""

## Analysis

### Training-Time Quantization Effects

The QLoRA approach applies 4-bit NF4 quantization during training, which has the following observed effects:

"""
    
    # Analyze the impact for each dataset
    positive_impacts = []
    negative_impacts = []
    
    for dataset, diff_data in differences['qlora_vs_lora'].items():
        if diff_data['quantization_impact'] == 'positive':
            positive_impacts.append(f"- **{dataset}**: +{diff_data['metric_difference']:.3f} ({diff_data['metric_difference_pct']:+.1f}%) [{diff_data['metric_name']}]")
        else:
            negative_impacts.append(f"- **{dataset}**: {diff_data['metric_difference']:.3f} ({diff_data['metric_difference_pct']:+.1f}%) [{diff_data['metric_name']}]")
    
    if positive_impacts:
        report += "#### Positive Impacts (QLoRA > LoRA)\n"
        report += "\n".join(positive_impacts) + "\n\n"
    
    if negative_impacts:
        report += "#### Negative Impacts (QLoRA < LoRA)\n"
        report += "\n".join(negative_impacts) + "\n\n"
    
    # Calculate VRAM and latency differences safely
    vram_reduction = ""
    if lora_metrics['vram_usage_gb'] > 0:
        vram_pct = ((lora_metrics['vram_usage_gb'] - qlora_metrics['vram_usage_gb']) / lora_metrics['vram_usage_gb'] * 100)
        vram_reduction = f"({vram_pct:+.1f}% reduction)"
    else:
        vram_reduction = "(N/A)"
    
    latency_change = ""
    if lora_metrics['avg_latency_ms'] > 0:
        latency_pct = ((qlora_metrics['avg_latency_ms'] - lora_metrics['avg_latency_ms']) / lora_metrics['avg_latency_ms'] * 100)
        latency_change = f"({latency_pct:+.1f}% change)"
    else:
        latency_change = "(N/A)"

    report += f"""### Resource Efficiency

- **VRAM Usage**: QLoRA uses {qlora_metrics['vram_usage_gb']:.1f} GB vs LoRA's {lora_metrics['vram_usage_gb']:.1f} GB {vram_reduction}
- **Latency Impact**: QLoRA: {qlora_metrics['avg_latency_ms']:.1f} ms vs LoRA: {lora_metrics['avg_latency_ms']:.1f} ms {latency_change}

### Conclusions

"""
    
    if differences['summary']['avg_qlora_vs_lora_diff'] > 0:
        report += f"QLoRA shows **superior performance** compared to LoRA with an average improvement of {differences['summary']['avg_qlora_vs_lora_diff']:.3f} points across all metrics. This suggests that 4-bit quantization during training may provide a beneficial regularization effect for this model and dataset combination.\n\n"
    else:
        report += f"LoRA shows **superior performance** compared to QLoRA with QLoRA performing {abs(differences['summary']['avg_qlora_vs_lora_diff']):.3f} points worse on average. This suggests that 4-bit quantization during training introduces some performance degradation, though it provides significant memory savings.\n\n"
    
    report += """### Recommendations

"""
    
    if differences['summary']['avg_qlora_vs_lora_diff'] > 0:
        report += "- **Use QLoRA** when memory is constrained, as it provides both better performance and lower VRAM usage\n"
        report += "- QLoRA appears to have a beneficial regularization effect for this model size and dataset\n"
    else:
        report += "- **Use LoRA** when maximum performance is required and memory is not constrained\n"
        report += "- **Use QLoRA** when memory is severely constrained and the performance trade-off is acceptable\n"
        report += f"- The performance penalty of QLoRA is relatively small ({abs(differences['summary']['avg_qlora_vs_lora_diff']):.3f} points) compared to the memory savings\n"
    
    report += f"""
---

*Study conducted on {differences['summary']['total_datasets_compared']} test datasets with TRUNC_EVAL=20 samples each.*
*Base model: Qwen/Qwen3-0.6B, Training dataset: OpenMath (1000 samples), LoRA rank: 64*
"""
    
    return report


def main():
    """Main function to generate the QLoRA vs LoRA comparison report."""
    metrics_dir = Path("Testing/metrics")
    study_results_dir = Path("study_results")
    study_results_dir.mkdir(exist_ok=True)
    
    # Define the models for our study
    target_models = {
        'base': 'Models__Qwen3-0.6B-base.json',
        'lora': 'Qwen3-0.6B-openmath_SFT_LoRa64_NoQuant.json',
        'qlora': 'Qwen3-0.6B-openmath_SFT_LoRa64_QLORA_w4_headbf16.json'
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
    
    # Calculate performance differences
    print("Calculating performance differences...")
    differences = calculate_performance_differences(
        key_metrics['base'], 
        key_metrics['lora'], 
        key_metrics['qlora']
    )
    
    # Generate markdown report
    print("Generating markdown report...")
    markdown_report = generate_markdown_report(
        key_metrics['base'], 
        key_metrics['lora'], 
        key_metrics['qlora'], 
        differences
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary JSON
    summary_data = {
        'study_type': 'qlora_vs_lora_comparison',
        'generated_at': datetime.now().isoformat(),
        'models_compared': list(target_models.keys()),
        'key_metrics': key_metrics,
        'performance_differences': differences,
        'conclusions': {
            'qlora_vs_lora_winner': 'qlora' if differences['summary']['avg_qlora_vs_lora_diff'] > 0 else 'lora',
            'avg_difference': differences['summary']['avg_qlora_vs_lora_diff'],
            'quantization_impact': differences['summary']['quantization_impact_summary']
        }
    }
    
    summary_path = study_results_dir / f"qlora_vs_lora_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Save markdown report
    report_path = study_results_dir / f"qlora_vs_lora_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"\n✅ Reports generated:")
    print(f"📊 Summary JSON: {summary_path}")
    print(f"📄 Markdown report: {report_path}")
    
    # Quick summary
    winner = "QLoRA" if differences['summary']['avg_qlora_vs_lora_diff'] > 0 else "LoRA"
    diff = abs(differences['summary']['avg_qlora_vs_lora_diff'])
    print(f"\n🎯 Study Result: {winner} performs better by {diff:.3f} accuracy points on average")


if __name__ == "__main__":
    main()