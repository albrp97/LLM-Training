#!/usr/bin/env python
"""SmoothQuant W8A8 Post-Training Quantization Study Report Generator.

This script generates a comprehensive analysis comparing SmoothQuant quantized models
against their unquantized counterparts, following the established systematic evaluation
framework used in previous quantization studies.

Compares:
- Base model vs Base + SmoothQuant W8A8
- Fine-tuned (NoPeft) vs Fine-tuned + SmoothQuant W8A8

Analysis includes accuracy metrics, degradation patterns, and quantization effectiveness.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load evaluation metrics from JSON file."""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Metrics file not found: {metrics_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {metrics_path}: {e}")
        return {}

def extract_key_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract key performance metrics from evaluation results."""
    if not metrics or 'datasets' not in metrics:
        return {}
    
    extracted = {}
    
    # Extract dataset-specific metrics from the 'datasets' key
    for dataset_file, data in metrics['datasets'].items():
        if isinstance(data, dict):
            # Extract clean dataset name from filename
            if 'OpenMathInstruct' in dataset_file:
                dataset = 'openmath'
            elif 'ai2_arc' in dataset_file:
                dataset = 'arc'
            elif 'squad' in dataset_file:
                dataset = 'squad'
            elif 'boolq' in dataset_file:
                dataset = 'boolq'
            else:
                continue  # Skip unknown datasets
            
            # Extract accuracy (all datasets have this)
            if 'accuracy' in data:
                extracted[f'{dataset}_accuracy'] = data.get('accuracy', 0.0)
            
            # Extract F1 if available in metrics subkey
            if 'metrics' in data:
                if 'macro_f1' in data['metrics']:
                    extracted[f'{dataset}_f1'] = data['metrics'].get('macro_f1', 0.0)
                if 'mcc' in data['metrics']:
                    extracted[f'{dataset}_mcc'] = data['metrics'].get('mcc', 0.0)
    
    return extracted

def calculate_quantization_impact(base_metrics: Dict[str, float], quant_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Calculate the impact of quantization on model performance."""
    impact = {}
    
    for metric, base_value in base_metrics.items():
        if metric in quant_metrics:
            quant_value = quant_metrics[metric]
            
            # All metrics are now accuracy or F1/MCC (higher is better)
            degradation = quant_value - base_value
            percent_change = ((quant_value - base_value) / base_value * 100) if base_value != 0 else 0
            winner = "SmoothQuant" if quant_value > base_value else "NoQuant"
            
            impact[metric] = {
                'base_value': base_value,
                'quant_value': quant_value,
                'absolute_change': degradation,
                'percent_change': percent_change,
                'winner': winner
            }
    
    return impact

def generate_dataset_analysis(impact_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    """Generate per-dataset analysis from impact data."""
    datasets = {}
    
    for metric, data in impact_data.items():
        # Extract dataset name from metric
        dataset_name = metric.split('_')[0]
        metric_type = '_'.join(metric.split('_')[1:])
        
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                'metrics': {},
                'summary': {'wins': 0, 'losses': 0, 'total_metrics': 0}
            }
        
        datasets[dataset_name]['metrics'][metric_type] = data
        datasets[dataset_name]['summary']['total_metrics'] += 1
        
        # Count wins/losses for SmoothQuant
        if data['winner'] == 'SmoothQuant':
            datasets[dataset_name]['summary']['wins'] += 1
        else:
            datasets[dataset_name]['summary']['losses'] += 1
    
    # Calculate win rate for each dataset
    for dataset_data in datasets.values():
        total = dataset_data['summary']['total_metrics']
        wins = dataset_data['summary']['wins']
        dataset_data['summary']['win_rate'] = (wins / total * 100) if total > 0 else 0
    
    return datasets

def format_performance_table(impact_data: Dict[str, Dict[str, float]]) -> str:
    """Format performance comparison as a markdown table."""
    if not impact_data:
        return "No performance data available.\n"
    
    table = "| Dataset | Metric | NoQuant | SmoothQuant | Change | Winner |\n"
    table += "|---------|--------|---------|-------------|---------|--------|\n"
    
    for metric, data in sorted(impact_data.items()):
        dataset = metric.split('_')[0]
        metric_type = '_'.join(metric.split('_')[1:])
        
        base_val = data['base_value']
        quant_val = data['quant_value']
        change = data['percent_change']
        winner = data['winner']
        
        # All metrics are percentages (accuracy, F1, MCC)
        base_str = f"{base_val:.1f}%"
        quant_str = f"{quant_val:.1f}%"
        change_str = f"{change:+.1f}%"
        
        table += f"| {dataset.upper()} | {metric_type} | {base_str} | {quant_str} | {change_str} | **{winner}** |\n"
    
    return table

def generate_summary_insights(base_analysis: Dict[str, Dict[str, Any]], 
                             finetuned_analysis: Dict[str, Dict[str, Any]]) -> List[str]:
    """Generate high-level insights from the analysis."""
    insights = []
    
    # Overall effectiveness analysis
    base_wins = sum(data['summary']['wins'] for data in base_analysis.values())
    base_total = sum(data['summary']['total_metrics'] for data in base_analysis.values())
    base_win_rate = (base_wins / base_total * 100) if base_total > 0 else 0
    
    finetuned_wins = sum(data['summary']['wins'] for data in finetuned_analysis.values())
    finetuned_total = sum(data['summary']['total_metrics'] for data in finetuned_analysis.values())
    finetuned_win_rate = (finetuned_wins / finetuned_total * 100) if finetuned_total > 0 else 0
    
    insights.append(f"**Overall SmoothQuant Win Rate**: Base model {base_win_rate:.1f}%, Fine-tuned model {finetuned_win_rate:.1f}%")
    
    # Model type comparison
    if base_win_rate > finetuned_win_rate:
        insights.append(f"SmoothQuant shows better effectiveness on base models (+{base_win_rate - finetuned_win_rate:.1f}% win rate)")
    elif finetuned_win_rate > base_win_rate:
        insights.append(f"SmoothQuant shows better effectiveness on fine-tuned models (+{finetuned_win_rate - base_win_rate:.1f}% win rate)")
    else:
        insights.append("SmoothQuant shows similar effectiveness across base and fine-tuned models")
    
    # Dataset-specific patterns
    if base_analysis and finetuned_analysis:
        all_datasets = set(base_analysis.keys()) | set(finetuned_analysis.keys())
        
        for dataset in all_datasets:
            base_rate = base_analysis.get(dataset, {}).get('summary', {}).get('win_rate', 0)
            ft_rate = finetuned_analysis.get(dataset, {}).get('summary', {}).get('win_rate', 0)
            
            if base_rate == 0 and ft_rate == 0:
                insights.append(f"**{dataset.upper()}**: SmoothQuant consistently underperforms on both model types")
            elif base_rate == 100 and ft_rate == 100:
                insights.append(f"**{dataset.upper()}**: SmoothQuant consistently outperforms on both model types")
            elif abs(base_rate - ft_rate) > 30:
                better_type = "base" if base_rate > ft_rate else "fine-tuned"
                insights.append(f"**{dataset.upper()}**: SmoothQuant works significantly better with {better_type} models")
    
    # W8A8 specific insights
    insights.append("**Technical Notes**: SmoothQuant W8A8 quantizes both weights and activations to 8-bit, providing balanced compression with moderate precision loss")
    
    return insights

def generate_markdown_report(base_impact: Dict[str, Dict[str, float]], 
                           finetuned_impact: Dict[str, Dict[str, float]],
                           base_analysis: Dict[str, Dict[str, Any]], 
                           finetuned_analysis: Dict[str, Dict[str, Any]]) -> str:
    """Generate comprehensive markdown report."""
    
    report = f"""# SmoothQuant W8A8 Post-Training Quantization Study

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the effectiveness of SmoothQuant W8A8 quantization on the Qwen3-0.6B model across both base and fine-tuned variants. SmoothQuant applies scaling factors to balance quantization between weights and activations, targeting both at 8-bit precision.

### Key Findings

{chr(10).join(f"- {insight}" for insight in generate_summary_insights(base_analysis, finetuned_analysis))}

## Base Model Comparison

**Models**: `Qwen3-0.6B-base` vs `Qwen3-0.6B-base_smoothquant_w8a8`

{format_performance_table(base_impact)}

### Base Model Analysis
"""
    
    for dataset, data in base_analysis.items():
        win_rate = data['summary']['win_rate']
        total_metrics = data['summary']['total_metrics']
        
        report += f"""
**{dataset.upper()} Dataset**: {win_rate:.1f}% win rate ({data['summary']['wins']}/{total_metrics} metrics)
"""
        
        for metric_type, metric_data in data['metrics'].items():
            change = metric_data['percent_change']
            winner = metric_data['winner']
            report += f"- {metric_type}: {change:+.1f}% change, **{winner}** wins\n"
    
    report += f"""

## Fine-tuned Model Comparison

**Models**: `Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant` vs `Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_smoothquant_w8a8`

{format_performance_table(finetuned_impact)}

### Fine-tuned Model Analysis
"""
    
    for dataset, data in finetuned_analysis.items():
        win_rate = data['summary']['win_rate']
        total_metrics = data['summary']['total_metrics']
        
        report += f"""
**{dataset.upper()} Dataset**: {win_rate:.1f}% win rate ({data['summary']['wins']}/{total_metrics} metrics)
"""
        
        for metric_type, metric_data in data['metrics'].items():
            change = metric_data['percent_change']
            winner = metric_data['winner']
            report += f"- {metric_type}: {change:+.1f}% change, **{winner}** wins\n"
    
    report += """

## Technical Details

### SmoothQuant Method
- **Algorithm**: SmoothQuant with per-channel scaling
- **Weight Quantization**: 8-bit signed integers
- **Activation Quantization**: 8-bit signed integers  
- **Calibration**: 100 OpenMath samples for scaling factor computation
- **Target**: Balanced W8A8 quantization with activation-aware scaling

### Evaluation Framework
- **Datasets**: OpenMath (math reasoning), Squad (reading comprehension), ARC (commonsense reasoning), BoolQ (yes/no questions)
- **Metrics**: Accuracy for Squad/ARC/BoolQ, Average Absolute Difference for OpenMath
- **Sample Size**: 20 samples per dataset (TRUNC_EVAL=20)
- **Methodology**: Direct comparison between quantized and unquantized model performance

### Model Details
- **Base Architecture**: Qwen3-0.6B (600M parameters)
- **Fine-tuning**: OpenMath dataset with supervised fine-tuning (SFT), no PEFT
- **Quantization Scope**: 197 Linear layers quantized per model
- **Preserved Components**: LM head kept in FP16 for stability

## Conclusions

The SmoothQuant W8A8 quantization study reveals **{generate_summary_insights(base_analysis, finetuned_analysis)[0].split(':')[1].strip()}** across the evaluated model variants and datasets.

This systematic evaluation provides insights into SmoothQuant's effectiveness as a post-training quantization method for efficient model deployment while maintaining acceptable performance levels.
"""
    
    return report

def generate_json_summary(base_impact: Dict[str, Dict[str, float]], 
                         finetuned_impact: Dict[str, Dict[str, float]],
                         base_analysis: Dict[str, Dict[str, Any]], 
                         finetuned_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate structured JSON summary of the analysis."""
    
    # Calculate overall statistics
    base_wins = sum(data['summary']['wins'] for data in base_analysis.values())
    base_total = sum(data['summary']['total_metrics'] for data in base_analysis.values())
    base_win_rate = (base_wins / base_total * 100) if base_total > 0 else 0
    
    finetuned_wins = sum(data['summary']['wins'] for data in finetuned_analysis.values())
    finetuned_total = sum(data['summary']['total_metrics'] for data in finetuned_analysis.values())
    finetuned_win_rate = (finetuned_wins / finetuned_total * 100) if finetuned_total > 0 else 0
    
    return {
        "study_type": "SmoothQuant W8A8 Post-Training Quantization",
        "generated_at": datetime.now().isoformat(),
        "models_compared": {
            "base": ["Qwen3-0.6B-base", "Qwen3-0.6B-base_smoothquant_w8a8"],
            "finetuned": ["Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant", "Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_smoothquant_w8a8"]
        },
        "overall_performance": {
            "base_model_smoothquant_win_rate": round(base_win_rate, 1),
            "finetuned_model_smoothquant_win_rate": round(finetuned_win_rate, 1),
            "overall_smoothquant_win_rate": round((base_win_rate + finetuned_win_rate) / 2, 1),
            "total_comparisons": base_total + finetuned_total
        },
        "detailed_results": {
            "base_model": {
                "impact_analysis": base_impact,
                "dataset_analysis": base_analysis
            },
            "finetuned_model": {
                "impact_analysis": finetuned_impact,
                "dataset_analysis": finetuned_analysis
            }
        },
        "quantization_config": {
            "method": "SmoothQuant",
            "weights_bits": 8,
            "activations_bits": 8,
            "calibration_samples": 100,
            "calibration_dataset": "openmath",
            "layers_quantized": 197,
            "preserve_lm_head": True
        },
        "evaluation_config": {
            "datasets": ["openmath", "squad", "arc", "boolq"],
            "samples_per_dataset": 20,
            "primary_metrics": ["accuracy", "avg_abs_diff", "f1", "mcc"]
        }
    }

def main():
    """Main execution function."""
    print("=== SmoothQuant W8A8 Quantization Study Report ===")
    
    # Define model paths and metrics files
    models = {
        'base_noquant': 'Testing/metrics/Qwen3-0.6B-base.json',
        'base_smoothquant': 'Testing/metrics/Qwen3-0.6B-base_smoothquant_w8a8.json',
        'finetuned_noquant': 'Testing/metrics/Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant.json',
        'finetuned_smoothquant': 'Testing/metrics/Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_smoothquant_w8a8.json'
    }
    
    # Load all metrics
    print("Loading evaluation metrics...")
    metrics = {}
    for key, path in models.items():
        metrics[key] = load_metrics(path)
        if metrics[key]:
            print(f"✓ Loaded {key}: {path}")
        else:
            print(f"✗ Failed to load {key}: {path}")
    
    # Extract key metrics for analysis
    extracted_metrics = {}
    for key, raw_metrics in metrics.items():
        extracted_metrics[key] = extract_key_metrics(raw_metrics)
        print(f"✓ Extracted {len(extracted_metrics[key])} metrics from {key}")
    
    # Calculate quantization impact
    print("\nCalculating quantization impact...")
    
    base_impact = calculate_quantization_impact(
        extracted_metrics['base_noquant'], 
        extracted_metrics['base_smoothquant']
    )
    
    finetuned_impact = calculate_quantization_impact(
        extracted_metrics['finetuned_noquant'], 
        extracted_metrics['finetuned_smoothquant']
    )
    
    print(f"✓ Base model analysis: {len(base_impact)} metrics compared")
    print(f"✓ Fine-tuned model analysis: {len(finetuned_impact)} metrics compared")
    
    # Generate dataset analysis
    base_analysis = generate_dataset_analysis(base_impact)
    finetuned_analysis = generate_dataset_analysis(finetuned_impact)
    
    print(f"✓ Dataset analysis: {len(base_analysis)} datasets for base, {len(finetuned_analysis)} for fine-tuned")
    
    # Generate reports
    print("\nGenerating reports...")
    
    # Markdown report
    markdown_report = generate_markdown_report(base_impact, finetuned_impact, base_analysis, finetuned_analysis)
    
    # JSON summary
    json_summary = generate_json_summary(base_impact, finetuned_impact, base_analysis, finetuned_analysis)
    
    # Save reports
    os.makedirs('study_results', exist_ok=True)
    
    # Save markdown report
    markdown_path = 'study_results/smoothquant_study_report.md'
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    print(f"✓ Markdown report saved: {markdown_path}")
    
    # Save JSON summary
    json_path = 'study_results/smoothquant_study_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(f"✓ JSON summary saved: {json_path}")
    
    # Print quick summary
    overall_rate = json_summary['overall_performance']['overall_smoothquant_win_rate']
    base_rate = json_summary['overall_performance']['base_model_smoothquant_win_rate']
    ft_rate = json_summary['overall_performance']['finetuned_model_smoothquant_win_rate']
    
    print(f"\n=== STUDY SUMMARY ===")
    print(f"SmoothQuant W8A8 Overall Win Rate: {overall_rate}%")
    print(f"Base Model Win Rate: {base_rate}%")
    print(f"Fine-tuned Model Win Rate: {ft_rate}%")
    print(f"Report generated successfully!")

if __name__ == "__main__":
    main()