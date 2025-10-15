#!/usr/bin/env python3
"""
BRECQ Post-Training Quantization Comparison Report Generator

Analyzes the effects of BRECQ (Block-wise Reconstruction-based Quantization) on both
base models and fine-tuned models by comparing performance across test datasets.
BRECQ uses mixed precision: W4 for MLP layers and W6 for attention layers.
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
                # Most datasets use accuracy (higher is better)
                main_metric = 'accuracy'
                main_value = dataset_metrics['accuracy']
            elif 'exact_match' in dataset_metrics:
                # Some use exact_match (higher is better)
                main_metric = 'exact_match'
                main_value = dataset_metrics['exact_match']
            elif 'mcc' in dataset_metrics:
                # Matthews Correlation Coefficient (higher is better)
                main_metric = 'mcc'
                main_value = dataset_metrics['mcc']
            elif 'f1' in dataset_metrics:
                # F1 score (higher is better)
                main_metric = 'f1'
                main_value = dataset_metrics['f1']
            
            if main_metric and main_value is not None:
                key_metrics['datasets'][dataset_name] = {
                    'metric_type': main_metric,
                    'value': main_value,
                    'num_samples': dataset_metrics.get('num_samples', 0),
                    'avg_latency_ms': dataset_metrics.get('avg_latency_ms', 0)
                }
                
                # Add to average latency calculation
                if dataset_metrics.get('avg_latency_ms', 0) > 0:
                    key_metrics['avg_latency_ms'] += dataset_metrics['avg_latency_ms']
    
    # Calculate average latency across datasets
    if len(key_metrics['datasets']) > 0:
        key_metrics['avg_latency_ms'] /= len(key_metrics['datasets'])
    
    return key_metrics


def calculate_quantization_impact(base_metrics: Dict[str, Any], quant_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the impact of quantization on performance metrics."""
    impact = {
        'datasets': {},
        'summary': {
            'total_datasets': 0,
            'improvements': 0,
            'degradations': 0,
            'avg_improvement': 0.0,
            'speed_improvement': 0.0,
            'vram_reduction': 0.0,
            'param_count_match': False
        }
    }
    
    # Compare dataset-specific metrics
    total_improvement = 0.0
    dataset_count = 0
    
    base_datasets = base_metrics.get('datasets', {})
    quant_datasets = quant_metrics.get('datasets', {})
    
    for dataset_name in base_datasets:
        if dataset_name in quant_datasets:
            base_value = base_datasets[dataset_name]['value']
            quant_value = quant_datasets[dataset_name]['value']
            metric_type = base_datasets[dataset_name]['metric_type']
            
            # Calculate improvement (positive means quantized is better)
            if metric_type == 'avg_abs_diff':  # Lower is better
                improvement = base_value - quant_value
            else:  # Higher is better for accuracy, mcc, f1, exact_match
                improvement = quant_value - base_value
            
            improvement_pct = (improvement / abs(base_value)) * 100 if base_value != 0 else 0
            
            impact['datasets'][dataset_name] = {
                'base_value': base_value,
                'quant_value': quant_value,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'metric_type': metric_type,
                'better': improvement > 0
            }
            
            total_improvement += improvement_pct
            dataset_count += 1
            
            if improvement > 0:
                impact['summary']['improvements'] += 1
            else:
                impact['summary']['degradations'] += 1
    
    impact['summary']['total_datasets'] = dataset_count
    impact['summary']['avg_improvement'] = total_improvement / dataset_count if dataset_count > 0 else 0.0
    
    # Calculate speed improvement (latency reduction)
    base_latency = base_metrics.get('avg_latency_ms', 0)
    quant_latency = quant_metrics.get('avg_latency_ms', 0)
    if base_latency > 0 and quant_latency > 0:
        speed_improvement = (base_latency - quant_latency) / base_latency
        impact['summary']['speed_improvement'] = speed_improvement
    
    # Calculate VRAM reduction
    base_vram = base_metrics.get('vram_usage_gb', 0)
    quant_vram = quant_metrics.get('vram_usage_gb', 0)
    if base_vram > 0:
        vram_reduction = (base_vram - quant_vram) / base_vram
        impact['summary']['vram_reduction'] = vram_reduction
    
    # Check if parameter counts match (they should for post-training quantization)
    base_params = base_metrics.get('total_params', 0)
    quant_params = quant_metrics.get('total_params', 0)
    impact['summary']['param_count_match'] = abs(base_params - quant_params) < 1000  # Allow small differences
    
    return impact


def generate_markdown_report(base_no_quant: Dict, base_brecq: Dict, 
                           nopeft_no_quant: Dict, nopeft_brecq: Dict,
                           base_impact: Dict, nopeft_impact: Dict) -> str:
    """Generate a comprehensive markdown report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate win rates
    base_wins = base_impact['summary']['improvements']
    base_total = base_impact['summary']['total_datasets']
    base_win_rate = (base_wins / base_total * 100) if base_total > 0 else 0
    
    nopeft_wins = nopeft_impact['summary']['improvements']
    nopeft_total = nopeft_impact['summary']['total_datasets']
    nopeft_win_rate = (nopeft_wins / nopeft_total * 100) if nopeft_total > 0 else 0
    
    report = f"""# BRECQ (Block-wise Reconstruction) Post-Training Quantization Study

**Generated**: {timestamp}

## Executive Summary

This report analyzes the effectiveness of BRECQ (Block-wise Reconstruction-based Quantization) on the Qwen3-0.6B model across both base and fine-tuned variants. BRECQ uses mixed precision quantization: W4 for MLP layers and W6 for attention layers with block-wise reconstruction to minimize quantization error.

### Key Findings

- **Overall BRECQ Win Rate**: Base model {base_win_rate:.1f}%, Fine-tuned model {nopeft_win_rate:.1f}%
- **Average Performance Impact**: Base {base_impact['summary']['avg_improvement']:+.3f}%, Fine-tuned {nopeft_impact['summary']['avg_improvement']:+.3f}%
- **Speed Improvement**: Base {base_impact['summary']['speed_improvement']*100:+.1f}%, Fine-tuned {nopeft_impact['summary']['speed_improvement']*100:+.1f}%
- **VRAM Reduction**: Base {base_impact['summary']['vram_reduction']*100:+.1f}%, Fine-tuned {nopeft_impact['summary']['vram_reduction']*100:+.1f}%

## Model Configurations

### Base Models
- **No Quantization**: `{base_no_quant['model_name']}`
- **BRECQ W4/W6**: `{base_brecq['model_name']}`

### Fine-tuned Models  
- **No Quantization**: `{nopeft_no_quant['model_name']}`
- **BRECQ W4/W6**: `{nopeft_brecq['model_name']}`

## Performance Analysis

### Base Model Results

| Dataset | No Quant | BRECQ W4/W6 | Change | Metric |
|---------|----------|-------------|---------|---------|
"""
    
    # Add base model dataset comparison
    for dataset_name, impact_data in base_impact['datasets'].items():
        base_val = impact_data['base_value']
        quant_val = impact_data['quant_value']
        change = impact_data['improvement_pct']
        metric_type = impact_data['metric_type']
        
        if metric_type == 'avg_abs_diff':
            change_str = f"{change:+.2f}%" if change != 0 else "0.00%"
            change_symbol = "✅" if change > 0 else "❌" if change < -1 else "➖"
        else:
            change_str = f"{change:+.2f}%" if change != 0 else "0.00%"
            change_symbol = "✅" if change > 1 else "❌" if change < -1 else "➖"
        
        report += f"| {dataset_name} | {base_val:.3f} | {quant_val:.3f} | {change_symbol} {change_str} | {metric_type} |\n"
    
    report += f"""
**Base Model Summary**: {base_wins}/{base_total} datasets improved ({base_win_rate:.1f}% win rate)

### Fine-tuned Model Results

| Dataset | No Quant | BRECQ W4/W6 | Change | Metric |
|---------|----------|-------------|---------|---------|
"""
    
    # Add fine-tuned model dataset comparison
    for dataset_name, impact_data in nopeft_impact['datasets'].items():
        base_val = impact_data['base_value']
        quant_val = impact_data['quant_value']
        change = impact_data['improvement_pct']
        metric_type = impact_data['metric_type']
        
        if metric_type == 'avg_abs_diff':
            change_str = f"{change:+.2f}%" if change != 0 else "0.00%"
            change_symbol = "✅" if change > 0 else "❌" if change < -1 else "➖"
        else:
            change_str = f"{change:+.2f}%" if change != 0 else "0.00%"
            change_symbol = "✅" if change > 1 else "❌" if change < -1 else "➖"
        
        report += f"| {dataset_name} | {base_val:.3f} | {quant_val:.3f} | {change_symbol} {change_str} | {metric_type} |\n"
    
    report += f"""
**Fine-tuned Model Summary**: {nopeft_wins}/{nopeft_total} datasets improved ({nopeft_win_rate:.1f}% win rate)

## Resource Efficiency

### VRAM Usage
- **Base Model**: {base_no_quant['vram_usage_gb']:.2f} GB → {base_brecq['vram_usage_gb']:.2f} GB ({base_impact['summary']['vram_reduction']*100:+.1f}%)
- **Fine-tuned Model**: {nopeft_no_quant['vram_usage_gb']:.2f} GB → {nopeft_brecq['vram_usage_gb']:.2f} GB ({nopeft_impact['summary']['vram_reduction']*100:+.1f}%)

### Inference Speed
- **Base Model Latency**: {base_no_quant['avg_latency_ms']:.2f} ms → {base_brecq['avg_latency_ms']:.2f} ms ({base_impact['summary']['speed_improvement']*100:+.1f}%)
- **Fine-tuned Model Latency**: {nopeft_no_quant['avg_latency_ms']:.2f} ms → {nopeft_brecq['avg_latency_ms']:.2f} ms ({nopeft_impact['summary']['speed_improvement']*100:+.1f}%)

### Model Size
- **Parameters**: {base_no_quant['total_params']:,} → {base_brecq['total_params']:,} ({"✅ Preserved" if base_impact['summary']['param_count_match'] else "❌ Changed"})

## Technical Details

### BRECQ Configuration
- **MLP Layer Quantization**: W4 (4-bit weights)
- **Attention Layer Quantization**: W6 (6-bit weights) 
- **Mixed Precision**: Enabled
- **Group Size**: 64
- **Reconstruction Method**: Block-wise error minimization

### Quantization Method
BRECQ performs block-wise reconstruction with mixed precision support:
1. **Block-wise Processing**: Quantizes weights in blocks with local reconstruction
2. **Mixed Precision**: Uses W6 for attention layers (Q,K,V,O projections) and W4 for MLP layers
3. **Error Minimization**: Iterative refinement to reduce quantization error per block
4. **Calibration**: Uses sample data to optimize quantization parameters

## Conclusions

### Overall Effectiveness
"""
    
    # Add conclusions based on results
    if base_impact['summary']['avg_improvement'] > 1 and nopeft_impact['summary']['avg_improvement'] > 1:
        report += "✅ **BRECQ shows positive results** on both base and fine-tuned models with consistent accuracy improvements.\n\n"
    elif base_impact['summary']['avg_improvement'] > 1 or nopeft_impact['summary']['avg_improvement'] > 1:
        report += "⚠️ **BRECQ shows mixed results** with improvements on one model type but not the other.\n\n"
    else:
        report += "❌ **BRECQ shows accuracy degradation** on both model types, suggesting the quantization is too aggressive.\n\n"
    
    # Resource efficiency conclusion
    avg_vram_reduction = (base_impact['summary']['vram_reduction'] + nopeft_impact['summary']['vram_reduction']) / 2
    avg_speed_improvement = (base_impact['summary']['speed_improvement'] + nopeft_impact['summary']['speed_improvement']) / 2
    
    if avg_vram_reduction > 0.1:
        report += f"💾 **Significant VRAM savings** of {avg_vram_reduction*100:.1f}% average reduction.\n\n"
    
    if avg_speed_improvement > 0.1:
        report += f"⚡ **Notable speed improvements** with {avg_speed_improvement*100:.1f}% average latency reduction.\n\n"
    elif avg_speed_improvement < -0.1:
        report += f"⚠️ **Speed degradation** with {abs(avg_speed_improvement)*100:.1f}% average latency increase.\n\n"
    
    # Model-specific recommendations
    report += "### Recommendations\n\n"
    
    if base_win_rate > 60:
        report += "✅ BRECQ is **recommended for base models** with strong performance retention.\n\n"
    elif base_win_rate > 40:
        report += "⚠️ BRECQ shows **mixed results for base models** - evaluate per use case.\n\n"
    else:
        report += "❌ BRECQ is **not recommended for base models** due to accuracy degradation.\n\n"
    
    if nopeft_win_rate > 60:
        report += "✅ BRECQ is **recommended for fine-tuned models** with strong performance retention.\n\n"
    elif nopeft_win_rate > 40:
        report += "⚠️ BRECQ shows **mixed results for fine-tuned models** - evaluate per use case.\n\n"
    else:
        report += "❌ BRECQ is **not recommended for fine-tuned models** due to accuracy degradation.\n\n"
    
    # Mixed precision note
    report += "### Mixed Precision Impact\n\n"
    report += "The mixed precision approach (W6 attention + W4 MLP) balances:\n"
    report += "- **Attention Preservation**: Higher precision for critical attention computations\n"
    report += "- **Efficiency**: Aggressive quantization for compute-heavy MLP layers\n"
    report += "- **Quality**: Block-wise reconstruction minimizes quantization artifacts\n\n"
    
    return report


def main():
    """Main function to generate the BRECQ comparison report."""
    print("🔍 BRECQ Post-Training Quantization Analysis")
    print("=" * 50)
    
    # Define paths
    workspace_root = Path(__file__).parent.parent
    metrics_dir = workspace_root / "Testing" / "metrics"
    study_results_dir = workspace_root / "study_results"
    
    # Ensure study results directory exists
    study_results_dir.mkdir(exist_ok=True)
    
    # Define target model patterns for BRECQ analysis
    target_models = {
        'base_no_quant': 'Models__Qwen3-0.6B-base.json',
        'base_brecq': 'Models__Qwen3-0.6B-base_brecq_w4g64_mix_attn6.json',
        'nopeft_no_quant': 'Models__Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant.json', 
        'nopeft_brecq': 'Models__Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant_brecq_w4g64_mix_attn6.json'
    }
    
    print("📁 Looking for model metrics files:")
    for model_type, filename in target_models.items():
        filepath = metrics_dir / filename
        exists = "✅" if filepath.exists() else "❌"
        print(f"  {exists} {model_type}: {filename}")
    
    # Check if all required files exist
    missing_files = []
    for model_type, filename in target_models.items():
        if not (metrics_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required metrics files:")
        for filename in missing_files:
            print(f"   - {filename}")
        print("\n💡 Generate missing models by:")
        print("   1. Creating BRECQ quantized models with tools/quantize.py")
        print("   2. Running evaluation with Testing/03_EvaluationOrchestrator.py")
        return
    
    print("\n📊 Loading and analyzing metrics...")
    
    # Load all metrics
    key_metrics = {}
    for model_type, filename in target_models.items():
        metrics = load_metrics_file(metrics_dir / filename)
        key_metrics[model_type] = extract_key_metrics(metrics)
        print(f"✅ Loaded {model_type}: {len(key_metrics[model_type]['datasets'])} datasets")
    
    # Calculate quantization impacts
    print("\n🔬 Calculating quantization impacts...")
    
    base_impact = calculate_quantization_impact(
        key_metrics['base_no_quant'], 
        key_metrics['base_brecq']
    )
    
    nopeft_impact = calculate_quantization_impact(
        key_metrics['nopeft_no_quant'], 
        key_metrics['nopeft_brecq']
    )
    
    # Generate markdown report
    print("📝 Generating markdown report...")
    markdown_report = generate_markdown_report(
        key_metrics['base_no_quant'],
        key_metrics['base_brecq'],
        key_metrics['nopeft_no_quant'],
        key_metrics['nopeft_brecq'],
        base_impact,
        nopeft_impact
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary JSON
    summary_data = {
        'study_type': 'brecq_post_training_quantization',
        'generated_at': datetime.now().isoformat(),
        'models_compared': list(target_models.keys()),
        'key_metrics': key_metrics,
        'base_model_impact': base_impact,
        'nopeft_model_impact': nopeft_impact,
        'conclusions': {
            'base_model_winner': 'brecq' if base_impact['summary']['avg_improvement'] > 0 else 'no_quant',
            'nopeft_model_winner': 'brecq' if nopeft_impact['summary']['avg_improvement'] > 0 else 'no_quant',
            'overall_brecq_effectiveness': 'positive' if (base_impact['summary']['avg_improvement'] + nopeft_impact['summary']['avg_improvement']) > 0 else 'negative',
            'mixed_precision_benefit': 'enabled',
            'quantization_levels': {'mlp': 'W4', 'attention': 'W6'}
        }
    }
    
    summary_path = study_results_dir / f"brecq_study_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Save markdown report
    report_path = study_results_dir / f"brecq_study_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"\n✅ Reports generated:")
    print(f"📊 Summary JSON: {summary_path}")
    print(f"📄 Markdown report: {report_path}")
    
    # Quick summary
    base_winner = "BRECQ" if base_impact['summary']['avg_improvement'] > 0 else "No Quantization"
    nopeft_winner = "BRECQ" if nopeft_impact['summary']['avg_improvement'] > 0 else "No Quantization"
    
    print(f"\n🎯 Study Results:")
    print(f"📈 Base Model: {base_winner} performs better by {abs(base_impact['summary']['avg_improvement']):.3f} points on average")
    print(f"📈 Fine-tuned Model: {nopeft_winner} performs better by {abs(nopeft_impact['summary']['avg_improvement']):.3f} points on average")
    
    # Mixed precision summary
    print(f"🔧 Mixed Precision: W6 attention + W4 MLP layers")
    print(f"💾 VRAM Reduction: {((base_impact['summary']['vram_reduction'] + nopeft_impact['summary']['vram_reduction']) / 2) * 100:.1f}% average")


if __name__ == "__main__":
    main()