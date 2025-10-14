#!/usr/bin/env python
"""
Debug Model Responses - Diagnostic Script

This script captures actual model responses for models showing suspicious metrics
like F1=0, avg_abs_diff=NaN, or unusual patterns. It runs a small sample (5 questions)
per problematic dataset and saves the raw responses for analysis.

Usage:
    python debug_model_responses.py <model_path>

The script replicates the exact inference process from Testing/02_TestModels.py
to ensure we capture the same responses that generated the problematic metrics.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

# Import the test module - need to import as module due to numeric prefix
import importlib.util
test_module_path = Path(__file__).parent / "Testing" / "02_TestModels.py"
spec = importlib.util.spec_from_file_location("test_models", test_module_path)
test_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_models)

# Extract functions we need
resolve_quant_context = test_models.resolve_quant_context
resolve_kv_dtype = test_models.resolve_kv_dtype  
load_model_with_quant = test_models.load_model_with_quant
chat = test_models.chat
extract_boxed = test_models.extract_boxed
datasets_info = test_models.datasets_info
compute_metrics = test_models.compute_metrics
normalize_arc_option = test_models.normalize_arc_option
normalize_bool = test_models.normalize_bool

def debug_model_responses(
    model_name: str,
    sample_size: int = 5,
    output_dir: str = "debug_responses"
):
    """
    Debug a model by capturing actual responses for problematic datasets.
    
    Args:
        model_name: Path to model directory
        sample_size: Number of samples per dataset to test
        output_dir: Directory to save debug results
    """
    
    print(f"🔍 Debugging model: {model_name}")
    print(f"📊 Sample size: {sample_size} per dataset")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Safe filename for output
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", model_name.replace("/", "__"))
    
    # Resolve quantization context (exactly like test script)
    try:
        quant_context = resolve_quant_context(model_name, "auto")
    except ValueError as exc:
        print(f"❌ Failed to resolve quantization context: {exc}")
        return
    
    # Resolve KV cache dtype
    resolved_kv_dtype = resolve_kv_dtype("auto", quant_context.spec)
    
    print(f"🔧 Quantization: {quant_context.method.value if quant_context.method else 'NoQuant'}")
    print(f"🔧 KV Cache dtype: {resolved_kv_dtype}")
    
    # Load model (exactly like test script)
    try:
        model, tokenizer = load_model_with_quant(model_name, quant_context, resolved_kv_dtype)
        print(f"✅ Model loaded successfully")
    except RuntimeError as exc:
        print(f"❌ Failed to load model: {exc}")
        return
    
    # Load test datasets
    datasets = list(Path("Datasets").glob("test-*.parquet"))
    print(f"📁 Found {len(datasets)} test datasets")
    
    # Debug results storage
    debug_results = {
        "model_name": model_name,
        "quantization": {
            "method": quant_context.method.value if quant_context.method else "NoQuant",
            "kv_cache_dtype": resolved_kv_dtype,
            "source": quant_context.source
        },
        "debug_timestamp": datetime.now().isoformat(),
        "sample_size": sample_size,
        "datasets": {}
    }
    
    # Process each dataset
    for dataset_path in sorted(datasets):
        dataset_name = dataset_path.name
        
        print(f"\n📋 Processing {dataset_name}...")
        
        # Load dataset
        df = pd.read_parquet(dataset_path)
        sample_df = df.head(sample_size)
        
        # Get dataset info
        info = datasets_info.get(dataset_name)
        if not info:
            print(f"⚠️  No info found for {dataset_name}, skipping")
            continue
        
        system_prompt = info["system_prompt"]
        use_context = info["context"]
        task_type = info["task"]
        
        print(f"   Task type: {task_type}")
        print(f"   Use context: {use_context}")
        
        # Debug responses for this dataset
        responses = []
        
        for idx, row in sample_df.iterrows():
            print(f"   🤖 Processing sample {idx + 1}/{sample_size}...")
            
            # Build prompt (exactly like test script)
            question = row.get("question", "")
            context = row.get("context", "") if use_context else ""
            answer = row.get("answer", row.get("answers", ""))
            
            # Create user prompt
            if use_context and context:
                user_prompt = f"Context: {context}\n\nQuestion: {question}"
            else:
                user_prompt = question
            
            # Generate response (exactly like test script)
            start_time = time.time()
            try:
                model_response = chat(
                    model=model,
                    tokenizer=tokenizer,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=1000
                )
                inference_time = time.time() - start_time
                success = True
                error_msg = None
            except Exception as e:
                model_response = ""
                inference_time = 0.0
                success = False
                error_msg = str(e)
                print(f"   ❌ Error generating response: {e}")
            
            # Extract boxed answer (exactly like test script)
            extracted_answer = extract_boxed(model_response)
            
            # Store debug info
            response_debug = {
                "sample_id": int(idx),
                "question": question,
                "context": context if use_context else None,
                "expected_answer": answer,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "model_response": model_response,
                "extracted_answer": extracted_answer,
                "inference_time_seconds": round(inference_time, 4),
                "success": success,
                "error": error_msg,
                "response_length": len(model_response),
                "has_boxed_format": extracted_answer is not None,
                "timestamp": datetime.now().isoformat()
            }
            
            responses.append(response_debug)
            
            # Log key info
            print(f"      Expected: {answer}")
            print(f"      Extracted: {extracted_answer}")
            print(f"      Success: {success}")
            print(f"      Time: {inference_time:.3f}s")
        
        # Store dataset results
        debug_results["datasets"][dataset_name] = {
            "task_type": task_type,
            "use_context": use_context,
            "total_samples": len(responses),
            "successful_responses": sum(1 for r in responses if r["success"]),
            "responses_with_boxed": sum(1 for r in responses if r["has_boxed_format"]),
            "responses": responses
        }
        
        # Calculate metrics for comparison (same as evaluation script)
        metrics_comparison = calculate_debug_metrics(responses, task_type)
        debug_results["datasets"][dataset_name]["calculated_metrics"] = metrics_comparison
        
        print(f"   ✅ Completed {dataset_name}: {len(responses)} samples processed")
        print(f"      📊 Calculated metrics: {metrics_comparison}")

def calculate_debug_metrics(responses: List[Dict], task_type: str) -> Dict[str, Any]:
    """
    Calculate the same metrics as the evaluation script for comparison.
    This helps verify we're reproducing the same issues.
    """
    
    # Extract predictions and gold answers
    golds = []
    preds = []
    
    for response in responses:
        if not response["success"]:
            # Failed generation - treat as no prediction
            preds.append(None)
        else:
            extracted = response["extracted_answer"]
            
            if task_type == "mcq4":
                # Normalize ARC answers
                pred = normalize_arc_option(extracted) if extracted else None
                preds.append(pred)
            elif task_type == "boolq":
                # Normalize boolean answers
                pred = normalize_bool(extracted) if extracted else None
                preds.append(pred)
            elif task_type == "squad_v2":
                # Squad uses extracted answer directly
                preds.append(extracted)
            elif task_type == "math_numeric":
                # Math uses extracted answer directly
                preds.append(extracted)
            else:
                # Generic: use extracted answer
                preds.append(extracted)
        
        # Extract gold answer
        expected = response["expected_answer"]
        if task_type == "mcq4":
            # ARC expects single letter
            golds.append(expected.upper() if isinstance(expected, str) else str(expected).upper())
        elif task_type == "boolq":
            # BoolQ expects boolean
            golds.append(expected)
        elif task_type == "squad_v2":
            # Squad expects answer(s) - handle both single string and list
            if isinstance(expected, list):
                golds.append(expected)
            else:
                golds.append([expected] if expected else [])
        elif task_type == "math_numeric":
            # Math expects numeric value
            try:
                gold_num = float(expected) if expected else 0.0
                golds.append(gold_num)
            except (ValueError, TypeError):
                golds.append(0.0)
        else:
            golds.append(expected)
    
    # Calculate metrics using the same function as evaluation script
    try:
        metrics_result = compute_metrics(task_type, golds, preds)
        
        # Add some debug info
        metrics_result["debug_info"] = {
            "total_responses": len(responses),
            "successful_responses": sum(1 for r in responses if r["success"]),
            "valid_extractions": sum(1 for p in preds if p is not None),
            "none_predictions": sum(1 for p in preds if p is None),
            "sample_predictions": preds[:3],  # First 3 for debugging
            "sample_golds": golds[:3] if len(golds) >= 3 else golds
        }
        
        return metrics_result
    
    except Exception as e:
        return {
            "error": f"Failed to calculate metrics: {str(e)}",
            "debug_info": {
                "total_responses": len(responses),
                "successful_responses": sum(1 for r in responses if r["success"]),
                "valid_extractions": sum(1 for p in preds if p is not None),
                "predictions_sample": preds[:3],
                "golds_sample": golds[:3] if len(golds) >= 3 else golds
            }
        }
    
    # Save debug results
    output_file = output_path / f"{safe_name}_debug_responses.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(debug_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Debug results saved to: {output_file}")
    
    # Print summary
    print(f"\n📊 SUMMARY for {model_name}")
    print("=" * 60)
    
    for dataset_name, dataset_results in debug_results["datasets"].items():
        total = dataset_results["total_samples"]
        success = dataset_results["successful_responses"] 
        boxed = dataset_results["responses_with_boxed"]
        metrics = dataset_results.get("calculated_metrics", {})
        
        print(f"{dataset_name}:")
        print(f"  • Total samples: {total}")
        print(f"  • Successful responses: {success}/{total} ({success/total*100:.1f}%)")
        print(f"  • Proper boxed format: {boxed}/{total} ({boxed/total*100:.1f}%)")
        
        # Print calculated metrics for comparison
        if "error" in metrics:
            print(f"  ❌ Metric calculation failed: {metrics['error']}")
        else:
            accuracy = metrics.get("accuracy", 0.0)
            print(f"  📊 Calculated accuracy: {accuracy:.1f}%")
            
            # Print specific metrics based on task type
            task_metrics = metrics.get("metrics", {})
            if "macro_f1" in task_metrics:
                f1 = task_metrics["macro_f1"]
                print(f"  📊 Calculated F1: {f1:.4f}")
            if "F1" in task_metrics:
                squad_f1 = task_metrics["F1"]
                print(f"  📊 Calculated Squad F1: {squad_f1:.2f}")
            if "avg_abs_diff" in task_metrics:
                avg_diff = task_metrics["avg_abs_diff"]
                print(f"  📊 Calculated avg_abs_diff: {avg_diff:.6f}")
        
        if success < total:
            print(f"  ⚠️  {total - success} failed responses!")
        if boxed < success:
            print(f"  ⚠️  {success - boxed} responses missing \\boxed{{}} format!")
        
        print()  # Empty line between datasets
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Debug model responses for suspicious metrics")
    parser.add_argument(
        "model_name",
        type=str,
        help="Path to the model directory to debug"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per dataset to test (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug_responses",
        help="Output directory for debug results (default: debug_responses)"
    )
    
    args = parser.parse_args()
    
    try:
        debug_model_responses(
            model_name=args.model_name,
            sample_size=args.samples,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
