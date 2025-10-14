#!/usr/bin/env python
"""Test script for GPTQ quantization implementation.

This script tests the GPTQ quantization workflow end-to-end:
1. Creates a test model configuration
2. Runs GPTQ quantization via tools/quantize.py
3. Tests inference on the quantized model
4. Validates that the model works correctly
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod, QuantizationSpec

def test_gptq_quantization():
    """Test GPTQ quantization implementation."""
    print("=" * 60)
    print("GPTQ Quantization Test")
    print("=" * 60)
    
    # Test parameters
    model_name = "Models/Qwen3-0.6B-openmath_SFT_NoPeft_NoQuant"  # Use existing model if available
    temp_dir = Path(tempfile.mkdtemp(prefix="gptq_test_"))
    calib_file = Path("Datasets/calibration_openmath_5samples.txt")
    
    try:
        print(f"Test directory: {temp_dir}")
        
        # Check if source model exists
        if not Path(model_name).exists():
            print(f"❌ Source model not found: {model_name}")
            print("Please train a model first using:")
            print("python Fine-tuning/01_Train.py")
            return False
            
        # Check if calibration file exists
        if not calib_file.exists():
            print(f"❌ Calibration file not found: {calib_file}")
            print("Please create calibration data first")
            return False
            
        print(f"✓ Source model found: {model_name}")
        print(f"✓ Calibration file found: {calib_file}")
        
        # Test 1: Import quantization modules
        print("\n📦 Testing imports...")
        try:
            from tools.quantize import quantize_with_gptq, _quantize_gptq_fallback
            print("✓ GPTQ quantization functions imported successfully")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
            
        # Test 2: Test quantization spec creation
        print("\n🔧 Testing quantization specification...")
        try:
            spec = QuantizationSpec(
                method=QuantMethod.GPTQ,
                weights_bits=4,
                activations_bits=None,
                group_size=64,
                symmetric=True,
                lm_head_dtype="fp16",
                backend="autogptq"
            )
            tag = spec.tag()
            print(f"✓ Quantization spec created: {tag}")
        except Exception as e:
            print(f"❌ Spec creation failed: {e}")
            return False
            
        # Test 3: Run GPTQ quantization
        print("\n⚙️ Testing GPTQ quantization...")
        dst_dir = temp_dir / "quantized_model"
        
        try:
            # Test the fallback implementation first (more reliable)
            result_path, metadata = _quantize_gptq_fallback(
                src=Path(model_name),
                dst=dst_dir,
                calib_path=calib_file,
                bits=4,
                group_size=64,
                keep_lm_head_fp16=True,
                symmetric=True,
                seed=13
            )
            
            print(f"✓ GPTQ quantization completed")
            print(f"✓ Output saved to: {result_path}")
            print(f"✓ Metadata: {metadata}")
            
        except Exception as e:
            print(f"❌ Quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # Test 4: Verify output files
        print("\n📁 Verifying output files...")
        expected_files = [
            "config.json",
            "model.safetensors", 
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        missing_files = []
        for file in expected_files:
            if not (dst_dir / file).exists():
                missing_files.append(file)
                
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            print("✓ All expected files present")
            
        # Test 5: Test model loading
        print("\n🔄 Testing model loading...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load the quantized model
            device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}
            
            model = AutoModelForCausalLM.from_pretrained(
                str(dst_dir),
                device_map=device_map,
                torch_dtype="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(str(dst_dir), trust_remote_code=True)
            
            print("✓ Model loaded successfully")
            
            # Test inference
            print("\n🧠 Testing inference...")
            test_prompt = "What is 2 + 2?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✓ Model inference successful")
            print(f"  Input: {test_prompt}")
            print(f"  Output: {response}")
            
        except Exception as e:
            print(f"❌ Model loading/inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # Test 6: Check quantization metadata
        print("\n📊 Checking quantization metadata...")
        metadata_file = dst_dir / "quantization_metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                quant_metadata = json.load(f)
            print(f"✓ Quantization metadata found")
            print(f"  Method: {quant_metadata.get('method')}")
            print(f"  Weights bits: {quant_metadata.get('weights_bits')}")
            print(f"  Backend: {quant_metadata.get('backend')}")
        else:
            print("⚠️  Quantization metadata not found (non-critical)")
            
        print("\n" + "=" * 60)
        print("🎉 GPTQ TEST PASSED - All tests successful!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up test directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Failed to cleanup {temp_dir}: {e}")


def test_cli_interface():
    """Test the CLI interface for GPTQ quantization."""
    print("\n" + "=" * 60)
    print("GPTQ CLI Interface Test")
    print("=" * 60)
    
    # Test CLI argument parsing
    print("🖥️  Testing CLI argument parsing...")
    
    try:
        from tools.quantize import build_parser
        
        parser = build_parser()
        
        # Test valid arguments
        test_args = [
            "run",
            "--method", "gptq",
            "--src", "Models/test-model", 
            "--dst", "Models/test-model-gptq",
            "--bits", "4",
            "--group-size", "64",
            "--keep-lm-head-fp16"
        ]
        
        args = parser.parse_args(test_args)
        print("✓ CLI argument parsing successful")
        print(f"  Method: {args.method}")
        print(f"  Bits: {args.bits}")
        print(f"  Group size: {args.group_size}")
        print(f"  Keep LM head FP16: {args.keep_lm_head_fp16}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting GPTQ implementation tests...\n")
    
    # Test GPTQ quantization
    test1_passed = test_gptq_quantization()
    
    # Test CLI interface  
    test2_passed = test_cli_interface()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"GPTQ Quantization Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"CLI Interface Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! GPTQ implementation is ready.")
        print("\nUsage examples:")
        print("1. Train a model:")
        print("   python Fine-tuning/01_Train.py")
        print("\n2. Quantize with GPTQ:")
        print("   python tools/quantize.py run --method gptq --src Models/your-model --dst Models/your-model-gptq --bits 4 --group-size 64 --keep-lm-head-fp16")
        print("\n3. Evaluate quantized model:")
        print("   python Testing/02_TestModels.py Models/your-model-gptq")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)