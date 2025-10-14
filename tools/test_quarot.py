#!/usr/bin/env python3
"""Test script to verify QuaRot quantization implementation."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod, tag_quant

def test_quarot_tagging():
    """Test QuaRot quantization tag generation."""
    print("Testing QuaRot tagging...")
    
    # Test basic W4A4 KV4 with default group size
    tag1 = tag_quant(
        QuantMethod.QUA_ROT,
        bits=4,
        group_size=64,
        acts_bits=4,
        kv_bits=4
    )
    expected1 = "QuaRot_w4_g64_a4_kv4"
    print(f"W4A4KV4 G64: {tag1} (expected: {expected1})")
    assert tag1 == expected1, f"Expected {expected1}, got {tag1}"
    
    # Test W4A8 KV4 (more practical configuration) - omit A8 as default
    tag2 = tag_quant(
        QuantMethod.QUA_ROT,
        bits=4,
        group_size=64,
        acts_bits=None,  # Omit A8 as it's default
        kv_bits=4
    )
    expected2 = "QuaRot_w4_g64_kv4"  # A8 omitted
    print(f"W4A8KV4 G64: {tag2} (expected: {expected2})")
    assert tag2 == expected2, f"Expected {expected2}, got {tag2}"
    
    # Test with head dtype
    tag3 = tag_quant(
        QuantMethod.QUA_ROT,
        bits=4,
        group_size=64,
        acts_bits=4,
        kv_bits=4,
        extras={"head": "fp16"}
    )
    expected3 = "QuaRot_w4_g64_a4_kv4_headfp16"
    print(f"W4A4KV4 G64 HeadFP16: {tag3} (expected: {expected3})")
    assert tag3 == expected3, f"Expected {expected3}, got {tag3}"
    
    # Test with non-default rotation method
    tag4 = tag_quant(
        QuantMethod.QUA_ROT,
        bits=4,
        group_size=64,
        acts_bits=4,
        kv_bits=4,
        extras={"rotation_method": "hadamard"}
    )
    expected4 = "QuaRot_w4_g64_a4_kv4_rotationmethodhadamard"  # Full key preserved
    print(f"W4A4KV4 G64 Hadamard: {tag4} (expected: {expected4})")
    assert tag4 == expected4, f"Expected {expected4}, got {tag4}"
    
    print("✓ All QuaRot tagging tests passed!")


def test_quarot_quantization_spec():
    """Test QuaRot QuantizationSpec creation."""
    print("\nTesting QuaRot QuantizationSpec...")
    
    # Import from training script
    training_path = Path(__file__).parent.parent / "Fine-tuning" / "01_Train.py"
    if training_path.exists():
        import importlib.util
        spec_module = importlib.util.spec_from_file_location("train_module", training_path)
        train_module = importlib.util.module_from_spec(spec_module)
        sys.modules["train_module"] = train_module
        spec_module.loader.exec_module(train_module)
        resolve_quantization_spec = train_module.resolve_quantization_spec
    else:
        raise ImportError("Could not find 01_Train.py")
    
    try:
        spec = resolve_quantization_spec(QuantMethod.QUA_ROT)
        
        print(f"Method: {spec.method}")
        print(f"Weights bits: {spec.weights_bits}")
        print(f"Activations bits: {spec.activations_bits}")
        print(f"KV cache bits: {spec.kv_cache_bits}")
        print(f"Group size: {spec.group_size}")
        print(f"Backend: {spec.backend}")
        print(f"Extras: {spec.extras}")
        
        # Verify expected values
        assert spec.method == QuantMethod.QUA_ROT
        assert spec.weights_bits == 4  # Should use PTQ_TARGET_WEIGHTS_BITS
        assert spec.activations_bits == 8  # Should use PTQ_TARGET_ACTS_BITS
        assert spec.kv_cache_bits == 8  # Should use PTQ_TARGET_KV_BITS
        assert spec.group_size == 128  # Should use PTQ_TARGET_GROUP_SIZE
        assert spec.backend == "custom"
        assert spec.extras["rotation_method"] == "pca"
        
        # Test tag generation
        tag = spec.tag()
        print(f"Generated tag: {tag}")
        
        # Should omit A8 and KV8 as they are defaults
        expected_tag = "QuaRot_w4_g128_headfp16"
        print(f"Expected tag: {expected_tag}")
        
        print("✓ QuaRot QuantizationSpec test passed!")
        
    except Exception as e:
        print(f"✗ QuaRot QuantizationSpec test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_quarot_tagging()
    test_quarot_quantization_spec()
    print("\n🎉 All QuaRot tests completed!")