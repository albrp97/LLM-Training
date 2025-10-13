#!/usr/bin/env python
"""Test script for AdaRound quantization implementation."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod, QuantizationSpec
from tools.quantize import quantize_with_adaround


def test_adaround_tagging():
    """Test that AdaRound generates proper tags."""
    print("Testing AdaRound tag generation...")
    
    # Test AdaRound with 4-bit, group size 128, skip LM head
    spec = QuantizationSpec(
        method=QuantMethod.ADA_ROUND,
        weights_bits=4,
        activations_bits=None,
        kv_cache_bits=None,
        group_size=128,
        lm_head_dtype="fp16",
        backend="custom"
    )
    
    tag = spec.tag()
    print(f"AdaRound tag: {tag}")
    
    # Expected: AdaRound_w4_g128_headfp16
    expected_parts = ["AdaRound", "w4", "g128", "headfp16"]
    tag_parts = tag.split("_")
    
    for expected_part in expected_parts:
        if expected_part not in tag_parts:
            print(f"ERROR: Expected '{expected_part}' in tag '{tag}'")
            return False
            
    print("✓ Tag generation test passed")
    return True


def test_quantmethod_parsing():
    """Test that AdaRound can be parsed from string."""
    print("\nTesting QuantMethod.from_any...")
    
    test_inputs = ["AdaRound", "adaround", "ADA_ROUND", "ada-round"]
    
    for input_str in test_inputs:
        try:
            method = QuantMethod.from_any(input_str)
            if method != QuantMethod.ADA_ROUND:
                print(f"ERROR: '{input_str}' parsed as {method}, expected {QuantMethod.ADA_ROUND}")
                return False
            print(f"✓ '{input_str}' -> {method}")
        except ValueError as e:
            print(f"ERROR: Failed to parse '{input_str}': {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("Testing AdaRound implementation...\n")
    
    success = True
    success &= test_adaround_tagging()
    success &= test_quantmethod_parsing()
    
    if success:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())