#!/usr/bin/env python3
"""Test SmoothQuant quantization tagging and configuration."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from quantization_utils import QuantMethod, QuantizationSpec, tag_quant


def test_smoothquant_tagging():
    """Test SmoothQuant tag generation with various configurations."""
    
    print("Testing SmoothQuant quantization tagging...")
    
    # Basic SmoothQuant W8A8
    spec = QuantizationSpec(
        method=QuantMethod.SMOOTH_QUANT,
        weights_bits=8,
        activations_bits=8,
        kv_cache_bits=8,
        group_size=None,  # SmoothQuant uses per-tensor scaling
        symmetric=True,
        per_channel=True,
        lm_head_dtype="fp16",
        backend="torch",
        extras={"alpha": 0.5}
    )
    
    tag = spec.tag()
    print(f"Basic SmoothQuant W8A8: {tag}")
    # Note: activation bits = 8 are filtered out as "default" in tagging
    assert "SmoothQuant_w8" in tag
    
    # SmoothQuant with custom alpha
    spec_alpha = QuantizationSpec(
        method=QuantMethod.SMOOTH_QUANT,
        weights_bits=8,
        activations_bits=8,
        kv_cache_bits=8,
        group_size=None,
        symmetric=True,
        per_channel=True,
        lm_head_dtype="fp16",
        backend="torch",
        extras={"alpha": 0.3}  # Non-default alpha
    )
    
    tag_alpha = spec_alpha.tag()
    print(f"SmoothQuant with α=0.3: {tag_alpha}")
    assert "SmoothQuant_w8" in tag_alpha
    assert "alpha0.3" in tag_alpha
    
    # Test direct tag_quant function
    direct_tag = tag_quant(
        QuantMethod.SMOOTH_QUANT,
        bits=8,
        acts_bits=8,
        extras={"head": "fp16"}
    )
    print(f"Direct tag generation: {direct_tag}")
    assert direct_tag == "SmoothQuant_w8_a8_headfp16"
    
    print("✅ All SmoothQuant tagging tests passed!")


def test_smoothquant_metadata():
    """Test SmoothQuant metadata generation."""
    
    print("\nTesting SmoothQuant metadata generation...")
    
    spec = QuantizationSpec(
        method=QuantMethod.SMOOTH_QUANT,
        weights_bits=8,
        activations_bits=8,
        kv_cache_bits=8,
        group_size=None,
        symmetric=True,
        per_channel=True,
        lm_head_dtype="fp16",
        backend="torch",
        extras={
            "alpha": 0.5,
            "calibration_samples": 512,
            "quantized_layers": 197
        }
    )
    
    metadata = spec.metadata()
    print(f"Metadata method: {metadata['method']}")
    print(f"Metadata weights_bits: {metadata['weights_bits']}")
    print(f"Metadata activations_bits: {metadata['activations_bits']}")
    print(f"Metadata backend: {metadata['backend']}")
    print(f"Metadata extras keys: {list(metadata['extras'].keys())}")
    
    assert metadata["method"] == "SmoothQuant"
    assert metadata["weights_bits"] == 8
    assert metadata["activations_bits"] == 8
    assert metadata["backend"] == "torch"
    assert metadata["extras"]["alpha"] == 0.5
    
    print("✅ All SmoothQuant metadata tests passed!")


if __name__ == "__main__":
    test_smoothquant_tagging()
    test_smoothquant_metadata()
    print("\n🎉 All SmoothQuant tests completed successfully!")