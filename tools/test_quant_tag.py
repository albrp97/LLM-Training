import subprocess
import sys
from pathlib import Path

from quantization_utils import QuantMethod, QuantizationSpec, tag_quant


def test_tag_quant_matrix():
    cases = [
        (("NoQuant",), {}, "NoQuant"),
        (("GPTQ",), {"bits": 4, "group_size": 64}, "GPTQ_w4_g64"),
        (
            ("AWQ",),
            {"bits": 4, "group_size": 128, "extras": {"head": "fp16"}},
            "AWQ_w4_g128_headfp16",
        ),
        (("HQQ",), {"bits": 8, "extras": {"head": "fp16"}}, "HQQ_w8_headfp16"),
    ]
    for (method_args, kwargs, expected) in cases:
        assert tag_quant(*method_args, **kwargs) == expected


def test_quant_spec_tag_drops_default_activations():
    spec = QuantizationSpec(
        method=QuantMethod.GPTQ,
        weights_bits=4,
        activations_bits=8,
        kv_cache_bits=8,
        group_size=64,
        lm_head_dtype="fp16",
    )
    assert spec.tag() == "GPTQ_w4_g64_headfp16"


def test_quantize_cli_missing_dependency(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    src = tmp_path / "src"
    src.mkdir()
    calib = tmp_path / "calib.txt"
    calib.write_text("prompt one\nprompt two\n", encoding="utf-8")
    dst = tmp_path / "dst"

    cmd = [
        sys.executable,
        "tools/quantize.py",
        "run",
        "--method",
        "gptq",
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--calib",
        str(calib),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    assert result.returncode == 2
    assert "auto-gptq" in result.stderr.lower()
