#!/usr/bin/env python
"""Post-training quantisation entry-point.

This script acts as a thin orchestration layer that normalises CLI inputs,
computes bookkeeping metadata, and dispatches to method-specific handlers.
Each handler is currently a stub that documents the expected integration
points with the corresponding quantisation library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from utils.quantization_utils import QuantMethod, QuantizationSpec, tag_quant

# ---------------------------------------------------------------------------
# Mappings and helpers
# ---------------------------------------------------------------------------

METHOD_CHOICES = tuple(
    method for method in QuantMethod if method in {
        QuantMethod.GPTQ,
        QuantMethod.QUA_ROT,
        QuantMethod.ADA_ROUND,
        QuantMethod.BRECQ,
        QuantMethod.AWQ,
        QuantMethod.HQQ,
        QuantMethod.SMOOTH_QUANT,
    }
)

BACKEND_HINTS: Dict[QuantMethod, str] = {
    QuantMethod.GPTQ: "autogptq",
    QuantMethod.AWQ: "awq",
    QuantMethod.HQQ: "hqq",
    QuantMethod.SMOOTH_QUANT: "custom",
    QuantMethod.QUA_ROT: "custom",
    QuantMethod.ADA_ROUND: "custom",
    QuantMethod.BRECQ: "custom",
}


def _method_choices() -> Tuple[str, ...]:
    return tuple(method.value.lower() for method in METHOD_CHOICES)


def _normalise_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return value if value > 0 else None


def _load_calibration(path: Path) -> Tuple[int, str]:
    """Return the number of non-empty prompts and a stable hash."""
    raw = path.read_text(encoding="utf-8")
    prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    joined = "\n".join(prompts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return len(prompts), digest


# ---------------------------------------------------------------------------
# Handler stubs
# ---------------------------------------------------------------------------

HandlerResult = Tuple[Path, Dict[str, str]]
Handler = Callable[[Path, Path, QuantizationSpec, argparse.Namespace], HandlerResult]


def _require(package: str, message: str) -> None:
    """Try importing a package and raise a helpful error otherwise."""
    try:
        __import__(package)
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise NotImplementedError(message) from exc


def quantize_gptq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "auto_gptq",
        "GPTQ quantisation requires the `auto-gptq` package. Install it with `pip install auto-gptq`.",
    )
    raise NotImplementedError("GPTQ quantisation stub: integrate your AutoGPTQ workflow here.")


def quantize_quarot(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    raise NotImplementedError("QuaRot quantisation is not yet implemented. Please wire your QuaRot pipeline.")


def quantize_adaround(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    raise NotImplementedError("AdaRound quantisation is not yet implemented. Integrate the respective toolkit.")


def quantize_brecq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    raise NotImplementedError("BRECQ quantisation is not yet implemented. Integrate the respective toolkit.")


def quantize_awq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "awq",
        "AWQ quantisation requires the `autoawq`/`awq` package. Install it with `pip install autoawq`.",
    )
    raise NotImplementedError("AWQ quantisation stub: integrate your AWQ workflow here.")


def quantize_hqq(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    _require(
        "hqq",
        "HQQ quantisation requires the `hqq` package. Install it with `pip install hqq`.",
    )
    raise NotImplementedError("HQQ quantisation stub: integrate your HQQ workflow here.")


def quantize_smoothquant(src: Path, dst: Path, spec: QuantizationSpec, args: argparse.Namespace) -> HandlerResult:
    raise NotImplementedError(
        "SmoothQuant quantisation is not implemented. Hook up SmoothQuant/torchao pipeline here."
    )


HANDLERS: Dict[QuantMethod, Handler] = {
    QuantMethod.GPTQ: quantize_gptq,
    QuantMethod.QUA_ROT: quantize_quarot,
    QuantMethod.ADA_ROUND: quantize_adaround,
    QuantMethod.BRECQ: quantize_brecq,
    QuantMethod.AWQ: quantize_awq,
    QuantMethod.HQQ: quantize_hqq,
    QuantMethod.SMOOTH_QUANT: quantize_smoothquant,
}


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantise a fine-tuned model using PTQ techniques.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--src", required=True, type=Path, help="Path to the FP16/BF16 source model.")
    common.add_argument("--dst", required=True, type=Path, help="Destination directory for quantised output.")
    common.add_argument(
        "--calib",
        type=Path,
        default=Path("Datasets/calibration_prompts.txt"),
        help="Calibration prompts file (default: %(default)s).",
    )
    common.add_argument("--bits", type=int, default=4, help="Target weight bits (default: %(default)s).")
    common.add_argument("--group-size", type=int, default=64, help="Quantisation group size (default: %(default)s).")
    common.add_argument(
        "--keep-lm-head-fp16",
        action="store_true",
        help="Keep the LM head in fp16 instead of quantising it.",
    )
    common.add_argument("--acts-bits", type=int, default=8, help="Activation quantisation bits (default: %(default)s).")
    common.add_argument("--kv-bits", type=int, default=8, help="KV-cache quantisation bits (default: %(default)s).")
    common.add_argument("--seed", type=int, default=13, help="Calibration/data seed (default: %(default)s).")
    common.add_argument(
        "--method",
        required=True,
        choices=_method_choices(),
        help="Quantisation method to apply.",
    )

    run_parser = subparsers.add_parser("run", parents=[common], help="Execute the quantisation workflow.")
    run_parser.set_defaults(func=cmd_run)

    list_parser = subparsers.add_parser("list", help="List supported quantisation methods.")
    list_parser.set_defaults(func=cmd_list)

    return parser


def _spec_from_args(method: QuantMethod, args: argparse.Namespace, calib_stats: Tuple[int, str]) -> QuantizationSpec:
    calib_size, calib_hash = calib_stats
    backend = BACKEND_HINTS.get(method, "custom")
    lm_head_dtype = "fp16" if args.keep_lm_head_fp16 else f"w{args.bits}"
    spec = QuantizationSpec(
        method=method,
        weights_bits=args.bits,
        activations_bits=_normalise_optional_int(args.acts_bits),
        kv_cache_bits=_normalise_optional_int(args.kv_bits),
        group_size=_normalise_optional_int(args.group_size),
        symmetric=None,
        per_channel=None,
        lm_head_dtype=lm_head_dtype,
        backend=backend,
        calibration={
            "size": calib_size,
            "source": str(args.calib),
            "hash": calib_hash,
            "seed": args.seed,
        },
        extras={
            "keep_lm_head_fp16": args.keep_lm_head_fp16,
        },
    )
    spec.extras.setdefault("seed", args.seed)
    return spec


def cmd_list(_args: argparse.Namespace) -> int:
    print("Supported quantisation methods:")
    for method in METHOD_CHOICES:
        backend = BACKEND_HINTS.get(method, "custom")
        print(f"- {method.value.lower():<12} (backend hint: {backend})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    src: Path = args.src
    dst: Path = args.dst
    if not src.exists():
        print(f"[ERROR] Source model '{src}' does not exist.", file=sys.stderr)
        return 2

    dst.mkdir(parents=True, exist_ok=True)

    try:
        method = QuantMethod.from_any(args.method)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    handler = HANDLERS.get(method)
    if handler is None:
        print(f"[ERROR] Method '{method.value}' is not handled yet.", file=sys.stderr)
        return 2

    try:
        calib_stats = _load_calibration(args.calib)
    except FileNotFoundError:
        print(f"[ERROR] Calibration file '{args.calib}' not found.", file=sys.stderr)
        return 2

    spec = _spec_from_args(method, args, calib_stats)
    acts_for_tag = None if spec.activations_bits in (None, 8, 16) else spec.activations_bits
    kv_for_tag = None if spec.kv_cache_bits in (None, 8, 16) else spec.kv_cache_bits
    quant_tag = tag_quant(
        method,
        bits=spec.weights_bits,
        group_size=spec.group_size,
        acts_bits=acts_for_tag,
        kv_bits=kv_for_tag,
        extras={"head": spec.lm_head_dtype} if spec.lm_head_dtype else None,
    )

    metadata = spec.metadata()
    metadata.update(
        {
            "quant_tag": quant_tag,
            "source_model": str(src),
            "destination": str(dst),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
        }
    )

    try:
        handler(src, dst, spec, args)
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    metadata_path = dst / "quantization_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    print(f"[OK] Quantisation complete. Metadata written to {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        print("\n[WARN] Quantisation interrupted by user.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
