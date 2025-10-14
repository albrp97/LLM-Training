"""Shared quantization helpers for training and evaluation scripts.

This module centralises the list of supported quantization strategies,
provides utilities to make their identifiers consistent, and exposes a
simplified tagging + metadata surface that both the training and
inference flows can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union


class QuantMethod(str, Enum):
    """Unified registry for supported quantization approaches."""

    NO_QUANT = "NoQuant"
    QLORA = "QLORA"
    GPTQ = "GPTQ"
    QUA_ROT = "QuaRot"
    ADA_ROUND = "AdaRound"
    BRECQ = "BRECQ"
    AWQ = "AWQ"
    HQQ = "HQQ"
    SMOOTH_QUANT = "SmoothQuant"

    @classmethod
    def from_any(cls, value: Union[str, "QuantMethod", None]) -> "QuantMethod":
        """Normalise user supplied values to a QuantMethod."""
        if isinstance(value, QuantMethod):
            return value
        if value is None:
            raise ValueError("Quantization method cannot be None")

        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Quantization method cannot be empty")

        # Accept case-insensitive matches and a few pragmatic aliases.
        canonical = normalized.replace("-", "").replace("_", "").lower()
        for method in cls:
            compare = method.value.replace("-", "").replace("_", "").lower()
            if canonical == compare:
                return method

        raise ValueError(f"Unsupported quantization method: {value}")

    @property
    def is_ptq(self) -> bool:
        """Whether the method is a post-training quantization (PTQ) flavour."""
        return self in PTQ_METHODS

    @property
    def tag_prefix(self) -> str:
        """Tag-friendly prefix that remains human readable."""
        return self.value


# Explicit tuple to preserve order for presentation/validation.
ALL_METHODS: Tuple[QuantMethod, ...] = tuple(QuantMethod)
PTQ_METHODS: Tuple[QuantMethod, ...] = (
    QuantMethod.GPTQ,
    QuantMethod.QUA_ROT,
    QuantMethod.ADA_ROUND,
    QuantMethod.BRECQ,
    QuantMethod.AWQ,
    QuantMethod.HQQ,
    QuantMethod.SMOOTH_QUANT,
)
TRAIN_TIME_QUANT_METHODS: Tuple[QuantMethod, ...] = (QuantMethod.QLORA,)


def _sanitize_segment(value: Union[str, int, float]) -> str:
    """Turn arbitrary values into safe tag segments."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    return "".join(ch for ch in str(value) if ch.isalnum() or ch in {"-", "."})


def _extras_to_segments(extras: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not extras:
        return ()
    segments = []
    for key in sorted(extras):
        val = extras[key]
        safe_key = _sanitize_segment(key)
        if isinstance(val, bool):
            if not val:
                continue
            segments.append(safe_key)
        elif val is None:
            continue
        else:
            segments.append(f"{safe_key}{_sanitize_segment(val)}")
    return segments


def tag_quant(
    method: Union[str, QuantMethod],
    *,
    bits: Optional[int] = None,
    group_size: Optional[int] = None,
    acts_bits: Optional[int] = None,
    kv_bits: Optional[int] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> str:
    """Generate a concise tag that encodes the chosen quantization settings."""

    quant_method = QuantMethod.from_any(method)
    parts = [quant_method.tag_prefix]

    def add(part_label: str, value: Optional[int]) -> None:
        if value is None:
            return
        parts.append(f"{part_label}{value}")

    add("w", bits)
    add("g", group_size)
    add("a", acts_bits)
    add("kv", kv_bits)
    parts.extend(_extras_to_segments(extras))

    return "_".join(parts)


@dataclass
class QuantizationSpec:
    """Lightweight container describing quant-related run-time metadata."""

    method: QuantMethod
    weights_bits: Optional[int] = None
    activations_bits: Optional[int] = None
    kv_cache_bits: Optional[int] = None
    group_size: Optional[int] = None
    symmetric: Optional[bool] = None
    per_channel: Optional[bool] = None
    lm_head_dtype: Optional[str] = None
    backend: Optional[str] = None
    calibration: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    tag_extras: Dict[str, Any] = field(default_factory=dict)

    def tag(self) -> str:
        """Generate a stable identifier for folder naming."""
        if self.method is QuantMethod.NO_QUANT:
            bits_for_tag = None
        else:
            bits_for_tag = self.weights_bits

        if bits_for_tag in (None, 16) and self.method is QuantMethod.NO_QUANT:
            bits_for_tag = None

        acts_for_tag = self.activations_bits
        if acts_for_tag in (None, 16, 8):
            acts_for_tag = None

        kv_for_tag = self.kv_cache_bits
        if kv_for_tag in (None, 16, 8):
            kv_for_tag = None

        extras = dict(self.tag_extras)
        if self.method is not QuantMethod.NO_QUANT and self.lm_head_dtype and "head" not in extras:
            extras["head"] = self.lm_head_dtype

        # Add method-specific tags
        if self.method == QuantMethod.BRECQ and self.extras:
            if self.extras.get("mixed_precision"):
                extras["mix"] = True
            attention_bits = self.extras.get("attention_bits")
            if attention_bits and attention_bits != bits_for_tag:
                extras[f"attn{attention_bits}"] = True
        elif self.method == QuantMethod.SMOOTH_QUANT and self.extras:
            # SmoothQuant uses W8A8 format by default
            alpha = self.extras.get("alpha")
            if alpha and alpha != 0.5:  # Only include non-default alpha
                extras[f"alpha{alpha}"] = True

        return tag_quant(
            self.method,
            bits=bits_for_tag,
            group_size=self.group_size,
            acts_bits=acts_for_tag,
            kv_bits=kv_for_tag,
            extras=extras or None,
        )

    def metadata(self) -> Dict[str, Any]:
        """Return a dict ready to be embedded in metadata JSON."""
        return {
            "method": self.method.value,
            "weights_bits": self.weights_bits,
            "activations_bits": self.activations_bits,
            "kv_cache_bits": self.kv_cache_bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "lm_head_dtype": self.lm_head_dtype,
            "backend": self.backend,
            "calibration": self.calibration,
            "extras": dict(self.extras) if self.extras else {},
        }


def load_quant_metadata(metadata: Mapping[str, Any]) -> Optional[QuantizationSpec]:
    """Best-effort reconstruction of QuantizationSpec from stored metadata."""
    if not metadata:
        return None
    try:
        method = QuantMethod.from_any(metadata.get("method"))
    except ValueError:
        return None

    return QuantizationSpec(
        method=method,
        weights_bits=metadata.get("weights_bits"),
        activations_bits=metadata.get("activations_bits"),
        kv_cache_bits=metadata.get("kv_cache_bits"),
        group_size=metadata.get("group_size"),
        symmetric=metadata.get("symmetric"),
        per_channel=metadata.get("per_channel"),
        lm_head_dtype=metadata.get("lm_head_dtype"),
        backend=metadata.get("backend"),
        calibration=metadata.get("calibration"),
        extras=dict(metadata.get("extras", {})),
    )


def detect_method_from_path(path: Union[str, Path]) -> Optional[QuantMethod]:
    """Infer the quantization method from a model path name."""
    name = Path(path).name.lower()
    for method in QuantMethod:
        if method.value.lower() in name:
            return method
    return None
