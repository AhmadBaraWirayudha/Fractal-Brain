from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


@dataclass
class QuantizationSettings:
    mode: str
    quantization_config: Any | None
    torch_dtype: Any | None


def build_quantization_settings(mode: str = 'auto') -> QuantizationSettings:
    mode = (mode or 'none').lower()
    # Pure-Python build: quantization is not applied, but we preserve the API.
    if mode not in ('auto', 'none'):
        warnings.warn(
            f"quantization mode {mode!r} was requested, but this pure-Python build has "
            "no quantization backend -- bits/compute_dtype/double_quant/quant_type are "
            "all accepted for forward-compat but nothing reads them. See "
            "docs/UNIFIED_PIPELINE.md.",
            stacklevel=2,
        )
    if mode == 'auto':
        return QuantizationSettings('none', None, None)
    return QuantizationSettings(mode, None, None)
