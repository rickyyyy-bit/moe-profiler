"""Small, explicit accelerator capability database for roofline analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Precision = Literal["fp16", "fp8", "int4"]


class DeviceSpec(BaseModel):
    """Peak device capabilities in decimal SI units.

    FLOP fields contain dense operations/second, not a TFLOPs shorthand or the
    doubled 2:4 structured-sparsity figure. A zero value means that NVIDIA does
    not publish a native peak for that precision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    peak_bw_gbps: float = Field(gt=0.0)
    peak_flops_fp16: float = Field(gt=0.0)
    peak_flops_fp8: float = Field(ge=0.0)
    peak_flops_int4: float = Field(ge=0.0)
    tdp_w: float = Field(gt=0.0)
    approx_price_usd: float = Field(gt=0.0)

    def peak_flops(self, precision: Precision = "fp16") -> float:
        """Return peak operations/second for ``precision``."""
        value = {
            "fp16": self.peak_flops_fp16,
            "fp8": self.peak_flops_fp8,
            "int4": self.peak_flops_int4,
        }[precision]
        if value == 0:
            raise ValueError(f"{self.name} does not expose a {precision} peak")
        return value


_DEVICES = {
    "A100-80GB-SXM": DeviceSpec(
        name="A100-80GB-SXM",
        peak_bw_gbps=2039.0,
        peak_flops_fp16=312e12,
        peak_flops_fp8=0.0,
        peak_flops_int4=1248e12,
        tdp_w=400.0,
        approx_price_usd=15_000.0,
    ),
    "RTX-A6000": DeviceSpec(
        name="RTX-A6000",
        peak_bw_gbps=768.0,
        peak_flops_fp16=154.8e12,
        peak_flops_fp8=0.0,
        peak_flops_int4=619.3e12,
        tdp_w=300.0,
        approx_price_usd=4_650.0,
    ),
    "RTX-4090": DeviceSpec(
        name="RTX-4090",
        peak_bw_gbps=1008.0,
        peak_flops_fp16=165.2e12,
        peak_flops_fp8=330.3e12,
        peak_flops_int4=1321.2e12,
        tdp_w=450.0,
        approx_price_usd=1_599.0,
    ),
    "H100-SXM": DeviceSpec(
        name="H100-SXM",
        peak_bw_gbps=3350.0,
        peak_flops_fp16=989e12,
        peak_flops_fp8=1979e12,
        peak_flops_int4=0.0,
        tdp_w=700.0,
        approx_price_usd=30_000.0,
    ),
    "H20": DeviceSpec(
        name="H20",
        peak_bw_gbps=4000.0,
        peak_flops_fp16=148e12,
        peak_flops_fp8=296e12,
        peak_flops_int4=0.0,
        tdp_w=500.0,
        approx_price_usd=15_000.0,
    ),
}

_ALIASES = {name.lower(): name for name in _DEVICES}
_ALIASES.update(
    {
        "a100": "A100-80GB-SXM",
        "a100-80gb": "A100-80GB-SXM",
        "a6000": "RTX-A6000",
        "4090": "RTX-4090",
        "h100": "H100-SXM",
    }
)


def get_device(name: str) -> DeviceSpec:
    """Look up a device by canonical name or a short case-insensitive alias."""
    normalized = name.strip().lower()
    try:
        canonical = _ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(_DEVICES)
        message = f"unknown device {name!r}; expected one of: {choices}"
        raise ValueError(message) from exc
    return _DEVICES[canonical]


def list_devices() -> tuple[DeviceSpec, ...]:
    """Return every known device in stable insertion order."""
    return tuple(_DEVICES.values())
