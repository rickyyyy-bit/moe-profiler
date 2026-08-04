"""Offline tests for profiler validation and roofline analysis."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from moe_profiler.cost.devices import get_device, list_devices
from moe_profiler.metrics.roofline import (
    RooflinePoint,
    arithmetic_intensity,
    classify,
    ridge_point,
    roof_performance,
)
from moe_profiler.profiling.torch_prof import (
    ProfiledBandwidth,
    profiled_bandwidth_from_events,
    record_profiled_bandwidth,
    validate_profiled_bandwidth,
)
from moe_profiler.runner.record import RunResult
from moe_profiler.viz.roofline_plot import plot_roofline


def test_device_database_contains_required_accelerators() -> None:
    devices = {device.name: device for device in list_devices()}

    assert {
        "A100-80GB-SXM",
        "RTX-A6000",
        "RTX-4090",
        "H100-SXM",
        "H20",
    } <= set(devices)
    assert get_device("a100") is devices["A100-80GB-SXM"]
    assert get_device("h100").peak_flops("fp8") == 1979e12
    with pytest.raises(ValueError, match="does not expose"):
        get_device("A100-80GB-SXM").peak_flops("fp8")


def test_roofline_arithmetic_and_classification() -> None:
    device = get_device("H100")
    ridge = ridge_point(device)
    decode = RooflinePoint(
        label="decode",
        intensity=ridge / 10,
        achieved_flops=50e12,
    )
    prefill = RooflinePoint(
        label="prefill",
        intensity=ridge * 2,
        achieved_flops=500e12,
    )

    assert arithmetic_intensity(2000.0, 100.0) == 20.0
    assert ridge == pytest.approx(989e12 / (3350e9))
    assert classify(decode, device) == "memory"
    assert classify(prefill, device) == "compute"
    assert roof_performance(ridge / 2, device) == pytest.approx(
        ridge / 2 * device.peak_bw_gbps * 1e9
    )
    assert roof_performance(ridge * 2, device) == pytest.approx(device.peak_flops_fp16)


def test_profiled_bandwidth_prefers_hardware_dram_counters() -> None:
    events = [
        {
            "args": {
                "dram__bytes_read.sum": 600e6,
                "dram__bytes_write.sum": 400e6,
            }
        }
    ]

    result = profiled_bandwidth_from_events(
        events,
        elapsed_s=0.002,
        analytical_dram_bytes=123.0,
    )

    assert result.byte_source == "hardware_counter"
    assert result.dram_bytes == 1e9
    assert result.profiled_bw_gbps == pytest.approx(500.0)


def test_profiled_bandwidth_uses_explicit_analytical_fallback() -> None:
    result = profiled_bandwidth_from_events(
        [{"device_memory_usage": 999e9}],
        elapsed_s=0.004,
        analytical_dram_bytes=2e9,
    )

    assert result.byte_source == "analytical_estimate"
    assert result.profiled_bw_gbps == pytest.approx(500.0)
    with pytest.raises(RuntimeError, match="provide bytes_per_step"):
        profiled_bandwidth_from_events([], elapsed_s=0.01)


def test_profile_validation_and_run_result_update() -> None:
    validation = validate_profiled_bandwidth(
        analytic_s_mbu=0.5,
        profiled_bw_gbps=510.0,
        peak_bw_gbps=1000.0,
        tolerance=0.05,
    )
    measurement = ProfiledBandwidth(
        dram_bytes=1e9,
        elapsed_s=0.002,
        profiled_bw_gbps=500.0,
        byte_source="hardware_counter",
    )

    assert validation.profiled_utilization == pytest.approx(0.51)
    assert validation.relative_error == pytest.approx(0.02)
    assert validation.within_tolerance is True
    updated = record_profiled_bandwidth(_run_result(), measurement)
    assert updated.profiled_bw_gbps == 500.0


def test_roofline_plot_contains_prefill_and_decode_points() -> None:
    device = get_device("H100")
    points = [
        RooflinePoint(label="decode", intensity=10.0, achieved_flops=20e12),
        RooflinePoint(label="prefill", intensity=500.0, achieved_flops=500e12),
    ]
    with TemporaryDirectory(prefix="profiling-test-", dir="results") as directory:
        output = plot_roofline(
            device,
            points,
            output_path=Path(directory) / "roofline.png",
        )

        assert output.exists()
        assert output.stat().st_size > 0


def _run_result() -> RunResult:
    return RunResult(
        run_id="profiling-test",
        timestamp=datetime.now(UTC),
        model="offline/fake-moe",
        model_revision="a" * 40,
        backend="vllm",
        backend_version="test",
        device="H100-SXM",
        workload="synthetic",
        batch_size=1,
        input_len=8,
        output_len=4,
        ttft_s=0.01,
        tpot_s=0.002,
        e2e_s=0.02,
        throughput_tok_s=100.0,
        prefill_throughput_tok_s=800.0,
    )
