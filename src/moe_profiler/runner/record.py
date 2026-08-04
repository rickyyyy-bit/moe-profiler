"""Stable result schema and append-only CSV persistence."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RunResult(BaseModel):
    """One measured sweep point and all fields used by later stages."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    run_id: str = Field(min_length=1)
    timestamp: datetime
    model: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    backend_version: str = Field(min_length=1)
    quant: str | None = None
    device: str = Field(min_length=1)
    workload: str = Field(min_length=1)
    batch_size: int = Field(gt=0)
    input_len: int = Field(gt=0)
    output_len: int = Field(gt=1)
    ttft_s: float = Field(ge=0.0)
    tpot_s: float = Field(ge=0.0)
    e2e_s: float = Field(ge=0.0)
    throughput_tok_s: float = Field(ge=0.0)
    prefill_throughput_tok_s: float = Field(ge=0.0)
    activated_param_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    vanilla_mbu: float | None = Field(default=None, ge=0.0)
    s_mbu: float | None = Field(default=None, ge=0.0)
    vanilla_mfu: float | None = Field(default=None, ge=0.0)
    s_mfu: float | None = Field(default=None, ge=0.0)
    achieved_bw_gbps: float | None = Field(default=None, ge=0.0)
    peak_bw_gbps: float | None = Field(default=None, ge=0.0)
    profiled_bw_gbps: float | None = Field(default=None, ge=0.0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    power_w: float | None = Field(default=None, ge=0.0)


class ResultWriter:
    """Append validated results to one run-specific CSV file."""

    def __init__(self, run_id: str, results_dir: str | Path = "results") -> None:
        if not run_id:
            raise ValueError("run_id cannot be empty")
        self.run_id = run_id
        self.results_dir = Path(results_dir)
        self.path = self.results_dir / f"{run_id}.csv"

    def append(self, result: RunResult) -> Path:
        """Append one row, creating the directory and header when necessary."""
        if result.run_id != self.run_id:
            raise ValueError(
                f"result run_id {result.run_id!r} does not match writer "
                f"run_id {self.run_id!r}"
            )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(RunResult.model_fields)
        self._validate_existing_header(fieldnames)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(result.model_dump(mode="json"))
        return self.path

    def append_many(self, results: list[RunResult]) -> Path:
        """Append results in order and return the run CSV path."""
        if not results:
            raise ValueError("results cannot be empty")
        for result in results:
            self.append(result)
        return self.path

    def _validate_existing_header(self, expected: list[str]) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open(encoding="utf-8", newline="") as input_file:
            actual = next(csv.reader(input_file), [])
        if actual != expected:
            raise ValueError(
                f"existing CSV header in {self.path} does not match RunResult schema"
            )
