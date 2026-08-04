# Stage 13 reproducibility report

## Scope and evidence status

This report covers the implemented Stage 0–8 system: serving latency and
throughput, model/KV statistics, expert activation, sparse utilisation metrics,
profiler validation, and roofline analysis. Stages 9–12 were intentionally
skipped, so there is no quantisation accuracy study, CAP radar, or agentic-vs-chat
comparison. The committed `demo_sweep.csv` is a deterministic synthetic fixture.
It verifies the analysis pipeline; it is not evidence of H100 or model speed.

Run `python scripts/regenerate_figures.py` from the repository root to create all
five report figures. Each output has one declared generator and no hand-authored
image is part of the report:

| Figure | Input and generator |
| --- | --- |
| `throughput_vs_batch.png` | `demo_sweep.csv` → `plot_throughput_vs_batch` |
| `tpot_vs_batch.png` | `demo_sweep.csv` → `plot_tpot_vs_batch` |
| `activated_ratio_vs_batch.png` | `demo_sweep.csv` → `plot_activated_ratio_vs_batch` |
| `kv_vs_seqlen.png` | fixed architecture dimensions → `plot_kv_vs_seqlen` |
| `roofline.png` | device DB + declared points → `plot_roofline` |

## Sparsity, batching, and memory

The demonstration activation ratios are 0.125, 0.22, 0.42, 0.72, 0.90, and
1.00 for batches 1, 2, 4, 8, 16, and 32. They illustrate the intended Stage 6
shape: a low batch-one base that rises sub-linearly toward full expert coverage.
The recorder computes the expert union separately for each forward pass and then
averages pass ratios. It never unions unrelated prompts across all trace passes,
which would artificially raise the batch-one baseline and flatten the curve.

KV cache is analytical and grows exactly with `batch × sequence length` under a
fixed model/dtype. The reproduction plot uses Qwen2-MoE-like dimensions (24
layers, 16 KV heads, head dimension 128, two bytes per element). This gives
196,608 bytes per token before batch and sequence scaling. Unlike expert weights,
KV memory is not reduced by sparse routing.

## Utilisation and roofline interpretation

Vanilla MBU charges every expert weight to each decode step; S-MBU charges only
experts observed in the activation matrix plus attention and KV bytes. The dense
special case is unit-tested: with one expert active, S-MBU equals MBU. MFU has the
same total-vs-active distinction but uses token FLOPs and peak operations. A
profiler validation helper compares analytical achieved bandwidth with available
DRAM counters and carries the counter source so an analytical fallback remains
visible.

The reproducible roofline uses the stored H100-SXM FP16 peaks and two explicitly
illustrative points. Decode at 12 FLOPs/byte is left of the ridge and classified
memory-bound; prefill at 450 FLOPs/byte is right of the ridge and classified
compute-bound. These positions encode the typical relationship that the real
profiler is designed to test. They are not measured kernel coordinates.

## What remains for an empirical report

A real release should run the same model revision, prompts, precision, backend
version, and device across all batch sizes; retain the generated RunResult CSV;
record actual activation ratios and profiler counter coverage; and replace the
demonstration source passed to `--csv`. The report can then state measured values
and uncertainty. The model must be a trained production MoE: a tiny/random fixture
that activates nearly every expert at batch 1 is an integration test or null
result, not evidence of sparsity collapse with batching. CAP/accuracy and agentic
findings remain out of scope unless Stages 9–12 are implemented and backed by
committed result tables.
