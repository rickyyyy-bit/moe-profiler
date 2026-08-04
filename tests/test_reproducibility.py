"""Stage 13 clean-checkout figure reproduction checks."""

from pathlib import Path

from scripts.regenerate_figures import regenerate_figures


def test_regenerate_figures_builds_complete_manifest(tmp_path: Path) -> None:
    outputs = regenerate_figures("results/demo_sweep.csv", tmp_path)

    assert {output.name for output in outputs} == {
        "activated_ratio_vs_batch.png",
        "kv_vs_seqlen.png",
        "roofline.png",
        "throughput_vs_batch.png",
        "tpot_vs_batch.png",
    }
    assert all(output.is_file() and output.stat().st_size > 0 for output in outputs)
