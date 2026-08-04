# Upstream contribution notes

## Status

Prepared and locally tested; **not committed, pushed, or submitted**. The project
owner requires reviewing and making every commit. Consequently there is no PR URL
or `v0.1.0` tag yet; this hand-off does not invent one.

## Upstream target

- Repository: <https://github.com/Auto-CAP/MoE-CAP>
- Inspected commit: `834276c3443050ce6b4ccb227a8269be755e6a3d`
- Inspection date: 2026-08-04
- Contribution class: focused performance-metric bug fix, explicitly welcomed by
  the upstream README.

The current client-side fallback in
`moe_cap/runner/openai_api_profile.py` calculates TPOT as
`(latency - ttft) / output_len`. TTFT already ends when the first output token
arrives, so a completion of `N` tokens has only `N - 1` inter-token intervals.
Dividing by `N` under-reports TPOT. A one-token response has no interval and must
not enter the mean.

## Review-ready patch

The portable patch is [`contributions/moe-cap-client-tpot.patch`](contributions/moe-cap-client-tpot.patch).
It:

1. extracts the fallback calculation into `_mean_client_tpot`;
2. divides post-TTFT time by `output_len - 1` only when `output_len > 1`;
3. adds tests for the denominator, one-token early EOS, and mixed valid/empty
   responses.

It was applied to the inspected upstream commit in a temporary clone. The focused
upstream test command passed: **6 passed** across the new test and existing
request-TTFT aggregation test.

Suggested PR title: `fix(metrics): calculate client TPOT over inter-token intervals`

Suggested PR description:

> The client fallback subtracts TTFT from end-to-end latency but divided the
> remaining time by all output tokens. Because the first token is represented by
> TTFT, only `output_len - 1` inter-token intervals remain. This change uses that
> denominator and excludes one-token completions, for which TPOT is undefined.
> Focused tests cover the corrected value and early-EOS behavior. Server-derived
> TPOT remains unchanged.

## Owner actions after review

```bash
# In a fork/clone of Auto-CAP/MoE-CAP at the inspected revision:
git apply /path/to/moe-profiler/contributions/moe-cap-client-tpot.patch
python -m pytest tests/test_client_tpot_fallback.py tests/test_request_ttft_aggregation.py

# After reviewing the diff, the owner creates the upstream commit and PR.
# In this repository, after reviewing all local changes:
git add -A
git commit -m "docs: add reproducible report and upstream patch"
git tag v0.1.0
```

Before tagging, also run the complete local acceptance commands shown in the
README. Do not tag until those commands are green and the owner has inspected the
working tree.
