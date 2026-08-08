# Upstream baseline and provenance

The proxy is derived from the Apache-2.0 licensed vllm-ascend project.

| Field | Value |
| --- | --- |
| Repository | `https://github.com/vllm-project/vllm-ascend` |
| Baseline tag | `v0.23.0rc1` |
| Baseline release date | `2026-07-20` |
| Upstream path | `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` |

This directory contains an unmodified snapshot of
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` from
the vllm-ascend tag `v0.23.0rc1`.

The baseline file is stored byte-for-byte in
[`load_balance_proxy_server_example.py`](load_balance_proxy_server_example.py).
The repository-root file is the KV-cache-aware candidate intended for
experimentation and eventual upstream contribution.

The initial implementation is motivated by
[vllm-ascend issue #12196](https://github.com/vllm-project/vllm-ascend/issues/12196).
It keeps Decoder selection load-balanced and applies affinity only to
Prefillers, where reusable prefix KV is located.

It exists solely to make the A/B/C experiment reproducible:

- A: this upstream baseline;
- B: the candidate proxy with KV-aware routing disabled;
- C: the candidate proxy with KV-aware routing enabled.

Do not add experimental changes to this file.

The benchmark accepts both the upstream reset endpoint's empty HTTP 200
response and the candidate's structured Prefiller-only reset response. If the
upstream fan-out fails on a Decoder without prefix caching, it retries reset
directly against every Prefiller.

When changing the baseline tag, update this tag, the snapshot, compatibility
notes, and published experiment metadata together. Results from different
baseline tags must not be silently mixed.
