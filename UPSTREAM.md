# Upstream provenance

The proxy is derived from the Apache-2.0 licensed vllm-ascend project.

| Field | Value |
| --- | --- |
| Repository | `https://github.com/vllm-project/vllm-ascend` |
| Baseline commit | `8f3fd59a203c3b35f29b6a77f459c9a78da7d5c0` |
| Baseline commit date | `2026-08-02T04:31:52+08:00` |
| Upstream path | `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` |
| Baseline SHA-256 | `97a5ded534e4b3a41186926946056bca98cbebe9861d7be3a9cfb1634f0d0bf3` |
| Candidate SHA-256 | `bac21a6d6d456a3d03d2ff1e288bf1d96f134dddb15b63baed64efc1a0e67656` |

The baseline file is stored byte-for-byte in
[`baseline/load_balance_proxy_server_example.py`](baseline/load_balance_proxy_server_example.py).
The repository-root file is the KV-cache-aware candidate intended for
experimentation and eventual upstream contribution.

The initial implementation is motivated by
[vllm-ascend issue #12196](https://github.com/vllm-project/vllm-ascend/issues/12196).
It keeps Decoder selection load-balanced and applies affinity only to
Prefillers, where reusable prefix KV is located.

The fork URL was re-fetched before the current candidate was generated. Its
`main` branch and the official upstream `main` both resolved to the baseline
commit and blob recorded above.

When rebasing onto a newer vllm-ascend revision, update the baseline snapshot,
commit metadata, checksums, compatibility notes, and published experiment
metadata together. Results from different baselines must not be silently mixed.
