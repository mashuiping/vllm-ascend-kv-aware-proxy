# Upstream baseline

This directory contains an unmodified snapshot of
`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` from
vllm-ascend commit `8f3fd59a203c3b35f29b6a77f459c9a78da7d5c0`.

It exists solely to make the A/B/C experiment reproducible:

- A: this upstream baseline;
- B: the candidate proxy with KV-aware routing disabled;
- C: the candidate proxy with KV-aware routing enabled.

Do not add experimental changes to this file. See [`../UPSTREAM.md`](../UPSTREAM.md)
for provenance and checksums.

