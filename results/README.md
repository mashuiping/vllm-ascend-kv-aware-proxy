# Experiment results

Raw runs are written under `results/runs/` and ignored by Git because request
payloads and logs may be large or sensitive. Publish only reviewed and redacted
artifacts under `results/published/<experiment-id>/`.

## Known invalid local runs

The six `20260810T002611Z-abc` through `20260810T010902Z-abc` runs are invalid.
Their A-group health samples contain candidate-only `prefill_loads` and
`active_tokens` fields, which means the declared baseline source was not
mounted. Do not use these runs for performance claims. New Pod-mode runs verify
the mounted proxy SHA-256, variant, active-token weight and affinity flag before
sending traffic.

Every published experiment should contain:

```text
<experiment-id>/
├── README.md
├── metadata.json
├── workload.jsonl.manifest.json
├── baseline-summary.json
├── candidate-off-summary.json
├── candidate-on-summary.json
└── comparison.json
```

`metadata.json` must record at least:

- repository commit and upstream baseline commit;
- vLLM, vllm-ascend, CANN and driver versions;
- image tag and immutable digest when available;
- NPU type, node topology and model identifier;
- Prefill/Decode configuration;
- complete benchmark arguments and random seed;
- workload profile, SHA-256 and whether model-token counts were verified;
- run order, reset procedure and repetition count.

Do not publish API keys, full production prompts, Kubernetes Secrets, internal
hostnames, Pod IPs, model credentials, or unreviewed raw logs.
