# Experiment results

Raw runs are written under `results/runs/` and ignored by Git because request
payloads and logs may be large or sensitive. Publish only reviewed and redacted
artifacts under `results/published/<experiment-id>/`.

Every published experiment should contain:

```text
<experiment-id>/
├── README.md
├── metadata.json
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
- run order, reset procedure and repetition count.

Do not publish API keys, full production prompts, Kubernetes Secrets, internal
hostnames, Pod IPs, model credentials, or unreviewed raw logs.

