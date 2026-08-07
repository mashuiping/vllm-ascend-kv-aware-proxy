# Benchmark methodology

## Comparison groups

Every result should include all three groups on the same P/D deployment:

| Group | Implementation | Affinity |
| --- | --- | --- |
| A | exact pinned upstream baseline | unavailable |
| B | candidate | disabled |
| C | candidate | enabled |

Use B versus C to isolate affinity. Use A versus B to detect effects from the
restored Prefiller load lifecycle and other candidate integration changes. Use
A versus C only as the end-to-end comparison with stock upstream.

## Control variables

- Keep P/D Pods, model, image, scheduler arguments and hardware unchanged.
- Restart only the proxy between groups by default.
- Drain all Prefillers, reset backend prefix caches and verify every direct
  Prefiller is cold before every measured run.
- Use identical prompts, seeds, session IDs and concurrency.
- Alternate group order across repetitions instead of always running A/B/C.
- Record warm-up separately from measured requests.
- Generate one immutable workload and use its SHA-256 in every A/B/C group.
- Define prompt sizes in model tokens and verify them with the served tokenizer.
- Run enough repetitions to distinguish a stable effect from normal variance.

At minimum, record the image digest, model identifier, NPU type, CANN and driver
versions, P/D topology, repository commit, upstream baseline commit, run order,
benchmark command, per-Prefiller reset outcome and experiment validity result.

## Required workloads

| Scenario | Purpose |
| --- | --- |
| `session-long` | Main multi-turn session-locality case |
| `shared-prefix` | Cross-session shared system-prompt locality |
| `short` | Negative control below useful cache granularity |
| `one-shot` | Measures overhead and LRU pollution without reuse |
| `hot-key` | Exposes hotspot risk without spillover |

Start with low concurrency to prove cache behavior, then use a concurrency
ladder appropriate for the deployed capacity. Short output lengths emphasize
Prefill and TTFT. For legacy generated workloads, `prefix-words` is only an
approximation; always report actual server `prompt_tokens` from results.

The preferred runner uses the checked-in profiles under `benchmarks/profiles/`
instead of the legacy `prefix-words` generator. See
[`benchmarks/README.md`](../benchmarks/README.md) for exact-token generation,
cache-fill and prefix-probe phases, per-stage Poisson QPS results and capacity
sizing.

## Required metrics

- request success and error count;
- warm-turn TTFT p50, p95 and p99;
- E2E p50, p95 and p99;
- prompt, cached and computed tokens;
- cached-token request rate and cached-token ratio;
- per-stage cache and latency results for shared-prefix probes and QPS levels;
- Prefill and Decode queue time;
- request and token throughput;
- per-Prefiller cache/query/load distribution;
- NPU memory pressure where available.

The controlled deployment enables prompt-token details. A missing
`cached_tokens` field is retained as `null` in raw JSONL but normalized to zero
for aggregate request-rate, token-ratio and computed-token metrics. Excluding
missing values would calculate cache ratios over hits only.

An affinity hit is not itself proof of value. The desired chain is:

```text
stable route -> higher cached tokens -> less Prefill work -> lower warm TTFT
```

If cached tokens rise but TTFT does not, inspect KV transfer, Decoder queueing
and proxy overhead before widening affinity.

The shared-prefix profile primes one prompt per group and measures a different
same-prefix prompt in `prefix-probe`. This prevents the cache-fill phase from
replicating every shared prefix onto all four Prefillers before the routing
comparison. Its QPS stages are a continuous ramp; run each rate independently
when the question is an isolated capacity point.

## Running

The benchmark writes request JSONL, configuration, metric snapshots, reset
verification, validity and a summary under the selected output directory. The
helper maps each experiment group onto deployment variables. Comparison output
is marked invalid unless all group checks pass.

For a production comparison, generate one workload and run the complete matrix:

```bash
MODEL=qwen3-32b PREFILL_NODE=<node> \
  bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

This defaults to the in-cluster Benchmark Pod and verifies every Prefiller.
The legacy local/port-forward path cannot discover and probe all four
Prefillers automatically, so its comparison is marked invalid and should be
used only for debugging.

The individual commands below remain useful for debugging and legacy generated
workloads:

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-32b \
  bash scripts/run_experiment.sh baseline

BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-32b \
  bash scripts/run_experiment.sh candidate-off

BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-32b \
  bash scripts/run_experiment.sh candidate-on
```

To generate `comparison.json`, point a treatment run at an existing baseline:

```bash
COMPARE_WITH=results/runs/<baseline>/summary.json \
BASE_URL=http://127.0.0.1:8000 MODEL=qwen3-32b \
  bash scripts/run_experiment.sh candidate-on
```

Review and redact artifacts according to [`results/README.md`](../results/README.md)
before publishing them.
