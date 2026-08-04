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
- Restart only the proxy between groups.
- Reset backend prefix caches before every measured run.
- Use identical prompts, seeds, session IDs and concurrency.
- Alternate group order across repetitions instead of always running A/B/C.
- Record warm-up separately from measured requests.
- Run enough repetitions to distinguish a stable effect from normal variance.

At minimum, record the image digest, model identifier, NPU type, CANN and driver
versions, P/D topology, repository commit, upstream baseline commit, run order,
benchmark command and reset outcome.

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
Prefill and TTFT. The generated `prefix-words` value controls workload size;
report actual server `prompt_tokens` from results.

## Required metrics

- request success and error count;
- warm-turn TTFT p50, p95 and p99;
- E2E p50, p95 and p99;
- prompt, cached and computed tokens;
- cached-token request rate and cached-token ratio;
- Prefill and Decode queue time;
- request and token throughput;
- per-Prefiller cache/query/load distribution;
- NPU memory pressure where available.

An affinity hit is not itself proof of value. The desired chain is:

```text
stable route -> higher cached tokens -> less Prefill work -> lower warm TTFT
```

If cached tokens rise but TTFT does not, inspect KV transfer, Decoder queueing
and proxy overhead before widening affinity.

## Running

The benchmark writes request JSONL, configuration, metric snapshots and a
summary under the selected output directory. The helper maps each experiment
group onto deployment variables:

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

