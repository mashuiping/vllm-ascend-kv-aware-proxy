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
- Generate one immutable workload and use its SHA-256 in every A/B/C group.
- Define prompt sizes in model tokens and verify them with the served tokenizer.
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
Prefill and TTFT. For legacy generated workloads, `prefix-words` is only an
approximation; always report actual server `prompt_tokens` from results.

The preferred runner uses the checked-in profiles under `benchmarks/profiles/`
instead of the legacy `prefix-words` generator. See
[`benchmarks/README.md`](../benchmarks/README.md) for exact-token generation,
cache-fill phases, Poisson QPS stages and capacity sizing.

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

For a production comparison, generate one workload and run the complete matrix:

```bash
BASE_URL=http://127.0.0.1:8000 \
TOKENIZER_URL=http://PREFILLER:7100 \
MODEL=qwen3-32b PREFILL_NODE=<node> \
  bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

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
