# Benchmark workloads

The production benchmark uses one immutable JSONL workload for all A/B/C
groups. Never generate a separate workload per group: the JSONL file fixes
session IDs, messages, request order, load-stage offsets, output limits and the
random seed.

## Profiles

| Profile | Shape | Purpose |
| --- | --- | --- |
| `session-affinity.json` | 60 sessions, 5 turns, 2,048-token system prompt, 200-token turn input, concurrency 20 | Higress-style multi-turn session locality |
| `shared-prefix-capacity.json` | 32 groups × 8 prompts, 2,048-token shared prefix, 256-token unique question, Poisson QPS ladder | llm-d-style cross-session prefix locality |
| `smoke.json` | 2 sessions × 2 turns | Offline generator and test smoke case |

The lengths in production profiles are model-token counts, not words or
characters. Generate them against the exact served model through a direct vLLM
endpoint exposing `/tokenize` and `/detokenize`:

```bash
python scripts/generate_benchmark_workload.py \
  --profile benchmarks/profiles/session-affinity.json \
  --output results/runs/session-affinity.jsonl \
  --tokenizer-url http://PREFILLER:7100 \
  --model qwen3-32b
```

The adjacent `*.manifest.json` must say `"token_count_verified": true` for a
publishable run. `--allow-unverified-token-counts` exists only for offline smoke
tests:

```bash
python scripts/generate_benchmark_workload.py \
  --profile benchmarks/profiles/smoke.json \
  --output /tmp/kv-aware-smoke.jsonl \
  --allow-unverified-token-counts
```

## Phases

Every run executes these phases in order:

```text
system warm-up (excluded)
  -> reset backend prefix caches and candidate affinity LRUs
  -> before metrics snapshot
  -> cache-fill
  -> after-cache-fill metrics snapshot
  -> measured turns or QPS stages
  -> final metrics snapshot
```

For `session-long`, turn 0 is `cache-fill` and later turns are `measure`. The
assistant messages in later turns are fixed in the workload instead of copying
live model output, so all groups receive byte-identical prompts.

For `shared-prefix`, every group/prompt pair is sent once during `cache-fill`.
Measured requests then replay the frozen corpus with deterministic Poisson
arrival offsets. Session headers are disabled so this workload isolates prefix
routing.

## Capacity sizing

Size a shared-prefix workload relative to aggregate Prefiller KV capacity:

```text
resident tokens ≈ groups × (system tokens + prompts per group × question tokens)
working-set ratio = resident tokens / aggregate Prefiller KV token capacity
```

Run at approximately 25%, 50%, 75% and 110% to cover ample capacity, normal
pressure, cache stress and eviction. Record the observed vLLM KV capacity and
usage; do not infer it only from model configuration.

## Running A/B/C

The orchestrator generates the workload once, then passes the same file to all
groups:

```bash
export PREFILL_NODE='your-prefill-node'
export BASE_URL='http://127.0.0.1:8000'
export TOKENIZER_URL='http://PREFILLER:7100'
export MODEL='qwen3-32b'

bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

Common runner arguments are forwarded to every group. For example, add all
Prefiller and Decoder metrics endpoints without changing the workload:

```bash
bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json \
  --prefill-metrics-url http://P0:7100/metrics \
  --prefill-metrics-url http://P1:7100/metrics \
  --decode-metrics-url http://D0:7100/metrics
```

The experiment directory contains the shared workload and manifest, one fixed
directory per group, and `comparison.json`. Comparison generation fails if the
three recorded workload SHA-256 values are not identical.

Use `GROUP_ORDER` to balance order across repetitions, for example:

```bash
GROUP_ORDER='candidate-off candidate-on baseline' \
  bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

At least six repetitions should cover the six A/B/C permutations. Treat
`candidate-off` versus `candidate-on` as the affinity comparison; the upstream
baseline versus candidate-off comparison measures other candidate changes.

The workload design follows the public multi-turn shape described by Higress
and the shared-prefix, cache-capacity and staged-load methodology published by
`llm-d-benchmark` and `inference-perf`.
