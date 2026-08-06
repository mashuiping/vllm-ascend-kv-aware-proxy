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

## Dataset construction and load model

The benchmark does not generate prompts independently for each A/B/C group. It
first creates one immutable JSONL workload and then reuses the same bytes for
`baseline`, `candidate-off`, and `candidate-on`. Each record contains the
session ID, complete message list, turn, phase, stage, scheduled offset,
`max_tokens`, and temperature. The workload SHA-256 is recorded in every
group's `config.json`; comparison fails if the hashes differ.

### `session-affinity.json`

The production session-locality profile is:

| Setting | Value |
| --- | ---: |
| Sessions | 60 |
| Turns per session | 5 |
| System prompt per session | 2,048 model tokens |
| New user input per turn | 200 model tokens |
| Maximum generated output | 32 tokens |
| Temperature | 0.0 |
| Concurrency | 20 |

This produces exactly 300 requests: 60 requests in `turn-0` (`cache-fill`)
and 60 requests in each of `turn-1` through `turn-4` (`measure`). Each session
gets a deterministic, session-specific system prompt. Later turns contain the
full prior conversation, with the assistant message fixed to
`Acknowledged.` rather than copied from a live model response. Consequently,
all three groups receive byte-identical prompts while later turns preserve the
same session prefix and session ID.

The load type is `turn-barrier`, not a target QPS. The runner uses a thread
pool with `max_workers=20`, submits the 60 requests for one turn, and waits for
that turn to finish before starting the next turn. At most 20 requests are in
flight; there is no think-time delay between turns unless explicitly supplied
to the runner.

### `shared-prefix-capacity.json`

The prefix-locality profile builds a corpus of 32 groups with 8 prompts per
group:

```text
32 groups × 8 prompts = 256 cache-fill prompts
```

Within one group, all eight prompts share the same 2,048-token system prompt
and have a different 256-token question. Session headers are disabled for this
profile, so the test isolates cross-session prefix reuse and prefix-aware
routing rather than session-key routing. Every request has `max_tokens=32` and
`temperature=0.0`.

The 256 cache-fill prompts are scheduled at 10 requests per second, with
offsets `0.0s`, `0.1s`, ..., `25.5s`. The measured stages use deterministic
Poisson inter-arrival offsets and a maximum of 64 in-flight requests:

| Stage | Target rate | Duration | Expected requests |
| --- | ---: | ---: | ---: |
| `qps-2` | 2 req/s | 30 s | about 60 |
| `qps-5` | 5 req/s | 30 s | about 150 |
| `qps-10` | 10 req/s | 30 s | about 300 |
| `qps-20` | 20 req/s | 30 s | about 600 |

The Poisson generator keeps only arrivals whose offset is less than the stage
duration, so the exact request count is allowed to vary around the expectation.
With the checked-in seed, the current generated counts are 53, 153, 307, and
595 respectively (1,108 measured requests plus 256 cache-fill requests).
The configured rate is a request-start schedule; if 64 requests are already
pending, the runner applies backpressure and the achieved rate can be lower
than the target.

### Token accuracy and output limits

For a publishable run, `generate_benchmark_workload.py` calls the served model's
`/tokenize` and `/detokenize` endpoints. It expands a deterministic candidate
string until it has enough token IDs, detokenizes the selected IDs, tokenizes
the result again, and rejects the record unless the round trip has exactly the
requested token count. The production profiles therefore measure model-token
lengths, not character or word approximations. Workload generation happens
once inside the benchmark Pod and is not repeated for individual A/B/C groups.

The profile's `max_tokens=32` is sent with every measured request, which keeps
decode output bounded. The separate infrastructure warm-up requests use
`max_tokens=1` and are excluded from the measured workload.

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

The Kubernetes A/B/C orchestrator uses eight sacrificial system warm-up
requests for every group. These requests exercise the HTTP path and model
runtime, but are not included in the workload summary. It then calls the
proxy's reset endpoint before taking the measurement snapshots, clearing the
backend prefix caches and candidate affinity LRUs so that each group starts
its measured `cache-fill` phase from the same logical state.

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

## Running A/B/C in Kubernetes

The orchestrator creates one temporary benchmark Pod in `NAMESPACE` (default
`qwen-pd`). The Pod calls `pd-prefill-0.pd-prefill:7100` for exact tokenization
and `pd-proxy:8000` for load, avoiding local port-forward connections. The local
script only changes the proxy group and copies final artifacts back:

```bash
export PREFILL_NODE='your-prefill-node'
export VLLM_IMAGE='your-tested-vllm-ascend-image'
export MODEL='qwen3-32b'

bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

Override the in-cluster endpoints only when the Services use different names:

```bash
export IN_CLUSTER_TOKENIZER_URL='http://custom-prefill:7100'
export IN_CLUSTER_BASE_URL='http://custom-proxy:8000'
```

On success, the temporary Pod and ConfigMap are removed after results are
copied to `results/runs/`. On failure they are retained and the script prints an
inspection command. Set `KEEP_BENCHMARK_POD=true` to retain them after success.

Common runner arguments are forwarded to every group. For example, add all
Prefiller and Decoder metrics endpoints using their in-cluster DNS names:

```bash
bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json \
  --prefill-metrics-url http://pd-prefill-0.pd-prefill:7100/metrics \
  --prefill-metrics-url http://pd-prefill-1.pd-prefill:7100/metrics \
  --decode-metrics-url http://pd-decode-0.pd-decode:7200/metrics
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

For compatibility, the previous local/port-forward execution path remains
available:

```bash
BENCHMARK_EXECUTION_MODE=local \
BASE_URL=http://127.0.0.1:8000 \
TOKENIZER_URL=http://127.0.0.1:7100 \
MODEL=qwen3-32b PREFILL_NODE=<node> \
  bash scripts/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

The workload design follows the public multi-turn shape described by Higress
and the shared-prefix, cache-capacity and staged-load methodology published by
`llm-d-benchmark` and `inference-perf`.
