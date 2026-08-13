# Benchmark workloads and tooling

The production benchmark uses one immutable JSONL workload for all A/B/C
groups. Never generate a separate workload per group: the JSONL file fixes
session IDs, messages, request order, load-stage offsets, output limits and the
random seed.

All benchmark generators, runners, comparison tools and report renderers live
in this directory alongside the checked-in workload profiles.

## Profiles


| Profile                       | Shape                                                                                           | Purpose                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `session-affinity.json`       | 60 sessions, 5 turns, 2,048-token system prompt, 200-token turn input, concurrency 20           | Higress-style multi-turn session locality |
| `shared-prefix-capacity.json` | 32 groups × 8 prompts, one prime plus one prefix probe per group, 2,048-token shared prefix, 256-token unique question, Poisson QPS ladder | llm-d-style cross-session prefix locality |
| `load-balance-active-tokens.json` | Unique 512/2,048/4,096-token prompt classes, 2/3/4.5 QPS for 30/60/90 seconds, concurrency 192, 256-token outputs, Proxy uses 2 Decoders | Targeted Prefill active-token lifecycle validation around the measured capacity knee |
| `session-affinity-zipf.json`  | Same shape as `session-affinity.json` but measure-turn traffic is Zipf-skewed (`zipf_alpha` 1.2) onto hot sessions | Single-node hot-spot pressure for the affinity overload guard |
| `shared-prefix-zipf.json`     | 32 groups × 8 prompts, QPS-stage group selection Zipf-skewed (`zipf_alpha` 1.2)                 | Cross-session hot-prefix pressure for the overload guard |
| `shared-prefix-capacity-pressure.json` | 96 groups × 8,192-token shared prefix (~786K prefix tokens total), Poisson QPS ladder up to 20 | KV eviction churn: total prefix corpus sized to 2–3× one node's KV capacity |
| `smoke.json`                  | 2 sessions × 2 turns                                                                            | Offline generator and test smoke case     |

The Zipf profiles set the optional `data.zipf_alpha` field: after a first
uniform coverage pass (so every session/group establishes a binding), request
volume is sampled with probability proportional to `1 / rank^alpha`. Omitting
the field keeps the original uniform behavior, so existing profiles are
unaffected. `shared-prefix-capacity-pressure.json` assumes the single-node KV
capacity measured on this deployment (gpu-mem-util 0.90, block-size 128,
max-model-len 32768); re-measure per the deploy README and adjust
`num_groups` if your topology differs.


The lengths in production profiles are model-token counts, not words or
characters. Generate them against the exact served model through a direct vLLM
endpoint exposing `/tokenize` and `/detokenize`:

```bash
python benchmarks/generate_benchmark_workload.py \
  --profile benchmarks/profiles/session-affinity.json \
  --output results/runs/session-affinity.jsonl \
  --tokenizer-url http://PREFILLER:7100 \
  --model qwen3-32b
```

The adjacent `*.manifest.json` must say `"token_count_verified": true` for a
publishable run. `--allow-unverified-token-counts` exists only for offline smoke
tests:

```bash
python benchmarks/generate_benchmark_workload.py \
  --profile benchmarks/profiles/smoke.json \
  --output /tmp/kv-aware-smoke.jsonl \
  --allow-unverified-token-counts
```

For the session and shared-prefix profiles, `data.system_prompt_tokens` can be
overridden at runtime without editing the JSON file. Set the environment
variable on the experiment command; the value is applied once while generating
the shared workload used by all three groups:

```bash
SYSTEM_PROMPT_TOKENS=8192 \
bash benchmarks/run_abc_experiment.sh \
  benchmarks/profiles/session-affinity.json
```

The same override works with `shared-prefix-capacity.json`. The generated
manifest records the effective profile, while the checked-in profile remains
unchanged. It is intentionally not supported by the heterogeneous
`load-balance-active-tokens.json` profile, whose prompt sizes are defined by
its individual prompt classes.



## Dataset construction and load model

The benchmark does not generate prompts independently for each A/B/C group. It
first creates one immutable JSONL workload and then reuses the same bytes for
`baseline`, `candidate-off`, and `candidate-on`. Each record contains the
session ID, complete message list, turn, phase, stage, scheduled offset,
`max_tokens`, and temperature. The workload SHA-256 is recorded in every
group's `config.json`; comparison fails if the hashes differ.

### `session-affinity.json`

The production session-locality profile is:


| Setting                   | Value              |
| ------------------------- | ------------------ |
| Sessions                  | 60                 |
| Turns per session         | 5                  |
| System prompt per session | 2,048 model tokens |
| New user input per turn   | 200 model tokens   |
| Maximum generated output  | 32 tokens          |
| Temperature               | 0.0                |
| Concurrency               | 20                 |


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

### `load-balance-active-tokens.json`

This profile is designed to expose the transient Prefill load signal. It is
not a cache-locality test:
every request has a unique prompt, session headers are disabled, and the
primary comparison is Prefill queue time, TTFT tail latency and per-node load
balance.

#### Lifecycle that creates a B/C routing difference

For a Prefiller, let `Q` be the score of requests whose Prefill RPC is still
running, and let `W` be the score of requests whose Prefill has completed but
whose Decoder has not returned its first response chunk. Because Prefill KV is
reserved across both phases, `active_kv_cache = Q + W`.

In the automatically selected `active-token` comparison mode, the effective
priorities are:

```text
A (exact upstream)       = 0.3 * (Q + W)
B (candidate weight=0)  = 0.3 * (Q + W)
C (candidate weight=1)  = Q + 0.3 * (Q + W)
                        = 1.3Q + 0.3W
```

The baseline source retains the `active_tokens + 0.3 * active_kv_cache`
formula, but never charges `active_tokens` on a Prefiller, so its effective
value is the A formula above. The candidate charges both fields when Prefill
is selected, releases `active_tokens` when Prefill completes, and retains
`active_kv_cache` until the Decoder produces its first chunk.

If almost every outstanding request is still in Prefill (`W` is near zero),
B computes `0.3Q` and C computes `1.3Q`. That is only a constant scale factor,
so both heaps rank the Prefillers in the same order. Merely creating more
overlapping Prefill work therefore does not exercise the behavioral change.
The nodes must have different lifecycle mixtures. For example:

| Prefiller | Running Prefill `Q` | Waiting for Decoder `W` | B priority | C priority |
| --------- | ------------------: | ----------------------: | ---------: | ---------: |
| P0 | 4,000 | 0 | 1,200 | 5,200 |
| P1 | 0 | 8,000 | 2,400 | 2,400 |

A/B select P0 while C selects P1. More generally, this two-node example
diverges when `Q0 < W1 < 4.33 * Q0`.

#### Why the previous profile showed little difference

The previous version used four routed Decoders, `max_tokens=4`, concurrency
128, and sustained 8/16/24/32 QPS Prefill traffic. The deployment allows up to
64 sequences per Decoder, so its nominal Decode concurrency was 256. Four-token
responses released Decoders quickly, and a completed Prefill normally moved
through the `W` phase before the next one-second health sample. Meanwhile the
high input rate kept many requests in `Q` on every Prefiller.

That workload successfully created Prefill overlap, but usually created one
of two low-information states:

- `W` was close to zero, making B a constant rescaling of A; or
- every Prefiller had a similar `Q/W` ratio, preserving the same ordering.

Long steady Poisson stages also average the short/medium/long request mix over
all four Prefillers. Aggregate request/token CV can consequently look the same
even if a small number of transient decisions differ. This is why increasing
QPS and prompt length alone did not produce a clear A/B result.

#### Capacity-knee pressure model

The revised profile keeps the three unique prompt-size classes (512, 2,048
and 4,096 system tokens, with correspondingly different user inputs), but uses
the following diagnostic configuration:

| Stage | Target rate | Duration | Expected requests | Concurrency |
| ----- | ----------- | -------- | ----------------- | ----------- |
| `below-knee-qps-2` | 2 req/s | 30 s | about 60 | 192 |
| `at-knee-qps-3` | 3 req/s | 60 s | about 180 | 192 |
| `above-knee-qps-4.5` | 4.5 req/s | 90 s | about 405 | 192 |

The profile sets `max_tokens=256`, tells the Proxy to route to Decoder 0 and 1,
and raises the client concurrency ceiling to 192. All four Decoder Pods stay
running with their normal `MAX_NUM_SEQS=64`; only the Proxy backend list is
reduced, so no Decoder restart is required. The A/B/C groups use the same two
Decoders.

Longer outputs do not extend their own Prefiller KV reservation: that
reservation is released on the first Decoder chunk. Instead, the longer-lived
Decode requests occupy the two visible Decoders and make later requests wait
longer before their first chunk. Those later requests remain in `W`, while
heterogeneous prompts keep other requests in `Q`. Reducing the early QPS stages
also avoids immediately turning every Prefiller into the same saturated-Q
state.

Health sampling is reduced from one second to 100 ms because the relevant
phase transition is transient. The initial mechanism-validation run showed the
expected routing divergence only near and above the capacity knee, so the
formal profile spends 30/60/90 seconds below/at/above the knee respectively.
This preserves a low-pressure negative control while concentrating tail samples
in the informative 4.5 QPS stage. The runner drains the stage before starting
the next one and records independent per-stage engine-counter snapshots,
launch span, drain time and total wall time.

Workload construction verifies unique text against the served tokenizer with
16 concurrent workers by default. Set
`TOKENIZER_CONCURRENCY=1` to serialize generation or lower the value if the
tokenizer endpoint is resource constrained.

The generated system and user messages are unique per request, so cache hits
should remain negligible. The different prompt sizes create overlapping
Prefill work; candidate weight=1 can then avoid selecting a node whose active
Prefill compute load is high even when its KV load alone appears low.

Run it with the normal orchestrator:

```bash
bash benchmarks/run_abc_experiment.sh \
  benchmarks/profiles/load-balance-active-tokens.json
```

`run_abc_experiment.sh` reads `deployment.proxy_decoder_count` and
`observability.sample_interval_s` from this profile. Profiles without those
fields use four Decoders and one-second sampling, so `session-affinity.json`
and `shared-prefix-capacity.json` retain the normal 4P4D topology. For an
explicit override, set `BENCHMARK_DECODER_COUNT` or `SAMPLE_INTERVAL`; the
override is applied identically to A/B/C.

For this profile the orchestrator automatically uses `active-token` mode:
exact upstream, candidate weight=0, and candidate weight=1, all with affinity
disabled. Compare B and C by stage, focusing on Prefill queue time, TTFT p95/p99 and
`prefill_backend_balance` in each summary. Candidate health samples also expose
the instantaneous `active_tokens`, `active_kv_cache`, priority and selection
count for every Prefiller. Do not use cache-hit rate as the success metric;
near-zero hits are expected because every prompt is unique. B and C should be
close because this profile intentionally supplies no affinity key.

Before interpreting latency, verify that the workload actually created the
required lifecycle mixture:

```bash
jq '.proxy_prefill_load_balance_measurement | {
  lifecycle_samples,
  lifecycle_overlap_sample_rate,
  lifecycle_skew_sample_rate,
  shadow_heap_divergence_sample_rate,
  active_tokens_total_p50,
  estimated_waiting_kv_total_p50
}' results/runs/<timestamp>-abc/candidate-off/summary.json
```

`estimated_waiting_kv_total_p50` uses `max(active_kv_cache-active_tokens, 0)`;
this is valid for this proxy path because it reserves equal compute and KV
scores for each request. Lifecycle rates use only samples with non-zero
Prefiller KV load. `lifecycle_overlap_sample_rate` reports loaded snapshots
containing both running Prefill and waiting-KV work. `lifecycle_skew_sample_rate`
requires at least two loaded Prefillers to have different `Q/(Q+W)` ratios.
`shadow_heap_divergence_sample_rate` is the strongest quick check: for each
sample it computes the baseline and candidate heap choices from the same
candidate state and reports how often their selected Prefiller differs. It is
a sampled counterfactual, not the actual cross-run route difference, but a
zero value means the profile still did not expose the intended scoring choice.

This is a mechanism-validation topology, not the final production claim. If
it produces a repeatable A/B difference, rerun the comparison with
`BENCHMARK_DECODER_COUNT=4` on the normal 4P4D topology. A difference only in
the two-Decoder run proves that the lifecycle signal can affect routing under
Decode backpressure; it does not prove a material benefit in the production
capacity ratio.

### `shared-prefix-capacity.json`

The prefix-locality profile builds a corpus of 32 groups with 8 prompts per
group:

```text
32 groups × 8 prompts = 256 measured prompts
```

Within one group, all eight prompts share the same 2,048-token system prompt
and have a different 256-token question. Session headers are disabled for this
profile, so the test isolates cross-session prefix reuse and prefix-aware
routing rather than session-key routing. Every request has `max_tokens=32` and
`temperature=0.0`.

The cache-fill phase sends one representative prompt per group (32 requests).
A following `prefix-probe` stage sends one different, previously unseen
question per group. This keeps a shared prefix resident on one Prefiller before
the probe: baseline has roughly a 1/4 chance of returning to that node, while
prefix-aware routing should return to the owning node. The remaining corpus is
replayed by the QPS stages. The probe is included in measured results and is
reported separately under `per_stage.prefix-probe`.

The cache-fill and prefix-probe requests are scheduled at 10 requests per
second. The measured QPS stages use deterministic Poisson inter-arrival offsets
and a maximum of 64 in-flight requests:


| Stage    | Target rate | Duration | Expected requests |
| -------- | ----------- | -------- | ----------------- |
| `qps-2`  | 2 req/s     | 30 s     | about 60          |
| `qps-5`  | 5 req/s     | 30 s     | about 150         |
| `qps-10` | 10 req/s    | 30 s     | about 300         |
| `qps-20` | 20 req/s    | 30 s     | about 600         |


The Poisson generator keeps only arrivals whose offset is less than the stage
duration, so the exact request count is allowed to vary around the expectation.
With the checked-in seed, the current generated counts are 60, 136, 309, and
618 respectively (1,123 QPS requests, plus 32 prefix probes and 32 cache-fill
requests, for 1,187 total requests).
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
  -> wait until all four Prefillers are idle
  -> prime and verify one isolated cache probe on every Prefiller
  -> reset through the proxy and verify every direct Prefiller is cold
  -> final reset removes the verification probes
  -> before metrics snapshot
  -> cache-fill
  -> after-cache-fill metrics snapshot
  -> prefix-probe (shared-prefix only)
  -> measured turns or QPS stages
  -> final metrics snapshot
```

The Kubernetes A/B/C orchestrator uses eight sacrificial system warm-up
requests for every group. These requests exercise the HTTP path and model
runtime, but are not included in the workload summary. It then waits for all
four Prefillers to report zero running and waiting requests. Every Prefiller is
primed directly with an isolated prompt, the second request must report cached
tokens, a proxy fan-out reset is issued, and the next direct request must report
zero or omitted cached tokens. A final reset removes those probes before metric
snapshots and the measured workload. Any missing backend, reset failure, drain
timeout, or post-reset cache hit aborts the group instead of producing a
comparison from contaminated state.

The default failure action is to abort without restarting the model. For an
explicit slow fallback, set `RESET_FAILURE_ACTION=restart`; the orchestrator
then restarts the Prefill StatefulSet and retries that group once, but only
when `reset-validation.json` shows that reset verification was the failure.

The raw request JSONL preserves an omitted `cached_tokens` value as `null`.
For aggregate metrics it is treated as a cache miss (`0`), matching the vLLM
behavior observed with prompt-token details enabled. Consequently,
`cached_token_request_rate` uses every successful request in its denominator,
`cached_token_ratio` uses every prompt token, and `client_computed_tokens`
includes cold requests.

For `session-long`, turn 0 is `cache-fill` and later turns are `measure`. The
assistant messages in later turns are fixed in the workload instead of copying
live model output, so all groups receive byte-identical prompts.

For `shared-prefix`, one representative prompt per group is sent during
`cache-fill`, followed by one previously unseen same-prefix prompt in
`prefix-probe`. Measured QPS stages then replay the frozen corpus with
deterministic Poisson arrival offsets. Session headers are disabled so this
workload isolates prefix routing. The QPS stages are a continuous ramp and
therefore share cache state; use separate runs for independent QPS points.

## Capacity sizing

Size a shared-prefix workload relative to aggregate Prefiller KV capacity:

```text
resident tokens ≈ groups × (system tokens + prompts per group × question tokens)
working-set ratio = resident tokens / aggregate Prefiller KV token capacity
```

The formula describes ideal prefix-affine placement. Without affinity, the
shared system prefix can be duplicated on multiple Prefillers, increasing the
effective working set and causing eviction earlier. Record KV capacity and
usage from metrics; do not assume the default 32-group profile creates
capacity pressure on every deployment.

Run at approximately 25%, 50%, 75% and 110% to cover ample capacity, normal
pressure, cache stress and eviction. Record the observed vLLM KV capacity and
usage; do not infer it only from model configuration.

## End-to-end Kubernetes runbook

The following is the recommended execution order for a fresh benchmark. The
P/D workloads must be running before the A/B/C harness starts; the harness
creates only a temporary CPU-only Benchmark Pod and changes the proxy between
groups.

### 1. Deploy or verify the P/D services

If the fixed Qwen3-32B topology is not running yet:

```bash
cd deploy/kubernetes/qwen3-32b-4p4d-tp2

export NAMESPACE='qwen-pd'
export VLLM_IMAGE='your-tested-vllm-ascend-image'
export PREFILL_NODE='your-prefill-node'
export DECODE_NODE='your-decode-node'
export MODEL_HOST_PATH='/models'
export MODEL_PATH='/models/Qwen/Qwen3-32B'
export SERVED_MODEL_NAME='qwen3-32b'
export NPU_RESOURCE='huawei.com/Ascend910'
export NIC_NAME='eth0'

bash deploy.sh all
bash deploy.sh status
cd ../..
```

If the P/D services are already healthy, only verify the current context and
Pods:

```bash
kubectl config current-context
kubectl -n qwen-pd get pods,services -o wide
```

The expected in-cluster endpoints are `pd-prefill-0.pd-prefill:7100`,
`pd-decode-0.pd-decode:7200`, and `pd-proxy:8000`. No `port-forward` is needed
for the Pod-mode benchmark.

### 2. Set benchmark variables

Run the harness from the repository root. `MODEL` must match
`SERVED_MODEL_NAME`, and `VLLM_IMAGE` should be the same tested image used by
the P/D deployment:

```bash
cd /path/to/vllm-ascend-kv-aware-proxy

export NAMESPACE='qwen-pd'
export MODEL='qwen3-32b'
export PREFILL_NODE='your-prefill-node'
export VLLM_IMAGE='your-tested-vllm-ascend-image'
```

Override the default Service DNS names only if the deployment uses different
names:

```bash
export IN_CLUSTER_TOKENIZER_URL='http://custom-prefill:7100'
export IN_CLUSTER_BASE_URL='http://custom-proxy:8000'
```



### 3. Run the smoke check

Use the small profile to validate Pod scheduling, image dependencies, in-cluster
DNS, tokenizer access, proxy rollout, and result copying:

```bash
KEEP_BENCHMARK_POD=true \
  bash benchmarks/run_abc_experiment.sh benchmarks/profiles/smoke.json
```

The script checks that the image contains Python `requests` and `tar`. Once a
successful smoke run has been inspected, remove its retained resources:

```bash
cd deploy/kubernetes/qwen3-32b-4p4d-tp2
bash deploy.sh cleanup-benchmark
cd ../..
```



### 4. Run the production A/B/C benchmark

The session-locality profile is the default production starting point:

```bash
bash benchmarks/run_abc_experiment.sh \
  benchmarks/profiles/session-affinity.json
```

The orchestrator creates one Benchmark Pod, generates one tokenizer-verified
workload inside it, and runs the following groups in order:

```text
baseline       = baseline proxy, KV-aware=false
candidate-off  = candidate proxy, KV-aware=false
candidate-on   = candidate proxy, KV-aware=true
```

For every group it sends eight infrastructure warm-up requests, drains all
Prefillers, verifies the proxy fan-out reset against four direct cache probes,
runs the workload, and then proceeds to the next group. The workload itself is
never regenerated between groups.

Pod mode automatically supplies the four stable Prefill roots and all eight
4P4D `/metrics` endpoints. Shared-prefix groups additionally require a cold
cache-fill and a non-empty `prefix-probe` stage. Each group writes
`reset-validation.json`,
`reset-probe-requests.jsonl`, and `validity.json`. `comparison.json` and the
HTML report are valid for performance claims only when all three group validity
checks pass.

For the shared-prefix/QPS experiment, run:

```bash
bash benchmarks/run_abc_experiment.sh \
  benchmarks/profiles/shared-prefix-capacity.json
```



### 5. Add metrics sampling when needed

Pod mode already samples all four Prefill and four Decode endpoints. Additional
debugging endpoints passed manually should use in-cluster DNS names, for
example:

```bash
bash benchmarks/run_abc_experiment.sh \
  benchmarks/profiles/session-affinity.json \
  --metrics-url http://some-extra-endpoint:9000/metrics
```



### 6. Inspect results

Each run is copied to a unique directory under `results/runs/`:

```text
<timestamp>-abc/
  workload.jsonl
  workload.jsonl.manifest.json
  baseline/
  candidate-off/
  candidate-on/
  comparison.json
  report.html
```

Each group directory also contains:

```text
reset-validation.json       # four-node fan-out and cold-probe evidence
reset-probe-requests.jsonl  # per-Prefiller prime/hit/miss observations
validity.json               # required checks and final valid flag
```

Check the token verification and comparison summary:

```bash
python -m json.tool results/runs/<timestamp>-abc/workload.jsonl.manifest.json
python -m json.tool results/runs/<timestamp>-abc/comparison.json
open results/runs/<timestamp>-abc/report.html  # macOS; open the file directly elsewhere
```

The manifest must contain `"token_count_verified": true`. Comparison also
checks that all three group `config.json` files contain the same workload
SHA-256 and embeds each group validity result. `comparison.json.valid` must be
`true`; otherwise the HTML report treats improvements as diagnostic only. The
harness packs `/results` into a gzip archive inside the Benchmark Pod, copies
that archive locally, extracts it, then generates `report.html`.
To render a report manually from an existing result directory:

```bash
python benchmarks/render_benchmark_report.py results/runs/<timestamp>-abc
```

The self-contained report highlights the A/B/C overview, B → C KV-aware
improvements, latency/cache/throughput metrics, and links to the raw JSON
artifacts. It does not load JavaScript or assets from an external network.

### 7. Recover or clean up a failed run

Failed runs retry the result copy once after 10s (`RETRY_DELAY_SECONDS`).
If that still fails, the harness retains the Benchmark Pod and ConfigMap and
prints commands like these for inspection and artifact recovery:

```bash
kubectl -n qwen-pd get pods -l app.kubernetes.io/name=pd-benchmark
kubectl -n qwen-pd describe pod <benchmark-pod>
kubectl -n qwen-pd exec -it <benchmark-pod> -- bash
kubectl -n qwen-pd exec <benchmark-pod> -- tar -C /results -czf /tmp/benchmark-results.tar.gz .
kubectl -n qwen-pd cp <benchmark-pod>:/tmp/benchmark-results.tar.gz results/runs/<timestamp>-abc/benchmark-results.tar.gz
tar -xzf results/runs/<timestamp>-abc/benchmark-results.tar.gz -C results/runs/<timestamp>-abc
rm -f results/runs/<timestamp>-abc/benchmark-results.tar.gz
```

After collecting diagnostics:

```bash
cd deploy/kubernetes/qwen3-32b-4p4d-tp2
bash deploy.sh cleanup-benchmark
cd ../..
```



## Advanced A/B/C options

The experiment directory contains the shared workload and manifest, one fixed
directory per group, and `comparison.json`. Comparison generation fails if the
three recorded workload SHA-256 values are not identical.

Use `GROUP_ORDER` to balance order across repetitions, for example:

```bash
GROUP_ORDER='candidate-off candidate-on baseline' \
  bash benchmarks/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

For normal profiles, B versus C isolates affinity. For `load-balance`,
automatic `active-token` mode makes A exact upstream, B candidate weight=0 and
C candidate weight=1; B versus C then isolates the active-token lifecycle.
Setting `ABC_COMPARISON_MODE=affinity-guard` makes all three groups run the
candidate source with KV-aware routing on: A and B keep the guards disabled
(A versus B is the same-policy noise control) while C enables
the guards (override the defaults 1.5 / 3 by exporting
`AFFINITY_OVERLOAD_FACTOR` and `AFFINITY_MISS_UNBIND_THRESHOLD`); A versus C
then isolates the guard mechanisms. Export `AFFINITY_CACHE_DISCOUNT_ALPHA`
(default 0 = off) to also enable cache-discounted reservations on C:
affinity-hit compute reservations are scaled by the per-binding EMA of
observed cached/prompt token ratio (KV pressure stays full-cost since the
decoder needs the complete prompt KV either way), so the overload guard
compares real compute instead of raw prompt size.

A typical overload-guard experiment pairs affinity-guard mode with a Zipf
profile. Without the cache discount, reservations over-state the load of
highly-cached bound nodes, so start the factor at 3 or higher; with the
discount enabled the accounting tracks real compute and smaller factors
become meaningful:

```bash
ABC_COMPARISON_MODE=affinity-guard \
AFFINITY_OVERLOAD_FACTOR=3 \
AFFINITY_MISS_UNBIND_THRESHOLD=3 \
AFFINITY_CACHE_DISCOUNT_ALPHA=0.3 \
MODEL=qwen3-32b PREFILL_NODE=<node> \
  bash benchmarks/run_abc_experiment.sh benchmarks/profiles/shared-prefix-zipf.json
```

When judging guard runs, look beyond the latency deltas: high
`session/prefix_affinity_stats.overflows` with a falling cached/prompt ratio
in `prefill_cache_stats_by_source` for the session/prefix sources means the
guard is evicting healthy bindings rather than relieving a real hot spot.

Set a different integer `WORKLOAD_SEED` on every repetition; it overrides only
the generated workload and is recorded in `metadata.json` and the workload
manifest. Start with the first three cyclic Latin-square rows below so every
group occupies each execution position once. If the result is not decisive or
needs full six-order evidence, run the remaining three reverse rows:

| Run | `GROUP_ORDER` | Example `WORKLOAD_SEED` |
| --- | --- | --- |
| 1 | `baseline candidate-off candidate-on` | `202608101` |
| 2 | `candidate-off candidate-on baseline` | `202608102` |
| 3 | `candidate-on baseline candidate-off` | `202608103` |
| 4 | `baseline candidate-on candidate-off` | `202608104` |
| 5 | `candidate-on candidate-off baseline` | `202608105` |
| 6 | `candidate-off baseline candidate-on` | `202608106` |

Every Pod-mode group records `proxy-identity.json` containing the expected and
actual source SHA-256, variant, active-token weight, KV-aware flag, Pod UID and
image ID. The run aborts before sending traffic if they differ. A global
`PROXY_SOURCE_PATH` is rejected because it can accidentally make all groups
run the same source; use `BASELINE_PROXY_SOURCE_PATH` and
`CANDIDATE_PROXY_SOURCE_PATH` for intentional overrides.

Aggregate valid repetitions with paired bootstrap confidence intervals:

```bash
python benchmarks/compare_repetitions.py \
  --comparison B_vs_C \
  results/runs/<run-1>-abc results/runs/<run-2>-abc results/runs/<run-3>-abc
```

The aggregate output reports whether all six group orders were covered and
whether every repetition used a distinct workload seed.

## Re-running the legacy affinity experiments

The 2026-08-06/07 affinity runs predate the proxy-identity verification, the
same-policy noise control and the affinity instrumentation, so they cannot
support performance claims. `run_affinity_repetitions.sh` re-runs
`session-affinity.json` and `shared-prefix-capacity.json` on the hardened
harness — six repetitions per profile with the full Latin-square order rotation
and distinct seeds — then aggregates and applies the noise-gate verdict:

```bash
MODEL=qwen3-32b PREFILL_NODE=<node> \
  bash benchmarks/run_affinity_repetitions.sh
# or pick profiles / fewer repetitions explicitly:
REPETITIONS=3 bash benchmarks/run_affinity_repetitions.sh \
  benchmarks/profiles/shared-prefix-capacity.json
```

Judgment standard (same as the active-token analysis): the per-run |A vs B|
relative delta defines the noise band for each metric because B is
decision-identical to A; an effect is claimed only when the mean |A vs C|
delta exceeds that band with a consistent direction across repetitions, and
the paired bootstrap CI from `compare_repetitions.py` excludes zero.
`judge_affinity_repetitions.py` prints this verdict per metric and can also be
run standalone on any set of completed experiment directories.

For compatibility, the previous local/port-forward execution path remains
available:

```bash
BENCHMARK_EXECUTION_MODE=local \
BASE_URL=http://127.0.0.1:8000 \
TOKENIZER_URL=http://127.0.0.1:7100 \
MODEL=qwen3-32b PREFILL_NODE=<node> \
  bash benchmarks/run_abc_experiment.sh benchmarks/profiles/session-affinity.json
```

Local mode does not know the stable address of every Prefiller and therefore
cannot perform the four-node reset verification automatically. Its
`validity.json` and final comparison are intentionally marked invalid; use this
path for debugging only, not for performance claims. Use Pod mode for a valid
4P4D A/B/C comparison.

The workload design follows the public multi-turn shape described by Higress
and the shared-prefix, cache-capacity and staged-load methodology published by
`llm-d-benchmark` and `inference-perf`.
