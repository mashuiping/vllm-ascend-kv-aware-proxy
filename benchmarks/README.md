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
| `load-balance-active-tokens.json` | Unique 512/4,096/8,192-token prompt classes, continuous Poisson overlap at 8/16/24/32 QPS, concurrency 128 | A/B Prefill active-token load balancing |
| `smoke.json`                  | 2 sessions × 2 turns                                                                            | Offline generator and test smoke case     |


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

This profile is designed to expose the transient Prefill load signal in B
(`candidate-off`) versus A (`baseline`). It is not a cache-locality test:
every request has a unique prompt, session headers are disabled, and the
primary comparison is Prefill queue time, TTFT tail latency and per-node load
balance.

The workload cycles through three prompt-size classes (512, 4,096 and 8,192
system tokens, with correspondingly different user inputs) while requests
arrive according to deterministic Poisson offsets:

| Stage | Target rate | Duration | Concurrency |
| ----- | ----------- | -------- | ----------- |
| `qps-8` | 8 req/s | 45 s | 128 |
| `qps-16` | 16 req/s | 45 s | 128 |
| `qps-24` | 24 req/s | 45 s | 128 |
| `qps-32` | 32 req/s | 45 s | 128 |

With the checked-in seed, these stages generate 352, 690, 1,041 and 1,491
requests respectively (3,574 total). The longer stages provide enough samples
to distinguish a repeatable change from normal run-to-run noise. The 32 QPS
stage adds a stronger overlap point, and four-token outputs keep the profile
focused on Prefill rather than Decode. The exact count is deterministic, while
the inter-arrival offsets remain Poisson-distributed within each stage.

The generated system and user messages are unique per request, so cache hits
should remain negligible. The different prompt sizes create overlapping
Prefill work; B can then avoid selecting a node whose active Prefill compute
load is high even when its KV load alone appears low. C is still run by the
standard A/B/C script, but its affinity result is only a reference for this
profile.

Run it with the normal orchestrator:

```bash
bash benchmarks/run_abc_experiment.sh \
  benchmarks/profiles/load-balance-active-tokens.json
```

Compare A and B by stage, focusing on Prefill queue time, TTFT p95/p99 and
`prefill_backend_balance` in each summary. Candidate health samples also expose
the instantaneous `active_tokens`, `active_kv_cache`, priority and selection
count for every Prefiller. Do not use cache-hit rate as the success metric;
near-zero hits are expected because every prompt is unique. B and C should be
close because this profile intentionally supplies no affinity key.

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

Failed runs retain the Benchmark Pod and ConfigMap. The harness prints commands
like these for inspection and artifact recovery:

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
