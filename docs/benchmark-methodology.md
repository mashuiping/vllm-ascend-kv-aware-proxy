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

- Keep P/D Pods, model, image, scheduler arguments and hardware unchanged
  within every A/B/C comparison. A diagnostic profile may expose only a subset
  of already-running backends through the Proxy, but A, B and C must use the
  same subset and the result must be labelled as a mechanism-validation
  topology rather than a production-capacity comparison.
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
| `load-balance` | Heterogeneous Prefill load plus controlled Decode backpressure for A/B active-token lifecycle scoring |
| `short` | Negative control below useful cache granularity |
| `one-shot` | Measures overhead and LRU pollution without reuse |
| `hot-key` | Exposes hotspot risk without spillover |

Start with low concurrency to prove cache behavior, then use a concurrency
ladder appropriate for the deployed capacity. Short output lengths emphasize
Prefill and TTFT for locality profiles. The active-token lifecycle profile is
the deliberate exception: it uses longer output and fewer Proxy-visible
Decoders to retain completed Prefills in the wait-for-first-Decoder-chunk
phase. For legacy generated workloads, `prefix-words` is only an approximation;
always report actual server `prompt_tokens` from results.

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

The executable runbook lives in [`benchmarks/README.md`](../benchmarks/README.md).
It documents the Pod and local modes, workload generation, reset validation,
result artifacts and recovery procedures. A result is publishable only when
the recorded validity checks pass for every comparison group.

Review and redact artifacts according to [`results/README.md`](../results/README.md)
before publishing them.
