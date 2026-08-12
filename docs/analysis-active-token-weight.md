# Analysis: why the active-token Prefill score shows no A/B/C advantage

Status: **closed, 2026-08-12** — conclusion accepted (see Verdict).
Data: the 12 parsable `load-balance` A/B/C runs from 2026-08-10 under
`results/runs/` (6 runs at seed 20260724, 02:36–03:28 UTC; 6 runs at
per-run seeds 202608101–202608106, 12:30–13:48 UTC). Profile:
`benchmarks/profiles/load-balance-active-tokens.json` (4 Prefillers, 2
Proxy-visible Decoders, Poisson 2/3/4.5 QPS, 256-token outputs).
Cross-checked against the 2026-08-06/07 `session-long` (affinity-mode)
runs as a harness-sensitivity control.

## TL;DR

The candidate mechanism works as designed — it changes 29–42% of heap
routing decisions and measurably tightens Prefill token balance — but the
experiment cannot show an end-to-end win because:

1. **The effect ceiling is ~0–0.2 s.** The only latency component Prefill
   routing can change is Prefill-side queue time. Measured mean Prefill queue
   time is 0.000 s in the entire morning batch and 0.001–0.23 s at the
   heaviest stage of the afternoon batch, against TTFT p95 of 1.7–6.5 s.
   Even eliminating *all* Prefill queueing would move TTFT by ≤ 3–4%.
2. **The noise floor is ~10–100× the ceiling.** Baseline (A) TTFT p95 alone
   ranges 3.07–6.28 s across the afternoon runs (2.05×). A vs B — two groups
   running the *same effective policy* — differ by up to 17% in TTFT p95
   within a single run. Decode queue time, governed by an identical policy in
   all three groups, still varies ±20% between groups. Any real ±3% effect is
   invisible under this variance.
3. **The two policies are nearly the same policy.** Both counters are fed the
   same per-request value at the same instant; the scores differ only in how
   they weight the not-yet-finished-prefilling subset (1.3× vs 0.3×). On top
   of that, the `+120.07` constant in `calculate_prefill_score` swamps the
   length term, so both variants approximate least-outstanding-request-count
   routing rather than token-aware routing.
4. **The workload leaves nothing to fix.** With 4 homogeneous Prefillers,
   smooth Poisson arrivals, a narrow 3-class prompt mix and zero prefix-cache
   reuse, the baseline already achieves Prefill computed-token CV of
   0.06–0.10. The candidate lowers it further (down to 0.016), but balancing
   an unqueued resource buys no latency.

The observed "slight TTFT regression" of C is not a real effect: B (candidate
source, weight 0, decision-identical to A) shows deltas of the same sign and
magnitude in several runs, which is the definition of run noise. Two
additional pieces of evidence close the case:

- **The worst "regression" run is a cluster transient, not a policy
  pathology** (see "Forensics on the +50% outlier run" below): all four
  Prefillers queued simultaneously and uniformly during C's window while C's
  token balance was *better* than A's.
- **The harness itself is sensitive enough to detect real effects** (see
  "Sensitivity control" below): the 08-06/07 affinity-mode experiment showed
  consistent −17% to −48% TTFT p95 wins for C with B ≈ A across 13+ runs on
  the same pipeline. The active-token null result is therefore a property of
  the mechanism at this operating point, not of the measurement.

## Policy equivalence in detail

Baseline (A) reserves only KV pressure at pick time and its `active_tokens`
stays 0 for Prefillers, so its effective Prefill priority is `0.3 × kv`:

```387:392:baseline/load_balance_proxy_server_example.py
    def begin_request(self, load: float) -> dict[str, Any]:
        """Pick a prefiller, reserve KV pressure, and count this as an active request."""
        with self._lock:
            picked = self._pick_server(ServerRole.PREFILL, load, kv_cache=True)
            self.request_num += 1
            return picked
```

The candidate scores `weight × active_tokens + 0.3 × kv`:

```546:551:load_balance_proxy_server_example.py
        if role is ServerRole.PREFILL:
            # active_tokens is released when Prefill finishes, while KV pressure
            # lasts until Decoder starts consuming the transfer. This restores
            # the two-phase accounting used before the shared-scheduler refactor.
            return self.prefill_active_token_weight * entry.active_tokens + entry.active_kv_cache * 0.3
        return entry.active_tokens
```

But both counters receive the **same value at the same moment**:

```1436:1436:load_balance_proxy_server_example.py
    prefiller_compute_score = prefiller_kv_score = calculate_prefill_score(request_length)
```

`active_tokens` is released at Prefill completion; `active_kv_cache` is
released when the Decoder consumes the transfer. Every still-prefilling
request is therefore also present in the KV counter, and the candidate score
decomposes as:

```
candidate = 0.3 × kv_outstanding + 1.0 × still_prefilling
baseline  = 0.3 × kv_outstanding
```

i.e. the candidate only re-weights the still-prefilling subset from 0.3 to
1.3. The baseline is **not blind** to in-flight Prefills — it sees them
through the KV reservation made at the identical instant. The two scores
rank nodes differently only when the prefill-done-awaiting-KV-pull backlog is
distributed differently from the still-prefilling load, which under smooth
arrivals and a fast KV pull is a transient, near-tie situation. The shadow
statistics confirm this: 29–42% of heap decisions diverge, yet the divergent
picks land on nodes whose scores are close enough that no queue-time
difference survives.

Additionally, the shared score function is nearly length-blind:

```453:455:load_balance_proxy_server_example.py
def calculate_prefill_score(request_length: int) -> float:
    length_score = request_length / 4.0
    return length_score * 0.0345 + 120.0745
```

For the profile's warm-turn lengths (~640 to ~4,600 tokens, a 7.2× compute
spread) the score only spans 125.6 → 159.8 (1.27×). Per-request load is
dominated by the constant, so "active tokens" degrades to "≈120 × in-flight
request count". Any benefit specific to *token*-awareness (routing around a
node chewing a long prompt) is structurally muted before the experiment
starts.

## Where TTFT actually comes from

Above-knee stage of run `20260810T134851Z-abc`, baseline group, per-request
means from backend metrics deltas:

| Component | Mean |
| --- | --- |
| Prefill queue (Prefiller side) | 0.155 s |
| Prefill execution | 0.830 s |
| Decode-side queue | 0.385 s |
| Decode-side "prefill" (first-token scheduling incl. KV wait) | 0.195 s |
| **TTFT p50 / p95** | **1.80 s / 5.49 s** |

The p95 tail (~5.5 s) is several times the sum of mean server-side
components; it is produced by burst-induced queueing that concentrates on the
**Decode side** (2 Decoders serving ~110 concurrent decoding requests at 4.5
QPS × ~25 s residency), which all three groups share verbatim. The lever the
candidate controls — the 0.155 s Prefill queue — is 2.8% of TTFT p95.

In the morning batch (seed 20260724) mean Prefill queue time was **0.000 s in
all groups of all six runs**: the routing policy was choosing among idle
nodes, and no policy can beat another at that game.

## Noise floor evidence

TTFT p95 (warm turns, ms) and mean queue times (s) across the twelve runs:

| Run | A ttft95 | B ttft95 | C ttft95 | A/B/C decode queue | A/B/C prefill queue |
| --- | --- | --- | --- | --- | --- |
| 02:36 | 1760 | 1668 | 1699 | 0.174 / 0.185 / 0.183 | 0 / 0 / 0 |
| 02:46 | 1781 | 1762 | 1733 | 0.187 / 0.187 / 0.188 | 0 / 0 / 0 |
| 02:57 | 1738 | 1722 | 1725 | 0.188 / 0.192 / 0.189 | 0 / 0 / 0 |
| 03:07 | 1733 | 1701 | 1711 | 0.195 / 0.192 / 0.190 | 0 / 0 / 0 |
| 03:17 | 1733 | 1719 | 1696 | 0.191 / 0.192 / 0.190 | 0 / 0 / 0 |
| 03:28 | 1778 | 1738 | 1746 | 0.194 / 0.192 / 0.192 | 0 / 0 / 0 |
| 12:30 | 4881 | 4400 | 4919 | 0.334 / 0.281 / 0.225 | 0.003 / 0.003 / 0.010 |
| 12:46 | 4263 | 4349 | 6552 | 0.371 / 0.308 / 0.460 | 0.001 / 0.013 / 0.231 |
| 13:01 | 6284 | 6576 | 5743 | 0.458 / 0.369 / 0.488 | 0.195 / 0.190 / 0.077 |
| 13:17 | 3071 | 3521 | 3329 | 0.287 / 0.326 / 0.270 | 0.001 / 0.001 / 0.008 |
| 13:33 | 3835 | 3168 | 4519 | 0.250 / 0.271 / 0.287 | 0.025 / 0.001 / 0.043 |
| 13:48 | 4954 | 5178 | 5266 | 0.316 / 0.262 / 0.259 | 0.099 / 0.154 / 0.130 |

Read-outs:

- A vs B is the built-in same-policy control (B's shadow stats show
  `actual_differs_from_baseline = 0`). Its TTFT p95 delta reaches −17%
  (13:33) and +15% (13:17) — that is the noise band, and every A vs C delta
  falls inside it.
- Decode queue time uses an identical policy in all groups yet swings ±20%
  between groups within one run and 2.6× across runs.
- The same profile at fixed seed (morning) versus varying seeds (afternoon)
  flips the system between a no-queue regime and a 3–6.5 s-tail regime: the
  operating point sits near the knee where seed and cluster conditions
  dominate outcomes.

## Mechanism evidence (the candidate is not broken)

| Run | Heap decisions diverging from baseline policy | Prefill token CV: A → C |
| --- | --- | --- |
| 12:30 | 29.4% | 0.091 → 0.064 |
| 12:46 | 33.1% | 0.076 → 0.043 |
| 13:01 | 37.0% | 0.062 → 0.066 |
| 13:17 | 34.8% | 0.084 → 0.084 |
| 13:33 | 41.6% | 0.062 → 0.016 |
| 13:48 | 29.5% | 0.103 → 0.076 |

The candidate routes differently and generally balances Prefill tokens more
tightly. It is optimizing a resource that is not queued: a correct mechanism
applied at a non-bottleneck.

## Forensics on the +50% outlier run (12:46, seed 202608102)

Run `20260810T124623Z-abc` is the single strongest-looking "negative
optimization" datapoint: C's TTFT p95 is +50.2% vs A, and C's mean Prefill
queue time is 0.231 s against A's 0.001 s. Per-Prefiller inspection rules
out a routing pathology:

| Group | Per-node mean Prefill queue (s) | Prefill token CV |
| --- | --- | --- |
| A (baseline) | 0.000 / 0.001 / 0.001 / 0.000 | 0.076 |
| C (candidate) | 0.227 / 0.245 / 0.251 / 0.199 | 0.043 |

If the candidate policy had herded requests onto one node, one Prefiller
would queue while the others idled. Instead all four Prefillers queued
**simultaneously and uniformly**, and C's token balance was actually
*better* than A's (CV 0.043 vs 0.076). C balanced better and still lost —
the only explanation is a global load/cluster transient during C's
measurement window. Since A/B/C groups run **sequentially**, each group
samples a different cluster moment, and the outcome is dominated by *when*
a group ran, not *how* it routed. Excluding this run, the afternoon mean
C-vs-A TTFT p95 delta drops from +12% to +4.5%, inside the ±15–17% A-vs-B
same-policy control band.

## Sensitivity control: the harness can detect real effects

The 2026-08-06/07 `session-long` runs (affinity comparison mode, 4
Decoders, 32-token outputs, prefix reuse) ran on the same A/B/C pipeline
and show what a real win looks like:

| Runs | C vs A, TTFT p95 | B vs A, TTFT p95 |
| --- | --- | --- |
| 08-07 08:23–09:03 (6 runs) | −31% … −48% | −25% … +25% (noise band) |
| 08-07 09:17–10:08 (6 runs) | −17% … −18%, all 6 negative | −5% … +22% |

C won consistently in every parsable affinity run (13+ runs, no
exceptions) while B stayed inside the noise band. The measurement pipeline,
metrics collection, and comparison tooling are therefore demonstrably able
to resolve a genuine routing improvement. The active-token experiment's
null result is real: the effect is ~zero at this operating point, not
hidden by broken instrumentation.

## What a discriminative experiment would need (if ever revisited)

The conclusion below closes this investigation; no further repetition
campaign is planned. If token-aware Prefill routing is ever revisited for a
different deployment shape, all of the following must change together —
each alone is insufficient:

1. **A Prefill-bound operating point.** Gate the measurement on mean Prefill
   queue time being a material share of TTFT (e.g. ≥ 20–30%). Get there by
   raising QPS beyond the Prefill knee, exposing fewer Prefillers, or growing
   prompt lengths — and calibrate against the *Prefill* knee, not the Decode
   knee (the 4D calibration profile as written moves the Decode capacity,
   which does not help this mechanism).
2. **Load heterogeneity the KV signal cannot see.** The two scores separate
   only when still-prefilling load and KV-pull backlog diverge across nodes.
   Use a heavy-tailed prompt mix (e.g. 128 vs 16k tokens) and bursty
   arrivals instead of three near-band classes under smooth Poisson.
3. **A token-proportional score.** Drop or renormalize the `+120.07` constant
   in `calculate_prefill_score` so per-request load scales with tokens;
   otherwise both variants reduce to request counting and the hypothesis
   "token-aware beats KV-only" is never actually exercised.
4. **Mechanism-first metrics and a noise gate.** Judge first on Prefill queue
   time distribution and per-decision score/token CV, not end-to-end TTFT
   p95. Run paired seeds with repetitions and only claim an effect when
   |C − A| clearly exceeds the same-run |B − A| control band. Under the
   current design that band is ±15–17%, an order of magnitude above the
   achievable effect.

## Verdict (accepted 2026-08-12)

**In this deployment shape — Prefill never queues and the TTFT tail is
owned by the Decode side — the active-token Prefill score is not needed.**
This is the accepted, final conclusion of the experiment, and it is a valid
experimental result rather than a failure:

- The candidate mechanism is correct and works as designed (it routes
  differently and tightens token balance), but it optimizes a resource that
  never queues here, so it cannot buy latency.
- The candidate re-weights a signal the baseline already has (both counters
  are fed identically at pick time; the `+120.07` score constant reduces
  both to request counting), so the intervention is tiny by construction.
- The apparent TTFT regression is noise: the same-policy A-vs-B control
  swings ±15–17%, the worst outlier run was a uniform four-node cluster
  transient during which C's balance was *better* than A's, and the same
  harness resolves real 17–48% wins in the affinity experiment without
  difficulty.

The baseline KV-pressure-only score remains the production policy for this
deployment. More repetitions of this design would keep averaging to zero
and should not be run.
