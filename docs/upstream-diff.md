# Candidate versus upstream baseline

The candidate starts from the exact file documented in
[`baseline/README.md`](../baseline/README.md).
Its functional changes are intentionally limited to the following areas.

## Added

- Explicit `--enable-kv-cache-aware-routing` switch.
- Stable session-key extraction with header-over-body precedence.
- Safe text-only prefix canonicalization and fixed-size hashing.
- Bounded shared session and prefix LRUs.
- Session, then prefix, then heap Prefiller selection.
- Binding commit after successful Prefill.
- Binding cleanup when a Prefiller drains, is removed, or prefix cache resets.
- Failure-path rollback for both Prefill compute and KV-transfer pressure.
- Tests covering supported identifiers, safe skips, opt-in behavior, lifecycle,
  affinity precedence, invalid responses, eviction, removal and reset.

## Changed

- Prefiller reservation now charges both the existing `active_tokens` and
  `active_kv_cache` fields.
- The upstream `_pick_server()` evolves into a shared selection/reservation
  primitive. It can use either the normal heap or an affinity-selected key and
  accepts independent active-token and KV-cache loads, so Prefiller and Decoder
  accounting still update through one scheduler path.
- Prefill success releases `active_tokens`; KV pressure remains on its original
  lifecycle.
- Initial and recompute paths carry the same optional affinity keys.
- The CLI and scheduler constructor accept LRU and prefix-hash sizes.

## Unchanged

- The `BackendServer` state structure.
- The upstream Prefiller and Decoder priority formulas.
- Decoder selection and load balancing.
- OpenAI-compatible endpoints, streaming behavior and cached-token forwarding.
- Dynamic backend add/remove protocol.
- Multimodal and tool requests when no explicit session binding exists; they
  fall back to the normal load heap.

## Deliberately deferred

- Hot-prefix replication or multi-node buckets.
- Overload spillover from a live affinity owner (a priority-comparison escape
  valve was tried and removed in 0.2.x; see docs/design.md).
- Prometheus metrics and dashboards.
- Model-specific multimodal/tool prefix canonicalization.
- Distributed affinity state outside the proxy's shared scheduler process.

Keeping these items out of the initial candidate makes measurements easier to
attribute and keeps the eventual upstream review focused.
