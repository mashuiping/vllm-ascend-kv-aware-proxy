# Routing design

## Goal

Increase reusable Prefiller KV-cache hits for multi-turn and shared-prefix
traffic without changing Decoder load balancing or guessing token prefixes for
request types the proxy cannot canonicalize safely.

## Route order

With KV-aware routing enabled, Prefiller selection follows:

```text
stable session key
  -> text prefix hash
      -> existing load heap
```

A session identifier is an explicit locality contract, so it wins over inferred
prefix locality. The Decoder remains selected by the existing active-token
load heap because Prefiller prefix KV locality does not make a particular
Decoder more valuable.

Bindings are committed only after a successful Prefill response. A failed
Prefill therefore cannot claim ownership of KV that may not exist. Bindings to
removed or draining Prefillers are invalidated, and both maps use bounded LRU
eviction to constrain client-controlled cardinality.

## Session keys

Accepted fields are deliberately session-scoped:

```text
X-Session-ID
X-Claude-Code-Session-ID
session_id
session_params.session_id
```

Headers take precedence because a gateway can add routing metadata without
rewriting an OpenAI-compatible JSON payload. User and tenant IDs are excluded
because they can pin many unrelated conversations to one Prefiller. Request and
trace IDs are excluded because they change for every request.

The field name is included in the internal key namespace so values from
different contracts do not collide accidentally.

## Prefix keys

The proxy hashes only canonical forms it can reproduce deterministically:

- a string `prompt`;
- chat messages with string `role` and string `content`.

The hash namespace includes a canonicalization version, endpoint kind, model,
OpenAI `prompt_cache_key`, and the configured prefix slice. Only a fixed-size
BLAKE2s digest enters shared scheduler memory; prompt text is not retained as an
LRU key.

Tool definitions, tool calls, structured content and multimodal inputs are
skipped. Their actual token prefix depends on model-specific templates, so a
text-only approximation can create false locality.

## Two-phase Prefiller load

The upstream priority formula is retained:

```text
Prefiller priority = active_tokens + 0.3 * active_kv_cache
Decoder priority   = active_tokens
```

At Prefiller selection, the candidate reserves both compute load in
`active_tokens` and KV-transfer pressure in `active_kv_cache`. On successful
Prefill completion it releases compute load, while KV pressure remains until
the Decoder starts consuming the transfer or the request is cleaned up.

This restores the lifecycle used before the shared-scheduler refactor while
avoiding a new scheduler field or a parallel release mechanism.

## Opt-in behavior

Session and prefix affinity require `--enable-kv-cache-aware-routing`. Without
the flag, extraction returns no affinity keys and selection falls back to the
normal heap.

The restored Prefiller `active_tokens` lifecycle is independent of that flag.
Consequently, candidate-off is not byte-for-byte equivalent to the exact
upstream baseline; this is why experiments include three groups rather than
only an on/off comparison.

