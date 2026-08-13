# Limitations and safety boundaries

## A live binding currently wins over load

Session and prefix hits remain on their bound Prefiller while it is live and not
draining; there is no load-based spillover. An overload escape valve was tried
and removed in 0.2.x: instantaneous priority comparison only produced
false-positive escapes, and experiments showed a genuinely backlogged hot
Prefiller does not materialise while affinity keeps its cache hot (see
docs/design.md). A hot session or prefix can still create a hotspot in
principle. Always include a hot-key negative test, watch tail latency and
bound client concurrency.

## Affinity is an in-memory hint

LRU state is not persisted. Proxy restart, manual prefix-cache reset, Prefiller
removal and eviction discard bindings. This affects performance, not request
correctness: a miss returns to load-based routing and recomputes KV.

## Client identifiers are routing hints, not authentication

All supported affinity fields are client-controlled unless an upstream gateway
removes and replaces them. A caller can therefore choose a hot key and create a
localized load hotspot. Restrict direct access, overwrite
`X-Session-Affinity` at a trusted edge when it is used as an override, and do
not use an affinity match as an authorization decision.

The proxy hashes identifiers and rejects values larger than 256 UTF-8 bytes,
but it does not currently include an authenticated tenant namespace or secret
HMAC in the key. Deployments where unrelated tenants can choose identical IDs
should add a trusted tenant scope before hashing.

Some HTTP gateways reject or drop underscore-containing header names such as
`session_id`. They are accepted for Cline/Roo/Pi compatibility, but integrations
should prefer `Session-ID`, `X-Session-ID`, or `X-Session-Affinity` where they
control the contract.

## Native identifiers have different scopes

Codex `Thread-ID` is preferred over its broader `Session-ID`. Claude Code
subagents use the combination of session and agent ID so parallel agents do not
all share one hot binding. `X-Claude-Code-Parent-Agent-ID` is not a routing key.
Generic `X-Client-Request-ID`, request IDs, trace IDs, user IDs, and tenant IDs
are deliberately ignored even though a particular client may sometimes place a
stable value in one of them.

## Reusable-prefix gate depends on Prefiller details

`--enable-reusable-prefix-affinity-gate` only takes effect when Prefillers
report both `cached_tokens` and `created_cache_tokens` in `prompt_tokens_details`.
The upstream `/v1/completions` path currently exposes `cached_tokens` only and
falls back to the optimistic-bind branch with a warning. Older Prefiller builds
that omit `prompt_tokens_details` when `cached_tokens == 0` also fall back to a
bind. Gate behavior is an opt-in; without the flag, today's bind-everything
behavior is preserved.

## Prefix hashing is conservative

Only plain text requests are hashed. Multimodal inputs, structured content and
tool-related requests skip prefix affinity because their rendered token stream
is model specific. An explicit stable session ID may still provide affinity for
those requests; deployments should validate whether that is appropriate for
their application.

## Character length is not token length

`--prefix-hash-chars` slices canonical text by characters. It does not know the
backend tokenizer or KV block boundary. Tune it from observed `cached_tokens`
and TTFT, not from a model name alone.
