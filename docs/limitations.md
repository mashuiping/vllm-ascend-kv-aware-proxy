# Limitations and safety boundaries

## A live binding currently wins over load

Session and prefix hits remain on their bound Prefiller while it is live and not
draining. There is no overload spillover in the minimal candidate. A hot session
or prefix can therefore create a hotspot. Always include a hot-key negative
test, watch tail latency and bound client concurrency.

## Affinity is an in-memory hint

LRU state is not persisted. Proxy restart, manual prefix-cache reset, Prefiller
removal and eviction discard bindings. This affects performance, not request
correctness: a miss returns to load-based routing and recomputes KV.

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

## The initial K8s deployment is hardware-specific

The supplied manifests assume two 8-NPU Ascend 910B2 nodes, four TP2 instances
per role, specific driver hostPaths and Mooncake Ascend Direct. They require a
privileged security context. Review and adapt them before use in another
cluster.

## Experimental support policy

The repository is not an official vllm-ascend distribution. Pin the exact
vLLM, vllm-ascend, image, CANN and driver combination used by an experiment.
Re-run correctness and performance tests after every upstream rebase.

