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

The extractor separates conversation-scoped identifiers from request, user,
tenant, and telemetry identifiers. Its priority order is:

```text
trusted routing override: X-Session-Affinity
  -> conversation-specific native ID: Codex Thread-ID, Claude session+agent
  -> native client ID: OpenCode session, Cline/Roo task
  -> generic session header
  -> top-level or nested JSON session/task field
```

Headers take precedence because a gateway can add routing metadata without
rewriting an OpenAI-compatible JSON payload. `X-Session-Affinity` is an explicit
routing override and should only be trusted from a controlled client or
gateway. User and tenant IDs are excluded because they can pin unrelated
conversations to one Prefiller. Request and trace IDs are excluded because they
normally change for every request.

Accepted values are non-empty scalars no larger than 256 UTF-8 bytes. The
selected identity is converted to a versioned, fixed-size BLAKE2s digest before
it enters the shared scheduler. Equivalent header and body aliases therefore
reuse a binding without storing the raw identifier. If multiple non-equivalent
identifiers are present, the highest-priority value wins and a debug conflict
record names the sources without logging their values.

### Client compatibility

This table records the public documentation and source contracts checked for
the implementation. Header names are case-insensitive.

| Client | Observed stable field | Proxy behavior | Evidence |
| --- | --- | --- | --- |
| OpenCode | `X-OpenCode-Session`; otherwise `X-Session-Affinity` and `X-Session-ID` | Accept all three | [request construction](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/llm/request.ts) |
| Claude Code | `X-Claude-Code-Session-ID`, optional `X-Claude-Code-Agent-ID` and parent agent ID | Use session+agent for a subagent; parent is metadata only | [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol#request-headers) |
| Codex | `Session-ID`, `Thread-ID`; also sets `X-Client-Request-ID` from the thread | Prefer `Thread-ID`; do not use the ambiguous request-ID alias | [Codex request headers](https://github.com/openai/codex/blob/main/codex-rs/codex-api/src/requests/headers.rs) |
| Pi | Provider-dependent `Session-ID`, `session_id`, `X-Session-ID`, or `X-Session-Affinity` | Accept stable session forms; ignore `X-Client-Request-ID` | [Pi compatibility formats](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/types.ts) |
| Cline | `X-Task-ID`; Codex provider uses `session_id` | Accept both | [Cline request headers](https://github.com/cline/cline/blob/main/sdk/packages/llms/src/providers/request-headers.ts) |
| Roo Code | Interface documents `X-Roo-Task-ID`; current OpenAI providers send `session_id` from the task ID | Accept both; do not rely on the documented header being present | [interface](https://github.com/RooCodeInc/Roo-Code/blob/main/src/api/index.ts), [provider](https://github.com/RooCodeInc/Roo-Code/blob/main/src/api/providers/openai-native.ts) |
| Gemini CLI | Internal session UUID; Code Assist embeds `request.session_id` | Accept nested body field; no generic Gemini session header assumed | [Code Assist converter](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/code_assist/converter.ts) |
| Cursor/unknown | No verified public stable field | Continue to generic session fields, then prefix/load fallback | — |

Body compatibility covers `session_id`, `sessionId`, `task_id`, `taskId`,
`session_params.session_id`, and Gemini Code Assist's `request.session_id`.
OpenAI `prompt_cache_key` remains part of the separate prefix fingerprint: it
groups cache-compatible prompts but is not assumed to identify one conversation.

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

## Reusable-prefix affinity gate

With `--enable-reusable-prefix-affinity-gate`, the proxy decides whether to
commit session/prefix affinity from the Prefill response:

```text
reusable_prefix_tokens =
    prompt_tokens_details.cached_tokens
  + prompt_tokens_details.created_cache_tokens
```

Action of the gate:

| Condition | Action |
| --- | --- |
| Gate off | Bind session/prefix as today |
| Gate on, details complete, `reusable > 0` | Bind |
| Gate on, details complete, `reusable == 0` | Do not bind |
| Gate on, details missing or incomplete | Warning log + bind (optimistic default) |

`reusable == 0` does not clear an existing binding; sticky routing is
preserved so the Prefiller can rebuild cache after an LRU miss. Session and
prefix affinity share the same latch. The gate is a no-op without
`--enable-kv-cache-aware-routing`. Prefillers should run with
`--enable-prompt-tokens-details` (and prefix caching).
