import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROXY_PATH = Path(__file__).resolve().parents[1] / "load_balance_proxy_server_example.py"
SPEC = importlib.util.spec_from_file_location("load_balance_proxy_server_example", PROXY_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proxy
SPEC.loader.exec_module(proxy)


def make_scheduler(**kwargs):
    return proxy.SharedProxyScheduler(
        [("127.0.0.1", 8100), ("127.0.0.1", 8101)],
        [("127.0.0.1", 8200)],
        **kwargs,
    )


def test_extract_session_key_supports_common_stable_fields():
    header_cases = [
        ("X-Session-ID", "session"),
        ("X-Claude-Code-Session-ID", "claude"),
    ]
    for name, value in header_cases:
        assert proxy.extract_session_key({name: value}, {}) == f"header:{name.lower()}:{value}"

    headers = {
        "X-Session-ID": "session",
        "X-Claude-Code-Session-ID": "claude",
    }
    assert proxy.extract_session_key(headers, {}) == "header:x-session-id:session"
    assert proxy.extract_session_key(headers, {"session_id": "body"}) == "header:x-session-id:session"

    body_cases = [
        ({"session_params": {"session_id": "nested"}}, "body:session_params.session_id:nested"),
        ({"session_id": "session"}, "body:session_id:session"),
    ]
    for body, expected in body_cases:
        assert proxy.extract_session_key({}, body) == expected

    conflicting_body = {
        "session_id": "primary",
        "session_params": {"session_id": "secondary"},
    }
    assert proxy.extract_session_key({}, conflicting_body) == "body:session_id:primary"
    ignored_identifiers = [
        ({"X-User-ID": "user"}, {}),
        ({"X-Tenant-ID": "tenant"}, {}),
        ({"X-Request-ID": "request"}, {}),
        ({"X-Correlation-ID": "request"}, {}),
        ({}, {"metadata": {"user_id": "user"}}),
        ({}, {"user": "openai-user"}),
        ({}, {"user_id": "user"}),
    ]
    for ignored_headers, ignored_body in ignored_identifiers:
        assert proxy.extract_session_key(ignored_headers, ignored_body) is None


def test_extract_prefix_key_uses_only_supported_text_requests():
    first = {"model": "model", "prompt": "a" * 64 + "first"}
    same_prefix = {"model": "model", "prompt": "a" * 64 + "second"}
    different_model = {"model": "other", "prompt": "a" * 64 + "first"}

    first_key = proxy.extract_prefix_key(first, 64)
    assert first_key == proxy.extract_prefix_key(same_prefix, 64)
    assert first_key != proxy.extract_prefix_key(different_model, 64)
    assert proxy.extract_prefix_key({"prompt": "short"}, 64) is None

    text_chat = {"model": "model", "messages": [{"role": "user", "content": "a" * 64}]}
    assert proxy.extract_prefix_key(text_chat, 32) is not None
    same_chat_prefix = {
        "model": "model",
        "messages": [{"role": "user", "content": "a" * 64 + "different tail"}],
    }
    assert proxy.extract_prefix_key(text_chat, 32) == proxy.extract_prefix_key(same_chat_prefix, 32)
    assert proxy.extract_prefix_key({**text_chat, "tools": []}, 32) is None
    assert proxy.extract_prefix_key({**first, "tools": []}, 32) is None
    assert (
        proxy.extract_prefix_key(
            {"messages": [{"role": "assistant", "content": "a" * 64, "tool_calls": []}]},
            32,
        )
        is None
    )
    assert (
        proxy.extract_prefix_key(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "a" * 64}],
                    }
                ]
            },
            32,
        )
        is None
    )


def test_kv_cache_aware_routing_is_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["load_balance_proxy_server_example.py"])
    assert proxy.parse_args().enable_kv_cache_aware_routing is False
    monkeypatch.setattr(sys, "argv", ["load_balance_proxy_server_example.py", "--enable-kv-cache-aware-routing"])
    assert proxy.parse_args().enable_kv_cache_aware_routing is True

    headers = {"X-Session-ID": "session"}
    body = {"model": "model", "prompt": "a" * 64}

    # Updating the example must not silently reroute existing deployments that
    # do not opt in to the new policy, even when a request contains usable keys.
    assert proxy.extract_affinity_keys(headers, body, enabled=False, prefix_chars=32) == (None, None)

    session_key, prefix_key = proxy.extract_affinity_keys(headers, body, enabled=True, prefix_chars=32)
    assert session_key == "header:x-session-id:session"
    assert prefix_key == proxy.extract_prefix_key(body, 32)


def test_prefiller_active_tokens_and_kv_pressure_have_separate_lifetimes():
    scheduler = make_scheduler()
    picked = scheduler.begin_request(100.0, 200.0)
    server = scheduler.prefillers[picked["key"]]

    assert server.active_tokens == 100.0
    assert server.active_kv_cache == 200.0

    scheduler.complete_prefill(picked["key"], 100.0, None, None)
    assert server.active_tokens == 0.0
    assert server.active_kv_cache == 200.0

    scheduler.release_prefill_kv(picked["key"], 200.0)
    scheduler.finish_request(None, 0.0, None, 0.0, False)
    assert server.active_kv_cache == 0.0
    assert scheduler.request_num == 0


def test_decoder_picker_tracks_only_decoder_load():
    scheduler = make_scheduler()

    picked = scheduler.pick_decoder(25.0)

    assert scheduler.decoders[picked["key"]].active_tokens == 25.0
    assert all(server.active_tokens == 0.0 for server in scheduler.prefillers.values())
    assert all(server.active_kv_cache == 0.0 for server in scheduler.prefillers.values())

    scheduler.release_decoder(picked["key"], 25.0)
    assert scheduler.decoders[picked["key"]].active_tokens == 0.0


def test_session_affinity_is_committed_only_after_successful_prefill():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10.0, 10.0, "session")
    assert "session" not in scheduler.session_lru

    scheduler.complete_prefill(first["key"], 10.0, "session", None)
    scheduler.release_prefill_kv(first["key"], 10.0)

    second = scheduler.reserve_prefill_kv(10.0, 10.0, "session")
    assert second["key"] == first["key"]


def test_failed_prefill_does_not_commit_affinity_and_releases_pressure():
    scheduler = make_scheduler()
    picked = scheduler.begin_request(10.0, 20.0, "session")

    scheduler.abort_prefill_reservation(picked["key"], 10.0, 20.0, True)

    server = scheduler.prefillers[picked["key"]]
    assert scheduler.session_lru == {}
    assert server.active_tokens == 0.0
    assert server.active_kv_cache == 0.0
    assert scheduler.request_num == 0


def test_affinity_priority_is_session_then_prefix_then_heap():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10.0, 10.0, None, "prefix")
    scheduler.complete_prefill(first["key"], 10.0, None, "prefix")
    scheduler.release_prefill_kv(first["key"], 10.0)

    prefix_hit = scheduler.reserve_prefill_kv(10.0, 10.0, "new-session", "prefix")
    assert prefix_hit["key"] == first["key"]
    scheduler.complete_prefill(prefix_hit["key"], 10.0, "new-session", "prefix")
    scheduler.release_prefill_kv(prefix_hit["key"], 10.0)

    other_key = next(key for key in scheduler.prefillers if key != first["key"])
    scheduler._bind_affinity_no_lock(scheduler.prefix_lru, "other-prefix", other_key, 1)
    session_hit = scheduler.reserve_prefill_kv(10.0, 10.0, "new-session", "other-prefix")
    assert session_hit["key"] == first["key"]
    assert session_hit["route_source"] == "session"
    scheduler.complete_prefill(
        session_hit["key"],
        10.0,
        "new-session",
        "other-prefix",
        session_hit["route_source"],
    )
    assert scheduler.prefix_lru["other-prefix"] == other_key


@pytest.mark.parametrize("response_json", [ValueError("invalid json"), []], ids=["invalid-json", "non-object"])
def test_invalid_prefill_response_releases_reserved_pressure(monkeypatch, response_json):
    scheduler = make_scheduler()

    class FakeRuntime:
        async def schedule(self, method, *args, **kwargs):
            return getattr(scheduler, method)(*args, **kwargs)

        async def get_client(self, _role, _key):
            return object()

    class InvalidResponse:
        def json(self):
            if isinstance(response_json, Exception):
                raise response_json
            return response_json

    async def fake_send_request(*_args, **_kwargs):
        return InvalidResponse()

    monkeypatch.setattr(proxy, "get_runtime", lambda: FakeRuntime())
    monkeypatch.setattr(proxy, "get_global_args", lambda: SimpleNamespace(max_retries=0, retry_delay=0))
    monkeypatch.setattr(proxy, "send_request_to_service", fake_send_request)

    with pytest.raises(ValueError):
        asyncio.run(proxy.assign_instances("/completions", {}, 100, is_initial_request=True))

    assert scheduler.request_num == 0
    assert all(server.active_tokens == 0.0 for server in scheduler.prefillers.values())
    assert all(server.active_kv_cache == 0.0 for server in scheduler.prefillers.values())


def test_affinity_lru_eviction_removal_and_reset():
    scheduler = make_scheduler(session_lru_size=1, prefix_lru_size=1)
    keys = list(scheduler.prefillers)

    scheduler._bind_affinity_no_lock(scheduler.session_lru, "old", keys[0], 1)
    scheduler._bind_affinity_no_lock(scheduler.session_lru, "new", keys[1], 1)
    assert list(scheduler.session_lru) == ["new"]

    scheduler._bind_affinity_no_lock(scheduler.prefix_lru, "old-prefix", keys[0], 1)
    scheduler._bind_affinity_no_lock(scheduler.prefix_lru, "new-prefix", keys[1], 1)
    assert list(scheduler.prefix_lru) == ["new-prefix"]

    scheduler.remove_instances(proxy.ServerRole.PREFILL, [("127.0.0.1", 8101)])
    assert scheduler.session_lru == {}
    assert scheduler.prefix_lru == {}

    scheduler._bind_affinity_no_lock(scheduler.session_lru, "session", keys[0], 1)
    scheduler.clear_affinity_caches()
    assert scheduler.session_lru == {}
    assert scheduler.prefix_lru == {}
