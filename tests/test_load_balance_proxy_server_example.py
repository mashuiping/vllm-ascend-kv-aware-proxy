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


def test_reset_prefix_cache_reports_every_prefill(monkeypatch):
    scheduler = make_scheduler()

    class FakeResponse:
        status_code = 200
        text = '{"success": true}'

        def json(self):
            return {"success": True}

        def raise_for_status(self):
            return None

    class FakeClient:
        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    class FakeRuntime:
        def __init__(self):
            self.scheduler = scheduler

        async def sync_clients(self):
            return None

        async def get_client(self, _role, _key):
            return FakeClient()

    monkeypatch.setattr(proxy, "get_runtime", lambda: FakeRuntime())
    response = asyncio.run(proxy.reset_prefix_cache(SimpleNamespace(query_params={})))
    payload = __import__("json").loads(response.body)

    assert response.status_code == 200
    assert payload["success"] is True
    assert len(payload["backends"]) == 2


def test_reset_prefix_cache_rejects_explicit_backend_failure(monkeypatch):
    scheduler = make_scheduler()

    class FakeResponse:
        status_code = 200
        text = '{"success": false}'

        def json(self):
            return {"success": False}

        def raise_for_status(self):
            return None

    class FakeClient:
        async def post(self, *_args, **_kwargs):
            return FakeResponse()

    class FakeRuntime:
        def __init__(self):
            self.scheduler = scheduler

        async def sync_clients(self):
            return None

        async def get_client(self, _role, _key):
            return FakeClient()

    monkeypatch.setattr(proxy, "get_runtime", lambda: FakeRuntime())
    response = asyncio.run(proxy.reset_prefix_cache(SimpleNamespace(query_params={})))
    payload = __import__("json").loads(response.body)

    assert response.status_code == 500
    assert payload["success"] is False
    assert len(payload["failed"]) == 2


def test_extract_reusable_prefix_tokens_sums_cached_and_created():
    response = {
        "usage": {
            "prompt_tokens_details": {
                "cached_tokens": 256,
                "created_cache_tokens": 128,
            }
        }
    }
    assert proxy.extract_reusable_prefix_tokens(response) == 384


def test_extract_reusable_prefix_tokens_returns_zero_when_both_zero():
    response = {
        "usage": {
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "created_cache_tokens": 0,
            }
        }
    }
    assert proxy.extract_reusable_prefix_tokens(response) == 0


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"usage": {}},
        {"usage": {"prompt_tokens_details": None}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": 256}}},
        {"usage": {"prompt_tokens_details": {"created_cache_tokens": 128}}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": "256", "created_cache_tokens": 0}}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": 256, "created_cache_tokens": "0"}}},
        {"usage": {"prompt_tokens_details": [256, 0]}},
    ],
    ids=[
        "missing-usage",
        "empty-usage",
        "null-details",
        "missing-created",
        "missing-cached",
        "non-int-cached",
        "non-int-created",
        "non-dict-details",
    ],
)
def test_extract_reusable_prefix_tokens_incomplete_returns_none(response):
    assert proxy.extract_reusable_prefix_tokens(response) is None


def test_complete_prefill_with_allow_affinity_false_does_not_bind_session():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10.0, 10.0, "session")
    scheduler.complete_prefill(first["key"], 10.0, "session", None, route_source="session", allow_affinity=False)
    scheduler.release_prefill_kv(first["key"], 10.0)
    assert scheduler.session_lru == {}


def test_complete_prefill_with_allow_affinity_false_does_not_bind_prefix():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10.0, 10.0, None, "prefix")
    scheduler.complete_prefill(first["key"], 10.0, None, "prefix", route_source="prefix", allow_affinity=False)
    scheduler.release_prefill_kv(first["key"], 10.0)
    assert scheduler.prefix_lru == {}


def test_complete_prefill_allow_affinity_false_preserves_existing_bindings():
    scheduler = make_scheduler()
    first = scheduler.begin_request(10.0, 10.0, "session")
    scheduler.complete_prefill(first["key"], 10.0, "session", None)
    bound = scheduler.session_lru["session"]
    scheduler.release_prefill_kv(first["key"], 10.0)

    second = scheduler.begin_request(10.0, 10.0, "session")
    scheduler.complete_prefill(second["key"], 10.0, "session", None, route_source="session", allow_affinity=False)
    scheduler.release_prefill_kv(second["key"], 10.0)
    assert scheduler.session_lru["session"] == bound


def test_assign_instances_with_gate_off_binds_even_when_reusable_is_zero(monkeypatch):
    scheduler = make_scheduler()
    response_payload = {"usage": {"prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 0}}}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
            self.headers = {}

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.base_url = "http://test"

        async def post(self, *args, **kwargs):
            self.calls += 1
            return FakeResponse(response_payload)

    async def fake_send_request(client, *_args, **_kwargs):
        return await client.post()

    class FakeRuntime:
        def __init__(self):
            self.scheduler = scheduler

        async def schedule(self, method, *args, **kwargs):
            method_fn = getattr(scheduler, method)
            if method == "pick_decoder":
                return {"key": next(iter(scheduler.decoders)), "host": "127.0.0.1", "port": 8200}
            return method_fn(*args, **kwargs)

        async def get_client(self, _role, _key):
            return FakeClient()

    runtime = FakeRuntime()
    monkeypatch.setattr(proxy, "get_runtime", lambda: runtime)
    monkeypatch.setattr(
        proxy,
        "get_global_args",
        lambda: SimpleNamespace(
            max_retries=0,
            retry_delay=0,
            enable_kv_cache_aware_routing=True,
            enable_reusable_prefix_affinity_gate=False,
            prefix_hash_chars=0,
        ),
    )
    monkeypatch.setattr(proxy, "send_request_to_service", fake_send_request)

    instance_info = asyncio.run(
        proxy.assign_instances(
            "/completions",
            {"session_id": "s1"},
            100,
            is_initial_request=True,
            session_key=proxy.extract_session_key({}, {"session_id": "s1"}),
            prefix_key=None,
        )
    )
    scheduler.release_prefill_kv(instance_info.prefiller_key, instance_info.prefiller_score)
    scheduler.release_decoder(
        next(iter(scheduler.decoders)),
        instance_info.decoder_score,
    )
    assert "header:x-session-id:s1" in scheduler.session_lru or "body:session_id:s1" in scheduler.session_lru


def test_assign_instances_with_gate_on_skips_bind_when_reusable_zero(monkeypatch):
    scheduler = make_scheduler()
    response_payload = {"usage": {"prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 0}}}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self):
            self.base_url = "http://test"

        async def post(self, *args, **kwargs):
            return FakeResponse(response_payload)

    class FakeRuntime:
        def __init__(self):
            self.scheduler = scheduler

        async def schedule(self, method, *args, **kwargs):
            method_fn = getattr(scheduler, method)
            if method == "pick_decoder":
                return {"key": next(iter(scheduler.decoders)), "host": "127.0.0.1", "port": 8200}
            return method_fn(*args, **kwargs)

        async def get_client(self, _role, _key):
            return FakeClient()

    runtime = FakeRuntime()
    monkeypatch.setattr(proxy, "get_runtime", lambda: runtime)
    monkeypatch.setattr(
        proxy,
        "get_global_args",
        lambda: SimpleNamespace(
            max_retries=0,
            retry_delay=0,
            enable_kv_cache_aware_routing=True,
            enable_reusable_prefix_affinity_gate=True,
            prefix_hash_chars=0,
        ),
    )
    monkeypatch.setattr(proxy, "send_request_to_service", lambda client, *a, **kw: client.post())

    s_key = "body:session_id:s1"
    instance_info = asyncio.run(
        proxy.assign_instances(
            "/completions",
            {"session_id": "s1"},
            100,
            is_initial_request=True,
            session_key=s_key,
            prefix_key=None,
        )
    )
    scheduler.release_prefill_kv(instance_info.prefiller_key, instance_info.prefiller_score)
    scheduler.release_decoder(
        next(iter(scheduler.decoders)),
        instance_info.decoder_score,
    )
    assert scheduler.session_lru == {}


def test_assign_instances_with_gate_on_binds_and_warns_when_details_missing(monkeypatch, caplog):
    scheduler = make_scheduler()

    class FakeResponse:
        def json(self):
            return {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    class FakeClient:
        def __init__(self):
            self.base_url = "http://test"

        async def post(self, *args, **kwargs):
            return FakeResponse()

    class FakeRuntime:
        def __init__(self):
            self.scheduler = scheduler

        async def schedule(self, method, *args, **kwargs):
            method_fn = getattr(scheduler, method)
            if method == "pick_decoder":
                return {"key": next(iter(scheduler.decoders)), "host": "127.0.0.1", "port": 8200}
            return method_fn(*args, **kwargs)

        async def get_client(self, _role, _key):
            return FakeClient()

    runtime = FakeRuntime()
    monkeypatch.setattr(proxy, "get_runtime", lambda: runtime)
    monkeypatch.setattr(
        proxy,
        "get_global_args",
        lambda: SimpleNamespace(
            max_retries=0,
            retry_delay=0,
            enable_kv_cache_aware_routing=True,
            enable_reusable_prefix_affinity_gate=True,
            prefix_hash_chars=0,
        ),
    )
    monkeypatch.setattr(proxy, "send_request_to_service", lambda client, *a, **kw: client.post())

    s_key = "body:session_id:s1"
    with caplog.at_level("DEBUG", logger=proxy.logger.name):
        instance_info = asyncio.run(
            proxy.assign_instances(
                "/completions",
                {"session_id": "s1"},
                100,
                is_initial_request=True,
                session_key=s_key,
                prefix_key=None,
            )
        )
    scheduler.release_prefill_kv(instance_info.prefiller_key, instance_info.prefiller_score)
    scheduler.release_decoder(
        next(iter(scheduler.decoders)),
        instance_info.decoder_score,
    )
    assert s_key in scheduler.session_lru
    assert any("reusable_prefix_tokens" in record.getMessage() for record in caplog.records)
