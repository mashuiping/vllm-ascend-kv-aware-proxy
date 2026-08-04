from pathlib import Path

import pytest

from scripts.benchmark_workload import (
    PlannedRequest,
    TokenTextFactory,
    generate_workload,
    load_workload_jsonl,
    workload_manifest,
    write_workload_jsonl,
)


def session_profile():
    return {
        "version": 1,
        "name": "session-test",
        "scenario": "session-long",
        "seed": 42,
        "data": {
            "sessions": 3,
            "turns": 2,
            "system_prompt_tokens": 32,
            "turn_input_tokens": 8,
            "fixed_assistant_text": "Fixed response.",
        },
        "load": {"type": "turn-barrier", "concurrency": 2},
        "request": {"max_tokens": 4, "temperature": 0.0},
    }


def shared_prefix_profile():
    return {
        "version": 1,
        "name": "shared-test",
        "scenario": "shared-prefix",
        "seed": 43,
        "data": {
            "num_groups": 2,
            "prompts_per_group": 2,
            "system_prompt_tokens": 32,
            "question_tokens": 8,
        },
        "load": {
            "type": "poisson",
            "cache_fill_rate": 10,
            "stages": [{"name": "qps-20", "rate": 20, "duration": 1}],
        },
        "request": {"max_tokens": 4, "temperature": 0.0},
    }


def test_token_text_factory_uses_exact_token_round_trip():
    factory = TokenTextFactory(
        tokenize=lambda text: list(text.encode("utf-8")),
        detokenize=lambda tokens: bytes(tokens).decode("utf-8"),
    )

    text = factory.make("stable-label", 32)

    assert len(text.encode("utf-8")) == 32
    assert text.startswith("stable-label")
    assert factory.verified is True


def test_session_workload_is_deterministic_and_freezes_history():
    first = generate_workload(session_profile(), TokenTextFactory())
    second = generate_workload(session_profile(), TokenTextFactory())

    assert first == second
    assert len(first) == 6
    assert [record.phase for record in first].count("cache-fill") == 3
    assert [record.phase for record in first].count("measure") == 3
    assert len({record.session_id for record in first}) == 3
    warm = next(record for record in first if record.turn == 1)
    assert warm.messages[-2] == {"role": "assistant", "content": "Fixed response."}


def test_shared_prefix_workload_primes_full_corpus_then_runs_poisson_stage():
    records = generate_workload(shared_prefix_profile(), TokenTextFactory())
    cache_fill = [record for record in records if record.phase == "cache-fill"]
    measured = [record for record in records if record.phase == "measure"]

    assert len(cache_fill) == 4
    assert measured
    assert all(record.send_session_key is False for record in records)
    systems_by_group = {}
    for record in cache_fill:
        group = record.session_index // 2
        systems_by_group.setdefault(group, set()).add(record.messages[0]["content"])
    assert all(len(systems) == 1 for systems in systems_by_group.values())
    assert systems_by_group[0] != systems_by_group[1]
    assert [record.scheduled_offset_s for record in measured] == sorted(
        record.scheduled_offset_s for record in measured
    )


def test_workload_jsonl_round_trip_and_manifest(tmp_path: Path):
    records = generate_workload(session_profile(), TokenTextFactory())
    path = tmp_path / "workload.jsonl"

    write_workload_jsonl(path, records)

    assert load_workload_jsonl(path) == records
    manifest = workload_manifest(session_profile(), records, token_count_verified=False)
    assert manifest["requests_by_phase"] == {"cache-fill": 3, "measure": 3}
    assert manifest["token_count_verified"] is False


def test_workload_loader_rejects_non_contiguous_sequences(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    record = PlannedRequest(
        sequence=1,
        phase="measure",
        stage="turn-1",
        scheduled_offset_s=0.0,
        session_index=0,
        session_id="session",
        turn=1,
        scenario="session-long",
        send_session_key=True,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=1,
        temperature=0.0,
    )
    write_workload_jsonl(path, [record])

    with pytest.raises(ValueError, match="contiguous"):
        load_workload_jsonl(path)
