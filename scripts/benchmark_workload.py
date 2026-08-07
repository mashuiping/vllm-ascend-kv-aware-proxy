#!/usr/bin/env python3
"""Deterministic workload generation for KV-aware routing benchmarks."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WORKLOAD_VERSION = 1


@dataclass(frozen=True)
class PlannedRequest:
    sequence: int
    phase: str
    stage: str
    scheduled_offset_s: float
    session_index: int
    session_id: str
    turn: int
    scenario: str
    send_session_key: bool
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlannedRequest":
        return cls(
            sequence=int(value["sequence"]),
            phase=str(value["phase"]),
            stage=str(value["stage"]),
            scheduled_offset_s=float(value.get("scheduled_offset_s", 0.0)),
            session_index=int(value["session_index"]),
            session_id=str(value["session_id"]),
            turn=int(value["turn"]),
            scenario=str(value["scenario"]),
            send_session_key=bool(value.get("send_session_key", True)),
            messages=[dict(message) for message in value["messages"]],
            max_tokens=int(value["max_tokens"]),
            temperature=float(value.get("temperature", 0.0)),
        )


class TokenTextFactory:
    """Build labeled text with an exact model-token length when callbacks exist."""

    def __init__(
        self,
        tokenize: Callable[[str], list[int]] | None = None,
        detokenize: Callable[[list[int]], str] | None = None,
    ):
        if (tokenize is None) != (detokenize is None):
            raise ValueError("tokenize and detokenize must be supplied together")
        self._tokenize = tokenize
        self._detokenize = detokenize
        self.verified = tokenize is not None

    def make(self, label: str, target_tokens: int) -> str:
        if target_tokens <= 0:
            raise ValueError("target token length must be positive")
        candidate = f"{label}\n" + " ".join(
            f"benchmarkword{index % 997:03d}" for index in range(max(target_tokens * 4, 64))
        )
        if self._tokenize is None or self._detokenize is None:
            # Useful for offline smoke tests only. Production profiles require
            # server-tokenizer verification in generate_benchmark_workload.py.
            return " ".join(candidate.split()[:target_tokens])

        token_ids = self._tokenize(candidate)
        while len(token_ids) < target_tokens:
            candidate += " " + candidate
            token_ids = self._tokenize(candidate)
        text = self._detokenize(token_ids[:target_tokens])
        actual = self._tokenize(text)
        if len(actual) != target_tokens:
            raise ValueError(f"tokenizer round trip produced {len(actual)} tokens; expected {target_tokens}")
        return text


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("workload profile must be a JSON object")
    if profile.get("version") != WORKLOAD_VERSION:
        raise ValueError(f"unsupported workload profile version: {profile.get('version')!r}")
    if profile.get("scenario") not in ("session-long", "shared-prefix", "load-balance"):
        raise ValueError("profile scenario must be session-long, shared-prefix, or load-balance")
    return profile


def override_system_prompt_tokens(
    profile: dict[str, Any],
    system_prompt_tokens: int | None,
) -> dict[str, Any]:
    """Return a profile copy with an optional system-prompt size override."""
    if system_prompt_tokens is None:
        return profile
    if system_prompt_tokens <= 0:
        raise ValueError("system_prompt_tokens override must be positive")
    data = profile.get("data")
    if not isinstance(data, dict) or "system_prompt_tokens" not in data:
        raise ValueError(
            "system_prompt_tokens override is supported only for profiles with data.system_prompt_tokens"
        )
    effective = dict(profile)
    effective["data"] = dict(data)
    effective["data"]["system_prompt_tokens"] = system_prompt_tokens
    return effective


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _request_settings(profile: dict[str, Any]) -> tuple[int, float]:
    request = profile.get("request") or {}
    return _positive_int(request.get("max_tokens", 32), "request.max_tokens"), float(request.get("temperature", 0.0))


def generate_session_workload(
    profile: dict[str, Any],
    text_factory: TokenTextFactory,
) -> list[PlannedRequest]:
    data = profile.get("data") or {}
    seed = int(profile.get("seed", 20260724))
    sessions = _positive_int(data.get("sessions"), "data.sessions")
    turns = _positive_int(data.get("turns"), "data.turns")
    system_tokens = _positive_int(data.get("system_prompt_tokens"), "data.system_prompt_tokens")
    turn_tokens = _positive_int(data.get("turn_input_tokens"), "data.turn_input_tokens")
    fixed_assistant = str(data.get("fixed_assistant_text", "Acknowledged."))
    max_tokens, temperature = _request_settings(profile)
    scenario = str(profile["scenario"])
    rng = random.Random(seed)
    sequence = 0
    records: list[PlannedRequest] = []
    histories: list[list[dict[str, str]]] = []

    for session_index in range(sessions):
        system = text_factory.make(f"session-{seed}-{session_index}-system", system_tokens)
        first_user = text_factory.make(f"session-{seed}-{session_index}-turn-0", turn_tokens)
        histories.append(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": first_user},
            ]
        )

    for turn in range(turns):
        order = list(range(sessions))
        rng.shuffle(order)
        for session_index in order:
            records.append(
                PlannedRequest(
                    sequence=sequence,
                    phase="cache-fill" if turn == 0 else "measure",
                    stage=f"turn-{turn}",
                    scheduled_offset_s=0.0,
                    session_index=session_index,
                    session_id=f"bench-session-{seed}-{session_index:08d}",
                    turn=turn,
                    scenario=scenario,
                    send_session_key=True,
                    messages=[dict(message) for message in histories[session_index]],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            sequence += 1
        if turn + 1 < turns:
            for session_index in range(sessions):
                histories[session_index].append({"role": "assistant", "content": fixed_assistant})
                histories[session_index].append(
                    {
                        "role": "user",
                        "content": text_factory.make(
                            f"session-{seed}-{session_index}-turn-{turn + 1}",
                            turn_tokens,
                        ),
                    }
                )
    return records


def _poisson_offsets(rate: float, duration: float, rng: random.Random) -> list[float]:
    if rate <= 0 or duration <= 0:
        raise ValueError("stage rate and duration must be positive")
    offsets: list[float] = []
    current = 0.0
    while True:
        current += rng.expovariate(rate)
        if current >= duration:
            return offsets
        offsets.append(current)


def generate_shared_prefix_workload(
    profile: dict[str, Any],
    text_factory: TokenTextFactory,
) -> list[PlannedRequest]:
    data = profile.get("data") or {}
    load = profile.get("load") or {}
    seed = int(profile.get("seed", 20260724))
    groups = _positive_int(data.get("num_groups"), "data.num_groups")
    prompts_per_group = _positive_int(data.get("prompts_per_group"), "data.prompts_per_group")
    cache_fill_prompts_per_group = _positive_int(
        data.get("cache_fill_prompts_per_group", prompts_per_group),
        "data.cache_fill_prompts_per_group",
    )
    if cache_fill_prompts_per_group > prompts_per_group:
        raise ValueError("data.cache_fill_prompts_per_group cannot exceed data.prompts_per_group")
    system_tokens = _positive_int(data.get("system_prompt_tokens"), "data.system_prompt_tokens")
    question_tokens = _positive_int(data.get("question_tokens"), "data.question_tokens")
    cache_fill_rate = float(load.get("cache_fill_rate", 10.0))
    prefix_probe_rate = float(load.get("prefix_probe_rate", 0.0))
    if prefix_probe_rate < 0:
        raise ValueError("load.prefix_probe_rate must be non-negative")
    if prefix_probe_rate > 0 and cache_fill_prompts_per_group >= prompts_per_group:
        raise ValueError("load.prefix_probe_rate requires an unprimed prompt per group")
    stages = load.get("stages") or []
    if not isinstance(stages, list) or not stages:
        raise ValueError("load.stages must contain at least one QPS stage")
    max_tokens, temperature = _request_settings(profile)
    rng = random.Random(seed)

    corpus: list[tuple[int, int, list[dict[str, str]]]] = []
    for group_index in range(groups):
        system = text_factory.make(f"shared-group-{seed}-{group_index}-system", system_tokens)
        for prompt_index in range(prompts_per_group):
            question = text_factory.make(
                f"shared-group-{seed}-{group_index}-question-{prompt_index}",
                question_tokens,
            )
            corpus.append(
                (
                    group_index,
                    prompt_index,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                )
            )

    records: list[PlannedRequest] = []
    sequence = 0
    cache_fill = [item for item in corpus if item[1] < cache_fill_prompts_per_group]
    # Keep the QPS-stage random streams independent of the number of prime
    # requests. Changing cache_fill_prompts_per_group must not silently change
    # stage arrival offsets or prompt order.
    cache_fill_rng = random.Random(seed ^ 0x5A17)
    cache_fill_rng.shuffle(cache_fill)
    for index, (group_index, prompt_index, messages) in enumerate(cache_fill):
        records.append(
            PlannedRequest(
                sequence=sequence,
                phase="cache-fill",
                stage="cache-fill",
                scheduled_offset_s=index / cache_fill_rate,
                session_index=group_index * prompts_per_group + prompt_index,
                session_id=f"bench-shared-{seed}-{group_index:05d}-{prompt_index:05d}",
                turn=0,
                scenario="shared-prefix",
                send_session_key=False,
                messages=[dict(message) for message in messages],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        sequence += 1

    if prefix_probe_rate > 0:
        prefix_probe = [item for item in corpus if item[1] == cache_fill_prompts_per_group]
        for index, (group_index, prompt_index, messages) in enumerate(prefix_probe):
            records.append(
                PlannedRequest(
                    sequence=sequence,
                    phase="measure",
                    stage="prefix-probe",
                    scheduled_offset_s=index / prefix_probe_rate,
                    session_index=group_index * prompts_per_group + prompt_index,
                    session_id=f"bench-shared-{seed}-probe-{group_index:05d}",
                    turn=1,
                    scenario="shared-prefix",
                    send_session_key=False,
                    messages=[dict(message) for message in messages],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            sequence += 1

    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError("every load stage must be an object")
        rate = float(stage["rate"])
        duration = float(stage["duration"])
        stage_name = str(stage.get("name") or f"qps-{rate:g}")
        stage_rng = random.Random(rng.randrange(2**63))
        offsets = _poisson_offsets(rate, duration, stage_rng)
        order = list(corpus)
        rng.shuffle(order)
        for request_index, offset in enumerate(offsets):
            group_index, prompt_index, messages = order[request_index % len(order)]
            records.append(
                PlannedRequest(
                    sequence=sequence,
                    phase="measure",
                    stage=stage_name,
                    scheduled_offset_s=offset,
                    session_index=group_index * prompts_per_group + prompt_index,
                    session_id=(f"bench-shared-{seed}-{stage_index:03d}-{request_index:08d}"),
                    turn=stage_index + 1,
                    scenario="shared-prefix",
                    send_session_key=False,
                    messages=[dict(message) for message in messages],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            sequence += 1
    return records


def generate_load_balance_workload(
    profile: dict[str, Any],
    text_factory: TokenTextFactory,
) -> list[PlannedRequest]:
    """Generate a continuous, heterogeneous Prefill workload.

    Every request has a unique system prompt and no session header, so this
    profile measures transient Prefill load placement instead of cache
    affinity. Prompt classes intentionally have different token sizes; the
    resulting overlap gives candidate-off a chance to use its active-token
    accounting while the request is still in Prefill.
    """
    data = profile.get("data") or {}
    load = profile.get("load") or {}
    seed = int(profile.get("seed", 20260724))
    prompt_classes = data.get("prompt_classes") or []
    if not isinstance(prompt_classes, list) or not prompt_classes:
        raise ValueError("data.prompt_classes must contain at least one prompt class")
    classes: list[tuple[str, int, int]] = []
    for index, prompt_class in enumerate(prompt_classes):
        if not isinstance(prompt_class, dict):
            raise ValueError("every data.prompt_classes entry must be an object")
        name = str(prompt_class.get("name") or f"class-{index}")
        system_tokens = _positive_int(
            prompt_class.get("system_prompt_tokens"),
            f"data.prompt_classes[{index}].system_prompt_tokens",
        )
        input_tokens = _positive_int(
            prompt_class.get("turn_input_tokens"),
            f"data.prompt_classes[{index}].turn_input_tokens",
        )
        classes.append((name, system_tokens, input_tokens))

    stages = load.get("stages") or []
    if not isinstance(stages, list) or not stages:
        raise ValueError("load.stages must contain at least one QPS stage")
    max_tokens, temperature = _request_settings(profile)
    rng = random.Random(seed)
    records: list[PlannedRequest] = []
    sequence = 0
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError("every load stage must be an object")
        rate = float(stage.get("rate", 0.0))
        duration = float(stage.get("duration", 0.0))
        if rate <= 0 or duration <= 0:
            raise ValueError("load stage rate and duration must be positive")
        stage_name = str(stage.get("name") or f"qps-{rate:g}")
        stage_rng = random.Random(rng.randrange(2**63))
        offsets = _poisson_offsets(rate, duration, stage_rng)
        for request_index, offset in enumerate(offsets):
            class_name, system_tokens, input_tokens = classes[
                (request_index + stage_index) % len(classes)
            ]
            label = f"load-balance-{seed}-{stage_index:03d}-{request_index:08d}-{class_name}"
            records.append(
                PlannedRequest(
                    sequence=sequence,
                    phase="measure",
                    stage=stage_name,
                    scheduled_offset_s=offset,
                    session_index=sequence,
                    session_id=f"bench-load-balance-{seed}-{sequence:08d}",
                    turn=stage_index,
                    scenario="load-balance",
                    send_session_key=False,
                    messages=[
                        {
                            "role": "system",
                            "content": text_factory.make(f"{label}-system", system_tokens),
                        },
                        {
                            "role": "user",
                            "content": text_factory.make(f"{label}-input", input_tokens),
                        },
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            sequence += 1
    if not records:
        raise ValueError("load-balance stages produced no requests")
    return records


def generate_workload(
    profile: dict[str, Any],
    text_factory: TokenTextFactory,
) -> list[PlannedRequest]:
    if profile["scenario"] == "session-long":
        return generate_session_workload(profile, text_factory)
    if profile["scenario"] == "shared-prefix":
        return generate_shared_prefix_workload(profile, text_factory)
    return generate_load_balance_workload(profile, text_factory)


def write_workload_jsonl(path: Path, records: Iterable[PlannedRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def load_workload_jsonl(path: Path) -> list[PlannedRequest]:
    records: list[PlannedRequest] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(PlannedRequest.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid workload record at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"workload is empty: {path}")
    sequences = [record.sequence for record in records]
    if sequences != list(range(len(records))):
        raise ValueError("workload sequence values must be contiguous and start at zero")
    if any(not math.isfinite(record.scheduled_offset_s) or record.scheduled_offset_s < 0 for record in records):
        raise ValueError("scheduled offsets must be finite and non-negative")
    return records


def workload_manifest(
    profile: dict[str, Any],
    records: list[PlannedRequest],
    token_count_verified: bool,
) -> dict[str, Any]:
    by_phase: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for record in records:
        by_phase[record.phase] = by_phase.get(record.phase, 0) + 1
        by_stage[record.stage] = by_stage.get(record.stage, 0) + 1
    return {
        "version": WORKLOAD_VERSION,
        "name": profile.get("name"),
        "scenario": profile.get("scenario"),
        "seed": profile.get("seed"),
        "token_count_verified": token_count_verified,
        "requests": len(records),
        "requests_by_phase": by_phase,
        "requests_by_stage": by_stage,
        "profile": profile,
    }
