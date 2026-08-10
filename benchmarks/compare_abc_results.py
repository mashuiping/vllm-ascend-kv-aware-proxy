#!/usr/bin/env python3
"""Validate and compare one completed A/B/C benchmark experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from benchmark_session_affinity import compare_summaries
except ModuleNotFoundError:  # Imported as benchmarks.compare_abc_results in tests.
    from benchmarks.benchmark_session_affinity import compare_summaries


GROUPS = ("baseline", "candidate-off", "candidate-on")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def build_comparison(experiment_dir: Path) -> dict[str, Any]:
    summaries = {group: load_json(experiment_dir / group / "summary.json") for group in GROUPS}
    configs = {group: load_json(experiment_dir / group / "config.json") for group in GROUPS}
    raw_hashes = [config.get("workload_sha256") for config in configs.values()]
    workload_hashes = {value for value in raw_hashes if isinstance(value, str) and value}
    if len(workload_hashes) != 1 or len(workload_hashes) != len(set(raw_hashes)):
        raise ValueError(f"A/B/C workload SHA-256 values differ or are missing: {workload_hashes}")
    group_validity: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        path = experiment_dir / group / "validity.json"
        group_validity[group] = (
            load_json(path) if path.is_file() else {"valid": False, "checks": {"validity_artifact_present": False}}
        )
    identities = {group: configs[group].get("proxy_identity") for group in GROUPS}
    identity_present = any(identity is not None for identity in identities.values())
    identity_checks: dict[str, bool] = {}
    if identity_present:
        identity_checks = {
            "all_groups_present": all(isinstance(identity, dict) for identity in identities.values()),
            "group_names_match": all(
                isinstance(identity, dict) and identity.get("group") == group for group, identity in identities.items()
            ),
            "source_hashes_match_expected": all(
                isinstance(identity, dict)
                and identity.get("actual_source_sha256") == identity.get("expected_source_sha256")
                for identity in identities.values()
            ),
            "variants_match": all(
                isinstance(identity, dict) and identity.get("actual_variant") == identity.get("expected_variant")
                for identity in identities.values()
            ),
            "kv_aware_flags_match": all(
                isinstance(identity, dict) and identity.get("actual_kv_aware") == identity.get("expected_kv_aware")
                for identity in identities.values()
            ),
            "active_token_weights_match": all(
                isinstance(identity, dict)
                and identity.get("actual_active_token_weight") == identity.get("expected_active_token_weight")
                for identity in identities.values()
            ),
            "comparison_modes_match": len(
                {
                    identity.get("comparison_mode")
                    for identity in identities.values()
                    if isinstance(identity, dict) and identity.get("comparison_mode")
                }
            )
            == 1,
        }
        if identity_checks["all_groups_present"]:
            baseline_hash = identities["baseline"].get("actual_source_sha256")
            candidate_off_hash = identities["candidate-off"].get("actual_source_sha256")
            candidate_on_hash = identities["candidate-on"].get("actual_source_sha256")
            identity_checks["baseline_differs_from_candidate"] = bool(
                baseline_hash and candidate_off_hash and baseline_hash != candidate_off_hash
            )
            identity_checks["candidate_groups_match"] = bool(
                candidate_off_hash and candidate_off_hash == candidate_on_hash
            )
            comparison_mode = identities["baseline"].get("comparison_mode")
            try:
                off_weight = float(identities["candidate-off"].get("actual_active_token_weight"))
                on_weight = float(identities["candidate-on"].get("actual_active_token_weight"))
            except (TypeError, ValueError):
                off_weight = on_weight = -1.0
            if comparison_mode == "active-token":
                identity_checks["comparison_mode_semantics_match"] = (
                    identities["baseline"].get("actual_variant") == "baseline"
                    and identities["candidate-off"].get("actual_variant") == "candidate"
                    and identities["candidate-on"].get("actual_variant") == "candidate"
                    and identities["candidate-off"].get("actual_kv_aware") == "false"
                    and identities["candidate-on"].get("actual_kv_aware") == "false"
                    and off_weight == 0.0
                    and on_weight > 0.0
                )
            elif comparison_mode == "affinity":
                identity_checks["comparison_mode_semantics_match"] = (
                    identities["candidate-off"].get("actual_kv_aware") == "false"
                    and identities["candidate-on"].get("actual_kv_aware") == "true"
                    and off_weight == on_weight
                )
            else:
                identity_checks["comparison_mode_semantics_match"] = False
    identity_valid = not identity_present or (bool(identity_checks) and all(identity_checks.values()))
    valid = all(value.get("valid") is True for value in group_validity.values()) and identity_valid
    return {
        "valid": valid,
        "validity": {"groups": group_validity, "proxy_identity": identity_checks},
        "workload_sha256": workload_hashes.pop(),
        "A_vs_B": compare_summaries(summaries["baseline"], summaries["candidate-off"]),
        "B_vs_C": compare_summaries(summaries["candidate-off"], summaries["candidate-on"]),
        "A_vs_C": compare_summaries(summaries["baseline"], summaries["candidate-on"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison(args.experiment_dir)
    output = args.experiment_dir / "comparison.json"
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
