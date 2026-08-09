#!/usr/bin/env python3
"""Render a self-contained HTML report for one A/B/C benchmark run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

GROUPS = ("baseline", "candidate-off", "candidate-on")
GROUP_LABELS = {
    "baseline": "A · baseline",
    "candidate-off": "B · candidate-off",
    "candidate-on": "C · candidate-on",
}
COMPARISONS = (
    ("A_vs_B", "A → B · candidate changes"),
    ("B_vs_C", "B → C · KV-aware routing"),
    ("A_vs_C", "A → C · total effect"),
)

METRIC_LABELS = {
    "warm_turns.ttft_ms.p50": ("Warm TTFT p50", "ms", "lower"),
    "warm_turns.ttft_ms.p95": ("Warm TTFT p95", "ms", "lower"),
    "warm_turns.ttft_ms.p99": ("Warm TTFT p99", "ms", "lower"),
    "warm_turns.e2e_ms.p95": ("Warm E2E p95", "ms", "lower"),
    "warm_turns.cached_token_request_rate": ("Requests with cached tokens", "%", "higher"),
    "warm_turns.cached_token_ratio": ("Cached input-token ratio", "%", "higher"),
    "warm_turns.client_computed_tokens.mean": ("Mean client-computed tokens", "tokens", "lower"),
    "warm_turns.request_throughput_per_second": ("Request throughput", "req/s", "higher"),
    "warm_turns.output_token_throughput_per_second": ("Output-token throughput", "tok/s", "higher"),
    "overall.success_rate": ("Overall success rate", "%", "higher"),
    "measurement_metrics_delta_by_role.prefill.derived:prefix_cache_hit_rate": (
        "Prefill prefix-cache hit rate",
        "%",
        "higher",
    ),
    "measurement_metrics_delta_by_role.prefill.derived:mean_prefill_computed_tokens": (
        "Mean Prefill computed tokens",
        "tokens",
        "lower",
    ),
    "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_prefill_time_seconds": (
        "Mean Prefill time",
        "s",
        "lower",
    ),
    "measurement_metrics_delta_by_role.prefill.derived:mean:vllm:request_queue_time_seconds": (
        "Mean Prefill queue time",
        "s",
        "lower",
    ),
    "measurement_metrics_delta_by_role.decode.derived:mean:vllm:request_queue_time_seconds": (
        "Mean Decode queue time",
        "s",
        "lower",
    ),
    "proxy_prefill_load_balance_measurement.priority_cv_p95": (
        "Prefill priority CV p95",
        "ratio",
        "lower",
    ),
    "proxy_prefill_load_balance_measurement.session_affinity_stats_delta.derived_overload_fallback_rate": (
        "Session-affinity overload fallback rate",
        "%",
        "lower",
    ),
    "proxy_prefill_load_balance_measurement.prefix_affinity_stats_delta.derived_prefix_hit_rate": (
        "Prefix-affinity hit rate",
        "%",
        "higher",
    ),
    "proxy_prefill_load_balance_measurement.prefix_affinity_stats_delta.derived_spillover_rate": (
        "Prefix-affinity spillover rate",
        "%",
        "lower",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def format_value(value: Any, unit: str = "") -> str:
    parsed = number(value)
    if parsed is None:
        return "—"
    if unit == "%":
        return f"{parsed * 100:.1f}%"
    if unit == "ms":
        return f"{parsed:.1f} ms"
    if unit == "s":
        return f"{parsed:.4f} s"
    if unit == "ratio":
        return f"{parsed:.3f}"
    if unit == "tokens":
        return f"{parsed:,.1f}"
    if unit in {"req/s", "tok/s"}:
        return f"{parsed:,.2f} {unit}"
    return f"{parsed:,.3f}"


def format_percent(value: Any) -> str:
    parsed = number(value)
    return "—" if parsed is None else f"{parsed * 100:+.1f}%"


def metric_definition(path: str) -> tuple[str, str, str]:
    if path in METRIC_LABELS:
        return METRIC_LABELS[path]
    if path.startswith("per_stage."):
        _, stage, *suffix = path.split(".")
        base_path = "warm_turns." + ".".join(suffix)
        label, unit, direction = METRIC_LABELS.get(base_path, (".".join(suffix), "", "higher"))
        return f"{stage} · {label}", unit, direction
    return path, "", "higher"


def metric_rows(comparison: dict[str, Any], experiment_valid: bool = True) -> list[str]:
    rows: list[str] = []
    for path, value in sorted(comparison.items()):
        if not isinstance(value, dict):
            continue
        label, unit, _direction = metric_definition(path)
        improvement = number(value.get("improvement"))
        if not experiment_valid or improvement is None:
            status = "neutral"
        elif improvement > 0.005:
            status = "positive"
        elif improvement < -0.005:
            status = "negative"
        else:
            status = "neutral"
        rows.append(
            "<tr>"
            f'<td><span class="metric-label">{html.escape(label)}</span>'
            f'<span class="metric-path">{html.escape(path)}</span></td>'
            f"<td>{html.escape(format_value(value.get('baseline'), unit))}</td>"
            f"<td>{html.escape(format_value(value.get('treatment'), unit))}</td>"
            f"<td>{html.escape(format_percent(value.get('relative_change')))}</td>"
            f'<td class="{status}">{html.escape(format_percent(value.get("improvement")))}</td>'
            "</tr>"
        )
    if not rows:
        return ['<tr><td colspan="5" class="muted">No comparable metrics were recorded for this pair.</td></tr>']
    return rows


def summary_card(group: str, summary: dict[str, Any]) -> str:
    warm = summary.get("warm_turns") or {}
    prefix_probe = (summary.get("per_stage") or {}).get("prefix-probe") or {}
    focus = prefix_probe if prefix_probe else warm
    focus_label = "prefix-probe" if prefix_probe else "warm turns"
    overall = summary.get("overall") or {}
    success_rate = format_value(overall.get("success_rate"), "%")
    warm_e2e_p95 = format_value((warm.get("e2e_ms") or {}).get("p95"), "ms")
    cached_ratio = format_value(focus.get("cached_token_ratio"), "%")
    request_throughput = format_value(warm.get("request_throughput_per_second"), "req/s")
    return "".join(
        [
            '<article class="card">',
            f"<h3>{html.escape(GROUP_LABELS[group])}</h3>",
            f'<div class="big">{html.escape(format_value(focus.get("ttft_ms", {}).get("p95"), "ms"))}</div>',
            f'<div class="caption">{html.escape(focus_label)} TTFT p95</div>',
            "<dl>",
            f"<dt>Requests</dt><dd>{html.escape(str(overall.get('requests', '—')))}</dd>",
            f"<dt>Success rate</dt><dd>{html.escape(success_rate)}</dd>",
            f"<dt>Warm E2E p95</dt><dd>{html.escape(warm_e2e_p95)}</dd>",
            f"<dt>{html.escape(focus_label)} cached ratio</dt><dd>{html.escape(cached_ratio)}</dd>",
            f"<dt>Request throughput</dt><dd>{html.escape(request_throughput)}</dd>",
            "</dl>",
            "</article>",
        ]
    )


def phase_overview(summaries: dict[str, dict[str, Any]]) -> str:
    rows: list[str] = []
    for group in GROUPS:
        for phase, label in (("cache_fill", "cache-fill"), ("warm_turns", "warm turns")):
            summary = summaries[group].get(phase) or {}
            rows.append(
                "<tr>"
                f"<td>{html.escape(GROUP_LABELS[group])}</td>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(summary.get('requests', '—')))}</td>"
                f"<td>{html.escape(format_value(summary.get('success_rate'), '%'))}</td>"
                f"<td>{html.escape(format_value((summary.get('ttft_ms') or {}).get('p95'), 'ms'))}</td>"
                f"<td>{html.escape(format_value((summary.get('e2e_ms') or {}).get('p95'), 'ms'))}</td>"
                f"<td>{html.escape(format_value(summary.get('cached_token_ratio'), '%'))}</td>"
                "</tr>"
            )
        for stage, summary in sorted((summaries[group].get("per_stage") or {}).items()):
            if stage == "cache-fill":
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(GROUP_LABELS[group])}</td>"
                f"<td>{html.escape(stage)}</td>"
                f"<td>{html.escape(str(summary.get('requests', '—')))}</td>"
                f"<td>{html.escape(format_value(summary.get('success_rate'), '%'))}</td>"
                f"<td>{html.escape(format_value((summary.get('ttft_ms') or {}).get('p95'), 'ms'))}</td>"
                f"<td>{html.escape(format_value((summary.get('e2e_ms') or {}).get('p95'), 'ms'))}</td>"
                f"<td>{html.escape(format_value(summary.get('cached_token_ratio'), '%'))}</td>"
                "</tr>"
            )
    return "".join(rows)


def render(experiment_dir: Path, output: Path) -> None:
    comparison = load_json(experiment_dir / "comparison.json")
    summaries = {group: load_json(experiment_dir / group / "summary.json") for group in GROUPS}
    manifest_path = experiment_dir / "workload.jsonl.manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    workload_hash = comparison.get("workload_sha256", "—")
    verified = manifest.get("token_count_verified")
    experiment_valid = comparison.get("valid") is True
    cards = "".join(summary_card(group, summaries[group]) for group in GROUPS)
    comparison_sections = []
    for key, title in COMPARISONS:
        pair = comparison.get(key) or {}
        comparison_sections.append(
            f"<section><h2>{html.escape(title)}</h2>"
            "<table><thead><tr><th>Metric</th><th>Baseline</th><th>Treatment</th>"
            "<th>Relative change</th><th>Improvement</th></tr></thead><tbody>"
            + "".join(metric_rows(pair, experiment_valid))
            + "</tbody></table></section>"
        )
    raw_links = []
    for group in GROUPS:
        raw_links.append(f'<a href="{html.escape(group)}/summary.json">{html.escape(GROUP_LABELS[group])} summary</a>')
        raw_links.append(f'<a href="{html.escape(group)}/config.json">{html.escape(GROUP_LABELS[group])} config</a>')
        if (experiment_dir / group / "validity.json").is_file():
            raw_links.append(
                f'<a href="{html.escape(group)}/validity.json">{html.escape(GROUP_LABELS[group])} validity</a>'
            )
        if (experiment_dir / group / "reset-validation.json").is_file():
            raw_links.append(
                f'<a href="{html.escape(group)}/reset-validation.json">{html.escape(GROUP_LABELS[group])} reset</a>'
            )
    raw_links.extend(
        [
            '<a href="comparison.json">comparison.json</a>',
            '<a href="workload.jsonl.manifest.json">workload manifest</a>',
            '<a href="workload.jsonl">workload.jsonl</a>',
        ]
    )
    verification_class = "positive" if verified is True else "negative"
    validity_message = (
        "Experiment validity checks passed."
        if experiment_valid
        else (
            "Experiment is invalid: improvements are shown for diagnosis only "
            "and must not be used as performance claims."
        )
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark report · {html.escape(experiment_dir.name)}</title>
<style>
:root {{
  color-scheme: light;
  --ink:#172033; --muted:#667085; --line:#e5e7eb; --panel:#fff;
  --bg:#f6f8fb; --accent:#2563eb; --good:#087443; --bad:#b42318;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:28px; }}
h1 {{ margin:0 0 4px; font-size:28px; }} h2 {{ margin:30px 0 12px; font-size:20px; }}
h3 {{ margin:0 0 10px; font-size:16px; }}
.subtitle,.muted,.metric-path,.caption {{ color:var(--muted); }} .subtitle {{ margin-bottom:20px; }}
.meta,.cards,section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.meta {{ padding:16px 20px; display:flex; gap:28px; flex-wrap:wrap; }}
.meta strong {{ display:block; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; overflow:hidden; margin-top:20px; }}
.card {{ padding:20px; background:var(--panel); }} .big {{ font-size:27px; font-weight:700; color:var(--accent); }}
.caption {{ font-size:12px; margin-bottom:14px; }}
dl {{ display:grid; grid-template-columns:1fr auto; gap:5px 12px; margin:0; }} dt {{ color:var(--muted); }}
dd {{ margin:0; font-weight:600; text-align:right; }}
section {{ padding:18px; overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:760px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
th:first-child,td:first-child {{ text-align:left; }} tr:last-child td {{ border-bottom:0; }}
.metric-label {{ display:block; font-weight:600; }}
.metric-path {{ display:block; font-size:11px; word-break:break-all; }}
.positive {{ color:var(--good); font-weight:700; }} .negative {{ color:var(--bad); font-weight:700; }}
.neutral {{ color:var(--muted); }}
.validity-banner {{ margin:16px 0; padding:14px 18px; border-radius:10px; font-weight:650; }}
.validity-banner.valid {{ color:var(--good); background:#ecfdf3; border:1px solid #abefc6; }}
.validity-banner.invalid {{ color:var(--bad); background:#fef3f2; border:1px solid #fecdca; }}
.links {{ display:flex; gap:10px 18px; flex-wrap:wrap; }} a {{ color:var(--accent); }}
@media(max-width:800px) {{ main {{ padding:16px; }} .cards {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Benchmark report</h1>
<div class="subtitle">{html.escape(experiment_dir.name)}</div>
<div class="validity-banner {"valid" if experiment_valid else "invalid"}">
{validity_message}
</div>
<div class="meta">
<div><strong>Workload SHA-256</strong><code>{html.escape(str(workload_hash))}</code></div>
<div><strong>Token count verified</strong><span class="{verification_class}">{html.escape(str(verified))}</span></div>
<div><strong>Scenario</strong><span>{html.escape(str(manifest.get("scenario", "—")))}</span></div>
<div><strong>Requests</strong><span>{html.escape(str(manifest.get("requests", "—")))}</span></div>
<div><strong>Validity</strong><span>{html.escape(str(experiment_valid))}</span></div>
</div>
<h2>Group overview</h2><div class="cards">{cards}</div>
<section><h2>Phase overview</h2>
<table><thead><tr><th>Group</th><th>Phase</th><th>Requests</th><th>Success rate</th>
<th>TTFT p95</th><th>E2E p95</th><th>Cached token ratio</th></tr></thead>
<tbody>{phase_overview(summaries)}</tbody></table></section>
{"".join(comparison_sections)}
<section><h2>Raw artifacts</h2><div class="links">{"".join(raw_links)}</div></section>
</main></body></html>"""
    output.write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Output HTML path; defaults to <experiment_dir>/report.html")
    args = parser.parse_args()
    output = args.output or args.experiment_dir / "report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    render(args.experiment_dir, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
