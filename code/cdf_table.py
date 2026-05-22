"""CDF-metric-only table across every (recipe, K, alpha, scope) cell.

The CDF metric is the adaptive per-row core-set hit-rate defined in
``core_set_metric.py``. For each row of a row-stochastic plan T the
threshold ``mu_i + sigma_i`` (the mean+std of the row's mass
distribution --- equivalently, a point on the per-row mass CDF) defines
a "core set"

    core_i = { j : T[i, j] > mu_i + sigma_i }.

Two numbers summarise the metric:

    hit_rate       = fraction of rows where the ground-truth column
                     lies in core_i.
    mean_core_size = average |core_i| across rows.

Read together, they answer: "in what fraction of cells is the GT inside
the high-mass head of the row's distribution, and how many candidate
columns does that head contain on average?"

This script reads the already-built
    results/core_set_analysis/extended_table.csv
and renders a CDF-only view. No metric is recomputed --- this is purely
a column projection so the output is easy to read.

Outputs (under results/cdf_analysis/):
  - cdf_table.csv  (machine-readable)
  - cdf_table.md   (human-readable, grouped per recipe)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
IN_CSV = RES / "core_set_analysis" / "extended_table.csv"
OUT_DIR = RES / "cdf_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Order in which recipes appear in the markdown output.
RECIPE_ORDER = [
    "Random baseline",
    "GW (unsup, alpha=1.0)",
    "FGW direct (M = caption cos)",
    "FGW transitive (identity bridge)",
    "Text baseline (cosine sim, NOT row-stochastic)",
]


def _maybe(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _sort_key(row: dict) -> tuple[int, float, float, int]:
    """Stable order: recipe → K → alpha → scope."""
    try:
        recipe_rank = RECIPE_ORDER.index(row["method"])
    except ValueError:
        recipe_rank = len(RECIPE_ORDER)
    try:
        K = float(row["K"]) if row["K"] not in (None, "") else 0.0
    except (TypeError, ValueError):
        K = 0.0
    try:
        a = float(row["alpha"]) if row["alpha"] not in (None, "") else 0.0
    except (TypeError, ValueError):
        a = 0.0
    scope_rank = 0 if row["scope"] == "aggregate" else 1
    return (recipe_rank, K, a, scope_rank)


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(
            f"input table not found: {IN_CSV}\n"
            f"run `python code/extended_metrics_table.py` first."
        )

    rows: list[dict] = []
    with IN_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "method": r.get("method", ""),
                "K": r.get("K", ""),
                "alpha": r.get("alpha", ""),
                "scope": r.get("scope", ""),
                "core_hit": _maybe(r.get("core_hit", "")),
                "core_mean_size": _maybe(r.get("core_mean_size", "")),
                "metric_kind": r.get("metric_kind", ""),
            })

    # Drop rows whose metric_kind is not core_set (text baseline reports R@10/10).
    rows = [r for r in rows if r["metric_kind"] == "core_set"]
    rows.sort(key=_sort_key)

    # ---- CSV ----
    csv_out = OUT_DIR / "cdf_table.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "K", "alpha", "scope", "core_hit", "core_mean_size"])
        for r in rows:
            w.writerow([
                r["method"], r["K"], r["alpha"], r["scope"],
                _fmt(r["core_hit"]) if r["core_hit"] is not None else "",
                _fmt(r["core_mean_size"], digits=1) if r["core_mean_size"] is not None else "",
            ])

    # ---- Markdown: one section per recipe ----
    lines: list[str] = []
    lines.append("# CDF metric (core-set hit rate) across every experiment")
    lines.append("")
    lines.append(
        "Each cell reports `core_hit / |core|`: the fraction of rows in which "
        "the ground-truth target lies above the per-row `mu + sigma` mass "
        "threshold, and the mean size of that high-mass head. "
        "Higher hit and smaller `|core|` together indicate a sharp, correct plan."
    )
    lines.append("")

    by_method: dict[str, list[dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    for method in RECIPE_ORDER:
        if method not in by_method:
            continue
        lines.append(f"## {method}")
        lines.append("")
        lines.append("| K | alpha | scope | core_hit | |core| |")
        lines.append("|---:|---:|---|---:|---:|")
        for r in by_method[method]:
            K = r["K"] if r["K"] not in (None, "") else "—"
            a = r["alpha"] if r["alpha"] not in (None, "") else "—"
            hit = _fmt(r["core_hit"])
            size = _fmt(r["core_mean_size"], digits=1)
            lines.append(f"| {K} | {a} | {r['scope']} | {hit} | {size} |")
        lines.append("")

    md_out = OUT_DIR / "cdf_table.md"
    md_out.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_out}")
    print(f"wrote {md_out}")
    print(f"{len(rows)} core-set cells across {len(by_method)} recipes")


if __name__ == "__main__":
    main()
