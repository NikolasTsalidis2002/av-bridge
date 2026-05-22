"""Cumulative nucleus metric across every (recipe, K, alpha, scope) cell.

For each row i of a row-stochastic plan T:

  1. Sort T[i, :] in descending order.
  2. nucleus_size(p) = smallest k such that the cumulative top-k mass
     reaches at least p.
  3. nucleus_hit(p)  = 1 if the GT column gt[i] lies in that top-k
     prefix, else 0.

Two numbers summarise the metric at a given p:

  hit_rate       = mean over rows of nucleus_hit
  mean_nucleus   = mean over rows of nucleus_size

Read together they answer: "to cover p of the row's transport mass, how
many candidate columns does it take, and is the GT among them?"

Reported at p in {0.5, 0.9}. The text-only baseline is excluded because
its matrix is raw cosine similarity, not a probability distribution, so
a cumulative-mass quantile has no meaning.

Outputs (under results/cdf_analysis/):
  - nucleus_table.csv  (machine-readable)
  - nucleus_table.md   (human-readable, grouped per recipe)
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT_DIR = RES / "cdf_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_VALUES = (0.5, 0.9)
REUSABLE_K = 300
REUSABLE_ALPHA = 0.7
ALPHA_GRID = (0.0, 0.3, 0.5, 0.7, 0.9)
K_GRID = (10, 20, 50, 100, 160, 200, 300)


@dataclass
class Spec:
    label: str
    plan_dir: Path
    heldout_idx: Path
    # For sweepable recipes: ("K_a", K_GRID, ALPHA_GRID); single-cell:
    # (("single", filename, K, alpha))
    cells: list[tuple]


SPECS = [
    Spec(
        label="GW (unsup, alpha=1.0)",
        plan_dir=RES / "exp_unsup",
        heldout_idx=RES / "exp_unsup" / "heldout_compare_idx.npy",
        cells=[("single", "T_gw.npy", 0, 1.0)],
    ),
    Spec(
        label="FGW direct (M = caption cos)",
        plan_dir=RES / "exp_d" / "plans",
        heldout_idx=RES / "exp_d" / "heldout_compare_idx.npy",
        cells=[("alpha_only", a) for a in ALPHA_GRID],
    ),
    Spec(
        label="FGW transitive (identity bridge)",
        plan_dir=RES / "exp_c" / "plans",
        heldout_idx=RES / "exp_c" / "heldout_compare_idx.npy",
        cells=[("K_alpha", K, a) for K in K_GRID for a in ALPHA_GRID],
    ),
    Spec(
        label="Random baseline",
        plan_dir=RES / "exp_random",
        heldout_idx=RES / "exp_random" / "heldout_compare_idx.npy",
        cells=[("single", "T_random.npy", 0, float("nan"))],
    ),
]


def resolve_path(spec: Spec, cell: tuple) -> tuple[Path | None, int | str, float | str]:
    kind = cell[0]
    if kind == "single":
        _, filename, K, a = cell
        return spec.plan_dir / filename, K, a
    if kind == "alpha_only":
        _, a = cell
        per_a = spec.plan_dir / f"T__a{a:.2f}.npy"
        if per_a.exists():
            return per_a, 0, a
        if abs(a - REUSABLE_ALPHA) < 1e-9:
            canonical = spec.plan_dir.parent / "T_caption.npy"
            if canonical.exists():
                return canonical, 0, a
        return per_a, 0, a
    if kind == "K_alpha":
        _, K, a = cell
        per_cell = spec.plan_dir / f"T__K{K}__a{a:.2f}.npy"
        if per_cell.exists():
            return per_cell, K, a
        if K == REUSABLE_K and abs(a - REUSABLE_ALPHA) < 1e-9:
            canonical = spec.plan_dir.parent / "T_transitive.npy"
            if canonical.exists():
                return canonical, K, a
        return per_cell, K, a
    return None, "", ""


def nucleus_stats(T: np.ndarray, gt: np.ndarray, row_subset, p: float) -> tuple[float, float]:
    """Per-row top-p nucleus: smallest k with cum top-k mass >= p.

    Returns (hit_rate, mean_size) over the chosen row subset.
    Rows whose total mass is below p are still handled — the nucleus
    is then the entire row (size n).
    """
    n_rows, n_cols = T.shape
    if row_subset is None:
        row_subset = np.arange(n_rows)
    if row_subset.size == 0:
        return float("nan"), float("nan")

    # Normalize each row to sum to 1 so the cumulative mass threshold p is
    # interpretable as a fraction of the row's transported mass. Plans from
    # ot.gromov with uniform marginals have row sums equal to 1/n, not 1.
    row_sums = T.sum(axis=1, keepdims=True)
    T_norm = np.where(row_sums > 0, T / row_sums, T)

    order = np.argsort(-T_norm, axis=1)         # (n_rows, n_cols), descending column ids
    sorted_T = np.take_along_axis(T_norm, order, axis=1)
    cum = np.cumsum(sorted_T, axis=1)
    # rank within order: for each row, position (0..n-1) of column j in the sort.
    rank_pos = np.argsort(order, axis=1)        # rank of each column id

    # For each row i: nucleus_size = first k where cum[i, k-1] >= p
    # If no k satisfies, fall back to n_cols.
    nucleus_size = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        idx = np.searchsorted(cum[i], p, side="left")
        nucleus_size[i] = min(idx + 1, n_cols)

    # Hit: gt[i]'s rank position must be < nucleus_size[i]
    gt_pos = rank_pos[np.arange(n_rows), gt]    # 0-indexed position
    hits = (gt_pos < nucleus_size)

    return (float(hits[row_subset].mean()),
            float(nucleus_size[row_subset].mean()))


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def main() -> None:
    rows_out: list[dict] = []
    for spec in SPECS:
        S_compare = (np.load(spec.heldout_idx)
                     if spec.heldout_idx.exists() else None)

        for cell in spec.cells:
            plan_path, K, a = resolve_path(spec, cell)
            if plan_path is None or not plan_path.exists():
                continue

            T = np.load(plan_path)
            if T.ndim != 2 or T.shape[0] != T.shape[1]:
                continue
            n = T.shape[0]
            gt = np.arange(n)

            row_sums = T.sum(axis=1)
            if not np.allclose(row_sums, row_sums.mean(), atol=1e-3):
                # not row-stochastic → cumulative mass quantile is ill-defined
                continue

            heldout = (np.setdiff1d(np.arange(n), S_compare)
                       if S_compare is not None else None)

            for p in P_VALUES:
                hit_a, size_a = nucleus_stats(T, gt, np.arange(n), p)
                rows_out.append({
                    "method": spec.label, "K": K, "alpha": a, "scope": "aggregate",
                    "p": p, "nucleus_hit": hit_a, "nucleus_mean_size": size_a,
                })
                if heldout is not None and heldout.size > 0:
                    hit_h, size_h = nucleus_stats(T, gt, heldout, p)
                    rows_out.append({
                        "method": spec.label, "K": K, "alpha": a,
                        "scope": "heldout_like_c",
                        "p": p, "nucleus_hit": hit_h, "nucleus_mean_size": size_h,
                    })

    # ---- CSV ----
    csv_out = OUT_DIR / "nucleus_table.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "K", "alpha", "scope", "p",
                    "nucleus_hit", "nucleus_mean_size"])
        for r in rows_out:
            w.writerow([r["method"], r["K"], r["alpha"], r["scope"],
                        f"{r['p']:.1f}",
                        _fmt(r["nucleus_hit"]),
                        _fmt(r["nucleus_mean_size"], digits=1)])

    # ---- Markdown ----
    RECIPE_ORDER = [
        "Random baseline", "GW (unsup, alpha=1.0)",
        "FGW direct (M = caption cos)", "FGW transitive (identity bridge)",
    ]
    by_method: dict[str, list[dict]] = {}
    for r in rows_out:
        by_method.setdefault(r["method"], []).append(r)

    lines = []
    lines.append("# Cumulative nucleus metric")
    lines.append("")
    lines.append(
        "For each row of the plan, sort target columns by mass (descending) "
        "and take the smallest set whose cumulative mass reaches *p*. "
        "**nucleus_size** is the size of that set (averaged over rows); "
        "**nucleus_hit** is the fraction of rows where the GT column lies "
        "in that set. Reported at p ∈ {0.5, 0.9}."
    )
    lines.append("")
    lines.append("Text baseline is omitted because `T_text.npy` is a raw "
                 "similarity matrix, not row-stochastic.")
    lines.append("")

    for method in RECIPE_ORDER:
        if method not in by_method:
            continue
        lines.append(f"## {method}")
        lines.append("")
        lines.append("| K | α | scope | p | nucleus_hit | nucleus_size |")
        lines.append("|---:|---:|---|---:|---:|---:|")
        rows = sorted(by_method[method],
                      key=lambda r: (
                          (float(r["K"]) if r["K"] not in ("", None) else 0.0),
                          (float(r["alpha"]) if isinstance(r["alpha"], (int, float))
                           and not (isinstance(r["alpha"], float)
                                    and math.isnan(r["alpha"])) else 0.0),
                          0 if r["scope"] == "aggregate" else 1,
                          r["p"],
                      ))
        for r in rows:
            K = r["K"] if r["K"] not in ("", None) else "—"
            a = (r["alpha"] if (not isinstance(r["alpha"], float)
                                or not math.isnan(r["alpha"])) else "—")
            lines.append(
                f"| {K} | {a} | {r['scope']} | {r['p']:.1f} | "
                f"{_fmt(r['nucleus_hit'])} | {_fmt(r['nucleus_mean_size'], digits=1)} |"
            )
        lines.append("")

    (OUT_DIR / "nucleus_table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_out}")
    print(f"wrote {OUT_DIR / 'nucleus_table.md'}")
    print(f"{len(rows_out)} (recipe, K, α, scope, p) entries")


if __name__ == "__main__":
    main()
