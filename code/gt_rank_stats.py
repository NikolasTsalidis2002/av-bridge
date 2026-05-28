"""Ground-truth rank position stats per (recipe, K, alpha, scope) cell.

For each row i of a plan T, rank(gt_i) is the 1-indexed position of the
GT column among T[i, :] sorted descending. We summarise that rank
distribution per cell with:

  median, mean, p25, p75, p90, min, max
  pct_rank_eq_1   (fraction landing in the top spot)
  pct_rank_le_10  (R@10 for cross-check against the sweep CSV)

Recipes covered: every (K, alpha) cell of FGW transitive (identity
bridge), every alpha cell of FGW direct (cosine M), Pure-GW, Random,
and the Text baseline (raw cosine matrix — rank by similarity).

Outputs (under results/cdf_analysis/):
  gt_rank_stats.csv
  gt_rank_stats.md
  per_row_ranks/<recipe>__K{K}__a{alpha}.npy   (the raw rank arrays)
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
RANK_DIR = OUT_DIR / "per_row_ranks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANK_DIR.mkdir(parents=True, exist_ok=True)

REUSABLE_K = 300
REUSABLE_ALPHA = 0.7
ALPHA_GRID = (0.0, 0.3, 0.5, 0.7, 0.9)
K_GRID = (10, 20, 50, 100, 160, 200, 300)


@dataclass
class Spec:
    label: str
    plan_dir: Path
    heldout_idx: Path
    cells: list[tuple]  # ("single", filename, K, alpha) | ("alpha_only", a) | ("K_alpha", K, a)


SPECS = [
    Spec("Random baseline",
         RES / "exp_random", RES / "exp_random" / "heldout_compare_idx.npy",
         [("single", "T_random.npy", 0, float("nan"))]),
    Spec("GW (Pure-GW, alpha=1.0)",
         RES / "exp_unsup", RES / "exp_unsup" / "heldout_compare_idx.npy",
         [("single", "T_gw.npy", 0, 1.0)]),
    Spec("FGW direct (M = caption cos)",
         RES / "exp_d" / "plans", RES / "exp_d" / "heldout_compare_idx.npy",
         [("alpha_only", a) for a in ALPHA_GRID]),
    Spec("FGW transitive (identity bridge)",
         RES / "exp_c" / "plans", RES / "exp_c" / "heldout_compare_idx.npy",
         [("K_alpha", K, a) for K in K_GRID for a in ALPHA_GRID]),
    Spec("Text baseline (raw caption cosine)",
         RES / "exp_text", RES / "exp_text" / "heldout_compare_idx.npy",
         [("single", "T_text.npy", 0, float("nan"))]),
]


def resolve_plan_path(spec: Spec, cell: tuple) -> tuple[Path | None, int | str, float | str]:
    kind = cell[0]
    if kind == "single":
        _, fn, K, a = cell
        return spec.plan_dir / fn, K, a
    if kind == "alpha_only":
        _, a = cell
        p = spec.plan_dir / f"T__a{a:.2f}.npy"
        if p.exists():
            return p, 0, a
        if abs(a - REUSABLE_ALPHA) < 1e-9:
            canonical = spec.plan_dir.parent / "T_caption.npy"
            if canonical.exists():
                return canonical, 0, a
        return p, 0, a
    if kind == "K_alpha":
        _, K, a = cell
        p = spec.plan_dir / f"T__K{K}__a{a:.2f}.npy"
        if p.exists():
            return p, K, a
        if K == REUSABLE_K and abs(a - REUSABLE_ALPHA) < 1e-9:
            canonical = spec.plan_dir.parent / "T_transitive.npy"
            if canonical.exists():
                return canonical, K, a
        return p, K, a
    return None, "", ""


def ranks_for(T: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """1-indexed rank of GT column in each row (sorted descending)."""
    diag = T[np.arange(T.shape[0]), gt]
    return (T > diag[:, None]).sum(axis=1) + 1


def stats_dict(ranks: np.ndarray, n_cols: int) -> dict:
    if ranks.size == 0:
        return {k: float("nan") for k in
                ("median", "mean", "p25", "p75", "p90", "min", "max",
                 "pct_rank_eq_1", "pct_rank_le_10")}
    return {
        "median": float(np.median(ranks)),
        "mean":   float(ranks.mean()),
        "p25":    float(np.percentile(ranks, 25)),
        "p75":    float(np.percentile(ranks, 75)),
        "p90":    float(np.percentile(ranks, 90)),
        "min":    int(ranks.min()),
        "max":    int(ranks.max()),
        "pct_rank_eq_1":  float((ranks == 1).mean()),
        "pct_rank_le_10": float((ranks <= 10).mean()),
    }


def fmt_num(x, digits=1):
    if isinstance(x, float) and math.isnan(x):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    all_rows: list[dict] = []
    for spec in SPECS:
        S_compare = (np.load(spec.heldout_idx)
                     if spec.heldout_idx.exists() else None)

        for cell in spec.cells:
            plan_path, K, a = resolve_plan_path(spec, cell)
            if plan_path is None or not plan_path.exists():
                continue
            T = np.load(plan_path)
            if T.ndim != 2 or T.shape[0] != T.shape[1]:
                continue
            n = T.shape[0]
            gt = np.arange(n)
            ranks = ranks_for(T, gt)

            # persist rank array
            tag = (f"{spec.label.replace(' ', '_').replace('(', '').replace(')', '')[:48]}"
                   f"__K{int(K) if K != '' else 0}__a{a if isinstance(a, str) else f'{a:.2f}'}")
            np.save(RANK_DIR / f"{tag}.npy", ranks)

            # aggregate
            all_rows.append({
                "method": spec.label, "K": K, "alpha": a, "scope": "aggregate",
                **stats_dict(ranks, n),
            })
            # heldout
            if S_compare is not None:
                heldout = np.setdiff1d(np.arange(n), S_compare)
                if heldout.size > 0:
                    all_rows.append({
                        "method": spec.label, "K": K, "alpha": a,
                        "scope": "heldout_like_c",
                        **stats_dict(ranks[heldout], n),
                    })

    # ---- CSV ----
    fieldnames = ["method", "K", "alpha", "scope",
                  "median", "mean", "p25", "p75", "p90", "min", "max",
                  "pct_rank_eq_1", "pct_rank_le_10"]
    csv_out = OUT_DIR / "gt_rank_stats.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # ---- Markdown grouped per recipe ----
    RECIPE_ORDER = [s.label for s in SPECS]
    by_method: dict[str, list[dict]] = {}
    for r in all_rows:
        by_method.setdefault(r["method"], []).append(r)

    lines = []
    lines.append("# GT rank-position stats per (recipe, K, α, scope) cell")
    lines.append("")
    lines.append(
        "For each query, the GT's rank position among the 400 candidate "
        "audios (1 = top). Stats are summarised across rows. Lower = better; "
        "chance median ≈ 200."
    )
    lines.append("")
    lines.append("Columns:")
    lines.append("- **median / mean** of GT rank")
    lines.append("- **p25 / p75 / p90** — distribution quartiles")
    lines.append("- **min / max** — best/worst case")
    lines.append("- **pct_rank_eq_1** — fraction where GT is the top-1 prediction (= R@1)")
    lines.append("- **pct_rank_le_10** — fraction where GT is in top 10 (= R@10)")
    lines.append("")
    for method in RECIPE_ORDER:
        if method not in by_method:
            continue
        lines.append(f"## {method}")
        lines.append("")
        lines.append("| K | α | scope | median | mean | p25 | p75 | p90 | min | max | R@1 | R@10 |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows = sorted(by_method[method], key=lambda r: (
            float(r["K"]) if r["K"] not in (None, "") else 0.0,
            (float(r["alpha"]) if isinstance(r["alpha"], (int, float))
             and not (isinstance(r["alpha"], float) and math.isnan(r["alpha"])) else -1.0),
            0 if r["scope"] == "aggregate" else 1,
        ))
        for r in rows:
            K = r["K"] if r["K"] not in (None, "") else "—"
            a = (r["alpha"] if not (isinstance(r["alpha"], float) and math.isnan(r["alpha"]))
                 else "—")
            lines.append(
                f"| {K} | {a} | {r['scope']} | "
                f"{fmt_num(r['median'])} | {fmt_num(r['mean'])} | "
                f"{fmt_num(r['p25'])} | {fmt_num(r['p75'])} | {fmt_num(r['p90'])} | "
                f"{r['min']} | {r['max']} | "
                f"{fmt_num(r['pct_rank_eq_1'], 3)} | {fmt_num(r['pct_rank_le_10'], 3)} |"
            )
        lines.append("")
    md_out = OUT_DIR / "gt_rank_stats.md"
    md_out.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_out}")
    print(f"wrote {md_out}")
    print(f"wrote {len(list(RANK_DIR.iterdir()))} rank arrays under {RANK_DIR}")
    print(f"{len(all_rows)} (recipe, K, α, scope) entries")


if __name__ == "__main__":
    main()
