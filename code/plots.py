"""Phase 9 — tables and plots.

For each experiment, reads results/exp_*/sweep*.csv and produces:
  - grid_aggregate.csv and grid_heldout.csv (K × alpha grid of R@10 / routes)
  - sweep.png (R@k panel + structural panel at alpha=0.5)

Also writes comparison.csv and comparison_structure.csv at the top-level
results/ folder.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_sweep(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def grid_for(df: pd.DataFrame, scope: str, K_cl: int) -> pd.DataFrame:
    sub = df[df["scope"] == scope]
    if sub.empty:
        return pd.DataFrame()
    grid = sub.pivot_table(
        index="K", columns="alpha",
        values=["R@10", "routes_correct"], aggfunc="mean",
    )
    rows = []
    for K in grid.index:
        row = {"K": K}
        for alpha in sorted(set(sub["alpha"])):
            r10 = grid.loc[K, ("R@10", alpha)]
            rc = grid.loc[K, ("routes_correct", alpha)]
            row[f"alpha={alpha:.1f}"] = f"{r10:.3f} / {int(rc)}/{K_cl}"
        rows.append(row)
    return pd.DataFrame(rows)


def find_elbow(K_vals: list[int], r10: list[float]) -> int:
    """First K such that marginal gain on R@10 falls below 0.02, else max-R@10 K."""
    if not K_vals:
        return -1
    pairs = sorted(zip(K_vals, r10), key=lambda x: x[0])
    for i in range(1, len(pairs)):
        if pairs[i][1] - pairs[i - 1][1] < 0.02:
            return pairs[i - 1][0]
    return max(pairs, key=lambda x: x[1])[0]


def plot_sweep(df: pd.DataFrame, K_cl: int, out: Path, alpha_focus: float = 0.5) -> None:
    sub = df[(df["scope"] == "aggregate") & (df["alpha"] == alpha_focus)].sort_values("K")
    if sub.empty:
        # fall back to whatever alpha is present
        sub = df[df["scope"] == "aggregate"].sort_values(["alpha", "K"])
        if sub.empty:
            print(f"[plot] no data for {out}")
            return
        alpha_focus = sub["alpha"].iloc[0]
        sub = sub[sub["alpha"] == alpha_focus]

    K = sub["K"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for k_metric, marker in zip(["R@1", "R@5", "R@10", "R@20"], ["o", "s", "^", "D"]):
        axes[0].plot(K, sub[k_metric], marker=marker, label=k_metric)
    elbow = find_elbow(K, sub["R@10"].tolist())
    if elbow > 0:
        axes[0].axvline(elbow, linestyle="--", alpha=0.5, color="grey")
        axes[0].text(elbow, 0.02, f" elbow K={elbow}", color="grey")
    axes[0].set_xlabel("K (number of paired anchors)")
    axes[0].set_ylabel(f"Recall@k  (alpha={alpha_focus})")
    axes[0].set_title("Retrieval")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(K, sub["nmi"], marker="o", label="cluster NMI")
    route_acc = sub["routes_correct"] / sub["routes_total"]
    axes[1].plot(K, route_acc, marker="s", label="route acc")
    axes[1].plot(K, sub["pearson_r"], marker="^", label="Pearson r")
    if elbow > 0:
        axes[1].axvline(elbow, linestyle="--", alpha=0.5, color="grey")
    axes[1].set_xlabel("K (number of paired anchors)")
    axes[1].set_ylabel(f"structural metric (alpha={alpha_focus})")
    axes[1].set_title("Structural")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out}")


KCL_MAP = {"exp_a": 10, "exp_b": 20, "exp_c": 15}


def emit_experiment_artifacts(exp_dir: Path, sweep_name: str) -> None:
    name = exp_dir.name
    K_cl = KCL_MAP.get(name, 10)
    df = load_sweep(exp_dir / sweep_name)
    if df.empty:
        print(f"[skip] {exp_dir} {sweep_name}")
        return

    grid_a = grid_for(df, "aggregate", K_cl)
    grid_h = grid_for(df, "heldout", K_cl)
    if not grid_a.empty:
        grid_a.to_csv(exp_dir / "grid_aggregate.csv", index=False)
        print(f"[grid] wrote {exp_dir / 'grid_aggregate.csv'}")
    if not grid_h.empty:
        grid_h.to_csv(exp_dir / "grid_heldout.csv", index=False)
        print(f"[grid] wrote {exp_dir / 'grid_heldout.csv'}")

    plot_sweep(df, K_cl, exp_dir / "sweep.png", alpha_focus=0.5)


def emit_comparison() -> None:
    """Two comparison tables (retrieval + structure), method × experiment grid."""
    rows_ret: list[dict] = []
    rows_str: list[dict] = []

    def pick(df: pd.DataFrame, scope: str, K: int, alpha: float):
        sub = df[(df["scope"] == scope) & (df["K"] == K) & (df["alpha"] == alpha)]
        if sub.empty:
            return None
        return sub.iloc[0]

    a = load_sweep(RES / "exp_a" / "sweep.csv")
    b = load_sweep(RES / "exp_b" / "sweep.csv")
    c_dir = load_sweep(RES / "exp_c" / "sweep_direct.csv")
    c_tr = load_sweep(RES / "exp_c" / "sweep_transitive.csv")

    def elbow_K(df: pd.DataFrame, alpha: float = 0.5) -> int:
        sub = df[(df["scope"] == "aggregate") & (df["alpha"] == alpha)].sort_values("K")
        if sub.empty:
            return -1
        return find_elbow(sub["K"].tolist(), sub["R@10"].tolist())

    def cell_ret(df: pd.DataFrame, scope: str, K: int, alpha: float, K_cl: int):
        r = pick(df, scope, K, alpha)
        if r is None:
            return "—"
        return f"{r['R@10']:.3f} / {int(r['routes_correct'])}/{K_cl}"

    def cell_str(df: pd.DataFrame, scope: str, K: int, alpha: float):
        r = pick(df, scope, K, alpha)
        if r is None:
            return "—"
        return f"NMI={r['nmi']:.3f} knn={r['knn_overlap']:.3f} r={r['pearson_r']:.3f}"

    K_target = 300  # reusable plan K
    alpha = 0.5

    sources = [
        ("exp_a", a, 10),
        ("exp_b", b, 20),
        ("exp_c_direct", c_dir, 15),
    ]

    # Per-method rows: "elbow" and "K=300"
    for method, K_pick in [("elbow@a=0.5", None), ("K=300@a=0.5", K_target)]:
        ret_row = {"method": method}
        str_row = {"method": method}
        for name, df, K_cl in sources:
            K_use = elbow_K(df, alpha) if K_pick is None else K_target
            ret_row[name + "_agg"] = cell_ret(df, "aggregate", K_use, alpha, K_cl)
            ret_row[name + "_hel"] = cell_ret(df, "heldout", K_use, alpha, K_cl)
            str_row[name + "_agg"] = cell_str(df, "aggregate", K_use, alpha)
            str_row[name + "_hel"] = cell_str(df, "heldout", K_use, alpha)
        rows_ret.append(ret_row)
        rows_str.append(str_row)

    # Add C-transitive row
    if not c_tr.empty:
        agg = c_tr[c_tr["scope"] == "aggregate"]
        hel = c_tr[c_tr["scope"] == "heldout"]
        if not agg.empty:
            r = agg.iloc[0]
            ret_row = {"method": "C-transitive"}
            ret_row["exp_a_agg"] = "—"
            ret_row["exp_a_hel"] = "—"
            ret_row["exp_b_agg"] = "—"
            ret_row["exp_b_hel"] = "—"
            ret_row["exp_c_direct_agg"] = f"{r['R@10']:.3f} / {int(r['routes_correct'])}/15 (transitive)"
            ret_row["exp_c_direct_hel"] = (
                f"{hel.iloc[0]['R@10']:.3f} / {int(hel.iloc[0]['routes_correct'])}/15"
                if not hel.empty else "—"
            )
            rows_ret.append(ret_row)
            str_row = {"method": "C-transitive"}
            str_row["exp_a_agg"] = "—"
            str_row["exp_a_hel"] = "—"
            str_row["exp_b_agg"] = "—"
            str_row["exp_b_hel"] = "—"
            str_row["exp_c_direct_agg"] = (
                f"NMI={r['nmi']:.3f} knn={r['knn_overlap']:.3f} r={r['pearson_r']:.3f}"
            )
            if not hel.empty:
                rh = hel.iloc[0]
                str_row["exp_c_direct_hel"] = (
                    f"NMI={rh['nmi']:.3f} knn={rh['knn_overlap']:.3f} r={rh['pearson_r']:.3f}"
                )
            else:
                str_row["exp_c_direct_hel"] = "—"
            rows_str.append(str_row)

    cols = ["method",
            "exp_a_agg", "exp_a_hel",
            "exp_b_agg", "exp_b_hel",
            "exp_c_direct_agg", "exp_c_direct_hel"]
    with (RES / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_ret:
            w.writerow({c: r.get(c, "—") for c in cols})
    with (RES / "comparison_structure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_str:
            w.writerow({c: r.get(c, "—") for c in cols})
    print(f"[comparison] wrote {RES / 'comparison.csv'}")
    print(f"[comparison] wrote {RES / 'comparison_structure.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--comparison", action="store_true")
    args = ap.parse_args()

    if args.exp in ("a", "all"):
        emit_experiment_artifacts(RES / "exp_a", "sweep.csv")
    if args.exp in ("b", "all"):
        emit_experiment_artifacts(RES / "exp_b", "sweep.csv")
    if args.exp in ("c", "all"):
        emit_experiment_artifacts(RES / "exp_c", "sweep_direct.csv")

    if args.comparison or args.exp == "all":
        emit_comparison()


if __name__ == "__main__":
    main()
