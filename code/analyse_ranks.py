"""Empirical CDF of the ground-truth partner's rank under each plan.

For each saved cross-modal transport plan $T \\in \\R^{n \\times n}$
at the canonical encoder pair, and for each source row $i$ with
ground-truth partner $j^* = i$ (identity correspondence under the
clip-identity lemma), compute the rank of $j^*$ in $\\arg\\sort(-T[i, :])$.

The empirical CDF of these ranks tells us whether the recipe places
the true partner *near the top* even when $R@10$ is modest --- the
semantic-plausibility test that $R@k$ at any single $k$ cannot answer.

Outputs:
  results/exp_grid/plots/rank_distribution.png
  results/exp_grid/rank_distributions.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"

# (label, plan path, encoder pair note).
PLANS = [
    ("D (caption-cost FGW)",        RES / "exp_d"     / "T_caption.npy"),
    ("C-transitive (text bridge)",  RES / "exp_c"     / "T_transitive.npy"),
    ("Pure-GW (no signal)",         RES / "exp_unsup" / "T_gw.npy"),
    ("Text-only (cosine)",          RES / "exp_text"  / "T_text.npy"),
    # C-direct plan is not saved by default; uncomment if present.
    # ("C-direct (supervised)",       RES / "exp_c"     / "T_cdirect.npy"),
]


def ranks_for(plan: np.ndarray) -> np.ndarray:
    """Rank (1-indexed) of GT = diagonal entry within each row's sort."""
    n = plan.shape[0]
    # Higher T_{ij} = more confident pairing => sort descending.
    # rank of GT in row i = 1 + count of j with T[i, j] > T[i, i].
    diag = np.diag(plan)
    ranks = (plan > diag[:, None]).sum(axis=1) + 1
    return ranks


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[pd.DataFrame] = []
    loaded: list[tuple[str, np.ndarray]] = []
    for label, path in PLANS:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        T = np.load(path)
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            print(f"[skip] {label}: shape {T.shape} not square")
            continue
        loaded.append((label, T))
        r = ranks_for(T)
        records.append(pd.DataFrame({"recipe": label, "rank": r}))
        print(f"[loaded] {label}  shape={T.shape}  "
              f"median rank={np.median(r):.1f}  "
              f"P@1={(r == 1).mean():.3f}  "
              f"P@10={(r <= 10).mean():.3f}")
    if not loaded:
        return
    n = loaded[0][1].shape[0]

    long = pd.concat(records, ignore_index=True)
    long.to_csv(RES / "exp_grid" / "rank_distributions.csv", index=False)
    print(f"[wrote] {RES / 'exp_grid' / 'rank_distributions.csv'}")

    # ----- CDF plot -----
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("colorblind", n_colors=max(3, len(loaded)))
    ks = np.arange(1, n + 1)
    chance = ks / n  # ECDF of uniform random rank.
    ax.plot(ks, chance, color="grey", linestyle="--", lw=1.2,
            label="random (chance)")
    for (label, T), color in zip(loaded, palette):
        r = ranks_for(T)
        ecdf = np.array([(r <= k).mean() for k in ks])
        ax.plot(ks, ecdf, color=color, lw=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("rank $k$ of GT partner (log scale)")
    ax.set_ylabel(r"$\Pr[\,\mathrm{rank}(j^\star) \leq k\,]$")
    ax.set_title("Empirical CDF of the ground-truth partner's rank\n"
                 f"per cross-modal recipe at the canonical encoder pair "
                 f"($n = {n}$)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    out = PLOT_DIR / "rank_distribution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
