r"""Empirical CDF of the ground-truth partner's rank under each plan.

For each saved cross-modal transport plan $T \in \mathbb{R}^{n \times n}$
at a fixed encoder pair, and for each source row $i$ with ground-truth
partner $j^* = i$ (identity correspondence under the clip-identity
lemma), compute the rank of $j^*$ in $\arg\sort(-T[i, :])$.

The empirical CDF of these ranks tells us whether the recipe places
the true partner *near the top* even when $R@10$ is modest --- the
semantic-plausibility test that $R@k$ at any single $k$ cannot
answer.

This script overlays the rank-CDF for two reference encoder pairs:
  - Canonical (CLIP-L/14 + CLAP-HTSAT-unfused): text-aligned default.
  - Text-free (DINOv2-large + MERT-330m): no encoder--text alignment.

Recipes are colour-coded; the pair is encoded by linestyle (solid =
canonical, dashed = text-free).

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
import matplotlib.lines as mlines

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PLOT_DIR = RES / "exp_grid" / "plots"


# Reference pairs and the suffix used in directory names.
PAIRS = [
    {"label": "canonical (CLIP-L/14 + CLAP-HTSAT-unfused)",
     "short": "canonical",
     "suffix": "",
     "linestyle": "-"},
    {"label": "text-free (DINOv2-large + MERT-330m)",
     "short": "text-free",
     "suffix": "__dinov2-large__mert-330m",
     "linestyle": "--"},
]


# Per-recipe plan filename (within whichever exp_*<suffix> directory).
RECIPES = [
    {"label": "FGW with caption cost",       "exp": "exp_d",     "file": "T_caption.npy"},
    {"label": "Text-bridged composition",    "exp": "exp_c",     "file": "T_transitive.npy"},
    {"label": "GW (intra-modal geometry)",   "exp": "exp_unsup", "file": "T_gw.npy"},
    {"label": "Raw caption cosine",          "exp": "exp_text",  "file": "T_text.npy"},
]


def ranks_for(plan: np.ndarray) -> np.ndarray:
    """Rank (1-indexed) of GT = diagonal entry within each row's sort."""
    diag = np.diag(plan)
    ranks = (plan > diag[:, None]).sum(axis=1) + 1
    return ranks


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[pd.DataFrame] = []
    plot_data: list[tuple[str, str, str, np.ndarray, int]] = []
    # (recipe_label, pair_label, linestyle, rank_array, n)
    n_global: int | None = None

    for pair in PAIRS:
        for rec in RECIPES:
            path = RES / f"{rec['exp']}{pair['suffix']}" / rec["file"]
            if not path.exists():
                print(f"[skip] {pair['short']} / {rec['label']}: "
                      f"missing {path}")
                continue
            T = np.load(path)
            if T.ndim != 2 or T.shape[0] != T.shape[1]:
                print(f"[skip] {path}: shape {T.shape} not square")
                continue
            n = T.shape[0]
            if n_global is None:
                n_global = n
            r = ranks_for(T)
            plot_data.append((rec["label"], pair["label"],
                              pair["linestyle"], r, n))
            records.append(pd.DataFrame({
                "recipe": rec["label"], "pair": pair["short"],
                "rank": r,
            }))
            print(f"[loaded] {pair['short']:10s}  {rec['label']:30s}  "
                  f"n={n}  median rank={np.median(r):.1f}  "
                  f"P@1={(r == 1).mean():.3f}  "
                  f"P@10={(r <= 10).mean():.3f}")

    if not plot_data:
        print("[err] no plans loaded; nothing to render")
        return

    long = pd.concat(records, ignore_index=True)
    long.to_csv(RES / "exp_grid" / "rank_distributions.csv", index=False)
    print(f"[wrote] {RES / 'exp_grid' / 'rank_distributions.csv'}")

    # Per-recipe colour so the same recipe's two curves share a hue.
    recipe_labels = list(dict.fromkeys(r["label"] for r in RECIPES))
    palette = dict(zip(recipe_labels,
                       sns.color_palette("colorblind",
                                         n_colors=len(recipe_labels))))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ks = np.arange(1, n_global + 1)
    chance = ks / n_global
    ax.plot(ks, chance, color="grey", linestyle=":", lw=1.2,
            label="random (chance)")

    for recipe_label, pair_label, ls, r, n in plot_data:
        ecdf = np.array([(r <= k).mean() for k in ks[:n]])
        ax.plot(ks[:n], ecdf, color=palette[recipe_label],
                linestyle=ls, lw=1.6, alpha=0.95)

    ax.set_xscale("log")
    ax.set_xlabel(r"rank $k$ of GT partner (log scale)")
    ax.set_ylabel(r"$\Pr[\,\mathrm{rank}(j^\star) \leq k\,]$")
    ax.set_title(
        "Empirical CDF of the GT partner's rank, two reference encoder pairs"
    )
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    sns.despine(ax=ax)

    # Two-part legend: one block for recipe (hue), one for pair (linestyle).
    recipe_handles = [
        mlines.Line2D([], [], color=palette[r], lw=1.6, label=r)
        for r in recipe_labels
    ]
    pair_handles = [
        mlines.Line2D([], [], color="black",
                      linestyle=p["linestyle"], lw=1.6, label=p["label"])
        for p in PAIRS
    ]
    chance_handle = mlines.Line2D([], [], color="grey", linestyle=":",
                                  lw=1.2, label="random (chance)")
    leg1 = ax.legend(handles=recipe_handles, title="recipe",
                     fontsize=8, title_fontsize=9,
                     loc="lower right", bbox_to_anchor=(1.0, 0.0))
    leg2 = ax.legend(handles=pair_handles + [chance_handle],
                     title="encoder pair", fontsize=8, title_fontsize=9,
                     loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(leg1)

    fig.tight_layout()
    out = PLOT_DIR / "rank_distribution.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
