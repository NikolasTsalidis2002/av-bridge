"""Single scatter per scope: each marker = one (K, α) cell.

  x  = samples_to_threshold  (mean # columns above row's μ+σ)
  y  = recall_at_threshold   (= R@x with K=samples_to_threshold)
  hue   = α    (FGW blend)
  size  = K    (paired anchors per modality)

So one point tells you all four numbers at once. Single-cell baselines
(Random, Pure-GW) are plotted as separate labeled markers.

Output: results/cdf_analysis/recall_at_threshold.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook",
              palette="colorblind", font_scale=1.0)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
IN_CSV = RES / "core_set_analysis" / "extended_table.csv"
OUT = RES / "cdf_analysis" / "recall_at_threshold.png"


ALPHA_VALUES = (0.0, 0.3, 0.5, 0.7, 0.9)
ALPHA_COLORS = dict(zip(ALPHA_VALUES,
                        sns.color_palette("viridis", n_colors=len(ALPHA_VALUES))))
K_VALUES = (10, 20, 50, 100, 160, 200, 300)
K_SIZE_MIN, K_SIZE_MAX = 40, 320


def k_to_size(K: float) -> float:
    """Linearly map K to marker area between K_SIZE_MIN and K_SIZE_MAX."""
    return K_SIZE_MIN + (K_SIZE_MAX - K_SIZE_MIN) * (K - K_VALUES[0]) / (K_VALUES[-1] - K_VALUES[0])


BASELINE_LINE_STYLE = {
    "Random baseline":                  {"color": "#888888", "ls": ":",  "label": "Random (R@10 chance = 10/N)"},
    "GW (unsup, alpha=1.0)":             {"color": "#55a868", "ls": "--", "label": "Pure-GW"},
    "FGW direct (M = caption cos)":      {"color": "#dd8452", "ls": "-.", "label": "FGW direct (α=0.7)"},
    "Text baseline (cosine sim, NOT row-stochastic)": {
        "color": "#c44e52", "ls": (0, (3, 1, 1, 1)),
        "label": "Text-only (R@10)"},
}


def _panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    tr = df[(df.metric_kind == "core_set") &
            (df.method == "FGW transitive (identity bridge)")].copy()

    for _, row in tr.iterrows():
        if not (np.isfinite(row.core_mean_size) and np.isfinite(row.core_hit)):
            continue
        a = float(row.alpha)
        K = int(row.K)
        ax.scatter([row.core_mean_size], [row.core_hit],
                   s=k_to_size(K),
                   color=ALPHA_COLORS[a],
                   edgecolor="white", linewidth=0.8,
                   alpha=0.9, zorder=4)

    # Baselines as horizontal dashed lines spanning the whole x range.
    for method, style in BASELINE_LINE_STYLE.items():
        # Random: theoretical R@10 chance = 10 / N regardless of scope or data.
        if method.startswith("Random"):
            ax.axhline(10.0 / 400.0, color=style["color"], lw=1.4, ls=style["ls"],
                       alpha=0.85, label=style["label"], zorder=3)
            continue
        row = df[df.method == method]
        if row.empty:
            continue
        # Text-only: not row-stochastic. Use the actual R@10 value (data-driven).
        if method.startswith("Text"):
            y_vals = row.r_at_10.dropna()
        # FGW direct: canonical α=0.7 cell on the recall_at_threshold scale.
        elif method.startswith("FGW direct"):
            cand = row[np.isclose(row.alpha, 0.7)]
            y_vals = cand.core_hit.dropna() if not cand.empty else pd.Series(dtype=float)
        else:
            y_vals = row.core_hit.dropna()
        if y_vals.empty:
            continue
        y = float(y_vals.iloc[0])
        if not np.isfinite(y):
            continue
        ax.axhline(y, color=style["color"], lw=1.4, ls=style["ls"],
                   alpha=0.85, label=style["label"], zorder=3)

    ticks = list(range(0, 91, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0, 90)
    ax.grid(True, which="major", axis="x", alpha=0.4)
    ax.set_xlabel(r"samples_to_threshold  (mean # columns above $\mu+\sigma$)")
    ax.set_ylabel(r"recall_at_threshold  ($R@x$, where $x$ = samples_to_threshold)")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.02, 0.7)
    sns.despine(ax=ax)


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"input not found: {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    _panel(axes[0], df[df.scope == "aggregate"], "Aggregate (all 400 queries)")
    _panel(axes[1], df[df.scope == "heldout_like_c"],
           "Held-out (rows not used as anchors)")

    # ----- legends -----
    # α colour legend
    alpha_handles = [
        mlines.Line2D([], [], marker="o", linestyle="",
                      markersize=9, color=ALPHA_COLORS[a],
                      markeredgecolor="white",
                      label=f"α = {a:.1f}")
        for a in ALPHA_VALUES
    ]
    # K size legend (chosen anchor values)
    K_show = [10, 50, 100, 200, 300]
    k_handles = [
        mlines.Line2D([], [], marker="o", linestyle="",
                      markersize=np.sqrt(k_to_size(K)) * 0.6,
                      color="#444444",
                      markeredgecolor="white",
                      label=f"K = {K}")
        for K in K_show
    ]
    # baseline legend (horizontal lines)
    bl_handles = [
        mlines.Line2D([], [], linestyle=style["ls"], lw=2.0,
                      color=style["color"], label=style["label"])
        for style in BASELINE_LINE_STYLE.values()
    ]

    leg1 = fig.legend(handles=alpha_handles, title="α",
                      loc="center left", bbox_to_anchor=(0.92, 0.78),
                      fontsize=9, title_fontsize=10, frameon=False)
    leg2 = fig.legend(handles=k_handles, title="K",
                      loc="center left", bbox_to_anchor=(0.92, 0.50),
                      fontsize=9, title_fontsize=10, frameon=False)
    leg3 = fig.legend(handles=bl_handles, title="baseline",
                      loc="center left", bbox_to_anchor=(0.92, 0.20),
                      fontsize=9, title_fontsize=10, frameon=False)
    fig.add_artist(leg1); fig.add_artist(leg2)  # don't replace each other

    fig.suptitle(
        "FGW transitive — each marker is one (K, α) cell.\n"
        "Color = α, size = K. Upper-left = sharp & correct.",
        fontsize=12, y=1.005,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
