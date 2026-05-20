"""Quantify geometric similarity between every image / audio encoder pair.

Two complementary measures answer ``are the two embedding spaces
geometrically similar enough that a cross-modal alignment is possible?''

  - Linear CKA: a centred-Gram correlation between $X$ and $Y$ that
    is 1 iff the two inner-product structures are identical up to a
    rotation and a global scale (\\citet{kornblith2019similarity}).
  - Identity-permutation Pearson $r$: correlation between the upper
    triangles of $D_X[i,j] = \\|X_i - X_j\\|$ and
    $D_Y[i,j] = \\|Y_i - Y_j\\|$. Equal to one iff paired items
    occupy identical relative positions in the two spaces.

Both metrics are computed on the $n = 400$ row-aligned AVCaps pool.
Outputs:
  results/exp_grid/plots/cka_heatmap.png
  results/exp_grid/plots/identity_pearson_heatmap.png
  results/exp_grid/geometric_similarity.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sns.set_theme(style="whitegrid", context="notebook", palette="colorblind",
              font_scale=0.95)

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"
OUT_DIR = ROOT / "results" / "exp_grid"
PLOT_DIR = OUT_DIR / "plots"

IMAGE_ENCS = ["clip-base", "clip-large",
              "dinov2-small", "dinov2-base", "dinov2-large",
              "vit-mae-base"]
AUDIO_ENCS = ["clap-unfused", "clap-fused", "clap-larger",
              "mert-95m", "mert-330m"]


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two row-aligned matrices $X, Y \\in \\R^{n \\times d}$.

    CKA(X, Y) = \\|X_c^T Y_c\\|_F^2 / (\\|X_c^T X_c\\|_F * \\|Y_c^T Y_c\\|_F),
    where X_c = X - mean(X, axis=0). Range: [0, 1]; 1 iff identical
    inner-product structure up to global rotation.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    den = (np.linalg.norm(Xc.T @ Xc, ord="fro")
           * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    return float(num / den) if den > 0 else float("nan")


def identity_pearson(X: np.ndarray, Y: np.ndarray) -> float:
    """Pearson r between upper-triangles of pairwise distance matrices.

    Equal to the Pearson r that Pure-GW recovers under the identity
    permutation; it is therefore the natural pre-alignment baseline
    for the structural fidelity metric reported in the chapter.
    """
    n = X.shape[0]
    Dx = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    Dy = np.linalg.norm(Y[:, None] - Y[None, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    a, b = Dx[iu], Dy[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def heatmap(pivot: pd.DataFrame, title: str, out: Path,
            cmap: str = "magma", vmin=None, vmax=None, fmt: str = ".3f"):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=True, fmt=fmt, annot_kws={"fontsize": 9},
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": pivot.attrs.get("metric", "value"),
                          "shrink": 0.85})
    ax.set_xlabel("audio encoder")
    ax.set_ylabel("image encoder")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")


def main() -> None:
    rows = []
    for img in IMAGE_ENCS:
        x_path = EMB / f"vision_{img}.npy"
        if not x_path.exists():
            print(f"[skip] missing {x_path}")
            continue
        X = np.load(x_path)
        for aud in AUDIO_ENCS:
            y_path = EMB / f"audio_{aud}.npy"
            if not y_path.exists():
                continue
            Y = np.load(y_path)
            cka = linear_cka(X, Y)
            r_id = identity_pearson(X, Y)
            rows.append({"image_encoder": img, "audio_encoder": aud,
                         "cka": cka, "pearson_r_identity": r_id})
            print(f"  {img:14s} x {aud:14s}  CKA={cka:.3f}  r_id={r_id:.3f}")

    if not rows:
        print("[err] no encoder pairs found; nothing to do")
        return

    df = pd.DataFrame(rows)
    csv = OUT_DIR / "geometric_similarity.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"[wrote] {csv}")

    for col, cmap, vmin, vmax, fname, label in [
        ("cka",                "magma", 0, None,
         "cka_heatmap.png",                "Linear CKA between image & audio encoders"),
        ("pearson_r_identity", "magma", 0, None,
         "identity_pearson_heatmap.png",
         "Identity-permutation Pearson $r$ (raw pairwise distances)"),
    ]:
        pv = df.pivot(index="image_encoder", columns="audio_encoder", values=col)
        pv = pv.sort_index().sort_index(axis=1)
        pv.attrs["metric"] = col
        heatmap(pv, label, PLOT_DIR / fname, cmap=cmap, vmin=vmin, vmax=vmax)


if __name__ == "__main__":
    main()
