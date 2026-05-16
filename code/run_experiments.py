"""Phase 3 + 6 + 7 + 8 — anchor sampling and the three experiments.

Usage:
  python code/run_experiments.py --exp a
  python code/run_experiments.py --exp b
  python code/run_experiments.py --exp c-direct
  python code/run_experiments.py --exp c-transitive

Encoders default to clip-large + clap-unfused (matching the spec's CLIP/CLAP
spirit). Override with --image-encoder / --audio-encoder.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from algorithms import build_bridge, recipe, transitive_plan
from metrics import evaluate, evaluate_heldout

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"
RES = ROOT / "results"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Sweep grids set by the plan (rescaled for n=400).
K_GRID = [10, 20, 50, 100, 160, 200, 300, 400]
ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]

# K_cl values per experiment, per the spec.
KCL = {"a": 10, "b": 20, "c-direct": 15, "c-transitive": 15}

# Reusable-plan K for C-transitive (= 300 here, rescaled from spec's 350).
REUSABLE_K = 300
REUSABLE_ALPHA = 0.7

CSV_COLS = [
    "K", "alpha", "scope",
    "R@1", "R@5", "R@10", "R@20",
    "routes_correct", "routes_total",
    "knn_overlap", "pearson_r", "nmi",
]


def kmeans_stratified_indices(
    X: np.ndarray,
    n: int,
    n_clusters: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Pick n indices by walking KMeans clusters round-robin, taking the
    cluster centre's nearest unused member each turn.
    """
    n = int(min(n, X.shape[0]))
    if n == X.shape[0]:
        return np.arange(n)
    n_clusters = int(min(n_clusters, n))
    n_clusters = max(n_clusters, 1)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    chosen: list[int] = []
    taken: set[int] = set()
    for round_idx in range(n * 4):
        if len(chosen) == n:
            break
        c = round_idx % n_clusters
        members = [i for i in np.where(km.labels_ == c)[0] if i not in taken]
        if not members:
            continue
        dists = np.linalg.norm(
            X[members] - km.cluster_centers_[c], axis=1
        )
        pick = members[int(np.argmin(dists))]
        chosen.append(pick)
        taken.add(pick)
    return np.array(chosen, dtype=int)


def load_anchors(X_image: np.ndarray, n: int = 400) -> np.ndarray:
    path = EMB / "anchor_indices.npy"
    if path.exists():
        return np.load(path)
    idx = kmeans_stratified_indices(X_image, n=n, n_clusters=min(10, n), seed=SEED)
    np.save(path, idx)
    return idx


def load_embedding(name: str) -> np.ndarray:
    """Loads a saved L2-normalised embedding by file stem."""
    path = EMB / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing embedding: {path}")
    return np.load(path).astype(np.float32)


def select_for_anchors(M: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return M[anchors]


def fmt_cell(metrics: dict, K_cl: int) -> str:
    r10 = metrics.get("R@10", float("nan"))
    rc = metrics.get("routes_correct", 0)
    return f"{r10:.3f} / {rc}/{K_cl}"


# --- Sweep driver ----------------------------------------------------------
def run_sweep(
    X: np.ndarray,
    Y: np.ndarray,
    K_cl: int,
    csv_path: Path,
    save_plan_at: tuple[int, float] | None = None,
    plan_path: Path | None = None,
    K_grid: list[int] | None = None,
    alpha_grid: list[float] | None = None,
) -> None:
    """Run the (K, alpha) sweep, evaluate, append rows to csv_path."""
    n = X.shape[0]
    gt = np.arange(n)
    K_grid = K_grid or K_GRID
    alpha_grid = alpha_grid or ALPHA_GRID

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()

        for K in K_grid:
            if K > n:
                continue
            S = kmeans_stratified_indices(
                X, n=K, n_clusters=min(10, K), seed=SEED
            )
            for alpha in alpha_grid:
                print(f"  K={K:4d}  alpha={alpha:.2f}")
                T = recipe(X, Y, S, S, alpha=alpha, eps=0.005, lam=1.0)

                agg = evaluate(T, X, Y, gt, K_cl, seed=SEED)
                w.writerow({"K": K, "alpha": alpha, "scope": "aggregate", **agg})

                hel = evaluate_heldout(T, X, Y, gt, S, K_cl, seed=SEED)
                w.writerow({"K": K, "alpha": alpha, "scope": "heldout", **hel})

                if (save_plan_at is not None
                        and (K, alpha) == save_plan_at
                        and plan_path is not None):
                    np.save(plan_path, T)
                    print(f"    saved plan -> {plan_path.name}")
                f.flush()


# --- Experiments -----------------------------------------------------------
def exp_a(image_name: str, K_grid: list[int] | None = None) -> None:
    print(f"[Exp A] image={image_name}  target=visual captions")
    X_image = load_embedding(f"vision_{image_name}")
    ZV = load_embedding("ZV_text")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = ZV[anchors]
    run_sweep(
        X, Y,
        K_cl=KCL["a"],
        csv_path=RES / "exp_a" / "sweep.csv",
        save_plan_at=(REUSABLE_K, REUSABLE_ALPHA),
        plan_path=RES / "exp_a" / "T_iv.npy",
        K_grid=K_grid,
    )


def exp_b(audio_name: str, K_grid: list[int] | None = None) -> None:
    print(f"[Exp B] audio={audio_name}  target=audio captions")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZA = load_embedding("ZA_text")
    X_image = load_embedding(f"vision_clip-large")  # for shared anchor ordering
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = Y_audio[anchors]
    Y = ZA[anchors]
    run_sweep(
        X, Y,
        K_cl=KCL["b"],
        csv_path=RES / "exp_b" / "sweep.csv",
        save_plan_at=(REUSABLE_K, REUSABLE_ALPHA),
        plan_path=RES / "exp_b" / "T_ac.npy",
        K_grid=K_grid,
    )


def exp_c_direct(image_name: str, audio_name: str,
                 K_grid: list[int] | None = None) -> None:
    print(f"[Exp C-direct] image={image_name}  audio={audio_name}")
    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    run_sweep(
        X, Y,
        K_cl=KCL["c-direct"],
        csv_path=RES / "exp_c" / "sweep_direct.csv",
        K_grid=K_grid,
    )


def exp_c_transitive(image_name: str, audio_name: str,
                     top_k: int = 20, tau: float = 0.1,
                     K_per_leg: int = REUSABLE_K,
                     alpha_per_leg: float = REUSABLE_ALPHA) -> None:
    """Compose Exp A and Exp B plans through the text bridge."""
    print(f"[Exp C-transitive] top_k={top_k} tau={tau} "
          f"K_per_leg={K_per_leg} alpha_per_leg={alpha_per_leg}")

    T_iv = np.load(RES / "exp_a" / "T_iv.npy")
    T_ac = np.load(RES / "exp_b" / "T_ac.npy")

    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")
    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV_a = ZV[anchors]
    ZA_a = ZA[anchors]

    B = build_bridge(ZV_a, ZA_a, top_k=top_k, tau=tau)
    T = transitive_plan(T_iv, B, T_ac)

    gt = np.arange(X.shape[0])
    K_cl = KCL["c-transitive"]
    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED)
    # Held-out for transitive: rows in NEITHER A's nor B's K=REUSABLE_K index set.
    S_a = kmeans_stratified_indices(X, n=K_per_leg, n_clusters=min(10, K_per_leg), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=K_per_leg, n_clusters=min(10, K_per_leg), seed=SEED)
    S_both = np.unique(np.concatenate([S_a, S_b]))
    hel = evaluate_heldout(T, X, Y, gt, S_both, K_cl, seed=SEED)

    out = RES / "exp_c" / "sweep_transitive.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra_cols = ["bridge_top_k", "bridge_tau", "K_per_leg", "alpha_per_leg"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS + extra_cols)
        w.writeheader()
        extras = {
            "bridge_top_k": top_k, "bridge_tau": tau,
            "K_per_leg": K_per_leg, "alpha_per_leg": alpha_per_leg,
        }
        w.writerow({"K": K_per_leg, "alpha": alpha_per_leg, "scope": "aggregate", **agg, **extras})
        w.writerow({"K": K_per_leg, "alpha": alpha_per_leg, "scope": "heldout", **hel, **extras})
    print(f"  wrote {out}")
    np.save(RES / "exp_c" / "T_transitive.npy", T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp", choices=["a", "b", "c-direct", "c-transitive", "all"],
        required=True,
    )
    ap.add_argument("--image-encoder", default="clip-large")
    ap.add_argument("--audio-encoder", default="clap-unfused")
    ap.add_argument("--K-grid", type=str, default=None,
                    help="Override K sweep, e.g. '50,100,200,300,400'")
    ap.add_argument("--bridge-top-k", type=int, default=20)
    ap.add_argument("--bridge-tau", type=float, default=0.1)
    args = ap.parse_args()

    K_grid = (
        [int(x) for x in args.K_grid.split(",")]
        if args.K_grid else None
    )

    if args.exp in ("a", "all"):
        exp_a(args.image_encoder, K_grid=K_grid)
    if args.exp in ("b", "all"):
        exp_b(args.audio_encoder, K_grid=K_grid)
    if args.exp in ("c-direct", "all"):
        exp_c_direct(args.image_encoder, args.audio_encoder, K_grid=K_grid)
    if args.exp in ("c-transitive", "all"):
        exp_c_transitive(
            args.image_encoder, args.audio_encoder,
            top_k=args.bridge_top_k, tau=args.bridge_tau,
        )


if __name__ == "__main__":
    main()
