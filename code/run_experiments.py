"""Phase 3 + 6 + 7 + 8 — anchor sampling and the experiments.

Usage:
  python code/run_experiments.py --exp a
  python code/run_experiments.py --exp b
  python code/run_experiments.py --exp c-direct
  python code/run_experiments.py --exp c-transitive
  python code/run_experiments.py --exp d

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

from algorithms import (
    build_bridge,
    caption_cost_recipe,
    pure_gw,
    recipe,
    text_only_retrieval,
    transitive_plan,
)
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
KCL = {
    "a": 10, "b": 20,
    "c-direct": 15, "c-transitive": 15,
    "d": 15, "unsup": 15, "text": 15,
}

# Reusable-plan K for C-transitive (= 300 here, rescaled from spec's 350).
REUSABLE_K = 300
REUSABLE_ALPHA = 0.7

CSV_COLS = [
    "K", "alpha", "scope",
    "R@1", "R@5", "R@10", "R@20",
    "routes_correct", "routes_total",
    "knn_overlap", "pearson_r",
    "nmi", "ami", "ari", "v_measure", "homogeneity", "completeness",
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
DEFAULT_IMAGE = "clip-large"
DEFAULT_AUDIO = "clap-unfused"


def _suffix_for(default: str, picked: str) -> str:
    """Empty when running at the default encoder; otherwise '__<encoder>'."""
    return "" if picked == default else f"__{picked}"


def exp_a(image_name: str, K_grid: list[int] | None = None) -> None:
    print(f"[Exp A] image={image_name}  target=visual captions")
    X_image = load_embedding(f"vision_{image_name}")
    ZV = load_embedding("ZV_text")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = ZV[anchors]
    out_dir = RES / f"exp_a{_suffix_for(DEFAULT_IMAGE, image_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_sweep(
        X, Y,
        K_cl=KCL["a"],
        csv_path=out_dir / "sweep.csv",
        save_plan_at=(REUSABLE_K, REUSABLE_ALPHA),
        plan_path=out_dir / "T_iv.npy",
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
    out_dir = RES / f"exp_b{_suffix_for(DEFAULT_AUDIO, audio_name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_sweep(
        X, Y,
        K_cl=KCL["b"],
        csv_path=out_dir / "sweep.csv",
        save_plan_at=(REUSABLE_K, REUSABLE_ALPHA),
        plan_path=out_dir / "T_ac.npy",
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


def exp_d_caption(image_name: str, audio_name: str,
                  alpha_grid: list[float] | None = None) -> None:
    """Experiment D — caption-cost FGW for image -> audio.

    No anchors, no ridge: M is built directly from caption distances
    (ZV vs ZA) and FGW combines it with the intra-modal structural
    costs of the image and audio embeddings.
    """
    print(f"[Exp D] image={image_name}  audio={audio_name}  "
          f"M = caption sqdist (ZV vs ZA)")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]
    ZV = ZV[anchors]
    ZA = ZA[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["d"]
    alpha_grid = alpha_grid or ALPHA_GRID

    out_dir = RES / "exp_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # "Same-rows view" partition for comparability with C-transitive's held-out:
    # rows lying outside the union of A's and B's K=REUSABLE_K anchor sets.
    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    saved_plan = None
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for alpha in alpha_grid:
            print(f"  alpha={alpha:.2f}")
            T = caption_cost_recipe(X, Y, ZV, ZA, alpha=alpha, eps=0.005)
            agg = evaluate(T, X, Y, gt, K_cl, seed=SEED)
            # K column is 0: no anchors / no ridge in this experiment.
            w.writerow({"K": 0, "alpha": alpha, "scope": "aggregate", **agg})
            if abs(alpha - REUSABLE_ALPHA) < 1e-12:
                # Same-rows view at the canonical alpha — flagged as not a
                # real held-out (D has no anchors), only for comparability
                # with C-direct/C-transitive at the same out-of-A∪B rows.
                cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED)
                w.writerow({"K": 0, "alpha": alpha, "scope": "heldout_like_c", **cmp})
                saved_plan = T
            f.flush()

    if saved_plan is not None:
        np.save(out_dir / "T_caption.npy", saved_plan)
        print(f"  wrote {csv_path}")
        print(f"  wrote {out_dir / 'T_caption.npy'}")


def exp_unsupervised_gw(image_name: str, audio_name: str) -> None:
    """Fully unsupervised image -> audio alignment with pure entropic GW.

    No anchors, no ridge, no captions: the plan is determined entirely by
    the alignment of the image and audio intra-modal distance geometries.
    This is the lower-bound reference (Regime 2 of the methodology) for
    every other image-audio variant.
    """
    print(f"[Exp unsup-GW] image={image_name}  audio={audio_name}  "
          f"M = none, pure entropic GW(C1, C2)")

    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")
    anchors = load_anchors(X_image, n=X_image.shape[0])
    X = X_image[anchors]
    Y = Y_audio[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["unsup"]

    print("  solving entropic GW...")
    T = pure_gw(X, Y, eps=0.005)
    print(f"  plan shape={T.shape}  sum={T.sum():.6f}")

    out_dir = RES / "exp_unsup"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # Same-rows view: 100 rows outside the union of A's and B's K=300
    # anchor sets, for direct comparison with C-direct held-out,
    # C-transitive held-out, and Experiment D same-rows.
    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED)
    cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        # K=0 sentinel, alpha=1.0 (pure GW endpoint).
        w.writerow({"K": 0, "alpha": 1.0, "scope": "aggregate", **agg})
        w.writerow({"K": 0, "alpha": 1.0, "scope": "heldout_like_c", **cmp})

    np.save(out_dir / "T_gw.npy", T)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_gw.npy'}")


def exp_text_only(image_name: str, audio_name: str) -> None:
    """Pure text-caption retrieval baseline for image -> audio.

    No OT, no FGW, no image or audio embeddings at all. The "plan" is
    the raw cosine-similarity matrix between the visual-caption and
    audio-caption pools, fed to the same metric suite as the other
    experiments so the numbers are directly comparable.

    The image_name and audio_name arguments are accepted only so that
    the structural metrics (kNN overlap, Pearson r, cluster NMI) can
    be computed on the same X, Y as the other image-audio experiments;
    they do NOT enter the construction of the plan itself.
    """
    print(f"[Exp text-only] retrieval = cosine(ZV, ZA) directly  (no OT)")

    ZV = load_embedding("ZV_text")
    ZA = load_embedding("ZA_text")
    X_image = load_embedding(f"vision_{image_name}")
    Y_audio = load_embedding(f"audio_{audio_name}")

    anchors = load_anchors(X_image, n=X_image.shape[0])
    ZV = ZV[anchors]
    ZA = ZA[anchors]
    X = X_image[anchors]
    Y = Y_audio[anchors]

    gt = np.arange(X.shape[0])
    K_cl = KCL["text"]

    T = text_only_retrieval(ZV, ZA)
    print(f"  similarity matrix shape={T.shape}  range=[{T.min():.3f},{T.max():.3f}]")

    out_dir = RES / "exp_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    # Same-rows view: rows outside the union of A's and B's K=300 anchor
    # sets, matching every other image-audio variant for direct comparison.
    S_a = kmeans_stratified_indices(X, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_b = kmeans_stratified_indices(Y, n=REUSABLE_K, n_clusters=min(10, REUSABLE_K), seed=SEED)
    S_compare = np.unique(np.concatenate([S_a, S_b]))

    agg = evaluate(T, X, Y, gt, K_cl, seed=SEED)
    cmp = evaluate_heldout(T, X, Y, gt, S_compare, K_cl, seed=SEED)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        # K=0, alpha=NaN sentinels: no anchors, no FGW blend.
        w.writerow({"K": 0, "alpha": float("nan"), "scope": "aggregate", **agg})
        w.writerow({"K": 0, "alpha": float("nan"), "scope": "heldout_like_c", **cmp})

    np.save(out_dir / "T_text.npy", T)
    print(f"  wrote {csv_path}")
    print(f"  wrote {out_dir / 'T_text.npy'}")


def _discover_encoders() -> tuple[list[str], list[str]]:
    """List all (image, audio) encoder names that have a saved embedding."""
    vision = sorted(p.stem.replace("vision_", "") for p in EMB.glob("vision_*.npy"))
    audio = sorted(p.stem.replace("audio_", "") for p in EMB.glob("audio_*.npy"))
    return vision, audio


GRID_CSV_COLS = [
    "experiment", "image_encoder", "audio_encoder",
    "K", "alpha", "scope",
    "R@1", "R@5", "R@10", "R@20",
    "routes_correct", "routes_total",
    "knn_overlap", "pearson_r",
    "nmi", "ami", "ari", "v_measure", "homogeneity", "completeness",
]


def exp_encoder_grid(
    image_only: list[str] | None = None,
    audio_only: list[str] | None = None,
) -> None:
    """Canonical-operating-point ablation across the full encoder registry.

    For each combination of available image and audio encoders, run
        - C-direct  at (K = REUSABLE_K, alpha = 0.5),
        - D         at alpha = 0.7,
        - Unsup     (pure entropic GW),
        - Text      (raw caption cosine).
    For each image encoder run A at the canonical (K, alpha); same for B.
    Append every row to results/exp_grid/sweep.csv in long form.
    """
    vision_all, audio_all = _discover_encoders()
    vision = image_only if image_only else vision_all
    audio = audio_only if audio_only else audio_all
    print(f"[grid] vision={vision}")
    print(f"[grid] audio ={audio}")

    ZV_full = load_embedding("ZV_text")
    ZA_full = load_embedding("ZA_text")
    K_target = REUSABLE_K
    alpha_iv = 0.5  # canonical for the sweep experiments
    alpha_d  = 0.7  # canonical for D (matches RESULTS-headline)
    eps = 0.005

    out_dir = RES / "exp_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep.csv"

    def write_rows(rows: list[dict]) -> None:
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GRID_CSV_COLS)
            w.writeheader()
            w.writerows(rows)

    rows: list[dict] = []

    # ---- Unimodal A: image -> visual-caption text -----------------------
    for img in vision:
        try:
            X_full = load_embedding(f"vision_{img}")
        except FileNotFoundError as e:
            print(f"  ! skip A on {img}: {e}")
            continue
        anchors = load_anchors(X_full, n=X_full.shape[0])
        X = X_full[anchors]
        Y = ZV_full[anchors]
        S = kmeans_stratified_indices(X, n=K_target,
                                      n_clusters=min(10, K_target), seed=SEED)
        gt = np.arange(X.shape[0])
        print(f"  [A] image={img}")
        T = recipe(X, Y, S, S, alpha=alpha_iv, eps=eps, lam=1.0)
        for scope, m in [("aggregate", evaluate(T, X, Y, gt, KCL["a"], seed=SEED)),
                         ("heldout",   evaluate_heldout(T, X, Y, gt, S, KCL["a"], seed=SEED))]:
            rows.append({"experiment": "a", "image_encoder": img, "audio_encoder": "—",
                         "K": K_target, "alpha": alpha_iv, "scope": scope, **m})

    # ---- Unimodal B: audio -> audio-caption text ------------------------
    for aud in audio:
        try:
            Y_full = load_embedding(f"audio_{aud}")
        except FileNotFoundError as e:
            print(f"  ! skip B on {aud}: {e}")
            continue
        # Use the same anchor ordering as A (driven by the default image encoder
        # if available, otherwise just identity at n = n_full).
        anchor_ref = vision[0] if vision else None
        if anchor_ref is not None:
            anchors = load_anchors(load_embedding(f"vision_{anchor_ref}"),
                                   n=Y_full.shape[0])
        else:
            anchors = np.arange(Y_full.shape[0])
        X = Y_full[anchors]
        Y = ZA_full[anchors]
        S = kmeans_stratified_indices(X, n=K_target,
                                      n_clusters=min(10, K_target), seed=SEED)
        gt = np.arange(X.shape[0])
        print(f"  [B] audio={aud}")
        T = recipe(X, Y, S, S, alpha=alpha_iv, eps=eps, lam=1.0)
        for scope, m in [("aggregate", evaluate(T, X, Y, gt, KCL["b"], seed=SEED)),
                         ("heldout",   evaluate_heldout(T, X, Y, gt, S, KCL["b"], seed=SEED))]:
            rows.append({"experiment": "b", "image_encoder": "—", "audio_encoder": aud,
                         "K": K_target, "alpha": alpha_iv, "scope": scope, **m})

    # ---- Cross-modal grid: C-direct / D / Unsup / Text ------------------
    for img in vision:
        try:
            X_full = load_embedding(f"vision_{img}")
        except FileNotFoundError:
            continue
        anchors = load_anchors(X_full, n=X_full.shape[0])
        X = X_full[anchors]
        ZV = ZV_full[anchors]
        ZA = ZA_full[anchors]
        # Same-rows partition once per image (depends on X only).
        S_a = kmeans_stratified_indices(X, n=K_target,
                                        n_clusters=min(10, K_target), seed=SEED)
        gt = np.arange(X.shape[0])

        for aud in audio:
            try:
                Y_full = load_embedding(f"audio_{aud}")
            except FileNotFoundError as e:
                print(f"  ! skip cross on (image={img}, audio={aud}): {e}")
                continue
            Y = Y_full[anchors]
            S_b = kmeans_stratified_indices(Y, n=K_target,
                                            n_clusters=min(10, K_target), seed=SEED)
            S_compare = np.unique(np.concatenate([S_a, S_b]))
            print(f"  [grid] image={img}  audio={aud}")

            # C-direct at (K=REUSABLE_K, alpha=0.5)
            S = kmeans_stratified_indices(X, n=K_target,
                                          n_clusters=min(10, K_target), seed=SEED)
            T = recipe(X, Y, S, S, alpha=alpha_iv, eps=eps, lam=1.0)
            for scope, m in [("aggregate", evaluate(T, X, Y, gt, KCL["c-direct"], seed=SEED)),
                             ("heldout",   evaluate_heldout(T, X, Y, gt, S,
                                                            KCL["c-direct"], seed=SEED))]:
                rows.append({"experiment": "c-direct",
                             "image_encoder": img, "audio_encoder": aud,
                             "K": K_target, "alpha": alpha_iv,
                             "scope": scope, **m})

            # D at alpha=0.7
            T = caption_cost_recipe(X, Y, ZV, ZA, alpha=alpha_d, eps=eps)
            agg = evaluate(T, X, Y, gt, KCL["d"], seed=SEED)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["d"], seed=SEED)
            rows.append({"experiment": "d", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": alpha_d, "scope": "aggregate", **agg})
            rows.append({"experiment": "d", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": alpha_d, "scope": "heldout_like_c", **hel})

            # Unsup (pure entropic GW)
            T = pure_gw(X, Y, eps=eps)
            agg = evaluate(T, X, Y, gt, KCL["unsup"], seed=SEED)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["unsup"], seed=SEED)
            rows.append({"experiment": "unsup", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": 1.0, "scope": "aggregate", **agg})
            rows.append({"experiment": "unsup", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": 1.0, "scope": "heldout_like_c", **hel})

            # Text-only (the plan itself is encoder-independent, but structural
            # metrics depend on Y, so we evaluate per cell).
            T = text_only_retrieval(ZV, ZA)
            agg = evaluate(T, X, Y, gt, KCL["text"], seed=SEED)
            hel = evaluate_heldout(T, X, Y, gt, S_compare, KCL["text"], seed=SEED)
            rows.append({"experiment": "text", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": float("nan"), "scope": "aggregate", **agg})
            rows.append({"experiment": "text", "image_encoder": img, "audio_encoder": aud,
                         "K": 0, "alpha": float("nan"), "scope": "heldout_like_c", **hel})

            # Persist progressively so a crash doesn't lose hours of work.
            write_rows(rows)

    write_rows(rows)
    print(f"  wrote {out_path}  ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        choices=["a", "b", "c-direct", "c-transitive",
                 "d", "unsup", "text", "grid", "all"],
        required=True,
    )
    ap.add_argument("--grid-images", type=str, default=None,
                    help="Comma-separated subset of image encoders for --exp grid")
    ap.add_argument("--grid-audios", type=str, default=None,
                    help="Comma-separated subset of audio encoders for --exp grid")
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
    if args.exp in ("d", "all"):
        exp_d_caption(args.image_encoder, args.audio_encoder)
    if args.exp in ("unsup", "all"):
        exp_unsupervised_gw(args.image_encoder, args.audio_encoder)
    if args.exp in ("text", "all"):
        exp_text_only(args.image_encoder, args.audio_encoder)
    if args.exp == "grid":
        image_only = (args.grid_images.split(",")
                      if args.grid_images else None)
        audio_only = (args.grid_audios.split(",")
                      if args.grid_audios else None)
        exp_encoder_grid(image_only=image_only, audio_only=audio_only)


if __name__ == "__main__":
    main()
