"""Phase 5 — evaluation metrics for transport plans.

Every metric takes:
  - T   : transport plan, shape (n, m)
  - gt  : ground-truth pairing, shape (n,) with gt[i] = target row id
  - X_src, Y_tgt : source/target embeddings used for structural metrics
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances


def recall_at_k(T: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Top-k retrieval recall against gt."""
    n = T.shape[0]
    k = min(k, T.shape[1])
    topk = np.argpartition(-T, k - 1, axis=1)[:, :k]
    hits = 0
    for i in range(n):
        if gt[i] in topk[i]:
            hits += 1
    return float(hits / n) if n > 0 else 0.0


def cluster_routing(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> tuple[int, int]:
    """Cluster-to-cluster routing accuracy.

    Returns (correct, K_cl).
    """
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    correct = 0
    for s in range(K_cl):
        idx = np.where(src_lab == s)[0]
        if len(idx) == 0:
            continue
        mass = np.zeros(K_cl)
        for j in range(T.shape[1]):
            mass[tgt_lab[j]] += T[idx, j].sum()
        predicted = int(mass.argmax())
        gt_cluster = int(
            np.bincount(tgt_lab[gt[idx]], minlength=K_cl).argmax()
        )
        if predicted == gt_cluster:
            correct += 1
    return correct, K_cl


def knn_overlap(T: np.ndarray, X_src: np.ndarray, Y_tgt: np.ndarray, k: int = 5) -> float:
    """Mean overlap between source-knn (mapped through plan) and target-knn of partner."""
    if len(X_src) < 2 or len(Y_tgt) < 2:
        return float("nan")
    partners = T.argmax(axis=1)
    Ds = pairwise_distances(X_src)
    Dt = pairwise_distances(Y_tgt)
    k = min(k, len(X_src) - 1, len(Y_tgt) - 1)
    overlaps = []
    for i in range(len(X_src)):
        src_nn = np.argsort(Ds[i])[1:k + 1]
        tgt_nn = np.argsort(Dt[partners[i]])[1:k + 1]
        mapped = partners[src_nn]
        overlaps.append(len(set(mapped) & set(tgt_nn)) / k)
    return float(np.mean(overlaps))


def pearson_pairwise(T: np.ndarray, X_src: np.ndarray, Y_tgt: np.ndarray) -> float:
    """Pearson correlation of pairwise distances in source space vs partner-target space."""
    if len(X_src) < 3:
        return float("nan")
    partners = T.argmax(axis=1)
    Ds = pairwise_distances(X_src)
    Dt = pairwise_distances(Y_tgt[partners])
    iu = np.triu_indices(len(X_src), k=1)
    a = Ds[iu]
    b = Dt[iu]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def cluster_nmi(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> float:
    """Normalised mutual information between source clusters and target clusters of partners."""
    if len(X_src) < K_cl or len(Y_tgt) < K_cl:
        return float("nan")
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    partners = T.argmax(axis=1)
    return float(normalized_mutual_info_score(src_lab, tgt_lab[partners]))


def evaluate(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> dict:
    """Full metric suite. Used by run_experiments.py for each (K, alpha) cell."""
    r1 = recall_at_k(T, gt, 1)
    r5 = recall_at_k(T, gt, 5)
    r10 = recall_at_k(T, gt, 10)
    r20 = recall_at_k(T, gt, 20)
    r_correct, r_total = cluster_routing(T, X_src, Y_tgt, gt, K_cl, seed=seed)
    kno = knn_overlap(T, X_src, Y_tgt, k=5)
    pr = pearson_pairwise(T, X_src, Y_tgt)
    nmi = cluster_nmi(T, X_src, Y_tgt, K_cl, seed=seed)
    return {
        "R@1": r1, "R@5": r5, "R@10": r10, "R@20": r20,
        "routes_correct": r_correct, "routes_total": r_total,
        "knn_overlap": kno, "pearson_r": pr, "nmi": nmi,
    }


def evaluate_heldout(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    gt: np.ndarray,
    S: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> dict:
    """Same suite but restricted to rows whose index is NOT in S.

    For recall_at_k this is a row mask: we keep the full plan (so the
    columns the held-out rows can match against are still the whole target
    set, which is the honest setup) but only score rows not in S.
    For structural metrics we recompute on the masked subset.
    """
    n = T.shape[0]
    in_S = np.zeros(n, dtype=bool)
    in_S[np.asarray(S, dtype=int)] = True
    mask = ~in_S
    held = np.where(mask)[0]
    if held.size == 0:
        return {
            "R@1": float("nan"), "R@5": float("nan"),
            "R@10": float("nan"), "R@20": float("nan"),
            "routes_correct": 0, "routes_total": K_cl,
            "knn_overlap": float("nan"),
            "pearson_r": float("nan"),
            "nmi": float("nan"),
        }

    # Row-masked recall.
    T_h = T[held]
    gt_h = gt[held]
    r1 = recall_at_k(T_h, gt_h, 1)
    r5 = recall_at_k(T_h, gt_h, 5)
    r10 = recall_at_k(T_h, gt_h, 10)
    r20 = recall_at_k(T_h, gt_h, 20)

    # Structural metrics on the held-out subset.
    X_h = X_src[held]
    # gt[held] indexes into Y_tgt; we evaluate Y_tgt at the partner side.
    # For cluster_routing we need a remapping: cluster labels for X_h and Y_tgt's full set,
    # then aggregate plan mass restricted to held rows.
    Kc = min(K_cl, len(X_h), len(Y_tgt))
    if Kc < 2:
        r_correct, r_total = 0, K_cl
        kno = float("nan")
        pr = float("nan")
        nmi = float("nan")
    else:
        src_lab = KMeans(Kc, random_state=seed, n_init=10).fit_predict(X_h)
        tgt_lab = KMeans(Kc, random_state=seed, n_init=10).fit_predict(Y_tgt)
        partners_h = T_h.argmax(axis=1)

        correct = 0
        for s in range(Kc):
            idx = np.where(src_lab == s)[0]
            if len(idx) == 0:
                continue
            mass = np.zeros(Kc)
            for j in range(T.shape[1]):
                mass[tgt_lab[j]] += T_h[idx, j].sum()
            predicted = int(mass.argmax())
            gt_cluster = int(
                np.bincount(tgt_lab[gt_h[idx]], minlength=Kc).argmax()
            )
            if predicted == gt_cluster:
                correct += 1
        r_correct, r_total = correct, Kc

        # knn_overlap on held-out
        Ds = pairwise_distances(X_h)
        Dt = pairwise_distances(Y_tgt)
        k = min(5, len(X_h) - 1, len(Y_tgt) - 1)
        overlaps = []
        for i in range(len(X_h)):
            src_nn = np.argsort(Ds[i])[1:k + 1]
            tgt_nn = np.argsort(Dt[partners_h[i]])[1:k + 1]
            mapped = partners_h[src_nn]
            overlaps.append(len(set(mapped) & set(tgt_nn)) / k)
        kno = float(np.mean(overlaps)) if overlaps else float("nan")

        # pearson on held-out, partner side
        Dt_p = pairwise_distances(Y_tgt[partners_h])
        iu = np.triu_indices(len(X_h), k=1)
        a = Ds[iu]
        b = Dt_p[iu]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            pr = float("nan")
        else:
            pr = float(pearsonr(a, b)[0])

        nmi = float(
            normalized_mutual_info_score(src_lab, tgt_lab[partners_h])
        )

    return {
        "R@1": r1, "R@5": r5, "R@10": r10, "R@20": r20,
        "routes_correct": r_correct, "routes_total": r_total,
        "knn_overlap": kno, "pearson_r": pr, "nmi": nmi,
    }
