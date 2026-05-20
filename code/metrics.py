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
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    pairwise_distances,
    v_measure_score,
)


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


_AGREEMENT_NANS: dict = {
    "nmi": float("nan"),
    "ami": float("nan"),
    "ari": float("nan"),
    "v_measure": float("nan"),
    "homogeneity": float("nan"),
    "completeness": float("nan"),
}


def cluster_agreement(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> dict:
    """Cluster-agreement metrics between source clusters and partner-target
    clusters under the plan's hard argmax assignment.

    Returns a dict with six metrics:

    - ``nmi`` — normalised mutual information (arithmetic-mean normalisation).
      Not chance-corrected; expected value of independent labellings grows
      with K_cl / n.
    - ``ami`` — chance-corrected mutual information. ≈ 0 under independent
      labellings; positive values have a "lift over chance" reading.
    - ``ari`` — adjusted Rand index. Chance-corrected pairwise agreement:
      asks, of all source pairs (i, i'), whether they end up in the same
      target cluster iff they were in the same source cluster.
    - ``v_measure`` — symmetric harmonic mean of homogeneity and
      completeness; coincides with NMI under arithmetic-mean normalisation.
    - ``homogeneity`` — each predicted cluster contains members of a single
      source class.
    - ``completeness`` — all members of a source class end up in the same
      predicted cluster. Fragmentation vs impurity decomposes NMI into
      these two complementary axes.
    """
    if len(X_src) < K_cl or len(Y_tgt) < K_cl:
        return dict(_AGREEMENT_NANS)
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    partners = T.argmax(axis=1)
    mapped = tgt_lab[partners]
    return {
        "nmi":          float(normalized_mutual_info_score(src_lab, mapped)),
        "ami":          float(adjusted_mutual_info_score(src_lab, mapped)),
        "ari":          float(adjusted_rand_score(src_lab, mapped)),
        "v_measure":    float(v_measure_score(src_lab, mapped)),
        "homogeneity":  float(homogeneity_score(src_lab, mapped)),
        "completeness": float(completeness_score(src_lab, mapped)),
    }


def cluster_nmi_ami(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> tuple[float, float]:
    """Back-compat wrapper returning (NMI, AMI) only."""
    a = cluster_agreement(T, X_src, Y_tgt, K_cl, seed=seed)
    return a["nmi"], a["ami"]


def cluster_nmi(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
) -> float:
    """Back-compat wrapper returning NMI only."""
    return cluster_agreement(T, X_src, Y_tgt, K_cl, seed=seed)["nmi"]


def cluster_confusion(
    T: np.ndarray,
    X_src: np.ndarray,
    Y_tgt: np.ndarray,
    K_cl: int,
    seed: int = 42,
    mode: str = "soft",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-cluster × target-cluster confusion matrix induced by the plan.

    - ``mode="soft"`` (default): row s, col t of the K × K matrix is the
      total mass T[i, j] aggregated over source rows i in source-cluster s
      and target columns j in target-cluster t. Uses the full plan, not
      argmax — informative for diffuse OT plans.
    - ``mode="hard"``: counts of source rows whose argmax partner falls
      into each target cluster. Equivalent to the hard-partner confusion
      used inside ``cluster_routing``.

    The returned matrix is row-normalised (each source-cluster row sums to
    one when non-empty); a row of zeros indicates an empty source cluster.
    Also returns the source and target K-means label vectors so callers
    can apply a Hungarian column permutation for block-diagonal display.
    """
    src_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(X_src)
    tgt_lab = KMeans(K_cl, random_state=seed, n_init=10).fit_predict(Y_tgt)
    C = np.zeros((K_cl, K_cl), dtype=np.float64)
    if mode == "soft":
        for s in range(K_cl):
            src_idx = np.where(src_lab == s)[0]
            if src_idx.size == 0:
                continue
            row_mass = T[src_idx].sum(axis=0)
            for t in range(K_cl):
                tgt_idx = np.where(tgt_lab == t)[0]
                if tgt_idx.size == 0:
                    continue
                C[s, t] = row_mass[tgt_idx].sum()
    elif mode == "hard":
        partners = T.argmax(axis=1)
        for s in range(K_cl):
            src_idx = np.where(src_lab == s)[0]
            for i in src_idx:
                C[s, tgt_lab[partners[i]]] += 1.0
    else:
        raise ValueError(f"unknown confusion mode: {mode!r}")
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    C = C / row_sums
    return C, src_lab, tgt_lab


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
    agree = cluster_agreement(T, X_src, Y_tgt, K_cl, seed=seed)
    return {
        "R@1": r1, "R@5": r5, "R@10": r10, "R@20": r20,
        "routes_correct": r_correct, "routes_total": r_total,
        "knn_overlap": kno, "pearson_r": pr,
        **agree,
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
            **_AGREEMENT_NANS,
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
        agree = dict(_AGREEMENT_NANS)
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

        mapped_h = tgt_lab[partners_h]
        agree = {
            "nmi":          float(normalized_mutual_info_score(src_lab, mapped_h)),
            "ami":          float(adjusted_mutual_info_score(src_lab, mapped_h)),
            "ari":          float(adjusted_rand_score(src_lab, mapped_h)),
            "v_measure":    float(v_measure_score(src_lab, mapped_h)),
            "homogeneity":  float(homogeneity_score(src_lab, mapped_h)),
            "completeness": float(completeness_score(src_lab, mapped_h)),
        }

    return {
        "R@1": r1, "R@5": r5, "R@10": r10, "R@20": r20,
        "routes_correct": r_correct, "routes_total": r_total,
        "knn_overlap": kno, "pearson_r": pr,
        **agree,
    }
