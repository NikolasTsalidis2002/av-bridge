"""Estimate the K=300 supervision sets S_a and S_b from the saved plans,
then re-evaluate identity-bridge vs cosine-bridge on the held-out rows only.

We assume the saved plans were produced at K=300 (REUSABLE_K in
run_experiments.py), so 300 of 400 rows per leg were ridge-supervised.
Ridge-supervised rows sit sharply at T[i, i]; held-out rows do not.
We pick the bottom 100 by diag(T) per leg as the proxy held-out set.

Validation: their results/comparison.csv reports cosine-bridge held-out
R@10 = 0.067.  If our proxy gives ~0.067 on the cosine plan we know the
proxy is calibrated; then trust the identity-bridge number under the
same mask.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

ROOT = Path("/tmp/av-bridge")
R = ROOT / "results"
K_SUP = 300
N_HELD = 100  # 400 - K_SUP


def recall_at_k(T: np.ndarray, gt: np.ndarray, k: int) -> float:
    n = T.shape[0]
    k = min(k, T.shape[1])
    topk = np.argpartition(-T, k - 1, axis=1)[:, :k]
    return float(sum(gt[i] in topk[i] for i in range(n)) / n)


def row_norm(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    s = np.where(s > 1e-12, s, 1.0)
    return M / s


def main():
    T_iv = np.load(R / "exp_a" / "T_iv.npy")
    T_ac = np.load(R / "exp_b" / "T_ac.npy")
    T_cosine = np.load(R / "exp_c" / "T_transitive.npy")
    n = T_iv.shape[0]
    gt = np.arange(n)

    # ---- Estimate supervised vs held-out from the diagonals ----------
    d_iv = np.diag(T_iv)
    d_ac = np.diag(T_ac)

    # Sort ascending; bottom 100 (smallest diag) are the proxy held-out.
    order_iv = np.argsort(d_iv)
    order_ac = np.argsort(d_ac)
    S_a_est = set(order_iv[N_HELD:].tolist())       # estimated supervised (top 300)
    S_b_est = set(order_ac[N_HELD:].tolist())
    H_a_est = set(order_iv[:N_HELD].tolist())       # estimated held-out (bottom 100)
    H_b_est = set(order_ac[:N_HELD].tolist())

    # Strict held-out: rows out of S for BOTH legs.
    H_both = np.array(sorted(H_a_est & H_b_est), dtype=int)
    H_either = np.array(sorted(H_a_est | H_b_est), dtype=int)
    print(f"|H_a| (image leg held-out estimate) = {len(H_a_est)}")
    print(f"|H_b| (audio leg held-out estimate) = {len(H_b_est)}")
    print(f"|H_a ∩ H_b| (held-out for BOTH)     = {len(H_both)}")
    print(f"|H_a ∪ H_b| (held-out for EITHER)   = {len(H_either)}")
    print()

    # Quick histogram of diag(T_iv) and diag(T_ac) to show separation.
    print("Diagonal mass percentiles:")
    print(f"  diag(T_iv): 10%={np.percentile(d_iv,10):.3e}  "
          f"25%={np.percentile(d_iv,25):.3e}  50%={np.percentile(d_iv,50):.3e}  "
          f"75%={np.percentile(d_iv,75):.3e}  90%={np.percentile(d_iv,90):.3e}")
    print(f"  diag(T_ac): 10%={np.percentile(d_ac,10):.3e}  "
          f"25%={np.percentile(d_ac,25):.3e}  50%={np.percentile(d_ac,50):.3e}  "
          f"75%={np.percentile(d_ac,75):.3e}  90%={np.percentile(d_ac,90):.3e}")
    print(f"  Bottom 100 vs top 300 ratio (T_iv): "
          f"{d_iv[order_iv[:N_HELD]].mean():.3e} / {d_iv[order_iv[N_HELD:]].mean():.3e}")
    print(f"  Bottom 100 vs top 300 ratio (T_ac): "
          f"{d_ac[order_ac[:N_HELD]].mean():.3e} / {d_ac[order_ac[N_HELD:]].mean():.3e}")
    print()

    # ---- Identity-bridge plan ---------------------------------------
    T_identity = row_norm(T_iv @ T_ac.T)

    def eval_subset(T, mask_idx, label):
        T_sub = T[mask_idx]
        gt_sub = mask_idx
        rs = [recall_at_k(T_sub, gt_sub, k) for k in (1, 5, 10, 20)]
        print(f"{label:<40s}  R@1={rs[0]:.3f}  R@5={rs[1]:.3f}  "
              f"R@10={rs[2]:.3f}  R@20={rs[3]:.3f}  (n={len(mask_idx)})")

    print("=== Full aggregate (n=400) ===")
    eval_subset(T_cosine,   np.arange(n), "cosine bridge   (full)")
    eval_subset(T_identity, np.arange(n), "identity bridge (full)")
    print()

    print(f"=== Held-out for BOTH legs (proxy, n={len(H_both)}) ===")
    if len(H_both) > 0:
        eval_subset(T_cosine,   H_both, "cosine bridge   (strict held-out)")
        eval_subset(T_identity, H_both, "identity bridge (strict held-out)")
    else:
        print("  (empty intersection)")
    print()

    print(f"=== Held-out for EITHER leg (proxy, n={len(H_either)}) ===")
    eval_subset(T_cosine,   H_either, "cosine bridge   (loose held-out)")
    eval_subset(T_identity, H_either, "identity bridge (loose held-out)")
    print()

    print(f"=== Held-out for image leg only (proxy, n={len(H_a_est)}) ===")
    H_a_arr = np.array(sorted(H_a_est), dtype=int)
    eval_subset(T_cosine,   H_a_arr, "cosine bridge   (image-leg heldout)")
    eval_subset(T_identity, H_a_arr, "identity bridge (image-leg heldout)")
    print()

    print(f"=== Held-out for audio leg only (proxy, n={len(H_b_est)}) ===")
    H_b_arr = np.array(sorted(H_b_est), dtype=int)
    eval_subset(T_cosine,   H_b_arr, "cosine bridge   (audio-leg heldout)")
    eval_subset(T_identity, H_b_arr, "identity bridge (audio-leg heldout)")
    print()

    print("Reference (results/comparison.csv):")
    print("  C-transitive cosine bridge — aggregate R@10 = 0.175, held-out R@10 = 0.067")
    print("  If our cosine held-out lands near 0.067, the diag-based proxy is calibrated.")


if __name__ == "__main__":
    main()
