"""Compare the cosine-softmax bridge B vs the identity bridge in the
transitive composition  T = T_iv @ B @ T_ac.T.

Uses only the saved plan matrices in results/.  No embeddings, no videos.
gt = arange(n) because both unimodal legs share the same anchor reordering,
so clip at anchor-position i in T_iv corresponds to clip at anchor-position
i in T_ac.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

ROOT = Path("/tmp/av-bridge")
R = ROOT / "results"


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
    T_cosine = np.load(R / "exp_c" / "T_transitive.npy")  # their built-in cosine bridge
    n = T_iv.shape[0]
    gt = np.arange(n)

    print(f"n = {n}")
    print(f"T_iv  shape={T_iv.shape}  sum={T_iv.sum():.4f}  diag_mean={np.diag(T_iv).mean():.6f}")
    print(f"T_ac  shape={T_ac.shape}  sum={T_ac.sum():.4f}  diag_mean={np.diag(T_ac).mean():.6f}")
    print()

    # ---- Identity bridge: B = I, so T = T_iv @ T_ac.T ---------------------
    T_identity = row_norm(T_iv @ T_ac.T)
    print(f"T_identity (B = I)   diag_mean={np.diag(T_identity).mean():.6f}")
    print(f"T_cosine   (B = cos) diag_mean={np.diag(T_cosine).mean():.6f}")
    print()

    print("=== R@k on image -> audio (gt = same anchor position) ===")
    print(f"{'method':<28s}  R@1     R@5     R@10    R@20")
    for name, T in [
        ("cosine bridge (theirs)", T_cosine),
        ("identity bridge (B=I)",  T_identity),
    ]:
        rs = [recall_at_k(T, gt, k) for k in (1, 5, 10, 20)]
        print(f"{name:<28s}  {rs[0]:.3f}   {rs[1]:.3f}   {rs[2]:.3f}   {rs[3]:.3f}")

    # ---- Sanity: how diagonal are the two unimodal legs? ------------------
    print()
    print("=== Unimodal-leg diagonal quality (rank of own anchor) ===")
    for name, T in [("T_iv (image -> visual_text)", T_iv),
                    ("T_ac (audio -> audio_text)",  T_ac)]:
        order = np.argsort(-T, axis=1)
        ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
        print(f"{name:<32s}  top1={ (ranks<1).mean():.3f}  "
              f"top5={(ranks<5).mean():.3f}  top10={(ranks<10).mean():.3f}  "
              f"top20={(ranks<20).mean():.3f}  median_rank={int(np.median(ranks))}")


if __name__ == "__main__":
    main()
