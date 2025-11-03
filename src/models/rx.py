# src/models/rx.py
import numpy as np

def rx_score(cube_bhw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Robust RX anomaly score on a single cube.
    Input:  cube_bhw : (B, H, W) float32 in [0,1]
    Output: score_hw : (H, W) float32 ~ [0, 1] normalized
    """
    B, H, W = cube_bhw.shape
    X = cube_bhw.reshape(B, -1).T  # (N, B)
    mu = X.mean(axis=0, keepdims=True)         # (1, B)
    Xc = X - mu
    # covariance (B,B) with small ridge for stability
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    cov.flat[::B+1] += eps                      # ridge on diagonal
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # extra ridge if singular
        cov.flat[::B+1] += 1e-3
        inv_cov = np.linalg.inv(cov)

    # Mahalanobis distance squared = (x - mu)^T inv_cov (x - mu)
    m = (Xc @ inv_cov) * Xc
    d2 = m.sum(axis=1)                          # (N,)
    score = d2.reshape(H, W).astype(np.float32)

    # min-max normalize to [0,1] for visualization
    smin, smax = np.percentile(score, 1), np.percentile(score, 99)
    if smax > smin:
        score = (score - smin) / (smax - smin)
        score = np.clip(score, 0, 1)
    else:
        score = np.zeros_like(score, dtype=np.float32)
    return score