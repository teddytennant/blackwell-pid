"""Closed-form two-source Blackwell union and redundancy.

C is the normalized posterior covariance of the target given a source.
mutual_info(C) = -1/2 log det C. The union channel is the least informative
channel more informative than both, which for two sources is the pointwise
minimum of generalized eigenvalues of (C2, C1).
"""

import numpy as np
from scipy import linalg


def mutual_info(C):
    C = np.asarray(C, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("C must be positive definite")
    return float(-0.5 * logdet)


def channel_posterior(H):
    """Posterior covariance of standard normal Y given X = H Y + noise.

    H is (n_obs, d). C = (I + H.T @ H)^{-1}.
    """
    H = np.asarray(H, dtype=np.float64)
    d = H.shape[1]
    gram = H.T @ H
    return np.linalg.inv(np.eye(d) + gram)


def normalized_posterior(sigma_y, sigma_yx, sigma_x):
    """Whiten Y, drop degenerate X columns, return posterior covariance of Y.

    C = I - L^{-1} Sigma_{YX} Sigma_X^{+} Sigma_{XY} L^{-T}, with Sigma_Y = L L.T.
    """
    sigma_y = np.asarray(sigma_y, dtype=np.float64)
    sigma_yx = np.asarray(sigma_yx, dtype=np.float64)
    sigma_x = np.asarray(sigma_x, dtype=np.float64)
    L = np.linalg.cholesky(sigma_y)
    var = np.diag(sigma_x)
    keep = var >= 1e-12
    if not np.any(keep):
        return np.eye(sigma_y.shape[0])
    sigma_x = sigma_x[np.ix_(keep, keep)]
    sigma_yx = sigma_yx[:, keep]
    mid = sigma_yx @ np.linalg.pinv(sigma_x) @ sigma_yx.T
    solved = linalg.solve(L, mid)
    C = np.eye(sigma_y.shape[0]) - solved @ np.linalg.inv(L).T
    C = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, 0.0, None)
    return (evecs * evals) @ evecs.T


def union_closed_form(C1, C2):
    """Least informative channel more informative than both.

    Returns (union information, W) where W is the union posterior covariance.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    C1 = 0.5 * (C1 + C1.T)
    C2 = 0.5 * (C2 + C2.T)
    lam, V = linalg.eigh(C2, C1)
    # V.T @ C1 @ V = I, so inv(V).T @ inv(V) = C1.
    inv_vt = np.linalg.inv(V).T
    W = (inv_vt * np.minimum(1.0, lam)) @ np.linalg.inv(V)
    W = 0.5 * (W + W.T)
    i1 = mutual_info(C1)
    i2 = mutual_info(C2)
    value = 0.5 * (i1 + i2) + 0.25 * float(np.sum(np.abs(np.log(lam))))
    return value, W


def redundancy_fastpath(C1, C2):
    """BROJA-style redundancy when R = C1 + C2 - W is a covariance (max eig <= 1).

    Returns (value or None, R). None means the fast path does not apply and a
    convex program is required.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    union, W = union_closed_form(C1, C2)
    R = 0.5 * ((C1 + C2 - W) + (C1 + C2 - W).T)
    max_eig = float(np.max(np.linalg.eigvalsh(R)))
    if max_eig <= 1.0 + 1e-8:
        value = mutual_info(C1) + mutual_info(C2) - union
        return value, R
    return None, R
