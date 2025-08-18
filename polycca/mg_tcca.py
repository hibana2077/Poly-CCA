from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class MGTCCAResult:
    Ws: List[np.ndarray]
    means: List[np.ndarray]
    correlations: np.ndarray


def mg_tcca(views: List[np.ndarray], reg: float=1e-4, n_components: int|None=None) -> MGTCCAResult:
    """Multi-group (multi-view) TCCA simplified as averaging pairwise whitened cross-covariances.
    This is a lightweight surrogate capturing shared subspace.
    views: list of (n,d_g)
    """
    G = len(views)
    n = views[0].shape[0]
    means = [v.mean(0) for v in views]
    centered = [v - m for v,m in zip(views, means)]
    covs = []
    Ws_white = []
    for V in centered:
        d = V.shape[1]
        S = (V.T @ V)/(n-1) + reg * np.eye(d)
        E,U = np.linalg.eigh(S)
        E = np.clip(E,1e-12,None)
        W = U @ np.diag(E**-0.5) @ U.T
        Ws_white.append(W)
    # build average cross-cov operator in common latent space dimension = min d
    d_min = min(v.shape[1] for v in views)
    # project each whitened view to d_min with PCA on variance (here identity since whitened) -> just slice
    # Form averaged matrix A = sum_{g<h} Wg Cgh Wh^T normalized
    A = np.zeros((d_min,d_min))
    count=0
    for i in range(G):
        for j in range(i+1,G):
            Ci = Ws_white[i] @ ((centered[i].T @ centered[j])/(n-1)) @ Ws_white[j]
            A += Ci[:d_min,:d_min]
            count+=1
    A /= max(count,1)
    # SVD of symmetric (not guaranteed symmetrical) use SVD
    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    if n_components is None:
        n_components = d_min
    components = U[:, :n_components]
    Ws = []
    for Wwhite in Ws_white:
        Ws.append(Wwhite.T @ components)
    return MGTCCAResult(Ws=Ws, means=means, correlations=S[:n_components])


def transform(result: MGTCCAResult, X: np.ndarray, view_index: int) -> np.ndarray:
    return (X - result.means[view_index]) @ result.Ws[view_index]
