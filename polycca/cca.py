from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CCAResult:
    Wx: np.ndarray
    Wy: np.ndarray
    correlations: np.ndarray
    X_mean: np.ndarray
    Y_mean: np.ndarray


def cca(X: np.ndarray, Y: np.ndarray, reg: float = 1e-4, n_components: int | None = None) -> CCAResult:
    """Simple linear CCA with ridge regularization.

    X: (n,d1) Y:(n,d2)
    reg: ridge added to diagonal of covariances
    """
    n, d1 = X.shape
    n, d2 = Y.shape
    if n_components is None:
        n_components = min(d1,d2)
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Sxx = (Xc.T @ Xc)/ (n-1) + reg * np.eye(d1)
    Syy = (Yc.T @ Yc)/ (n-1) + reg * np.eye(d2)
    Sxy = (Xc.T @ Yc)/ (n-1)
    # whitening
    Ex, Ux = np.linalg.eigh(Sxx)
    Ey, Uy = np.linalg.eigh(Syy)
    Ex = np.clip(Ex, 1e-12, None)
    Ey = np.clip(Ey, 1e-12, None)
    Wx_whiten = Ux @ np.diag(Ex**-0.5) @ Ux.T
    Wy_whiten = Uy @ np.diag(Ey**-0.5) @ Uy.T
    T = Wx_whiten @ Sxy @ Wy_whiten
    # SVD
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    Wx = Wx_whiten.T @ U[:, :n_components]
    Wy = Wy_whiten.T @ Vt.T[:, :n_components]
    return CCAResult(Wx=Wx, Wy=Wy, correlations=S[:n_components], X_mean=X.mean(0), Y_mean=Y.mean(0))


def transform(result: CCAResult, X: np.ndarray, side: str = 'X') -> np.ndarray:
    if side.upper()=='X':
        return (X - result.X_mean) @ result.Wx
    else:
        return (X - result.Y_mean) @ result.Wy
