from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from .datasets import read_promoter_dataset, read_splice_dataset, batch_kmer_counts, polynomial_features, perturb_substitution
from .cca import cca, transform as cca_transform
from .mg_tcca import mg_tcca, transform as mg_transform
from .simulate import PolySimConfig, generate_dataset
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

@dataclass
class ExperimentResult:
    name: str
    macro_f1: float
    roc_auc: float | None
    correlations: np.ndarray


def train_eval_lr(X, y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X,y)
    pred = lr.predict(X)
    proba = None
    try:
        proba = lr.predict_proba(X)
    except Exception:
        pass
    macro = f1_score(y, pred, average='macro')
    roc = None
    if proba is not None and len(np.unique(y))==2:
        roc = roc_auc_score(y, proba[:,1])
    return macro, roc, lr


def run_poly_cca_promoter(data_root: Path, k=3, degree=2, noise_p=0.05) -> ExperimentResult:
    data = read_promoter_dataset(data_root)
    y = np.array([r[0] for r in data])
    seqs = [r[1] for r in data]
    Xk, _ = batch_kmer_counts(seqs,k)
    Phi = polynomial_features(Xk, degree=degree)
    # create perturbed view
    seqs_noise = perturb_substitution(seqs, p=noise_p)
    Xk2,_ = batch_kmer_counts(seqs_noise,k)
    Phi2 = polynomial_features(Xk2, degree=degree)
    result = cca(Phi, Phi2, reg=1e-3, n_components=16)
    Z = cca_transform(result, Phi, 'X')
    macro, roc, _ = train_eval_lr(Z, y)
    return ExperimentResult(name='Poly-CCA Promoter', macro_f1=macro, roc_auc=roc, correlations=result.correlations)


def run_kmer_baseline_promoter(data_root: Path, k=3) -> ExperimentResult:
    data = read_promoter_dataset(data_root)
    y = np.array([r[0] for r in data])
    seqs = [r[1] for r in data]
    Xk,_ = batch_kmer_counts(seqs,k)
    macro, roc, _ = train_eval_lr(Xk,y)
    return ExperimentResult(name='k-mer baseline', macro_f1=macro, roc_auc=roc, correlations=np.array([]))


def run_mg_tcca_sim(cfg: PolySimConfig, k=3, degree=2, ps=(0.0,0.05,0.1)) -> ExperimentResult:
    data = generate_dataset(cfg)
    y = np.array([r[0] for r in data])
    seqs = [r[1] for r in data]
    # create groups by noise levels
    views = []
    for p in ps:
        seqs_p = perturb_substitution(seqs, p=p)
        Xk,_ = batch_kmer_counts(seqs_p,k)
        Phi = polynomial_features(Xk, degree=degree)
        views.append(Phi)
    res = mg_tcca(views, reg=1e-3, n_components=16)
    # pick first view transform for classification
    Z = (views[0] - views[0].mean(0)) @ res.Ws[0]
    macro, roc, _ = train_eval_lr(Z,y)
    return ExperimentResult(name='MG-TCCA Sim', macro_f1=macro, roc_auc=roc, correlations=res.correlations)
