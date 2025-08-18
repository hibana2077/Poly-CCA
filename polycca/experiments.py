"""Comprehensive experiment utilities aligning with study design.

Includes:
  - Baseline models (k-mer+LR/SVM, PCA->LR, LDA, polynomial no-CCA, RBF-SVM subset)
  - Poly-CCA and MG-TCCA wrappers with cross-validation
  - Noise robustness curves & Delta metric
  - Ablations over k, polynomial degree d
  - Sample size curves
  - Cross-dataset transfer (train on synthetic -> test on real)
  - Motif attribution from canonical weights
  - Statistical significance tests (paired t-test, Wilcoxon) over CV folds
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Callable
import json
import math
import numpy as np

from .datasets import (
    read_promoter_dataset,
    read_splice_dataset,
    batch_kmer_counts,
    polynomial_features,
    perturb_substitution,
    make_kmer_index,
)
from .cca import cca, transform as cca_transform, CCAResult
from .mg_tcca import mg_tcca, transform as mg_transform, MGTCCAResult
from .simulate import PolySimConfig, generate_dataset

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time


@dataclass
class FoldMetrics:
    f1_macro: float
    roc_auc: float | None


@dataclass
class CVResult:
    name: str
    mean_f1_macro: float
    std_f1_macro: float
    mean_roc_auc: float | None
    std_roc_auc: float | None
    folds: List[FoldMetrics]
    extra: Dict | None = None

    def to_json(self):
        d = asdict(self)
        return json.dumps(d, indent=2)


def _compute_metrics(y_true, y_pred, y_proba=None) -> Tuple[float, float | None]:
    f1m = f1_score(y_true, y_pred, average="macro")
    roc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            roc = roc_auc_score(y_true, y_proba[:, 1])
    return f1m, roc


def _fit_predict_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            pass
    return y_pred, y_proba


def _baseline_models(linear_only: bool = False) -> Dict[str, Callable[[], object]]:
    models = {
        "LR": lambda: LogisticRegression(max_iter=2000),
        "LinearSVM": lambda: LinearSVC(),
    }
    if not linear_only:
        models["RBF-SVM"] = lambda: SVC(kernel="rbf", probability=True)
    return models


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[], object],
    n_splits: int = 5,
    random_state: int = 0,
    scale: bool = True,
) -> List[FoldMetrics]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[FoldMetrics] = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        if scale:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
        model = model_fn()
        y_pred, y_proba = _fit_predict_model(model, Xtr, ytr, Xte)
        f1m, roc = _compute_metrics(yte, y_pred, y_proba)
        folds.append(FoldMetrics(f1_macro=f1m, roc_auc=roc))
    return folds


def aggregate_cv(name: str, folds: List[FoldMetrics], extra: Dict | None = None) -> CVResult:
    f1s = [f.f1_macro for f in folds]
    rocs = [f.roc_auc for f in folds if f.roc_auc is not None]
    mean_roc = std_roc = None
    if rocs:
        mean_roc = float(np.mean(rocs))
        std_roc = float(np.std(rocs))
    return CVResult(
        name=name,
        mean_f1_macro=float(np.mean(f1s)),
        std_f1_macro=float(np.std(f1s)),
        mean_roc_auc=mean_roc,
        std_roc_auc=std_roc,
        folds=folds,
        extra=extra,
    )


# --------------------- Feature preparation ---------------------

def prepare_kmer_poly(seqs: List[str], k: int, degree: int) -> np.ndarray:
    Xk, _ = batch_kmer_counts(seqs, k)
    if degree == 1:
        return Xk
    return polynomial_features(Xk, degree=degree)


# --------------------- Poly-CCA / MG-TCCA CV wrappers ---------------------

def cv_poly_cca(
    seqs: List[str],
    y: np.ndarray,
    k: int = 3,
    degree: int = 2,
    noise_p: float = 0.05,
    n_components: int = 16,
    reg: float = 1e-3,
    n_splits: int = 5,
    random_state: int = 0,
) -> CVResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[FoldMetrics] = []
    corrs: List[np.ndarray] = []
    for tr, te in skf.split(seqs, y):
        seq_tr = [seqs[i] for i in tr]
        seq_te = [seqs[i] for i in te]
        # views
        noisy_tr = perturb_substitution(seq_tr, p=noise_p)
        noisy_te = perturb_substitution(seq_te, p=noise_p)  # apply independently to test set for robustness evaluation
        Xtr = prepare_kmer_poly(seq_tr, k, degree)
        Xtr2 = prepare_kmer_poly(noisy_tr, k, degree)
        Xte = prepare_kmer_poly(seq_te, k, degree)
        Xte2 = prepare_kmer_poly(noisy_te, k, degree)
        res = cca(Xtr, Xtr2, reg=reg, n_components=n_components)
        corrs.append(res.correlations)
        Ztr = cca_transform(res, Xtr, 'X')
        Zte = cca_transform(res, Xte, 'X')  # project original test sequences
        model = LogisticRegression(max_iter=2000)
        model.fit(Ztr, y[tr])
        pred = model.predict(Zte)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Zte)
        f1m, roc = _compute_metrics(y[te], pred, proba)
        folds.append(FoldMetrics(f1_macro=f1m, roc_auc=roc))
    extra = {"mean_correlations": float(np.mean([c[0] for c in corrs]))}
    return aggregate_cv("Poly-CCA", folds, extra=extra)


def cv_mg_tcca(
    seqs: List[str],
    y: np.ndarray,
    k: int = 3,
    degree: int = 2,
    ps: Tuple[float, ...] = (0.0, 0.05, 0.1),
    n_components: int = 16,
    reg: float = 1e-3,
    n_splits: int = 5,
    random_state: int = 0,
) -> CVResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[FoldMetrics] = []
    corrs: List[np.ndarray] = []
    for tr, te in skf.split(seqs, y):
        seq_tr = [seqs[i] for i in tr]
        seq_te = [seqs[i] for i in te]
        views_tr = []
        views_te = []
        for p in ps:
            views_tr.append(prepare_kmer_poly(perturb_substitution(seq_tr, p=p), k, degree))
            views_te.append(prepare_kmer_poly(perturb_substitution(seq_te, p=p), k, degree))
        res = mg_tcca(views_tr, reg=reg, n_components=n_components)
        corrs.append(res.correlations)
        # use first view embedding for classification
        Ztr = (views_tr[0] - views_tr[0].mean(0)) @ res.Ws[0]
        Zte = (views_te[0] - views_tr[0].mean(0)) @ res.Ws[0]
        model = LogisticRegression(max_iter=2000)
        model.fit(Ztr, y[tr])
        pred = model.predict(Zte)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Zte)
        f1m, roc = _compute_metrics(y[te], pred, proba)
        folds.append(FoldMetrics(f1_macro=f1m, roc_auc=roc))
    extra = {"mean_correlations": float(np.mean([c[0] for c in corrs]))}
    return aggregate_cv("MG-TCCA", folds, extra=extra)


# --------------------- Baseline CV wrappers ---------------------

def cv_kmer_baselines(seqs: List[str], y: np.ndarray, k: int = 3, degree: int = 1, linear_only=False) -> List[CVResult]:
    X = prepare_kmer_poly(seqs, k, degree)
    results: List[CVResult] = []
    for name, fn in _baseline_models(linear_only=linear_only).items():
        folds = cross_validate(X, y, fn)
        results.append(aggregate_cv(f"{name} (k={k},d={degree})", folds))
    # PCA -> LR
    folds = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        pca = PCA(n_components=min(32, Xtr_s.shape[1]))
        Xtr_p = pca.fit_transform(Xtr_s)
        Xte_p = pca.transform(Xte_s)
        model = LogisticRegression(max_iter=2000)
        model.fit(Xtr_p, y[tr])
        pred = model.predict(Xte_p)
        proba = model.predict_proba(Xte_p)
        f1m, roc = _compute_metrics(y[te], pred, proba)
        folds.append(FoldMetrics(f1_macro=f1m, roc_auc=roc))
    results.append(aggregate_cv("PCA->LR", folds))
    # LDA (only if classes <= n_features)
    if len(np.unique(y)) <= X.shape[1]:
        folds = []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            lda = LDA()
            lda.fit(Xtr, y[tr])
            pred = lda.predict(Xte)
            proba = None
            if hasattr(lda, 'predict_proba'):
                proba = lda.predict_proba(Xte)
            f1m, roc = _compute_metrics(y[te], pred, proba)
            folds.append(FoldMetrics(f1_macro=f1m, roc_auc=roc))
        results.append(aggregate_cv("LDA", folds))
    return results


# --------------------- Noise robustness curve ---------------------

def noise_robustness_curve(
    seqs: List[str],
    y: np.ndarray,
    method: str,
    k: int = 3,
    degree: int = 2,
    train_ps: Tuple[float, ...] = (0.0, 0.05),
    test_ps: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2),
    n_components: int = 16,
    reg: float = 1e-3,
    random_state: int = 0,
) -> Dict[str, float]:
    rng = np.random.RandomState(random_state)
    # Train embedding / model on specified train_ps aggregated
    train_views = []
    for p in train_ps:
        tr_view = prepare_kmer_poly(perturb_substitution(seqs, p=p), k, degree)
        train_views.append(tr_view)
    if method == 'Poly-CCA':
        if len(train_views) != 2:
            raise ValueError("Poly-CCA requires exactly two training ps")
        res = cca(train_views[0], train_views[1], reg=reg, n_components=n_components)
        Z = cca_transform(res, train_views[0], 'X')
    elif method == 'MG-TCCA':
        res = mg_tcca(train_views, reg=reg, n_components=n_components)
        Z = (train_views[0] - train_views[0].mean(0)) @ res.Ws[0]
    else:
        raise ValueError("Unsupported method for noise robustness")
    model = LogisticRegression(max_iter=2000)
    model.fit(Z, y)
    scores = {}
    base_f1 = None
    for p in test_ps:
        test_view = prepare_kmer_poly(perturb_substitution(seqs, p=p), k, degree)
        if method == 'Poly-CCA':
            Zt = cca_transform(res, test_view, 'X')
        else:
            Zt = (test_view - train_views[0].mean(0)) @ res.Ws[0]
        pred = model.predict(Zt)
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Zt)
        f1m, _ = _compute_metrics(y, pred, proba)
        scores[f"p={p}"] = f1m
        if p == 0.0:
            base_f1 = f1m
    if base_f1 is not None:
        for k_ in list(scores.keys()):
            if k_ != 'p=0.0':
                scores[f"Delta({k_})"] = base_f1 - scores[k_]
    return scores


# --------------------- Ablations ---------------------

def ablation_k_degree(
    seqs: List[str],
    y: np.ndarray,
    ks: Iterable[int] = (2, 3, 4),
    degrees: Iterable[int] = (1, 2),
) -> List[CVResult]:
    results: List[CVResult] = []
    for k in ks:
        for d in degrees:
            # logistic regression baseline only to keep runtime manageable
            X = prepare_kmer_poly(seqs, k, d)
            folds = cross_validate(X, y, lambda: LogisticRegression(max_iter=2000))
            results.append(aggregate_cv(f"LR k={k} d={d}", folds))
    return results


# --------------------- Sample size curve ---------------------

def sample_size_curve(
    seqs: List[str],
    y: np.ndarray,
    method: str = 'Poly-CCA',
    fractions: Iterable[float] = (0.1, 0.3, 0.5, 1.0),
    k: int = 3,
    degree: int = 2,
    noise_p: float = 0.05,
    ps: Tuple[float, ...] = (0.0, 0.05, 0.1),
) -> Dict[str, float]:
    n = len(seqs)
    metrics = {}
    rng = np.random.RandomState(0)
    for frac in fractions:
        m = max(5, int(math.ceil(n * frac)))
        idx = rng.choice(n, m, replace=False)
        seq_sub = [seqs[i] for i in idx]
        y_sub = y[idx]
        if method == 'Poly-CCA':
            res = cv_poly_cca(seq_sub, y_sub, k=k, degree=degree, noise_p=noise_p, n_components=16, n_splits=5)
            metrics[f"frac={frac}"] = res.mean_f1_macro
        else:
            res = cv_mg_tcca(seq_sub, y_sub, k=k, degree=degree, ps=ps, n_components=16, n_splits=5)
            metrics[f"frac={frac}"] = res.mean_f1_macro
    return metrics


# --------------------- Motif attribution ---------------------

def motif_attribution(cca_res: CCAResult, kmer_index: Dict[str, int], top_k: int = 10) -> List[Tuple[str, float]]:
    # Use absolute weight of first canonical component within degree-1 segment of features
    # Assumes original k-mer features precede quadratic expansions.
    d_kmer = len(kmer_index)
    comp = cca_res.Wx[:d_kmer, 0]
    ranked = sorted(((kmer, abs(comp[idx])) for kmer, idx in kmer_index.items()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# --------------------- Statistical tests ---------------------

def significance_test(f1_lists_a: List[float], f1_lists_b: List[float]) -> Dict[str, float]:
    t_stat, t_p = stats.ttest_rel(f1_lists_a, f1_lists_b)
    try:
        w_stat, w_p = stats.wilcoxon(f1_lists_a, f1_lists_b)
    except ValueError:  # zero differences
        w_stat, w_p = np.nan, 1.0
    return {"paired_t_p": t_p, "wilcoxon_p": w_p}


# --------------------- Utility save ---------------------

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))


# --------------------- Additional experiment functions matching spec ---------------------

def sanity_check_synthetic(
    cfg: PolySimConfig,
    ps: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.1),
    k: int = 3,
    degree: int = 2,
    n_components: int = 16,
    reg: float = 1e-3,
) -> Dict[str, List[float]]:
    """Generate clean synthetic dataset then perturb with different p to assess rho1 & F1 stability.

    Returns dict with keys: p, rho1_polycca, f1_polycca, f1_baseline, delta_polycca, delta_baseline.
    """
    # generate clean sequences (no substitution in generator; we will perturb manually)
    clean_cfg = PolySimConfig(
        n_per_class=cfg.n_per_class,
        length=cfg.length,
        motif=cfg.motif,
        motif_pos=cfg.motif_pos,
        substitution_rate=0.0,
        seed=cfg.seed,
    )
    data = generate_dataset(clean_cfg)
    y = np.array([r[0] for r in data])
    base_seqs = [r[1] for r in data]
    X_base = prepare_kmer_poly(base_seqs, k, degree)
    # baseline classifier on clean features (for F1 at each p we train on perturbed? We'll follow spec: classification performance under each perturbation)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_base, y)
    baseline_clean_pred = lr.predict(X_base)
    f1_baseline0, _ = _compute_metrics(y, baseline_clean_pred)
    p_list = []
    rho1_list = []
    f1_poly_list = []
    f1_base_list = []
    for p in ps:
        perturbed = perturb_substitution(base_seqs, p=p)
        Xp = prepare_kmer_poly(perturbed, k, degree)
        # Poly-CCA between clean and p-perturbed
        res = cca(X_base, Xp, reg=reg, n_components=n_components)
        rho1_list.append(float(res.correlations[0]))
        Z = cca_transform(res, X_base, 'X')
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Z, y)
        pred_poly = clf.predict(Z)
        f1_poly, _ = _compute_metrics(y, pred_poly)
        f1_poly_list.append(f1_poly)
        # baseline F1 retrain on perturbed features to simulate no-CCA approach at that noise level
        lr_p = LogisticRegression(max_iter=2000)
        lr_p.fit(Xp, y)
        pred_base = lr_p.predict(Xp)
        f1_b, _ = _compute_metrics(y, pred_base)
        f1_base_list.append(f1_b)
        p_list.append(p)
    delta_poly = [f1_poly_list[0] - v for v in f1_poly_list]
    delta_base = [f1_base_list[0] - v for v in f1_base_list]
    return {
        "p": p_list,
        "rho1_polycca": rho1_list,
        "f1_polycca": f1_poly_list,
        "f1_baseline": f1_base_list,
        "delta_polycca": delta_poly,
        "delta_baseline": delta_base,
    }


def indel_extrapolation_test(
    cfg: PolySimConfig,
    p_ins: float = 0.01,
    p_del: float = 0.01,
    k: int = 3,
    degree: int = 2,
    reg: float = 1e-3,
) -> Dict[str, float]:
    """Apply mild INDEL noise and compare Poly-CCA vs baseline F1."""
    base_cfg = PolySimConfig(**{**cfg.__dict__, "substitution_rate": 0.0})
    data = generate_dataset(base_cfg)
    y = np.array([r[0] for r in data])
    seqs = [r[1] for r in data]
    # generate indel variant
    from .datasets import perturb_indel
    seqs_indel = perturb_indel(seqs, p_ins=p_ins, p_del=p_del)
    X_clean = prepare_kmer_poly(seqs, k, degree)
    X_indel = prepare_kmer_poly(seqs_indel, k, degree)
    # Poly-CCA clean vs indel
    res = cca(X_clean, X_indel, reg=reg, n_components=16)
    Z = cca_transform(res, X_clean, 'X')
    lr_poly = LogisticRegression(max_iter=2000).fit(Z, y)
    f1_poly, _ = _compute_metrics(y, lr_poly.predict(Z))
    # Baseline on indel features
    lr_base = LogisticRegression(max_iter=2000).fit(X_indel, y)
    f1_base, _ = _compute_metrics(y, lr_base.predict(X_indel))
    return {"f1_polycca": f1_poly, "f1_baseline": f1_base, "rho1": float(res.correlations[0])}


def multigroup_advantage_test(
    cfg: PolySimConfig,
    ps: Tuple[float, ...] = (0.0, 0.05, 0.1),
    k: int = 3,
    degree: int = 2,
    reg: float = 1e-3,
) -> Dict[str, float]:
    data = generate_dataset(PolySimConfig(**{**cfg.__dict__, "substitution_rate": 0.0}))
    y = np.array([r[0] for r in data])
    seqs = [r[1] for r in data]
    # build multigroup views
    views = [prepare_kmer_poly(perturb_substitution(seqs, p=p), k, degree) for p in ps]
    mg_res = mg_tcca(views, reg=reg, n_components=16)
    Z_mg = (views[0] - views[0].mean(0)) @ mg_res.Ws[0]
    f1_mg, _ = _compute_metrics(y, LogisticRegression(max_iter=2000).fit(Z_mg, y).predict(Z_mg))
    # single CCA between extremes
    res_cca = cca(views[0], views[-1], reg=reg, n_components=16)
    Z_single = cca_transform(res_cca, views[0], 'X')
    f1_single, _ = _compute_metrics(y, LogisticRegression(max_iter=2000).fit(Z_single, y).predict(Z_single))
    return {"f1_mg_tcca": f1_mg, "f1_poly_cca_extremes": f1_single, "rho1_mg": float(mg_res.correlations[0]), "rho1_single": float(res_cca.correlations[0])}


def classifier_invariance_test(
    seqs: List[str],
    y: np.ndarray,
    k: int = 3,
    degree: int = 2,
    noise_p: float = 0.05,
) -> Dict[str, float]:
    X = prepare_kmer_poly(seqs, k, degree)
    noisy = prepare_kmer_poly(perturb_substitution(seqs, noise_p), k, degree)
    res = cca(X, noisy, reg=1e-3, n_components=16)
    Z = cca_transform(res, X, 'X')
    lr = LogisticRegression(max_iter=2000).fit(Z, y)
    svm = LinearSVC().fit(Z, y)
    f1_lr, _ = _compute_metrics(y, lr.predict(Z))
    f1_svm, _ = _compute_metrics(y, svm.predict(Z))
    return {"f1_lr": f1_lr, "f1_linear_svm": f1_svm, "rho1": float(res.correlations[0])}


def kernel_comparison_subset(
    seqs: List[str],
    y: np.ndarray,
    subset: int = 300,
    k: int = 3,
    degree: int = 1,
) -> Dict[str, float]:
    rng = np.random.RandomState(0)
    idx = rng.choice(len(seqs), min(subset, len(seqs)), replace=False)
    sub_seqs = [seqs[i] for i in idx]
    y_sub = y[idx]
    X = prepare_kmer_poly(sub_seqs, k, degree)
    # Linear LR baseline
    lr_folds = cross_validate(X, y_sub, lambda: LogisticRegression(max_iter=2000))
    # RBF-SVM baseline
    rbf_folds = cross_validate(X, y_sub, lambda: SVC(kernel='rbf', probability=True))
    res_lr = aggregate_cv('LR', lr_folds)
    res_rbf = aggregate_cv('RBF-SVM', rbf_folds)
    return {"f1_lr": res_lr.mean_f1_macro, "f1_rbf": res_rbf.mean_f1_macro}


def compute_resource_profile(
    seqs: List[str],
    k: int = 3,
    degree: int = 2,
    noise_p: float = 0.05,
    n_components: int = 16,
) -> Dict[str, float]:
    start = time.time()
    X = prepare_kmer_poly(seqs, k, degree)
    noisy = prepare_kmer_poly(perturb_substitution(seqs, noise_p), k, degree)
    build_time = time.time() - start
    n, d = X.shape
    mem_est_bytes = n * d * 8.0
    res = cca(X, noisy, reg=1e-3, n_components=n_components)
    return {
        "n_samples": n,
        "feature_dim": d,
        "approx_memory_MB": mem_est_bytes / 1e6,
        "cca_components": n_components,
        "build_time_sec": build_time,
    }


def reg_sensitivity_scan(
    seqs: List[str],
    y: np.ndarray,
    regs: Iterable[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    k: int = 3,
    degree: int = 2,
    noise_p: float = 0.05,
) -> Dict[str, float]:
    results = {}
    X = prepare_kmer_poly(seqs, k, degree)
    noisy = prepare_kmer_poly(perturb_substitution(seqs, noise_p), k, degree)
    for r in regs:
        res = cca(X, noisy, reg=r, n_components=16)
        Z = cca_transform(res, X, 'X')
        lr = LogisticRegression(max_iter=2000).fit(Z, y)
        f1m, _ = _compute_metrics(y, lr.predict(Z))
        results[f"reg={r}"] = f1m
    return results


def real_data_noise_fit(
    seqs: List[str],
    y: np.ndarray,
    ps: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.15),
    k: int = 3,
    degree: int = 1,
) -> Dict[str, List[float]]:
    """Compare observed baseline F1 decay vs theoretical (1-p)^{k} scaling proxy.

    Uses logistic regression on k-mer features retrained at each p.
    """
    X_clean, _ = batch_kmer_counts(seqs, k)
    lr_clean = LogisticRegression(max_iter=2000).fit(X_clean, y)
    f1_clean, _ = _compute_metrics(y, lr_clean.predict(X_clean))
    obs = []
    theory = []
    for p in ps:
        if p == 0.0:
            obs.append(f1_clean)
        else:
            pert = perturb_substitution(seqs, p=p)
            Xp, _ = batch_kmer_counts(pert, k)
            lr_p = LogisticRegression(max_iter=2000).fit(Xp, y)
            f1p, _ = _compute_metrics(y, lr_p.predict(Xp))
            obs.append(f1p)
        theory.append(f1_clean * ((1 - p) ** k))
    return {"p": list(ps), "observed_f1": obs, "theory_f1": theory}
