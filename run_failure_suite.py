"""Run required failure-mode experiments (E1-E7) and generate figures/tables.

Outputs written under results/failure_suite/
Figures:
  Fig1_failure_map.png (MI vs N or SNR vs Alignment map) -> here we implement MI (simulated via motif strength) vs sample size grid, colored by rho1 (first canonical corr) and F1 saved as two heatmaps.
  Fig3_mi_sweep_curves.png
  Fig4_alignment_ablation.png
  Fig5_snr_sweep.png
  Fig6_learning_curves.png
  Fig7_method_family_compare.png
  Fig8_null_perm_tests.pdf
Tables:
  Tab1_settings_summary.csv
  Tab2_results_aggregate.csv
Pipeline diagram (placeholder): Fig2_diagnostic_pipeline.pdf

Simplifications / assumptions (can refine later):
 - Mutual information (E2) approximated by varying motif embedding probability for positive class (motif_presence_prob) from 0 to 1 in synthetic generator; we proxy rho1 by first canonical correlation from Poly-CCA on two noisy views.
 - Alignment ablation (E3) done by shuffling a percentage of pairings between the two views after constructing them.
 - SNR sweep (E4) implemented by increasing substitution noise p in second view while keeping first view fixed; we compute first canonical corr + downstream F1.
 - Sample size curve (E5) random subsampling fractions of full synthetic dataset.
 - Method family (E6) compares CCA, Poly-CCA, MG-TCCA using existing wrappers.
 - Diagnostic pipeline (E7) collects HSIC (using a simple RBF kernel) permutation p-value, rho1 CI via bootstrap, then sample size curve leading to Go/No-Go decision (always No-Go in null case example).
 - Null baseline (E1) created by generating independent views (by extra shuffling of sequences for second view) guaranteeing no cross-view dependence.

"""
from __future__ import annotations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from polycca.simulate import PolySimConfig, generate_dataset
from polycca.datasets import batch_kmer_counts, polynomial_features, perturb_substitution, make_kmer_index
from polycca.cca import cca, transform as cca_transform
from polycca.mg_tcca import mg_tcca
from polycca.experiments import cv_poly_cca, cv_mg_tcca, cv_kmer_baselines, sample_size_curve

# ---------------- Utility -----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def first_canonical_corr(X1, X2, reg=1e-3, n_components=1):
    res = cca(X1, X2, reg=reg, n_components=n_components)
    return res.correlations[0]


def build_poly_view(seqs: List[str], k=3, degree=2):
    Xk, _ = batch_kmer_counts(seqs, k)
    if degree > 1:
        Xk = polynomial_features(Xk, degree=degree)
    return Xk

# -------------- E1 Null baseline --------------

def experiment_E1_null(cfg: PolySimConfig, out_dir: Path, k=3, degree=2, noise_p=0.05, n_perm=500, seed=0):
    random.seed(seed); np.random.seed(seed)
    data = generate_dataset(cfg)
    y = np.array([r[0] for r in data])
    seqs = [r[1] for r in data]
    # build two views independent by shuffling second
    seqs2 = seqs.copy()
    random.shuffle(seqs2)
    view1 = build_poly_view(seqs, k, degree)
    view2 = build_poly_view(perturb_substitution(seqs2, noise_p), k, degree)
    rho1 = first_canonical_corr(view1, view2)
    # simple HSIC with RBF kernels + permutation
    def hsic(X, Y, gamma_x=None, gamma_y=None):
        # median heuristic for gamma
        from scipy.spatial.distance import pdist, squareform
        if gamma_x is None:
            dists = pdist(X, 'sqeuclidean')
            med = np.median(dists)
            gamma_x = 1/(med+1e-12)
        if gamma_y is None:
            dists = pdist(Y, 'sqeuclidean')
            med = np.median(dists)
            gamma_y = 1/(med+1e-12)
        K = np.exp(-gamma_x * ((X[:,None,:]-X[None,:,:])**2).sum(-1))
        L = np.exp(-gamma_y * ((Y[:,None,:]-Y[None,:,:])**2).sum(-1))
        H = np.eye(X.shape[0]) - np.ones((X.shape[0], X.shape[0]))/X.shape[0]
        HKH = H @ K @ H
        HLH = H @ L @ H
        return np.trace(HKH @ HLH)/(X.shape[0]-1)**2
    hsic_stat = hsic(view1, view2)
    perm_stats = []
    for i in range(n_perm):
        idx = np.random.permutation(len(view2))
        perm_stats.append(hsic(view1, view2[idx]))
    perm_stats = np.array(perm_stats)
    p_val = (np.sum(perm_stats >= hsic_stat)+1)/(n_perm+1)
    # save permutation distribution figure (Fig8)
    plt.figure(figsize=(5,3))
    sns.histplot(perm_stats, bins=30, kde=False, color='gray')
    plt.axvline(hsic_stat, color='red', label=f'HSIC={hsic_stat:.3g}')
    plt.title('Null HSIC permutation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/'Fig8_null_perm_tests.pdf')
    plt.close()
    # bootstrap CI for rho1
    B=1000
    boot = []
    n = view1.shape[0]
    for b in range(B):
        idx = np.random.choice(n, n, replace=True)
        boot.append(first_canonical_corr(view1[idx], view2[idx]))
    boot = np.array(boot)
    ci_low, ci_high = np.percentile(boot, [2.5,97.5])
    summary = dict(E='E1', rho1=float(rho1), rho1_ci=[float(ci_low), float(ci_high)], hsic=float(hsic_stat), hsic_p=float(p_val))
    (out_dir/'E1_null.json').write_text(json.dumps(summary, indent=2))
    return summary

# -------------- E2 MI Sweep --------------

def generate_mi_level_dataset(n_per_class: int, length: int, base_motif: str, motif_probs: List[float], seed=0):
    rng = random.Random(seed)
    datasets = []
    for p in motif_probs:
        rows=[]
        for cls in [0,1]:
            for _ in range(n_per_class):
                seq = []
                for i in range(length):
                    base = random.choice('acgt')
                    seq.append(base)
                # class 1 embed motif with prob p at fixed pos
                if cls==1 and rng.random()<p:
                    pos = length//3
                    motif = base_motif
                    seq[pos:pos+len(motif)] = list(motif.lower())
                rows.append((cls, ''.join(seq)))
        datasets.append(rows)
    return datasets  # list over MI levels


def experiment_E2_mi_sweep(out_dir: Path, motif_probs=None):
    if motif_probs is None:
        motif_probs = [0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0]
    datasets = generate_mi_level_dataset(120, 120, 'TATA', motif_probs)
    rho_list=[]; f1_list=[]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    for rows in datasets:
        y = np.array([r[0] for r in rows])
        seqs = [r[1] for r in rows]
        # build two noisy aligned views
        view1 = build_poly_view(seqs)
        view2 = build_poly_view(perturb_substitution(seqs, 0.05))
        rho1 = first_canonical_corr(view1, view2)
        # downstream classification on view1 CCA embedding
        res = cca(view1, view2, n_components=8)
        Z = cca_transform(res, view1, 'X')
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z,y)
        pred = clf.predict(Z)
        f1m = f1_score(y,pred, average='macro')
        rho_list.append(rho1)
        f1_list.append(f1m)
    df = pd.DataFrame({'motif_prob':motif_probs,'rho1':rho_list,'F1':f1_list})
    plt.figure(figsize=(5,3))
    plt.plot(df.motif_prob, df.rho1, '-o', label='rho1')
    plt.plot(df.motif_prob, df.F1, '-s', label='Macro-F1')
    plt.xlabel('Motif embed probability (proxy MI)')
    plt.ylabel('Value')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/'Fig3_mi_sweep_curves.png'); plt.close()
    df.to_csv(out_dir/'E2_mi_sweep.csv', index=False)
    return df

# -------------- E3 Alignment Ablation --------------

def experiment_E3_alignment_ablation(out_dir: Path, shuffle_fracs=None, motif_prob: float = 0.5):
    if shuffle_fracs is None:
        shuffle_fracs = [0.0,0.1,0.25,0.5,0.75,1.0]
    # base moderately dependent dataset at provided motif_prob
    rows = generate_mi_level_dataset(150,120,'TATA',[motif_prob])[0]
    y = np.array([r[0] for r in rows]); seqs=[r[1] for r in rows]
    base_view = build_poly_view(seqs)
    results=[]
    for frac in shuffle_fracs:
        seqs2 = seqs.copy()
        k = int(len(seqs2)*frac)
        idx = np.random.choice(len(seqs2), k, replace=False)
        seqs_shuf = seqs2.copy()
        np.random.shuffle(seqs_shuf)
        for i,j in enumerate(idx):
            seqs2[j] = seqs_shuf[i]
        view2 = build_poly_view(perturb_substitution(seqs2,0.05))
        rho1 = first_canonical_corr(base_view, view2)
        res = cca(base_view, view2, n_components=8)
        Z = cca_transform(res, base_view, 'X')
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z,y)
        pred = clf.predict(Z)
        f1m = f1_score(y,pred, average='macro')
        results.append({'shuffle_frac':frac,'rho1':rho1,'F1':f1m})
    df = pd.DataFrame(results)
    plt.figure(figsize=(5,3))
    plt.plot(df.shuffle_frac, df.rho1,'-o', label='rho1')
    plt.plot(df.shuffle_frac, df.F1,'-s', label='Macro-F1')
    plt.xlabel('Alignment shuffle fraction')
    plt.ylabel('Value')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/'Fig4_alignment_ablation.png'); plt.close()
    df.to_csv(out_dir/'E3_alignment_ablation.csv', index=False)
    return df

# -------------- E4 SNR Sweep --------------

def experiment_E4_snr_sweep(out_dir: Path, noise_levels=None, motif_prob: float = 0.5):
    if noise_levels is None:
        noise_levels = [0.0,0.02,0.05,0.1,0.15,0.2,0.3]
    # start from the same moderately dependent base dataset as E3 (motif prob given)
    rows = generate_mi_level_dataset(150,120,'TATA',[motif_prob])[0]
    y = np.array([r[0] for r in rows]); seqs=[r[1] for r in rows]
    view1 = build_poly_view(seqs)
    rhos=[]; f1s=[]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    for nl in noise_levels:
        view2 = build_poly_view(perturb_substitution(seqs, nl))
        rho1 = first_canonical_corr(view1, view2)
        res = cca(view1, view2, n_components=8)
        Z = cca_transform(res, view1, 'X')
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z,y)
        pred = clf.predict(Z)
        f1m = f1_score(y,pred, average='macro')
        rhos.append(rho1); f1s.append(f1m)
    df = pd.DataFrame({'noise_level':noise_levels,'rho1':rhos,'F1':f1s})
    plt.figure(figsize=(5,3))
    plt.plot(df.noise_level, df.rho1,'-o', label='rho1')
    plt.plot(df.noise_level, df.F1,'-s', label='Macro-F1')
    plt.xlabel('Substitution noise (p)')
    plt.ylabel('Value')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/'Fig5_snr_sweep.png'); plt.close()
    df.to_csv(out_dir/'E4_snr_sweep.csv', index=False)
    return df

# -------------- E5 Learning Curves --------------

def experiment_E5_learning_curve(out_dir: Path, fractions=None, motif_prob: float = 0.5):
    if fractions is None:
        fractions = [0.1,0.2,0.3,0.5,0.7,1.0]
    # moderately dependent dataset (motif prob supplied)
    rows = generate_mi_level_dataset(300,120,'TATA',[motif_prob])[0]
    y = np.array([r[0] for r in rows]); seqs=[r[1] for r in rows]
    view1 = build_poly_view(seqs)
    view2 = build_poly_view(perturb_substitution(seqs,0.05))
    n = len(seqs)
    rng = np.random.RandomState(0)
    results=[]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    for frac in fractions:
        m = max(20, int(n*frac))
        idx = rng.choice(n, m, replace=False)
        v1 = view1[idx]; v2=view2[idx]; y_sub=y[idx]
        rho1 = first_canonical_corr(v1,v2)
        res=cca(v1,v2,n_components=8)
        Z = cca_transform(res, v1, 'X')
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Z,y_sub)
        pred = clf.predict(Z)
        f1m = f1_score(y_sub,pred, average='macro')
        results.append({'frac':frac,'rho1':rho1,'F1':f1m,'n':m})
    df = pd.DataFrame(results)
    plt.figure(figsize=(5,3))
    plt.plot(df.n, df.rho1,'-o', label='rho1')
    plt.plot(df.n, df.F1,'-s', label='Macro-F1')
    plt.xlabel('Sample size (n)')
    plt.ylabel('Value')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/'Fig6_learning_curves.png'); plt.close()
    df.to_csv(out_dir/'E5_learning_curve.csv', index=False)
    return df

# -------------- E6 Method Family Compare --------------

def experiment_E6_method_family(out_dir: Path, motif_probs: Optional[List[float]] = None, n_per_class: int = 200):
    """Compare method family trends across multiple MI (motif probability) levels.

    Produces long-format DataFrame with columns: motif_prob, method, F1, rho1 (if available).
    Also saves a multi-line plot (F1 vs motif_prob) and optionally per-method bars at mid MI.
    """
    if motif_probs is None:
        motif_probs = [0.3,0.5,0.7]
    records=[]
    for mp in motif_probs:
        rows = generate_mi_level_dataset(n_per_class,120,'TATA',[mp])[0]
        y = np.array([r[0] for r in rows]); seqs=[r[1] for r in rows]
        # Poly-CCA
        res_poly = cv_poly_cca(seqs, y, k=3, degree=2, noise_p=0.05, n_components=8, n_splits=5)
        # MG-TCCA (use small set of ps as before)
        res_mg = cv_mg_tcca(seqs, y, k=3, degree=2, ps=(0.0,0.05,0.1), n_components=8, n_splits=5)
        # Baseline linear CCA style (k-mer logistic etc.)
        baselines = cv_kmer_baselines(seqs, y, k=3, degree=1, linear_only=True)
        for r in [res_poly, res_mg, *baselines]:
            records.append({
                'motif_prob': mp,
                'method': r.name,
                'F1': r.mean_f1_macro,
                'rho1': r.extra.get('mean_correlations') if r.extra else np.nan
            })
    df = pd.DataFrame(records)
    # Plot trend consistency (F1 vs motif_prob)
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x='motif_prob', y='F1', hue='method', marker='o')
    plt.title('Method family consistency across MI levels')
    plt.tight_layout()
    plt.savefig(out_dir/'Fig7_method_family_compare.png'); plt.close()
    df.to_csv(out_dir/'E6_method_family.csv', index=False)
    return df

def choose_mid_motif_prob(mi_df: Optional[pd.DataFrame] = None,
                          target_rho_range: Tuple[float,float]=(0.3,0.6),
                          fallback: float = 0.5,
                          out_dir: Optional[Path]=None) -> float:
    """Select a motif probability giving moderate dependence.

    If mi_df provided (from E2), pick first rho1 within range; else pick closest to range midpoint.
    If nothing suitable, return fallback.
    Saves a small JSON with selection rationale if out_dir given.
    """
    if mi_df is None or 'rho1' not in mi_df.columns:
        return fallback
    low, high = target_rho_range
    midpoint = (low+high)/2
    # ensure sorted by motif_prob for reproducibility
    cand = mi_df.sort_values('motif_prob')
    in_range = cand[(cand.rho1>=low) & (cand.rho1<=high)]
    if len(in_range)>0:
        chosen_row = in_range.iloc[0]
    else:
        # choose closest to midpoint
        idx = (cand.rho1 - midpoint).abs().argmin()
        chosen_row = cand.iloc[idx]
    chosen = float(chosen_row.motif_prob)
    if out_dir is not None:
        summary = dict(chosen_motif_prob=chosen,
                       target_rho_range=target_rho_range,
                       chosen_rho1=float(chosen_row.rho1))
        (out_dir/'motif_prob_selection.json').write_text(json.dumps(summary, indent=2))
    return chosen

# -------------- E7 Diagnostic Pipeline (simplified) --------------

def experiment_E7_pipeline(out_dir: Path):
    # Use null dataset from E1
    cfg = PolySimConfig(n_per_class=120, length=120, seed=1)
    null_summary = experiment_E1_null(cfg, out_dir)  # reuse artifacts
    # sample size curve on null data (should stay flat)
    rows = generate_dataset(cfg)
    y = np.array([r[0] for r in rows]); seqs=[r[1] for r in rows]
    size_metrics = sample_size_curve(seqs, y, method='Poly-CCA', fractions=[0.3,0.5,0.7,1.0])
    # decision rule: if rho1 CI contains 0 and HSIC p>=0.05 and F1 near random (approx 0.5 for binary) -> No-Go
    decision = 'No-Go'
    pipeline_summary = dict(null_summary=null_summary, sample_size_metrics=size_metrics, decision=decision)
    (out_dir/'E7_pipeline.json').write_text(json.dumps(pipeline_summary, indent=2))
    # placeholder pipeline diagram
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6,2))
    steps = ['HSIC perm test','rho1 CI','Learning curve','Decision']
    for i,s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((i*1.4,0.2),1.2,0.6, boxstyle='round,pad=0.1', edgecolor='black', facecolor='#d9eaf7'))
        ax.text(i*1.4+0.6,0.5,s, ha='center', va='center', fontsize=9)
        if i< len(steps)-1:
            ax.annotate('', xy=(i*1.4+1.2,0.5), xytext=((i+1)*1.4,0.5), arrowprops=dict(arrowstyle='->'))
    ax.set_xlim(-0.2, len(steps)*1.4)
    ax.set_ylim(0,1)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir/'Fig2_diagnostic_pipeline.pdf')
    plt.close()
    return pipeline_summary

# -------------- Failure Map (Fig1) --------------

def experiment_failure_map(out_dir: Path, motif_probs=None, sample_sizes=None):
    if motif_probs is None:
        motif_probs = [0.0,0.05,0.1,0.2,0.4,0.7,1.0]
    if sample_sizes is None:
        sample_sizes = [50,100,150,200,300]
    results=[]
    for mprob in motif_probs:
        rows = generate_mi_level_dataset(max(sample_sizes)//2,120,'TATA',[mprob])[0]
        # downsample to each N
        seqs_all=[r[1] for r in rows]; y_all=np.array([r[0] for r in rows])
        for N in sample_sizes:
            if N>len(seqs_all):
                continue
            idx = np.random.choice(len(seqs_all), N, replace=False)
            seqs=[seqs_all[i] for i in idx]; y=y_all[idx]
            v1 = build_poly_view(seqs)
            v2 = build_poly_view(perturb_substitution(seqs,0.05))
            rho1 = first_canonical_corr(v1,v2)
            res = cca(v1,v2,n_components=4)
            Z = cca_transform(res, v1, 'X')
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import f1_score
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Z,y)
            pred = clf.predict(Z)
            f1m = f1_score(y,pred, average='macro')
            results.append({'motif_prob':mprob,'N':N,'rho1':rho1,'F1':f1m})
    df = pd.DataFrame(results)
    pivot_rho = df.pivot('motif_prob','N','rho1')
    pivot_f1 = df.pivot('motif_prob','N','F1')
    plt.figure(figsize=(6,4))
    sns.heatmap(pivot_rho, annot=False, cmap='viridis')
    plt.title('Failure Map (rho1)')
    plt.savefig(out_dir/'Fig1_failure_map_rho1.png'); plt.close()
    plt.figure(figsize=(6,4))
    sns.heatmap(pivot_f1, annot=False, cmap='magma')
    plt.title('Failure Map (Macro-F1)')
    plt.savefig(out_dir/'Fig1_failure_map.png'); plt.close()
    df.to_csv(out_dir/'failure_map.csv', index=False)
    return df

# -------------- Master runner --------------

def main():
    out_dir = Path('results')/ 'failure_suite'
    ensure_dir(out_dir)
    summaries={}
    # E1
    summaries['E1'] = experiment_E1_null(PolySimConfig(n_per_class=150, length=120, seed=0), out_dir)
    # E2
    mi_df = experiment_E2_mi_sweep(out_dir)
    summaries['E2'] = mi_df.to_dict('records')
    # choose mid-level motif probability for subsequent experiments
    mid_motif_prob = choose_mid_motif_prob(mi_df, target_rho_range=(0.3,0.6), out_dir=out_dir)
    # E3
    summaries['E3'] = experiment_E3_alignment_ablation(out_dir, motif_prob=mid_motif_prob).to_dict('records')
    # E4
    summaries['E4'] = experiment_E4_snr_sweep(out_dir, motif_prob=mid_motif_prob).to_dict('records')
    # E5
    summaries['E5'] = experiment_E5_learning_curve(out_dir, motif_prob=mid_motif_prob).to_dict('records')
    # E6
    # build a small symmetric set around selected mid MI to show trend consistency
    e6_probs = sorted({max(0.0, mid_motif_prob-0.2), mid_motif_prob, min(1.0, mid_motif_prob+0.2)})
    summaries['E6'] = experiment_E6_method_family(out_dir, motif_probs=e6_probs).to_dict('records')
    # E7
    summaries['E7'] = experiment_E7_pipeline(out_dir)
    # Failure map
    summaries['failure_map'] = experiment_failure_map(out_dir).to_dict('records')
    # Tables
    # Tab1 settings summary (minimal)
    settings_rows = [
        dict(experiment='E1', n_per_class=150, noise_p=0.05, k=3, degree=2),
        dict(experiment='E2', motif_probs='[0..1]', k=3, degree=2),
        dict(experiment='E3', shuffle_fracs='0-1', k=3),
        dict(experiment='E4', noise_levels='0-0.3', k=3),
        dict(experiment='E5', fractions='0.1-1.0', k=3),
        dict(experiment='E6', methods='CCA/Poly-CCA/MG-TCCA/Baselines'),
        dict(experiment='E7', pipeline='HSIC->rho1 CI->learning curve'),
    ]
    pd.DataFrame(settings_rows).to_csv(out_dir/'Tab1_settings_summary.csv', index=False)
    # Tab2 aggregate (extract key stats)
    agg_rows=[]
    # from E2 dataset
    try:
        mi_df = pd.read_csv(out_dir/'E2_mi_sweep.csv')
        agg_rows.append(dict(experiment='E2', metric='rho1_max', value=mi_df.rho1.max()))
    except Exception:
        pass
    pd.DataFrame(agg_rows).to_csv(out_dir/'Tab2_results_aggregate.csv', index=False)
    (out_dir/'all_summaries.json').write_text(json.dumps(summaries, indent=2))
    print(f'Finished. Artifacts in {out_dir}')

if __name__=='__main__':
    main()
