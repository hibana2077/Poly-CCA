"""Run full experiment suite (excluding items 8,12,13) and output JSON summary.

Experiments implemented (mapping to exp.md numbering):
 1 Sanity synthetic (ρ1 & F1 across p)
 2 Substitution noise robustness curves (Poly-CCA, MG-TCCA, baseline)
 3 INDEL extrapolation test
 4 Multi-group advantage (MG-TCCA vs single Poly-CCA)
 5 Polynomial degree ablation (d=1 vs d=2 at k=3)
 6 k-mer length scan (k=2,3,4) with d=1 baseline LR
 7 Sample size curves (Poly-CCA & MG-TCCA)
 9 Classifier invariance (LR vs Linear SVM in canonical space)
 10 Motif attribution (top-k k-mers)
 11 Kernel method comparison subset (LR vs RBF-SVM)
 14 Regularization sensitivity scan
 15 Real data noise fit (observed vs theoretical decay)

Excluded per user request: 8 (cross-dataset transfer), 12 (statistical significance extensive CV), 13 (resource profiling report).

Output: writes a single JSON file results/all_experiments.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from polycca import (
    sanity_check_synthetic,
    noise_robustness_curve,
    indel_extrapolation_test,
    multigroup_advantage_test,
    ablation_k_degree,
    sample_size_curve,
    classifier_invariance_test,
    motif_attribution,
    kernel_comparison_subset,
    reg_sensitivity_scan,
    real_data_noise_fit,
    cv_kmer_baselines,
    cv_poly_cca,
    cv_mg_tcca,
    make_kmer_index,
    read_promoter_dataset,
)
from polycca.simulate import PolySimConfig, generate_dataset
from polycca.datasets import batch_kmer_counts, polynomial_features, perturb_substitution
from polycca.cca import cca

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    out = {}

    # 1 Sanity synthetic
    cfg = PolySimConfig(n_per_class=120, length=120, substitution_rate=0.0, seed=42)
    out['1_sanity_synthetic'] = sanity_check_synthetic(cfg, ps=(0.0,0.01,0.05,0.1))

    # Prepare real datasets (promoter & splice not both needed for every test to keep runtime small)
    data_root = Path('data')
    promoter = read_promoter_dataset(data_root)
    prom_y = np.array([y for y,_ in promoter])
    prom_seqs = [s for _,s in promoter]

    # Synthetic full (for MG / noise heavy tasks)
    synth_full = generate_dataset(cfg)
    synth_y = np.array([y for y,_ in synth_full])
    synth_seqs = [s for _,s in synth_full]

    # 2 Noise robustness curves (Poly-CCA, MG-TCCA, baseline LR)
    poly_curve = noise_robustness_curve(synth_seqs, synth_y, method='Poly-CCA', test_ps=(0.0,0.05,0.1,0.15))
    mg_curve = noise_robustness_curve(synth_seqs, synth_y, method='MG-TCCA', test_ps=(0.0,0.05,0.1,0.15))
    # Baseline: retrain LR per noise level
    base_scores = {}
    from polycca.experiments import prepare_kmer_poly, _compute_metrics
    from sklearn.linear_model import LogisticRegression
    for p in (0.0,0.05,0.1,0.15):
        view = prepare_kmer_poly(perturb_substitution(synth_seqs, p=p), 3, 2)
        lr = LogisticRegression(max_iter=2000).fit(view, synth_y)
        f1m,_ = _compute_metrics(synth_y, lr.predict(view))
        base_scores[f'p={p}'] = f1m
    base_delta = {f'Delta(p={p})': base_scores['p=0.0']-v for p,v in base_scores.items() if p!='p=0.0'}
    out['2_noise_robustness'] = {
        'poly_cca': poly_curve,
        'mg_tcca': mg_curve,
        'baseline_lr': {**base_scores, **base_delta}
    }

    # 3 INDEL extrapolation
    out['3_indel_extrapolation'] = indel_extrapolation_test(cfg, p_ins=0.01, p_del=0.01)

    # 4 Multi-group advantage
    out['4_multigroup_advantage'] = multigroup_advantage_test(cfg, ps=(0.0,0.05,0.1))

    # 5 Polynomial degree ablation (k=3)
    # Filter ablation results for k=3 only
    abl = ablation_k_degree(prom_seqs, prom_y, ks=(3,), degrees=(1,2))
    out['5_degree_ablation'] = {r.name: {'mean_f1': r.mean_f1_macro, 'std_f1': r.std_f1_macro} for r in abl}

    # 6 k-mer length scan (k=2,3,4; use degree=1 to reduce runtime)
    kscan = ablation_k_degree(prom_seqs, prom_y, ks=(2,3,4), degrees=(1,))
    out['6_kmer_scan'] = {r.name: {'mean_f1': r.mean_f1_macro, 'std_f1': r.std_f1_macro} for r in kscan}

    # 7 Sample size curves (Poly-CCA & MG-TCCA on synthetic)
    out['7_sample_size_polycca'] = sample_size_curve(synth_seqs, synth_y, method='Poly-CCA', fractions=(0.1,0.3,0.5,1.0))
    out['7_sample_size_mgtcca'] = sample_size_curve(synth_seqs, synth_y, method='MG-TCCA', fractions=(0.1,0.3,0.5,1.0))

    # 9 Classifier invariance (promoter set)
    out['9_classifier_invariance'] = classifier_invariance_test(prom_seqs, prom_y, k=3, degree=2)

    # 10 Motif attribution (use synthetic Poly-CCA between clean and p=0.05)
    from polycca.datasets import make_kmer_index, kmer_count
    # Build features
    X_clean, k_index = batch_kmer_counts(synth_seqs, 3)
    X_noisy,_ = batch_kmer_counts(perturb_substitution(synth_seqs, 0.05), 3)
    Phi_clean = polynomial_features(X_clean, degree=2)
    Phi_noisy = polynomial_features(X_noisy, degree=2)
    cca_res = cca(Phi_clean, Phi_noisy, reg=1e-3, n_components=8)
    top_kmers = motif_attribution(cca_res, k_index, top_k=10)
    out['10_motif_attribution'] = [{'kmer': kmer, 'weight_abs': w} for kmer,w in top_kmers]

    # 11 Kernel comparison subset (promoter sequences)
    out['11_kernel_subset'] = kernel_comparison_subset(prom_seqs, prom_y, subset=min(80,len(prom_seqs)))

    # 14 Regularization sensitivity (promoter)
    out['14_reg_sensitivity'] = reg_sensitivity_scan(prom_seqs, prom_y, regs=(1e-5,1e-4,1e-3,1e-2))

    # 15 Real data noise fit (promoter)
    out['15_real_noise_fit'] = real_data_noise_fit(prom_seqs, prom_y, ps=(0.0,0.02,0.05,0.1))

    # Write JSON
    outfile = RESULTS_DIR / 'all_experiments.json'
    outfile.write_text(json.dumps(out, indent=2))
    print(f"Written results to {outfile}")


if __name__ == '__main__':
    main()
