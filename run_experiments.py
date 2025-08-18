"""Lightweight sanity experiment runner.

Runs a small subset to verify major components:
  1. Promoter dataset baseline vs Poly-CCA
  2. Synthetic sanity check (ρ1 & F1 across p)
  3. Noise robustness curve (Poly-CCA)
  4. INDEL extrapolation test
  5. MG-TCCA multigroup advantage quick check

Outputs JSON-like summaries to stdout. Keep parameters small for speed.
"""
from __future__ import annotations
from pathlib import Path
import json
from polycca.pipeline import run_poly_cca_promoter, run_kmer_baseline_promoter, run_mg_tcca_sim
from polycca.simulate import PolySimConfig
from polycca import (
    sanity_check_synthetic,
    noise_robustness_curve,
    indel_extrapolation_test,
    multigroup_advantage_test,
)


def jprint(tag: str, obj):
    print(f"\n=== {tag} ===")
    try:
        print(json.dumps(obj, indent=2))
    except TypeError:
        print(obj)


def main():
    root = Path('data')
    # 1. Promoter baseline vs Poly-CCA
    base_res = run_kmer_baseline_promoter(root)
    poly_res = run_poly_cca_promoter(root)
    jprint('Promoter k-mer baseline', base_res.__dict__)
    jprint('Promoter Poly-CCA', poly_res.__dict__)

    # 2. Synthetic sanity (reduced sample size for speed)
    cfg = PolySimConfig(n_per_class=60, length=100, substitution_rate=0.0, seed=1)
    sanity = sanity_check_synthetic(cfg, ps=(0.0, 0.01, 0.05))
    jprint('Synthetic sanity', sanity)

    # 3. Noise robustness curve (Poly-CCA)
    # Re-use same synthetic sequences (inside function it perturbs)
    # We'll just regenerate to keep API simple
    cfg2 = PolySimConfig(n_per_class=60, length=100, substitution_rate=0.0, seed=2)
    data_for_curve = sanity_check_synthetic(cfg2, ps=(0.0, 0.05))  # produce base dataset indirectly
    # Not reusing; directly build seq list from generator for curve
    from polycca.simulate import generate_dataset
    seqs_dataset = generate_dataset(cfg2)
    seqs = [s for _, s in seqs_dataset]
    y = __import__('numpy').array([c for c, _ in seqs_dataset])
    curve = noise_robustness_curve(seqs, y, method='Poly-CCA', test_ps=(0.0,0.05,0.1))
    jprint('Noise robustness Poly-CCA', curve)

    # 4. INDEL extrapolation test
    indel = indel_extrapolation_test(cfg, p_ins=0.01, p_del=0.01)
    jprint('INDEL extrapolation', indel)

    # 5. MG-TCCA multigroup advantage
    mg_adv = multigroup_advantage_test(cfg, ps=(0.0, 0.05, 0.1))
    jprint('MG-TCCA advantage', mg_adv)

    print('\nAll sanity checks completed.')


if __name__ == '__main__':
    main()
