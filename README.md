# Poly-CCA

Prototype implementation of Poly-CCA / MG-TCCA pipeline for small-sample DNA classification (Promoter / Splice + synthetic PolySimDNA).

## Repository Structure

| Path | Purpose |
|------|---------|
| `polycca/datasets.py` | Load UCI datasets, k-mer counting, polynomial feature expansion, simple substitution perturbation. |
| `polycca/cca.py` | Ridge-regularized linear CCA implementation. |
| `polycca/mg_tcca.py` | Lightweight multi-group TCCA style shared subspace extraction. |
| `polycca/simulate.py` | Synthetic PolySimDNA generator (motif + GC bias + substitution noise). |
| `polycca/pipeline.py` | High-level experiment runners (baseline, Poly-CCA, MG-TCCA). |
| `run_experiments.py` | Example script to execute a small set of experiments. |
| `data/` | Raw UCI datasets already placed here. |

## Quick Start

Install dependencies (optional if environment already has numpy/scikit-learn):

```bash
pip install -r requirements.txt
```

Run demo experiments:

```bash
python run_experiments.py
```

Example output (values will vary):

```
ExperimentResult(name='k-mer baseline', macro_f1=0.72, roc_auc=0.80, ...)
ExperimentResult(name='Poly-CCA Promoter', macro_f1=0.78, roc_auc=0.85, correlations=[...])
ExperimentResult(name='MG-TCCA Sim', macro_f1=0.90, roc_auc=0.95, correlations=[...])
```

## Extending

Planned / easy extensions:

- Cross-validation & statistical tests
- Noise robustness sweeps (vary p, add indel noise model)
- Motif attribution: map canonical vectors back to k-mers (weight ranking)
- Save canonical embeddings for downstream models
- Add regularization scans & hyperparameter search

## Design Choices

- k=3 (64-dim) k-mer normalized counts -> degree-2 polynomial expansion (O( ~2K ) features)
- Simple substitution perturbation for second view(s) to approximate theoretical shrinkage model
- Ridge regularization (1e-3 / 1e-4) to stabilize covariance inverses in small-sample regime

## License

Research prototype – add explicit license if distributing.

## Citation (Draft)

If you use this prototype, please cite the forthcoming manuscript:

```text
Poly-CCA: Polynomial-Perturbation-Invariant Multi-Group Canonical Representations for Small-Sample DNA Classification (2025) (preprint forthcoming)
```
