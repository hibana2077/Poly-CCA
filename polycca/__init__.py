"""Poly-CCA package.

Public exports for convenience.
"""

from .datasets import read_promoter_dataset, read_splice_dataset, batch_kmer_counts, polynomial_features, perturb_substitution, perturb_indel, make_kmer_index
from .cca import cca, CCAResult
from .mg_tcca import mg_tcca, MGTCCAResult
from .simulate import PolySimConfig, generate_dataset
from .pipeline import run_poly_cca_promoter, run_kmer_baseline_promoter, run_mg_tcca_sim
from .experiments import (
	cv_poly_cca,
	cv_mg_tcca,
	cv_kmer_baselines,
	noise_robustness_curve,
	ablation_k_degree,
	sample_size_curve,
	motif_attribution,
	significance_test,
	sanity_check_synthetic,
	indel_extrapolation_test,
	multigroup_advantage_test,
	classifier_invariance_test,
	kernel_comparison_subset,
	compute_resource_profile,
	reg_sensitivity_scan,
	real_data_noise_fit,
)

__all__ = [
	'read_promoter_dataset','read_splice_dataset','batch_kmer_counts','polynomial_features','perturb_substitution','perturb_indel','make_kmer_index',
	'cca','CCAResult','mg_tcca','MGTCCAResult','PolySimConfig','generate_dataset',
	'run_poly_cca_promoter','run_kmer_baseline_promoter','run_mg_tcca_sim',
	'cv_poly_cca','cv_mg_tcca','cv_kmer_baselines','noise_robustness_curve','ablation_k_degree','sample_size_curve','motif_attribution','significance_test',
	'sanity_check_synthetic','indel_extrapolation_test','multigroup_advantage_test','classifier_invariance_test','kernel_comparison_subset','compute_resource_profile','reg_sensitivity_scan','real_data_noise_fit'
]
