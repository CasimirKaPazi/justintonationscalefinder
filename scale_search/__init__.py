"""
Just Intonation Scale Search
Evolutionary optimization of microtonal scales using multiple consonance metrics.

Copyright (C) 2025

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from scale_search.config import (
    SearchConfig, COMBINE_METHODS, validate_combine_method, normalize_notes,
)
from scale_search.util import (
    prime_factors, parse_ratios, reduce_to_octave, sanitize_scale,
    is_valid_ratio, interval_between, generate_candidate_ratios,
    snap_to_candidates,
)
from scale_search.metrics import (
    CONSONANCE_METRICS, SCALE_METRICS,
    tenney_height, euler_gradus, harmonic_entropy, plomp_levelt_roughness,
    wilson_complexity, barlow_harmonicity, prime_limit_penalty, odd_limit,
    step_size_penalty, prime_diversity_score, edo_penalty, edo_deviation,
    scale_balance, scale_propriety, scale_variety,
    init_normalization, get_normalization, quantile_normalize,
    combined_metric, cached_metric, cached_metric_nd, clear_metric_cache,
    parse_metrics, format_metric_description,
    scale_fingerprint, fingerprint_distance, fingerprint_diversity_score,
)
from scale_search.operators import (
    SIMPLE_INTERVALS, COMMAS,
    mutate, crossover, create_random_scale,
    scale_invert, scale_rotate, crossover_slotwise, repair_slots, weighted_sample,
)
from scale_search.evolution import (
    evaluate_scale, evolve, combine_penalties,
    evaluate_grouped_scale, subscale_of, slots_to_pitches,
)
from scale_search.analysis import (
    scale_cents, compare_scales, nearest_scales, edo_fits,
)
from scale_search.scala import (
    ScalaScale, parse_scl, load_scala_archive, load_reference_scales,
)
from scale_search.groups import (
    BUILTIN_STRUCTURES, GroupStructure, RANK_NAMES, build_polytope,
    parse_group_spec, describe_structures,
)
from scale_search.display import (
    load_seed_scales, find_matching_seed,
    format_scale, print_interval_matrix, evaluate_named_scale, print_scale_details,
    metric_breakdown, parse_pitches,
)

# main() is available via: python -m scale_search
# or: from scale_search.__main__ import main

__all__ = [
    # config
    'SearchConfig', 'COMBINE_METHODS', 'validate_combine_method', 'normalize_notes',
    # util
    'prime_factors', 'parse_ratios', 'reduce_to_octave', 'sanitize_scale',
    'is_valid_ratio', 'interval_between', 'generate_candidate_ratios',
    'snap_to_candidates',
    # metrics
    'CONSONANCE_METRICS', 'SCALE_METRICS',
    'tenney_height', 'euler_gradus', 'harmonic_entropy', 'plomp_levelt_roughness',
    'wilson_complexity', 'barlow_harmonicity', 'prime_limit_penalty', 'odd_limit',
    'step_size_penalty', 'prime_diversity_score', 'edo_penalty', 'edo_deviation',
    'scale_balance', 'scale_propriety', 'scale_variety',
    'init_normalization', 'get_normalization', 'quantile_normalize',
    'combined_metric', 'cached_metric', 'cached_metric_nd', 'clear_metric_cache',
    'parse_metrics', 'format_metric_description',
    'scale_fingerprint', 'fingerprint_distance', 'fingerprint_diversity_score',
    # operators
    'SIMPLE_INTERVALS', 'COMMAS',
    'mutate', 'crossover', 'create_random_scale',
    'scale_invert', 'scale_rotate', 'crossover_slotwise', 'repair_slots', 'weighted_sample',
    # evolution
    'evaluate_scale', 'evolve', 'combine_penalties',
    'evaluate_grouped_scale', 'subscale_of', 'slots_to_pitches',
    # analysis
    'scale_cents', 'compare_scales', 'nearest_scales', 'edo_fits',
    # scala
    'ScalaScale', 'parse_scl', 'load_scala_archive', 'load_reference_scales',
    # groups
    'BUILTIN_STRUCTURES', 'GroupStructure', 'RANK_NAMES', 'build_polytope',
    'parse_group_spec', 'describe_structures',
    # display
    'load_seed_scales', 'find_matching_seed',
    'format_scale', 'print_interval_matrix', 'evaluate_named_scale', 'print_scale_details',
    'metric_breakdown', 'parse_pitches',
]
