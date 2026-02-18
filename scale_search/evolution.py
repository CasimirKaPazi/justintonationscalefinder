"""Evolutionary search: evaluation, selection, and the main evolution loop."""

import math
import random
import bisect
from fractions import Fraction
from typing import List, Set, Tuple

from scale_search.config import SearchConfig
from scale_search.util import generate_candidate_ratios, sanitize_scale
from scale_search.metrics import (
    CONSONANCE_METRICS, SCALE_METRICS,
    cached_metric, combined_metric, get_normalization, init_normalization,
    edo_penalty, edo_deviation,
    scale_fingerprint, fingerprint_distance, fingerprint_diversity_score,
    format_metric_description,
)
from scale_search.operators import (
    mutate, crossover, create_random_scale,
    scale_invert, scale_rotate,
)

def evaluate_scale(scale: List[Fraction], 
                   metrics: List[Tuple[str, float]] = None,
                   octave_range: int = 1,
                   interval_cache: dict = None,
                   scale_metrics: List[Tuple[str, float]] = None,
                   edo: int = 0,
                   combine_method: str = 'geometric') -> Tuple[float, dict]:
    """
    Evaluate a scale's quality using the specified metric(s).
    Returns (fitness, details) where higher fitness = better scale.
    
    Args:
        metrics: List of (metric_name, weight) tuples for interval evaluation
        scale_metrics: List of (metric_name, weight) tuples for whole-scale properties
        octave_range: How many octaves above/below to consider
        interval_cache: Optional precomputed {(n,d): score} lookup for fast evaluation
        edo: If > 0, penalize notes far from this EDO's steps (for neji scales)
        combine_method: How to combine interval scores:
            'geometric' - geometric mean (default): one bad interval pulls down score
            'worst' - minimax: optimize the worst interval
            'mean' - arithmetic mean: bad intervals can be averaged out
            'power2', 'power3', etc. - power mean with given exponent
            'logloss' - log loss: heavily penalizes near-dissonant intervals
    
    All metrics are converted to quality scores (0-1, higher = better).
    Final score combines interval quality with scale metrics via weighted geometric mean.
    """
    if metrics is None:
        metrics = [('tenney', 1.0)]
    if scale_metrics is None:
        scale_metrics = []
    
    # Build full scale across octaves
    base_scale = [Fraction(1, 1)] + sorted(scale) + [Fraction(2, 1)]
    
    # Extend to multiple octaves - use tuples for speed
    full_scale = []
    for octave_shift in range(-octave_range, octave_range + 1):
        mult_n = 2 ** max(0, octave_shift)
        mult_d = 2 ** max(0, -octave_shift)
        for note in base_scale:
            n = note.numerator * mult_n
            d = note.denominator * mult_d
            g = math.gcd(n, d)
            full_scale.append((n // g, d // g))
    
    # Remove duplicates and sort by value
    full_scale = sorted(set(full_scale), key=lambda x: x[0] / x[1])
    
    interval_scores = []
    individual_scores = {name: [] for name, _ in metrics}
    
    # Compute all pairwise intervals
    for i in range(len(full_scale)):
        n1, d1 = full_scale[i]
        for j in range(i + 1, len(full_scale)):
            n2, d2 = full_scale[j]
            
            # Compute interval key directly (faster than using Fraction)
            int_n = n2 * d1
            int_d = d2 * n1
            g = math.gcd(int_n, int_d)
            int_n, int_d = int_n // g, int_d // g
            
            # Reduce to single octave
            while int_n >= 2 * int_d:
                int_d *= 2
                g = math.gcd(int_n, int_d)
                int_n, int_d = int_n // g, int_d // g
            while int_n < int_d:
                int_n *= 2
                g = math.gcd(int_n, int_d)
                int_n, int_d = int_n // g, int_d // g
            
            key = (int_n, int_d)
            
            # Use cache if available, otherwise compute and cache
            if interval_cache is not None:
                if key in interval_cache:
                    score = interval_cache[key]
                    interval_scores.append(score)
                    for name, _ in metrics:
                        if (key, name) in interval_cache:
                            individual_scores[name].append(interval_cache[(key, name)])
                else:
                    # Compute and cache
                    interval = Fraction(int_n, int_d)
                    score = combined_metric(interval, metrics, normalize=True)
                    interval_cache[key] = score
                    interval_scores.append(score)
                    for name, _ in metrics:
                        val = cached_metric(interval, name)
                        interval_cache[(key, name)] = val
                        individual_scores[name].append(val)
            else:
                interval = Fraction(int_n, int_d)
                score = combined_metric(interval, metrics, normalize=True)
                interval_scores.append(score)
                for name, _ in metrics:
                    individual_scores[name].append(cached_metric(interval, name))
    
    # Combine interval scores using specified method
    # interval_scores are penalties (0-1, lower = better)
    if not interval_scores:
        raw_interval_penalty = 0
    elif combine_method == 'worst':
        # Minimax: the worst interval determines the score
        raw_interval_penalty = max(interval_scores)
    elif combine_method == 'geometric':
        # Geometric mean: one bad interval pulls down the whole score
        # For penalties, we compute geometric mean directly
        # Clamp to avoid log(0)
        clamped = [max(0.001, min(0.999, s)) for s in interval_scores]
        log_sum = sum(math.log(s) for s in clamped)
        raw_interval_penalty = math.exp(log_sum / len(clamped))
    elif combine_method.startswith('power'):
        # Power mean: (1/n * Î£ x^p)^(1/p)
        # Higher p emphasizes worse (higher) penalties more
        # p=1 is arithmetic mean, p=2 is RMS, pâ†’âˆž approaches max
        try:
            p = int(combine_method[5:]) if len(combine_method) > 5 else 2
        except ValueError:
            p = 2
        p = max(1, min(10, p))  # Clamp to reasonable range
        # Clamp scores to avoid issues with very small values
        clamped = [max(0.001, s) for s in interval_scores]
        power_sum = sum(s ** p for s in clamped)
        raw_interval_penalty = (power_sum / len(clamped)) ** (1 / p)
    elif combine_method == 'logloss':
        # Log loss: heavily penalizes near-dissonant intervals
        # -log(1-p) grows unboundedly as pâ†’1
        # We compute mean log loss then squash back to [0,1)
        # Clamp away from 1 to avoid log(0)
        clamped = [min(0.999, s) for s in interval_scores]
        log_loss_raw = sum(-math.log(1 - s) for s in clamped) / len(clamped)
        # Squash to [0,1): penalty = 1 - exp(-log_loss)
        raw_interval_penalty = 1 - math.exp(-log_loss_raw)
    else:  # 'mean' or default
        # Arithmetic mean: bad intervals can be averaged out
        raw_interval_penalty = sum(interval_scores) / len(interval_scores)
    
    # Note count normalization:
    # 
    # Larger scales naturally have worse average interval scores because they
    # include more complex intervals. Without correction, the optimizer always
    # prefers fewer notes when searching across variable note counts.
    #
    # We divide the raw penalty by exp(1 - 1/n) which compensates for the
    # increasing average dissonance. This brings the quality spread across
    # note counts down to ~0.05 (e.g. 0.93 for 30-note vs 0.97 for 5-note
    # random scales), which is a reasonable residual bias: simpler scales
    # genuinely have better intervals on average.
    
    num_notes = len(scale) + 1  # +1 for 1/1 (2/1 is octave, not separately counted)
    
    note_count_factor = math.exp(1 - 1/num_notes) if num_notes > 1 else 1.0
    interval_penalty = raw_interval_penalty / note_count_factor
    
    # Convert to quality (higher = better)
    interval_quality = 1.0 - interval_penalty
    interval_quality = max(0.001, interval_quality)  # Avoid zero for geometric mean
    
    # Collect all quality scores for geometric mean combination
    # Each entry: (quality, weight) where quality is 0-1 and higher = better
    quality_scores = []
    scale_metric_scores = {}
    scale_metric_scores['num_notes'] = num_notes
    
    # Interval quality as first component
    interval_weight = sum(w for _, w in metrics)
    quality_scores.append((interval_quality, interval_weight))
    
    for metric_name, weight in scale_metrics:
        if metric_name == 'edo':
            raw_score = edo_penalty(scale, edo)
            scale_metric_scores['edo'] = raw_score
            scale_metric_scores['edo_deviation'] = edo_deviation(scale, edo)
        else:
            metric_fn, _ = SCALE_METRICS[metric_name]
            raw_score = metric_fn(scale)
            scale_metric_scores[metric_name] = raw_score
        
        # Normalize to 0-1
        min_val, max_val = get_normalization(metric_name)
        norm_score = (raw_score - min_val) / (max_val - min_val)
        norm_score = max(0.0, min(1.0, norm_score))
        
        # Convert to quality (higher = better)
        # 'primes' and 'variety' are already positive (more = better)
        # 'steps', 'balance', 'propriety', 'edo' are penalties (lower = better), so invert
        if metric_name in ('primes', 'variety'):
            quality = norm_score  # Already: higher = higher quality
        else:
            quality = 1.0 - norm_score  # Invert penalties
        
        quality = max(0.001, quality)  # Avoid zero for geometric mean
        quality_scores.append((quality, weight))
    
    # Combine using weighted geometric mean
    # Formula: product(q_i ^ w_i) ^ (1 / sum(w_i))
    # Equivalently: exp(sum(w_i * log(q_i)) / sum(w_i))
    total_weight = sum(w for _, w in quality_scores)
    
    if total_weight > 0:
        log_sum = sum(w * math.log(q) for q, w in quality_scores)
        final_quality = math.exp(log_sum / total_weight)
    else:
        final_quality = interval_quality
    
    # Fitness = quality (higher = better, range 0-1)
    fitness = final_quality
    
    metric_avgs = {name: sum(scores) / len(scores) if scores else 0 
                   for name, scores in individual_scores.items()}
    metric_avgs.update(scale_metric_scores)
    
    details = {
        'quality': final_quality,  # Higher = better, range 0-1
        'interval_quality': interval_quality,
        'raw_interval_penalty': raw_interval_penalty,
        'min_score': min(interval_scores) if interval_scores else 0,
        'max_score': max(interval_scores) if interval_scores else 0,
        'metrics': metrics,
        'scale_metrics': scale_metrics,
        'metric_averages': metric_avgs,
        'num_intervals': len(interval_scores),
        'octave_range': octave_range,
        'edo': edo,
    }
    
    return fitness, details


def local_search(scale: List[Fraction], candidates: List[Fraction], 
                 eval_fn, config: SearchConfig, max_iterations: int = 10,
                 candidate_cents: List[Tuple[float, int]] = None) -> Tuple[List[Fraction], float]:
    """
    Hill-climbing local search: try small improvements to a scale.
    Returns improved scale and its fitness.
    
    Args:
        candidate_cents: Optional pre-computed list of (cents, index) sorted by cents
                        for O(log n) neighbor lookups instead of O(n)
    
    Tries:
    1. Swapping each note with nearby candidates
    2. Single-note mutations
    
    Respects required_notes (won't modify) and excluded_notes (won't add).
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = config.excluded_notes
    
    current = scale.copy()
    current_fitness = eval_fn(current)
    scale_set = set(current)
    
    # Indices of notes that can be modified
    mutable_indices = [i for i, note in enumerate(current) if note not in required]
    
    for _ in range(max_iterations):
        improved = False
        
        # Try swapping each mutable note with a similar one
        for i in mutable_indices:
            note = current[i]
            note_cents = 1200 * math.log2(float(note))
            
            # Find candidates within 100 cents using binary search if available
            if candidate_cents is not None:
                # Binary search for range [note_cents - 100, note_cents + 100]
                lo = bisect.bisect_left(candidate_cents, (note_cents - 100, -1))
                hi = bisect.bisect_right(candidate_cents, (note_cents + 100, float('inf')))
                nearby = [candidates[idx] for cents, idx in candidate_cents[lo:hi]
                         if candidates[idx] not in scale_set and candidates[idx] not in excluded]
            else:
                # Fallback to linear scan
                nearby = [c for c in candidates 
                         if c not in scale_set and c not in excluded
                         and abs(1200 * math.log2(float(c)) - note_cents) < 100]
            
            for replacement in nearby[:5]:  # Try up to 5 nearby notes
                test_scale = current.copy()
                test_scale[i] = replacement
                test_fitness = eval_fn(test_scale)
                
                if test_fitness > current_fitness:
                    current = test_scale
                    current_fitness = test_fitness
                    scale_set = set(current)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try a single mutation (mutate already respects constraints)
            mutated = mutate(current, candidates, config)
            mutated_fitness = eval_fn(mutated)
            if mutated_fitness > current_fitness:
                current = mutated
                current_fitness = mutated_fitness
                scale_set = set(current)
            else:
                break  # No improvement found
    
    return current, current_fitness


def tournament_select(population: List[Tuple[float, List[Fraction]]], k: int) -> List[Fraction]:
    """Select an individual via tournament selection."""
    tournament = random.sample(population, k)
    winner = max(tournament, key=lambda x: x[0])
    return winner[1]


def select_elites(population: List[Tuple[float, List[Fraction]]], 
                  n_elites: int) -> List[Tuple[float, List[Fraction]]]:
    """
    Select elite individuals with increasing randomness.
    
    First elite is always the best (greedy). Subsequent elites are sampled
    from an increasingly large window of top candidates, introducing more
    randomness for lower-ranked selections while avoiding duplicates.
    
    This provides natural diversity without expensive similarity calculations.
    """
    if n_elites <= 0 or not population:
        return []
    
    # Sort by fitness
    sorted_pop = sorted(population, key=lambda x: x[0], reverse=True)
    
    elites = []
    seen_scales = set()
    
    for k in range(n_elites):
        # Window size grows: 1, 3, 6, 10, 15... (triangular numbers give smooth growth)
        # Or simpler: 1, 4, 8, 12, 16... (1 + 4*k)
        window_size = min(1 + 4 * k, len(sorted_pop))
        
        # Find candidates in window that aren't duplicates
        candidates = []
        for ind in sorted_pop[:window_size]:
            scale_key = frozenset((r.numerator, r.denominator) for r in ind[1])
            if scale_key not in seen_scales:
                candidates.append(ind)
        
        if candidates:
            if k == 0:
                # First one is always the best
                chosen = candidates[0]
            else:
                # Random from available candidates in window
                chosen = random.choice(candidates)
            
            elites.append(chosen)
            scale_key = frozenset((r.numerator, r.denominator) for r in chosen[1])
            seen_scales.add(scale_key)
        else:
            # All in window are duplicates, try rest of population
            for ind in sorted_pop[window_size:]:
                scale_key = frozenset((r.numerator, r.denominator) for r in ind[1])
                if scale_key not in seen_scales:
                    elites.append(ind)
                    seen_scales.add(scale_key)
                    break
    
    return elites


def select_elites_diverse(population: List[Tuple[float, List[Fraction]]],
                          n_elites: int,
                          get_fingerprint_fn) -> List[Tuple[float, List[Fraction]]]:
    """
    Select elite individuals with fingerprint-based diversity.
    
    Combines fitness ranking with metric-space diversity:
    - First elite is always the best (greedy)
    - Subsequent elites balance fitness and distance from already-selected elites
    
    This promotes exploration of different regions of the metric space.
    
    Args:
        population: List of (fitness, scale) tuples
        n_elites: Number of elites to select
        get_fingerprint_fn: Function to get fingerprint for a scale
    """
    if n_elites <= 0 or not population:
        return []
    
    # Sort by fitness
    sorted_pop = sorted(population, key=lambda x: x[0], reverse=True)
    
    elites = []
    elite_fingerprints = []
    seen_scales = set()
    
    for k in range(n_elites):
        best_candidate = None
        best_score = -float('inf')
        
        # Consider top candidates (larger window for later elites)
        window_size = min(10 + 5 * k, len(sorted_pop))
        
        for i, ind in enumerate(sorted_pop[:window_size]):
            scale_key = frozenset((r.numerator, r.denominator) for r in ind[1])
            if scale_key in seen_scales:
                continue
            
            fitness = ind[0]
            fp = get_fingerprint_fn(ind[1])
            
            if k == 0:
                # First elite: pure fitness
                score = fitness
            else:
                # Later elites: fitness + diversity bonus
                # Diversity = average distance to already-selected elites
                if elite_fingerprints and fp:
                    diversity = sum(fingerprint_distance(fp, efp) for efp in elite_fingerprints) / len(elite_fingerprints)
                else:
                    diversity = 0.5
                
                # Fitness rank penalty (prefer higher-ranked)
                rank_factor = 1.0 - (i / window_size) * 0.3  # 1.0 to 0.7
                
                # Combine: mostly fitness, but diversity helps break ties
                score = fitness * rank_factor + diversity * 0.1
            
            if score > best_score:
                best_score = score
                best_candidate = ind
        
        if best_candidate:
            elites.append(best_candidate)
            scale_key = frozenset((r.numerator, r.denominator) for r in best_candidate[1])
            seen_scales.add(scale_key)
            fp = get_fingerprint_fn(best_candidate[1])
            if fp:
                elite_fingerprints.append(fp)
    
    return elites


def evolve(config: SearchConfig, verbose: bool = True, 
           seed_scales: List[List[Fraction]] = None) -> Tuple[List[Fraction], float, dict]:
    """
    Run evolutionary search with simulated annealing and diversity preservation.
    
    Args:
        config: SearchConfig with all parameters
        verbose: Print progress
        seed_scales: Optional list of seed scales to initialize population.
                    If None, uses default seeds. If empty list [], uses no seeds.
    """
    candidates, candidate_weights = generate_candidate_ratios(config.ratio_limit)
    
    # Compute normalization ranges from actual candidates
    init_normalization(config.ratio_limit)
    
    metrics = config.metrics
    scale_metrics = config.scale_metrics
    octave_range = config.octave_range
    edo = config.edo
    combine_method = config.combine_method
    
    # Pre-compute cents for all candidates (for fast neighbor lookups)
    # Store as list of (cents, index) sorted by cents
    candidate_cents = [(1200 * math.log2(float(c)), i) for i, c in enumerate(candidates)]
    candidate_cents.sort()  # Sort by cents value
    
    # Build metric description string
    metric_desc = format_metric_description(metrics, scale_metrics, edo)
    
    note_desc = f"exactly {config.fixed_notes}" if config.fixed_notes else f"up to {config.max_notes}"
    octave_desc = f"{2 * octave_range + 1} octaves" if octave_range > 0 else "1 octave"
    
    # Note constraints
    required_notes = config.required_notes if config.required_notes else []
    excluded_notes = config.excluded_notes
    
    if verbose:
        print(f"Generated {len(candidates)} candidate ratios")
        if required_notes:
            print(f"Required notes: {', '.join(str(r) for r in required_notes)}")
        if excluded_notes:
            print(f"Excluded notes: {', '.join(str(r) for r in excluded_notes)}")
        print()
    
    # Lazy interval cache - compute on demand instead of precomputing all pairs
    # This avoids O(nÂ²) precomputation which becomes prohibitive for large ratio limits
    interval_cache = {}
    
    # Scale fitness cache: frozenset of (n,d) tuples -> fitness
    scale_cache = {}
    
    # Fingerprint cache: frozenset of (n,d) tuples -> fingerprint tuple
    fingerprint_cache = {}
    
    def eval_scale_cached(scale):
        """Evaluate scale and cache both fitness and fingerprint."""
        key = frozenset((r.numerator, r.denominator) for r in scale)
        if key not in scale_cache:
            fitness, details = evaluate_scale(scale, metrics, octave_range, interval_cache, scale_metrics, edo, combine_method)
            scale_cache[key] = fitness
            fingerprint_cache[key] = scale_fingerprint(details)
        return scale_cache[key]
    
    def get_fingerprint(scale):
        """Get fingerprint for a scale (must be evaluated first)."""
        key = frozenset((r.numerator, r.denominator) for r in scale)
        if key not in fingerprint_cache:
            eval_scale_cached(scale)  # This will populate the cache
        return fingerprint_cache.get(key, ())
    
    # Use provided seeds (empty list means no seeds)
    if seed_scales is None:
        seed_scales = []
    
    # Initialize population
    # Note: N notes = N-1 ratios (tonic 1/1 is implicit, octave 2/1 is repeat)
    population = []
    
    def check_constraints(scale):
        """Check if scale satisfies required/excluded constraints."""
        scale_set = set(scale)
        if not all(r in scale_set for r in required_notes):
            return False
        if any(e in scale_set for e in excluded_notes):
            return False
        return True
    
    def add_scale_if_valid(scale, target_len=None):
        """Add scale to population if it meets size and constraint requirements.
        
        If scale is shorter than target, adds random notes from candidates.
        If scale is longer than target, removes notes (keeping required ones).
        """
        # Sanitize: ensure all ratios are within octave, no duplicates
        scale = sanitize_scale(scale)
        
        if not scale or not check_constraints(scale):
            return
        
        if target_len is not None:
            if len(scale) == target_len:
                fitness = eval_scale_cached(scale)
                population.append((fitness, list(scale)))
            elif len(scale) > target_len:
                # Remove notes to reach target length, but keep required notes
                optional = [n for n in scale if n not in required_notes]
                num_optional = target_len - len(required_notes)
                if num_optional >= 0 and len(optional) >= num_optional:
                    trimmed = list(required_notes) + random.sample(optional, num_optional)
                    fitness = eval_scale_cached(trimmed)
                    population.append((fitness, trimmed))
            else:
                # Scale is shorter - add notes from candidates to reach target
                scale_set = set(scale)
                available = [c for c in candidates 
                            if c not in scale_set and c not in excluded_notes]
                num_to_add = target_len - len(scale)
                if len(available) >= num_to_add:
                    # Weight toward simpler ratios
                    available_weights = [candidate_weights[candidates.index(c)] 
                                        for c in available]
                    additions = []
                    for _ in range(num_to_add):
                        if not available:
                            break
                        # Weighted random selection
                        total_w = sum(available_weights)
                        r = random.random() * total_w
                        cumsum = 0
                        for idx, w in enumerate(available_weights):
                            cumsum += w
                            if cumsum >= r:
                                additions.append(available[idx])
                                available.pop(idx)
                                available_weights.pop(idx)
                                break
                    expanded = list(scale) + additions
                    if check_constraints(expanded):
                        fitness = eval_scale_cached(expanded)
                        population.append((fitness, expanded))
        else:
            if config.min_notes - 1 <= len(scale) <= config.max_notes - 1:
                fitness = eval_scale_cached(scale)
                population.append((fitness, list(scale)))
    
    # Add seed scales and their transformations
    target_len = config.fixed_notes - 1 if config.fixed_notes else None
    
    for seed in seed_scales:
        # Original seed
        add_scale_if_valid(seed, target_len)
        
        # Inverted version
        inverted = scale_invert(seed, config.ratio_limit)
        if inverted:
            add_scale_if_valid(inverted, target_len)
        
        # Rotated versions (try a few random rotations)
        for _ in range(min(3, len(seed))):
            rotated = scale_rotate(seed, limit=config.ratio_limit)
            if rotated:
                add_scale_if_valid(rotated, target_len)
    
    # Fill rest with random scales (biased toward simpler ratios)
    while len(population) < config.population_size:
        if config.fixed_notes:
            num_ratios = config.fixed_notes - 1
        else:
            num_ratios = random.randint(config.min_notes - 1, config.max_notes - 1)
        scale = create_random_scale(candidates, num_ratios, candidate_weights,
                                    required=required_notes, excluded=excluded_notes)
        fitness = eval_scale_cached(scale)
        population.append((fitness, scale))
    
    best_ever = max(population, key=lambda x: x[0])
    
    # Temperature schedule for simulated annealing
    temp_ratio = (config.final_temp / config.initial_temp) ** (1.0 / max(1, config.generations))
    temperature = config.initial_temp
    
    # Track for stagnation and early stopping
    stagnation_counter = 0
    last_reheat = 0  # Generation of last reheat
    
    for gen in range(config.generations):
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Track best ever using RAW fitness (re-evaluate top candidate to get true quality)
        top_scale = population[0][1]
        top_raw_fitness = eval_scale_cached(top_scale)  # This is raw, from cache
        if top_raw_fitness > best_ever[0]:
            best_ever = (top_raw_fitness, top_scale)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Reheat: when stagnating, bump temperature back up to explore more
        # Only reheat if: stagnating, enough time since last reheat, and temp is actually low
        reheat_target = config.initial_temp * 0.5
        if (stagnation_counter >= 15 and 
            (gen - last_reheat) > 30 and 
            temperature < reheat_target * 0.6):  # Only reheat if temp dropped below 60% of target
            temperature = reheat_target
            last_reheat = gen
            if verbose:
                print(f"Gen {gen:4d}: reheating (stagnation={stagnation_counter}), temp -> {temperature:.4f}")
        
        # Early stopping if converged (but give more time)
        if stagnation_counter > 30:
            if verbose:
                best_quality = best_ever[0]
                num_intervals = sum(1 for k in interval_cache if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int))
                print(f"Gen {gen:4d}: converged (no improvement for 30 gens), best quality = {best_quality:.4f}, intervals = {num_intervals}")
            break
        
        # Adaptive mutation rate (increase when stagnating)
        adaptive_mutation = config.mutation_rate
        if stagnation_counter > 5:
            adaptive_mutation = min(0.9, config.mutation_rate * (1 + stagnation_counter / 10))
        
        if verbose and gen % 10 == 0:
            # Print quality (higher = better)
            # Show raw quality (not boosted) for current top
            current_quality = eval_scale_cached(population[0][1])  # Raw from cache
            best_quality = best_ever[0]  # Already raw
            unique = len(set(frozenset((r.numerator, r.denominator) for r in ind[1]) for ind in population[:50]))
            
            # Compute fingerprint diversity (average pairwise distance in top 20)
            fp_sample = [get_fingerprint(ind[1]) for ind in population[:20]]
            fp_sample = [fp for fp in fp_sample if fp]
            if len(fp_sample) >= 2:
                fp_dists = []
                for i in range(len(fp_sample)):
                    for j in range(i+1, len(fp_sample)):
                        fp_dists.append(fingerprint_distance(fp_sample[i], fp_sample[j]))
                fp_div = sum(fp_dists) / len(fp_dists) if fp_dists else 0
            else:
                fp_div = 0
            
            if abs(current_quality - best_quality) < 0.0001:
                print(f"Gen {gen:4d}: quality = {current_quality:.4f}, temp = {temperature:.4f}, unique = {unique}/50, fp_div = {fp_div:.3f}")
            else:
                print(f"Gen {gen:4d}: quality = {current_quality:.4f} (best = {best_quality:.4f}), temp = {temperature:.4f}, unique = {unique}/50, fp_div = {fp_div:.3f}")
        
        # Create new population
        new_population = []
        
        # Always preserve best_ever (critical: don't lose the best solution found)
        new_population.append(best_ever)
        
        # Inject random individuals when stagnating (helps escape local optima)
        num_random_inject = 0
        if stagnation_counter > 10:
            num_random_inject = min(config.population_size // 5, stagnation_counter)  # Up to 20% random
            for _ in range(num_random_inject):
                if config.fixed_notes:
                    num_ratios = config.fixed_notes - 1
                else:
                    num_ratios = random.randint(config.min_notes - 1, config.max_notes - 1)
                fresh = create_random_scale(candidates, num_ratios, candidate_weights,
                                           required=required_notes, excluded=excluded_notes)
                fitness = eval_scale_cached(fresh)
                new_population.append((fitness, fresh))
        
        # Elite selection using fingerprint-based diversity
        elites = select_elites_diverse(population, config.elite_count + 3, get_fingerprint)
        for elite in elites[:config.elite_count + 1]:  # One less since we added best_ever
            if elite[1] != best_ever[1]:  # Don't duplicate best_ever
                new_population.append(elite)
        
        # Get population fingerprints for diversity scoring
        pop_fingerprints = [get_fingerprint(ind[1]) for ind in population[:50]]  # Top 50
        pop_fingerprints = [fp for fp in pop_fingerprints if fp]  # Filter empty
        
        # Breed rest
        while len(new_population) < config.population_size:
            parent1 = tournament_select(population, config.tournament_size)
            parent2 = tournament_select(population, config.tournament_size)
            
            if random.random() < 0.7:
                child = crossover(parent1, parent2, config)
            else:
                child = parent1.copy()
            
            if random.random() < adaptive_mutation:
                child = mutate(child, candidates, config)
            
            # Sanitize: ensure all ratios are within octave
            child = sanitize_scale(child)
            if not child:
                continue  # Skip if sanitization produced empty scale
            
            # Occasional drastic mutations: invert or rotate entire scale
            # These help escape local optima by exploring different modes/inversions
            # Only use if result has same length (or valid length for variable scales)
            target_len = len(child)
            
            if random.random() < 0.03:  # 3% chance to invert
                inverted = scale_invert(child, config.ratio_limit)
                if (inverted and len(inverted) == target_len and check_constraints(inverted)):
                    child = inverted
            elif random.random() < 0.05:  # ~5% chance to rotate
                rotated = scale_rotate(child, limit=config.ratio_limit)
                if (rotated and len(rotated) == target_len and check_constraints(rotated)):
                    child = rotated
            
            raw_fitness = eval_scale_cached(child)
            selection_fitness = raw_fitness
            
            # Add small diversity bonus based on metric-space distance from population
            # This encourages exploration of different regions of the quality landscape
            # Note: bonus only affects selection, not best_ever tracking
            if pop_fingerprints and stagnation_counter > 5:
                child_fp = get_fingerprint(child)
                if child_fp:
                    diversity = fingerprint_diversity_score(child_fp, pop_fingerprints, k=5)
                    # Small bonus (up to ~0.02) for diverse scales
                    diversity_bonus = diversity * 0.02 * min(1.0, stagnation_counter / 20)
                    selection_fitness = raw_fitness + diversity_bonus
            
            # Simulated annealing acceptance (uses boosted fitness for selection)
            if new_population and selection_fitness < new_population[-1][0]:
                delta = new_population[-1][0] - selection_fitness
                accept_prob = math.exp(-delta / temperature) if temperature > 0 else 0
                if random.random() > accept_prob:
                    if random.random() < 0.3:
                        if config.fixed_notes:
                            num_ratios = config.fixed_notes - 1
                        else:
                            num_ratios = random.randint(config.min_notes - 1, config.max_notes - 1)
                        child = create_random_scale(candidates, num_ratios, candidate_weights,
                                                    required=required_notes, excluded=excluded_notes)
                        raw_fitness = eval_scale_cached(child)
                        selection_fitness = raw_fitness
            
            # Track best_ever using RAW fitness (no diversity bonus)
            if raw_fitness > best_ever[0]:
                best_ever = (raw_fitness, child)
                stagnation_counter = 0
            
            # Store with selection_fitness for population dynamics
            new_population.append((selection_fitness, child))
        
        # Periodic local search on top individuals (every 25 generations when stagnating)
        if stagnation_counter > 10 and gen % 25 == 0:
            new_population.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(3, len(new_population))):
                improved_scale, improved_fitness = local_search(
                    new_population[i][1], candidates, eval_scale_cached, config, 
                    max_iterations=5, candidate_cents=candidate_cents)
                if improved_fitness > new_population[i][0]:
                    new_population[i] = (improved_fitness, improved_scale)
                    if improved_fitness > best_ever[0]:
                        best_ever = (improved_fitness, improved_scale)
                        stagnation_counter = 0
        
        population = new_population
        temperature *= temp_ratio
    
    # Final local search on best solution
    if verbose:
        print("Polishing best solution with local search...")
    best_scale, best_fitness = local_search(
        best_ever[1], candidates, eval_scale_cached, config, 
        max_iterations=20, candidate_cents=candidate_cents)
    if best_fitness > best_ever[0]:
        best_ever = (best_fitness, best_scale)
    
    # Return best found
    best_fitness, best_scale = best_ever
    _, details = evaluate_scale(best_scale, metrics, octave_range, interval_cache, scale_metrics, edo, combine_method)
    return sorted(best_scale), best_fitness, details


