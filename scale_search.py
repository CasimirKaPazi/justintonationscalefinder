#!/usr/bin/env python3
"""
Evolutionary search for consonant just intonation scales.
Uses Tenney height (log2(n*d) for ratio n/d) as consonance metric.
"""

import random
import math
import bisect
from fractions import Fraction
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
import argparse


@dataclass
class SearchConfig:
    """
    Configuration for scale search.
    
    Note: 'notes' counts total scale degrees INCLUDING the octave.
    So a 7-note scale has 6 ratios between 1/1 and 2/1, plus the octave.
    
    Defaults:
        max_notes: 7 (heptatonic)
        min_notes: 3 (minimum viable scale)
        ratio_limit: 32 (allows up to 31-limit intervals)
        population_size: 200
        generations: 500
        mutation_rate: 0.3
        tournament_size: 5
        elite_count: 2
        initial_temp: 0.1
        final_temp: 0.001
        octave_range: 1 (evaluate across 3 octaves)
    """
    max_notes: int = 7          # Maximum notes in scale (including octave)
    min_notes: int = 3          # Minimum notes in scale (including octave)
    fixed_notes: int = None     # If set, use exactly this many notes (including octave)
    ratio_limit: int = 32       # Max numerator/denominator (32 allows up to 31-limit)
    population_size: int = 200
    generations: int = 500
    mutation_rate: float = 0.3
    tournament_size: int = 5
    elite_count: int = 2
    metrics: List[Tuple[str, float]] = None  # Interval metrics: (metric_name, weight) pairs
    scale_metrics: List[Tuple[str, float]] = None  # Scale-level metrics: steps, primes, edo
    # Simulated annealing parameters
    initial_temp: float = 0.1   # Starting temperature (higher = more exploration)
    final_temp: float = 0.001   # Ending temperature
    # Deprecated - no longer used (diversity maintained via randomized elite selection)
    diversity_weight: float = 0.1
    # Multi-octave evaluation
    octave_range: int = 1       # Octaves above/below to consider (0=single, 1=3 total, 2=5 total)
    # EDO approximation (for neji scales)
    edo: int = 0                # If > 0, penalize notes far from this EDO's steps
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [('tenney', 1.0)]
        if self.scale_metrics is None:
            self.scale_metrics = []
        # Swap min/max if backwards
        if self.min_notes > self.max_notes:
            self.min_notes, self.max_notes = self.max_notes, self.min_notes


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def prime_factors(n: int) -> List[Tuple[int, int]]:
    """Return prime factorization as list of (prime, exponent) pairs."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors.append((d, exp))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors


# =============================================================================
# METRIC CACHING
# =============================================================================

# Cache for metric calculations - maps (n, d, metric_name) -> score
_metric_cache: dict = {}

def clear_metric_cache():
    """Clear the metric cache (useful if memory is a concern)."""
    global _metric_cache
    _metric_cache = {}

def cached_metric(ratio: Fraction, metric_name: str) -> float:
    """Get cached metric value, computing if necessary."""
    key = (ratio.numerator, ratio.denominator, metric_name)
    if key not in _metric_cache:
        metric_fn, _ = CONSONANCE_METRICS[metric_name]
        _metric_cache[key] = metric_fn(ratio)
    return _metric_cache[key]


# =============================================================================
# CONSONANCE METRICS
# =============================================================================

def tenney_height(ratio: Fraction) -> float:
    """
    Tenney height: log2(n * d) for ratio n/d. 
    Lower = more consonant.
    Simple and captures "small numbers = consonant" intuition.
    """
    return math.log2(ratio.numerator * ratio.denominator)


def euler_gradus(ratio: Fraction) -> float:
    """
    Euler's Gradus Suavitatis (degree of sweetness).
    Γ = 1 + Σ(p - 1) * e for prime factorization Πp^e
    Applied to n*d of ratio n/d.
    Lower = more consonant. Favors powers of 2 and 3.
    """
    n = ratio.numerator * ratio.denominator
    factors = prime_factors(n)
    if not factors:
        return 1
    return 1 + sum((p - 1) * e for p, e in factors)


# Precomputed simple ratios for harmonic entropy (computed once at module load)
_ENTROPY_SIMPLE_RATIOS: List[Tuple[float, int]] = []

def _init_entropy_ratios():
    """Initialize the simple ratios list for harmonic entropy."""
    global _ENTROPY_SIMPLE_RATIOS
    max_complexity = 100
    for d in range(1, 16):
        for n in range(d, 2 * d + 1):
            if gcd(n, d) == 1:
                complexity = n * d
                if complexity <= max_complexity:
                    _ENTROPY_SIMPLE_RATIOS.append((math.log2(n / d), complexity))

_init_entropy_ratios()


def harmonic_entropy(ratio: Fraction, spread: float = 0.017) -> float:
    """
    Harmonic entropy based on Paul Erlich's model.
    Measures how "ambiguous" an interval is - intervals near multiple simple ratios
    have higher entropy than those clearly identified with one simple ratio.
    
    The spread parameter (in octaves) controls the uncertainty window.
    0.017 octaves ≈ 20 cents, which is roughly a just-noticeable difference.
    
    Lower = more consonant (less ambiguous).
    """
    target = math.log2(float(ratio))
    spread_sq_2 = 2 * spread * spread  # Precompute
    
    weights = []
    for ratio_log, complexity in _ENTROPY_SIMPLE_RATIOS:
        distance = target - ratio_log
        # Gaussian weight divided by complexity
        weight = math.exp(-(distance * distance) / spread_sq_2) / complexity
        if weight > 1e-12:
            weights.append(weight)
    
    total = sum(weights)
    if total < 1e-12:
        return 6.0
    
    # Shannon entropy
    entropy = 0.0
    for w in weights:
        p = w / total
        if p > 1e-12:
            entropy -= p * math.log2(p)
    
    return entropy


def plomp_levelt_roughness(ratio: Fraction, base_freq: float = 260.0) -> float:
    """
    Roughness based on Plomp-Levelt critical bandwidth model.
    Models psychoacoustic beating/roughness with harmonic timbres.
    Lower = more consonant.
    """
    f1 = base_freq
    f2 = base_freq * float(ratio)
    
    def critical_bandwidth(f):
        return 25 + 75 * (1 + 1.4 * (f / 1000) ** 2) ** 0.69
    
    def pure_tone_roughness(fa, fb):
        if fa > fb:
            fa, fb = fb, fa
        cb = critical_bandwidth((fa + fb) / 2)
        x = (fb - fa) / cb
        if x < 0.001 or x > 1.2:
            return 0.0
        return x * math.exp(1 - x / 0.25) * 4
    
    # First 6 harmonics
    total_roughness = 0.0
    for h1 in range(1, 7):
        for h2 in range(1, 7):
            amplitude = 1.0 / (h1 * h2)
            total_roughness += amplitude * pure_tone_roughness(f1 * h1, f2 * h2)
    
    return total_roughness


def wilson_complexity(ratio: Fraction) -> float:
    """
    Erv Wilson-inspired harmonic complexity.
    Taxicab distance in harmonic lattice, ignoring octaves.
    Sum of (prime - 1) * |exponent| for odd primes.
    Lower = more consonant.
    """
    n_factors = prime_factors(ratio.numerator)
    d_factors = prime_factors(ratio.denominator)
    
    exponents = {}
    for p, e in n_factors:
        exponents[p] = exponents.get(p, 0) + e
    for p, e in d_factors:
        exponents[p] = exponents.get(p, 0) - e
    
    complexity = 0
    for p, e in exponents.items():
        if p > 2:
            complexity += (p - 1) * abs(e)
    
    return complexity


def step_size_penalty(scale: List[Fraction], threshold_cents: float = 80.0, 
                      decay_cents: float = 25.0) -> float:
    """
    Penalty for notes that are too close together.
    Uses exponential decay: identical notes are heavily penalized,
    notes 80+ cents apart get no penalty.
    
    The decay parameter controls how quickly penalty drops off.
    With threshold=80 and decay=25:
      - 0 cents (unison): penalty ≈ 1.0
      - 30 cents: penalty ≈ 0.3
      - 50 cents: penalty ≈ 0.14
      - 80 cents: penalty ≈ 0.04
    
    Returns average penalty across all adjacent steps.
    Lower = better (more distinct steps).
    """
    if len(scale) < 2:
        return 0.0
    
    full_scale = sorted([Fraction(1, 1)] + list(scale) + [Fraction(2, 1)])
    
    penalties = []
    for i in range(len(full_scale) - 1):
        step_ratio = full_scale[i + 1] / full_scale[i]
        step_cents = 1200 * math.log2(float(step_ratio))
        
        if step_cents < threshold_cents:
            # Exponential penalty that increases sharply as step gets smaller
            penalty = math.exp(-(step_cents / decay_cents))
        else:
            penalty = 0.0
        penalties.append(penalty)
    
    return sum(penalties) / len(penalties) if penalties else 0.0


def get_primes_used(scale: List[Fraction]) -> Set[int]:
    """Get the set of odd primes used in all ratios of a scale."""
    primes = set()
    for ratio in scale:
        for n in [ratio.numerator, ratio.denominator]:
            for p, _ in prime_factors(n):
                if p > 2:
                    primes.add(p)
    return primes


def prime_diversity_score(scale: List[Fraction]) -> float:
    """
    Measures harmonic color diversity based on prime factors used.
    
    Returns sum of 1/p for each distinct prime p > 5.
    
    This naturally weights lower primes more heavily:
    - Having 7 contributes 1/7 ≈ 0.143
    - Having 11 contributes 1/11 ≈ 0.091  
    - Having 13 contributes 1/13 ≈ 0.077
    - Having 17 contributes 1/17 ≈ 0.059
    - Having 19 contributes 1/19 ≈ 0.053
    
    Example scores:
    - 5-limit only: 0.0
    - 7-limit: 0.143
    - 7 + 11 limit: 0.234
    - 7 + 11 + 13 limit: 0.311
    - 7 + 11 + 13 + 17 + 19: 0.423
    """
    primes = get_primes_used(scale)
    
    return sum(1.0 / p for p in primes if p > 5)


def edo_penalty(scale: List[Fraction], edo: int = 12) -> float:
    """
    Penalty for notes falling near EDO midpoints (between steps).
    Used for generating "neji" (near-equal just intonation) scales.
    
    For N-EDO, step size is 1200/N cents.
    Notes at EDO steps get 0 penalty.
    Notes at midpoints (halfway between steps) get maximum penalty of 1.
    
    Uses cosine curve: penalty = (1 - cos(2π * distance / step_size)) / 2
    This gives smooth transitions with 0 at steps and 1 at midpoints.
    
    Returns average penalty across all scale notes (0 = perfect EDO match).
    """
    if edo <= 0:
        return 0.0
    
    step_size = 1200.0 / edo  # cents per EDO step
    
    penalties = []
    for note in scale:
        cents = 1200 * math.log2(float(note))
        
        # Distance to nearest EDO step
        step_number = cents / step_size
        distance = abs(step_number - round(step_number)) * step_size
        
        # Cosine penalty: 0 at EDO steps, 1 at midpoints
        # distance ranges from 0 to step_size/2
        # normalize to 0-1 range where 1 is at midpoint
        normalized_distance = distance / (step_size / 2)
        penalty = (1 - math.cos(math.pi * normalized_distance)) / 2
        
        penalties.append(penalty)
    
    return sum(penalties) / len(penalties) if penalties else 0.0


def edo_deviation(scale: List[Fraction], edo: int = 12) -> float:
    """
    Average deviation from nearest EDO step in cents.
    Useful for reporting how close a scale is to an EDO.
    """
    if edo <= 0:
        return 0.0
    
    step_size = 1200.0 / edo
    
    deviations = []
    for note in scale:
        cents = 1200 * math.log2(float(note))
        step_number = cents / step_size
        deviation = abs(step_number - round(step_number)) * step_size
        deviations.append(deviation)
    
    return sum(deviations) / len(deviations) if deviations else 0.0


def prime_limit_penalty(ratio: Fraction) -> float:
    """
    Alternative consonance metric that doesn't penalize higher primes as harshly.
    Uses sqrt of Tenney height, which flattens the preference curve.
    
    This allows 7/4 (Tenney=4.8) to be closer in score to 3/2 (Tenney=2.6)
    than raw Tenney would suggest.
    """
    return math.sqrt(tenney_height(ratio))


CONSONANCE_METRICS = {
    'tenney': (tenney_height, "Tenney height: log2(n*d)"),
    'euler': (euler_gradus, "Euler's Gradus Suavitatis"),
    'entropy': (harmonic_entropy, "Harmonic entropy (Erlich-style)"),
    'roughness': (plomp_levelt_roughness, "Plomp-Levelt roughness model"),
    'wilson': (wilson_complexity, "Wilson harmonic complexity"),
    'soft_tenney': (prime_limit_penalty, "Softened Tenney (sqrt) - less harsh on higher primes"),
}

# Scale-level metrics (operate on whole scale, not individual intervals)
# Note: 'edo' is handled specially - use 'edo12', 'edo19', etc. to specify which EDO
SCALE_METRICS = {
    'steps': (step_size_penalty, "Step size penalty (penalizes clustered notes)"),
    'primes': (prime_diversity_score, "Prime diversity (penalizes lack of 7/11/13-limit)"),
    # 'edoN' handled specially in parse_metrics - generates neji scales
}

# Approximate ranges for normalization (based on typical intervals)
# These help combine metrics that have very different scales
METRIC_NORMALIZATION = {
    'tenney': (1.0, 12.0),      # log2(2) to log2(64*64)
    'euler': (1.0, 20.0),       # simple to complex
    'entropy': (0.5, 4.0),      # low to high entropy
    'roughness': (0.0, 1.5),    # smooth to rough
    'wilson': (0.0, 12.0),      # simple to complex
    'soft_tenney': (1.0, 3.5),  # sqrt of tenney range
    'steps': (0.0, 0.5),        # 0 = no close steps, 0.5 = very clustered
    'primes': (0.0, 0.5),       # sum(1/p): 0=5-limit, 0.31=13-limit, 0.42=19-limit
    'edo': (0.0, 1.0),          # 0 = on EDO steps, 1 = at midpoints
}


import re

def parse_metrics(metric_str: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], int]:
    """
    Parse metric specification string.
    Format: "metric1:weight1,metric2:weight2" or just "metric1"
    Example: "tenney:1,roughness:0.5,steps:0.3" or "tenney"
    
    Special: "edo12:1.0" or "neji19:0.5" for EDO approximation (neji scales)
    
    Returns: (interval_metrics, scale_metrics, edo)
    - interval_metrics: applied to each interval pair
    - scale_metrics: applied to the whole scale (steps, primes, edo)
    - edo: the EDO number if specified, 0 otherwise
    """
    interval_metrics = []
    scale_metrics = []
    edo = 0
    
    for part in metric_str.split(','):
        part = part.strip()
        if ':' in part:
            name, weight = part.split(':', 1)
            name = name.strip()
            weight = float(weight)
        else:
            name = part
            weight = 1.0
        
        # Check for edo/neji pattern (e.g., edo12, neji19)
        edo_match = re.match(r'^(edo|neji)(\d+)$', name.lower())
        if edo_match:
            edo = int(edo_match.group(2))
            scale_metrics.append(('edo', weight))
        elif name in CONSONANCE_METRICS:
            interval_metrics.append((name, weight))
        elif name in SCALE_METRICS:
            scale_metrics.append((name, weight))
        else:
            all_metrics = list(CONSONANCE_METRICS.keys()) + list(SCALE_METRICS.keys()) + ['edoN/nejiN']
            raise ValueError(f"Unknown metric: {name}. Available: {all_metrics}")
    
    # Default to tenney if no interval metrics specified
    if not interval_metrics:
        interval_metrics = [('tenney', 1.0)]
    
    return interval_metrics, scale_metrics, edo


def combined_metric(ratio: Fraction, metrics: List[Tuple[str, float]], normalize: bool = True) -> float:
    """
    Compute weighted combination of multiple metrics.
    If normalize=True, scales each metric to roughly 0-1 range before combining.
    Uses caching for repeated ratio evaluations.
    """
    total = 0.0
    total_weight = 0.0
    
    for metric_name, weight in metrics:
        score = cached_metric(ratio, metric_name)
        
        if normalize:
            min_val, max_val = METRIC_NORMALIZATION[metric_name]
            score = (score - min_val) / (max_val - min_val)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        total += weight * score
        total_weight += weight
    
    return total / total_weight if total_weight > 0 else 0.0


def generate_candidate_ratios(limit: int) -> Tuple[List[Fraction], List[float]]:
    """
    Generate all ratios n/d where 1 < n/d < 2 and n,d <= limit, in lowest terms.
    Returns (ratios, weights) where weights are Zipf-biased toward simpler ratios.
    Weight = 1 / (n * d) gives strong preference to low-complexity ratios.
    """
    ratios = set()
    for d in range(1, limit + 1):
        for n in range(d + 1, 2 * d):  # ratios between 1 and 2
            if n <= limit:
                g = gcd(n, d)
                ratios.add(Fraction(n // g, d // g))
    
    sorted_ratios = sorted(ratios)
    # Zipf-like weights: 1/(n*d) strongly favors simpler ratios
    weights = [1.0 / (r.numerator * r.denominator) for r in sorted_ratios]
    
    return sorted_ratios, weights


def interval_between(r1: Fraction, r2: Fraction) -> Fraction:
    """Get the interval between two ratios (as a ratio within one octave)."""
    if r1 > r2:
        r1, r2 = r2, r1
    interval = r2 / r1
    # Reduce to within octave if needed
    while interval >= 2:
        interval = interval / 2
    while interval < 1:
        interval = interval * 2
    return interval


def interval_key(r1: Fraction, r2: Fraction) -> Tuple[int, int]:
    """Get a hashable key for the interval between two ratios, reduced to one octave."""
    if (r1.numerator * r2.denominator) > (r2.numerator * r1.denominator):
        r1, r2 = r2, r1
    # Compute r2/r1 = (r2.num * r1.den) / (r2.den * r1.num)
    n = r2.numerator * r1.denominator
    d = r2.denominator * r1.numerator
    g = gcd(n, d)
    n, d = n // g, d // g
    # Reduce to within octave
    while n >= 2 * d:
        n, d = n, d * 2
        g = gcd(n, d)
        n, d = n // g, d // g
    while n < d:
        n, d = n * 2, d
        g = gcd(n, d)
        n, d = n // g, d // g
    return (n, d)


def evaluate_scale(scale: List[Fraction], 
                   metrics: List[Tuple[str, float]] = None,
                   octave_range: int = 1,
                   interval_cache: dict = None,
                   scale_metrics: List[Tuple[str, float]] = None,
                   edo: int = 0) -> Tuple[float, dict]:
    """
    Evaluate a scale's quality using the specified metric(s).
    Returns (fitness, details) where higher fitness = better scale.
    
    metrics: List of (metric_name, weight) tuples for interval evaluation
    scale_metrics: List of (metric_name, weight) tuples for whole-scale properties
    octave_range: How many octaves above/below to consider
    interval_cache: Optional precomputed {(n,d): score} lookup for fast evaluation
    edo: If > 0, penalize notes far from this EDO's steps (for neji scales)
    
    All metrics are converted to quality scores (0-1, higher = better):
    - Interval metrics (tenney, etc.): quality = 1 - normalized_dissonance
    - primes: quality = normalized_diversity (already positive)
    - steps: quality = 1 - normalized_clustering
    - edo: quality = 1 - normalized_deviation
    
    Combination uses weighted geometric mean:
    - All metrics must be good for a high score
    - Can't compensate a bad metric with a good one
    - Weights act as exponents: quality_i ^ weight_i
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
            g = gcd(n, d)
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
            g = gcd(int_n, int_d)
            int_n, int_d = int_n // g, int_d // g
            
            # Reduce to single octave
            while int_n >= 2 * int_d:
                int_d *= 2
                g = gcd(int_n, int_d)
                int_n, int_d = int_n // g, int_d // g
            while int_n < int_d:
                int_n *= 2
                g = gcd(int_n, int_d)
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
    
    # Base score from intervals (this is a penalty: 0-1, lower = better)
    raw_interval_penalty = sum(interval_scores) / len(interval_scores) if interval_scores else 0
    
    # Normalize for note count: more notes naturally have higher average scores
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
    scale_metric_scores['note_count_factor'] = note_count_factor
    
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
        min_val, max_val = METRIC_NORMALIZATION[metric_name]
        norm_score = (raw_score - min_val) / (max_val - min_val)
        norm_score = max(0.0, min(1.0, norm_score))
        
        # Convert to quality (higher = better)
        # 'primes' is already positive (more diversity = better)
        # 'steps' and 'edo' are penalties (lower = better), so invert
        if metric_name == 'primes':
            quality = norm_score  # Already: higher diversity = higher quality
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
    
    # Fitness = quality (higher = better, range roughly 0-1)
    fitness = final_quality
    
    metric_avgs = {name: sum(scores) / len(scores) if scores else 0 
                   for name, scores in individual_scores.items()}
    metric_avgs.update(scale_metric_scores)
    
    details = {
        'intervals': [],
        'avg_score': 1.0 - final_quality,  # For backward compat: lower = better
        'quality': final_quality,  # New: higher = better
        'interval_quality': interval_quality,
        'interval_penalty': interval_penalty,
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


def create_random_scale(candidates: List[Fraction], num_ratios: int, 
                        weights: List[float] = None) -> List[Fraction]:
    """
    Create a random scale with the given number of ratios.
    If weights provided, uses weighted sampling (biased toward simpler ratios).
    """
    if weights is not None and len(weights) == len(candidates):
        # Weighted sampling without replacement
        selected = []
        available = list(range(len(candidates)))
        available_weights = list(weights)
        
        for _ in range(min(num_ratios, len(candidates))):
            if not available:
                break
            # Normalize weights
            total = sum(available_weights)
            if total <= 0:
                idx_in_available = random.randrange(len(available))
            else:
                r = random.random() * total
                cumsum = 0
                idx_in_available = 0
                for i, w in enumerate(available_weights):
                    cumsum += w
                    if r <= cumsum:
                        idx_in_available = i
                        break
            
            selected.append(candidates[available[idx_in_available]])
            available.pop(idx_in_available)
            available_weights.pop(idx_in_available)
        
        return selected
    else:
        return random.sample(candidates, min(num_ratios, len(candidates)))


def load_seed_scales(filepath: str) -> List[List[Fraction]]:
    """
    Load seed scales from a file. Supports both simple format and Scala .scl format.
    
    Comment styles:
    - Lines starting with # are comments
    - Lines starting with ! are comments (Scala format)
    - Blank lines separate scales (in simple format)
    
    Ratio formats:
    - n/d (e.g., 9/8, 5/4, 3/2)
    - n (e.g., 2 means 2/1)
    
    Notes:
    - 2/1 (or just 2) at the end of a scale is removed (it's implicit)
    - Ratios outside the octave (like 3/1, 9/4) ARE included for multi-octave scales
    - Cents values (containing '.') are skipped
    
    Example simple format:
        # My favorite scale
        9/8
        5/4
        4/3
        3/2
        5/3
        15/8
        2/1
        
        # Another scale
        8/7
        6/5
        4/3
        3/2
    
    Example Scala .scl format:
        ! scale_name.scl
        Description of the scale
        12
        !
        10/9
        8/7
        6/5
        5/4
        4/3
        3/2
        25/16
        8/5
        5/3
        7/4
        9/5
        2
    
    Returns a list of scales, where each scale is a list of Fractions.
    """
    scales = []
    current_scale = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments (# or !) and blank lines
                if line.startswith('#') or line.startswith('!') or not line:
                    # Blank line or comment - if we have a current scale, save it
                    if current_scale and not line.startswith('!'):
                        # Only save on blank lines or # comments, not ! comments
                        # (Scala format uses ! for inline comments within a scale)
                        if not line or line.startswith('#'):
                            # Remove trailing 2/1 if present (it's implicit)
                            if current_scale[-1] == Fraction(2, 1):
                                current_scale = current_scale[:-1]
                            if current_scale:
                                scales.append(current_scale)
                            current_scale = []
                    continue
                
                # Skip lines that look like Scala metadata (note count, description)
                # These are typically just numbers (note count) or text without /
                # But we need to catch "2" as a ratio, so check if it could be a ratio
                
                # Skip cents values (contain decimal point)
                if '.' in line:
                    continue
                
                # Try to parse as ratio
                try:
                    if '/' in line:
                        # Fraction format: n/d
                        parts = line.split('/')
                        if len(parts) == 2:
                            n = int(parts[0].strip())
                            d = int(parts[1].strip())
                            ratio = Fraction(n, d)
                            if ratio > 1:
                                current_scale.append(ratio)
                    else:
                        # Could be integer like "2" meaning 2/1
                        # But also could be Scala note count line
                        # Heuristic: if we already have notes, or it's a small int that makes sense as ratio
                        val = int(line)
                        if val >= 2 and (current_scale or val <= 4):
                            # Likely a ratio like 2, 3, 4 (meaning 2/1, 3/1, 4/1)
                            ratio = Fraction(val, 1)
                            current_scale.append(ratio)
                        # Otherwise skip (likely note count metadata)
                except (ValueError, ZeroDivisionError):
                    # Skip malformed lines (likely description text)
                    pass
            
            # Don't forget the last scale
            if current_scale:
                # Remove trailing 2/1 if present
                if current_scale[-1] == Fraction(2, 1):
                    current_scale = current_scale[:-1]
                if current_scale:
                    scales.append(current_scale)
    
    except FileNotFoundError:
        print(f"Warning: Seed file '{filepath}' not found.")
        return []
    except Exception as e:
        print(f"Warning: Error reading seed file '{filepath}': {e}")
        return []
    
    return scales


def reduce_to_octave(ratio: Fraction) -> Fraction:
    """Reduce a ratio to within one octave (1 <= r < 2)."""
    while ratio >= 2:
        ratio = ratio / 2
    while ratio < 1:
        ratio = ratio * 2
    return ratio


def is_valid_ratio(ratio: Fraction, limit: int) -> bool:
    """Check if ratio is within the complexity limit."""
    return ratio.numerator <= limit and ratio.denominator <= limit


# Simple intervals for transposition mutations
SIMPLE_INTERVALS = [
    Fraction(3, 2),   # Perfect fifth
    Fraction(4, 3),   # Perfect fourth
    Fraction(5, 4),   # Major third
    Fraction(6, 5),   # Minor third
    Fraction(5, 3),   # Major sixth
    Fraction(8, 5),   # Minor sixth
    Fraction(7, 4),   # Harmonic seventh
    Fraction(7, 6),   # Septimal minor third
    Fraction(9, 8),   # Major second
    Fraction(10, 9),  # Minor whole tone
    Fraction(8, 7),   # Septimal major second
    Fraction(11, 8),  # Undecimal tritone
    Fraction(11, 9),  # Undecimal neutral third
    Fraction(13, 8),  # Tridecimal neutral sixth
]

# Musical commas for subtle pitch shifts
COMMAS = [
    Fraction(81, 80),   # Syntonic comma (21.5 cents) - 5-limit
    Fraction(64, 63),   # Septimal comma (27.3 cents) - 7-limit
    Fraction(33, 32),   # Undecimal comma (53.3 cents) - 11-limit
    Fraction(128, 125), # Diesis (41.1 cents) - 5-limit
    Fraction(36, 35),   # Septimal diesis (48.8 cents) - 7-limit
    Fraction(49, 48),   # Septimal sixth-tone (35.7 cents) - 7-limit
    Fraction(225, 224), # Septimal kleisma (7.7 cents) - very small
    Fraction(126, 125), # Septimal semicomma (13.8 cents)
]


def mutate_transpose(note: Fraction, limit: int) -> Fraction:
    """Transpose a note by a simple interval (up or down)."""
    interval = random.choice(SIMPLE_INTERVALS)
    if random.random() < 0.5:
        new_note = note * interval
    else:
        new_note = note / interval
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate_comma_shift(note: Fraction, limit: int) -> Fraction:
    """Shift a note by a musical comma (subtle pitch adjustment)."""
    comma = random.choice(COMMAS)
    if random.random() < 0.5:
        new_note = note * comma
    else:
        new_note = note / comma
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate_combine(note1: Fraction, note2: Fraction, limit: int) -> Fraction:
    """Combine two notes by multiplication or division."""
    if random.random() < 0.5:
        new_note = note1 * note2
    else:
        new_note = note1 / note2
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate_stack(scale: List[Fraction], limit: int) -> Fraction:
    """
    La Monte Young style: stack intervals to create complex ratios.
    E.g., 7/4 × 7/4 × 3/2 × 3/2 × 3/2 = 1323/1024
    """
    # Choose 2-5 intervals to stack (from scale or simple intervals)
    num_stacks = random.randint(2, 5)
    pool = list(scale) + SIMPLE_INTERVALS[:8]
    
    result = Fraction(1, 1)
    for _ in range(num_stacks):
        interval = random.choice(pool)
        if random.random() < 0.8:  # Usually multiply
            result = result * interval
        else:
            result = result / interval
    
    result = reduce_to_octave(result)
    if is_valid_ratio(result, limit):
        return result
    return None


def mutate_mediant(note1: Fraction, note2: Fraction, limit: int) -> Fraction:
    """
    Create mediant of two ratios: (a+c)/(b+d) for a/b and c/d.
    This creates ratios "between" two existing ones in the Stern-Brocot tree.
    """
    a, b = note1.numerator, note1.denominator
    c, d = note2.numerator, note2.denominator
    new_note = Fraction(a + c, b + d)
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate(scale: List[Fraction], candidates: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Mutate a scale using musically-meaningful operations:
    1. Transpose a note by a simple interval (fifth, third, seventh, etc.)
    2. Comma shift (subtle pitch adjustment by syntonic/septimal comma)
    3. Combine two notes from the scale (like La Monte Young's 1323/1024)
    4. Stack intervals La Monte Young style (7/4 × 7/4 × 3/2 × 3/2...)
    5. Mediant - create ratio "between" two scale notes
    6. Random replacement from candidates (fallback)
    """
    scale = scale.copy()
    scale_set = set(scale)
    limit = config.ratio_limit
    
    def try_add_note(new_note, idx):
        """Helper: replace note at idx if new_note is valid."""
        if (new_note and 
            new_note not in scale_set and 
            new_note != Fraction(1,1) and 
            new_note != Fraction(2,1)):
            scale[idx] = new_note
            return True
        return False
    
    # Choose mutation type
    r = random.random()
    
    if r < 0.18 and scale:
        # Transpose by simple interval
        idx = random.randrange(len(scale))
        if try_add_note(mutate_transpose(scale[idx], limit), idx):
            return scale
    
    if r < 0.30 and scale:
        # Comma shift (subtle pitch adjustment)
        idx = random.randrange(len(scale))
        if try_add_note(mutate_comma_shift(scale[idx], limit), idx):
            return scale
    
    if r < 0.48 and len(scale) >= 2:
        # Combine two scale notes
        idx = random.randrange(len(scale))
        note1 = scale[idx]
        others = [n for n in scale if n != note1]
        note2 = random.choice(others) if others else note1
        if try_add_note(mutate_combine(note1, note2, limit), idx):
            return scale
    
    if r < 0.62 and scale:
        # La Monte Young style interval stacking
        idx = random.randrange(len(scale))
        if try_add_note(mutate_stack(scale, limit), idx):
            return scale
    
    if r < 0.74 and len(scale) >= 2:
        # Mediant of two scale notes
        idx = random.randrange(len(scale))
        note1 = scale[idx]
        others = [n for n in scale if n != note1]
        note2 = random.choice(others) if others else note1
        if try_add_note(mutate_mediant(note1, note2, limit), idx):
            return scale
    
    # Fallback: random replacement from candidates
    available = [r for r in candidates if r not in scale_set]
    if scale and available:
        idx = random.randrange(len(scale))
        scale[idx] = random.choice(available)
    
    return scale


def crossover_uniform(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Uniform crossover: combine both parents and randomly sample.
    Simple but doesn't preserve structure.
    """
    combined = list(set(parent1) | set(parent2))
    
    # Convert note count to ratio count: N notes = N-1 ratios
    if config.fixed_notes:
        num_ratios = config.fixed_notes - 1
        if len(combined) < num_ratios:
            return parent1.copy()
        return random.sample(combined, num_ratios)
    
    max_ratios = config.max_notes - 1
    min_ratios = config.min_notes - 1
    
    if len(combined) <= max_ratios:
        return combined
    
    num_ratios = random.randint(min_ratios, max_ratios)
    return random.sample(combined, min(num_ratios, len(combined)))


def crossover_blend(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Blend crossover: alternate taking notes from each parent (sorted by pitch).
    Preserves some structural properties from both parents.
    """
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Interleave from both parents
    child = []
    i, j = 0, 0
    use_p1 = random.random() < 0.5
    
    target_len = len(parent1) if config.fixed_notes else random.randint(
        config.min_notes - 1, config.max_notes - 1)
    
    seen = set()
    while len(child) < target_len and (i < len(p1_sorted) or j < len(p2_sorted)):
        if use_p1 and i < len(p1_sorted):
            note = p1_sorted[i]
            i += 1
        elif j < len(p2_sorted):
            note = p2_sorted[j]
            j += 1
        elif i < len(p1_sorted):
            note = p1_sorted[i]
            i += 1
        else:
            break
        
        if note not in seen:
            child.append(note)
            seen.add(note)
        use_p1 = not use_p1
    
    return child if child else parent1.copy()


def crossover_pivot(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Pivot crossover: choose a pivot pitch, take lower notes from one parent
    and higher notes from the other.
    """
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Choose pivot point (in cents)
    pivot = random.uniform(200, 1000)  # Somewhere in the middle of the octave
    
    # Take notes below pivot from parent1, above from parent2 (or vice versa)
    if random.random() < 0.5:
        p1_sorted, p2_sorted = p2_sorted, p1_sorted
    
    child = []
    seen = set()
    
    for note in p1_sorted:
        cents = 1200 * math.log2(float(note))
        if cents < pivot and note not in seen:
            child.append(note)
            seen.add(note)
    
    for note in p2_sorted:
        cents = 1200 * math.log2(float(note))
        if cents >= pivot and note not in seen:
            child.append(note)
            seen.add(note)
    
    # Adjust to target length
    target_len = (config.fixed_notes - 1) if config.fixed_notes else random.randint(
        config.min_notes - 1, config.max_notes - 1)
    
    if len(child) > target_len:
        child = random.sample(child, target_len)
    elif len(child) < target_len:
        # Fill from combined parents
        all_notes = list(set(parent1) | set(parent2))
        available = [n for n in all_notes if n not in seen]
        while len(child) < target_len and available:
            note = random.choice(available)
            child.append(note)
            available.remove(note)
    
    return child if child else parent1.copy()


def crossover_interval_preserving(parent1: List[Fraction], parent2: List[Fraction], 
                                   config: SearchConfig) -> List[Fraction]:
    """
    Interval-preserving crossover: try to preserve good interval relationships
    from both parents by keeping contiguous pitch regions.
    """
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Choose how many contiguous notes to take from each parent
    if len(p1_sorted) >= 2 and len(p2_sorted) >= 2:
        # Take a contiguous region from parent1
        start1 = random.randint(0, len(p1_sorted) - 1)
        end1 = random.randint(start1, len(p1_sorted))
        region1 = p1_sorted[start1:end1]
        
        # Take complementary region from parent2 (notes not overlapping in pitch)
        if region1:
            min_cents = 1200 * math.log2(float(min(region1)))
            max_cents = 1200 * math.log2(float(max(region1)))
            region2 = [n for n in p2_sorted 
                      if 1200 * math.log2(float(n)) < min_cents - 50 
                      or 1200 * math.log2(float(n)) > max_cents + 50]
        else:
            region2 = p2_sorted
        
        child = list(set(region1) | set(region2))
    else:
        child = list(set(parent1) | set(parent2))
    
    # Adjust to target length
    target_len = (config.fixed_notes - 1) if config.fixed_notes else random.randint(
        config.min_notes - 1, config.max_notes - 1)
    
    if len(child) > target_len:
        child = random.sample(child, target_len)
    elif len(child) < target_len:
        all_notes = list(set(parent1) | set(parent2))
        available = [n for n in all_notes if n not in child]
        while len(child) < target_len and available:
            note = random.choice(available)
            child.append(note)
            available.remove(note)
    
    return child if child else parent1.copy()


def crossover(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Combine two scales using a randomly chosen crossover operator.
    """
    r = random.random()
    if r < 0.3:
        return crossover_uniform(parent1, parent2, config)
    elif r < 0.5:
        return crossover_blend(parent1, parent2, config)
    elif r < 0.7:
        return crossover_pivot(parent1, parent2, config)
    else:
        return crossover_interval_preserving(parent1, parent2, config)


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
    """
    current = scale.copy()
    current_fitness = eval_fn(current)
    scale_set = set(current)
    
    for _ in range(max_iterations):
        improved = False
        
        # Try swapping each note with a similar one
        for i in range(len(current)):
            note = current[i]
            note_cents = 1200 * math.log2(float(note))
            
            # Find candidates within 100 cents using binary search if available
            if candidate_cents is not None:
                # Binary search for range [note_cents - 100, note_cents + 100]
                lo = bisect.bisect_left(candidate_cents, (note_cents - 100, -1))
                hi = bisect.bisect_right(candidate_cents, (note_cents + 100, float('inf')))
                nearby = [candidates[idx] for cents, idx in candidate_cents[lo:hi]
                         if candidates[idx] not in scale_set]
            else:
                # Fallback to linear scan
                nearby = [c for c in candidates 
                         if c not in scale_set 
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
            # Try a single mutation
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
    metrics = config.metrics
    scale_metrics = config.scale_metrics
    octave_range = config.octave_range
    edo = config.edo
    
    # Pre-compute cents for all candidates (for fast neighbor lookups)
    # Store as list of (cents, index) sorted by cents
    candidate_cents = [(1200 * math.log2(float(c)), i) for i, c in enumerate(candidates)]
    candidate_cents.sort()  # Sort by cents value
    
    # Build metric description string
    if len(metrics) == 1:
        _, metric_desc = CONSONANCE_METRICS[metrics[0][0]]
    else:
        parts = [f"{name}:{weight}" for name, weight in metrics]
        metric_desc = "Combined: " + ", ".join(parts)
    
    if scale_metrics:
        scale_parts = []
        for name, weight in scale_metrics:
            if name == 'edo':
                scale_parts.append(f"edo{edo}:{weight}")
            else:
                scale_parts.append(f"{name}:{weight}")
        metric_desc += " + scale[" + ", ".join(scale_parts) + "]"
    
    note_desc = f"exactly {config.fixed_notes}" if config.fixed_notes else f"up to {config.max_notes}"
    octave_desc = f"{2 * octave_range + 1} octaves" if octave_range > 0 else "1 octave"
    
    if verbose:
        print(f"Generated {len(candidates)} candidate ratios")
        print(f"Searching for scales with {note_desc} notes")
        print(f"Evaluating across {octave_desc}")
        print(f"Using metric: {metric_desc}")
        print()
    
    # Lazy interval cache - compute on demand instead of precomputing all pairs
    # This avoids O(n²) precomputation which becomes prohibitive for large ratio limits
    interval_cache = {}
    
    # Scale fitness cache: frozenset of (n,d) tuples -> fitness
    scale_cache = {}
    
    def eval_scale_cached(scale):
        key = frozenset((r.numerator, r.denominator) for r in scale)
        if key not in scale_cache:
            fitness, _ = evaluate_scale(scale, metrics, octave_range, interval_cache, scale_metrics, edo)
            scale_cache[key] = fitness
        return scale_cache[key]
    
    # Use provided seeds (empty list means no seeds)
    if seed_scales is None:
        seed_scales = []
    
    # Initialize population
    # Note: N notes = N-1 ratios (tonic 1/1 is implicit, octave 2/1 is repeat)
    population = []
    
    # Add seed scales (filtered to appropriate size)
    for seed in seed_scales:
        if config.fixed_notes:
            target_len = config.fixed_notes - 1
            if len(seed) == target_len:
                fitness = eval_scale_cached(seed)
                population.append((fitness, seed.copy()))
            elif len(seed) > target_len:
                # Randomly sample to target length
                trimmed = random.sample(seed, target_len)
                fitness = eval_scale_cached(trimmed)
                population.append((fitness, trimmed))
        else:
            if config.min_notes - 1 <= len(seed) <= config.max_notes - 1:
                fitness = eval_scale_cached(seed)
                population.append((fitness, seed.copy()))
    
    # Fill rest with random scales (biased toward simpler ratios)
    while len(population) < config.population_size:
        if config.fixed_notes:
            num_ratios = config.fixed_notes - 1
        else:
            num_ratios = random.randint(config.min_notes - 1, config.max_notes - 1)
        scale = create_random_scale(candidates, num_ratios, candidate_weights)
        fitness = eval_scale_cached(scale)
        population.append((fitness, scale))
    
    best_ever = max(population, key=lambda x: x[0])
    
    # Temperature schedule for simulated annealing
    temp_ratio = (config.final_temp / config.initial_temp) ** (1.0 / max(1, config.generations))
    temperature = config.initial_temp
    
    # Track for stagnation and early stopping
    stagnation_counter = 0
    
    for gen in range(config.generations):
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Track best ever found (fitness is -score, so higher fitness = better)
        if population[0][0] > best_ever[0]:
            best_ever = population[0]
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Early stopping if converged
        if stagnation_counter > 20:
            if verbose:
                best_quality = best_ever[0]
                num_intervals = sum(1 for k in interval_cache if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int))
                print(f"Gen {gen:4d}: converged (no improvement for 20 gens), best quality = {best_quality:.4f}, intervals = {num_intervals}")
            break
        
        # Adaptive mutation rate (increase when stagnating)
        adaptive_mutation = config.mutation_rate
        if stagnation_counter > 10:
            adaptive_mutation = min(0.8, config.mutation_rate * (1 + stagnation_counter / 20))
        
        if verbose and gen % 10 == 0:
            # Print quality (higher = better)
            current_quality = population[0][0]
            best_quality = best_ever[0]
            unique = len(set(frozenset((r.numerator, r.denominator) for r in ind[1]) for ind in population[:50]))
            # Count unique intervals cached (keys that are tuples of 2 ints, not (tuple, string))
            num_intervals = sum(1 for k in interval_cache if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], int))
            if abs(current_quality - best_quality) < 0.0001:
                print(f"Gen {gen:4d}: quality = {current_quality:.4f}, temp = {temperature:.4f}, unique = {unique}/50, scales = {len(scale_cache)}, intervals = {num_intervals}")
            else:
                print(f"Gen {gen:4d}: quality = {current_quality:.4f} (best = {best_quality:.4f}), temp = {temperature:.4f}, unique = {unique}/50, scales = {len(scale_cache)}, intervals = {num_intervals}")
        
        # Create new population
        new_population = []
        
        # Always preserve best_ever (critical: don't lose the best solution found)
        new_population.append(best_ever)
        
        # Elite selection with increasing randomness (avoids expensive similarity calculations)
        elites = select_elites(population, config.elite_count + 3)
        for elite in elites[:config.elite_count + 1]:  # One less since we added best_ever
            if elite[1] != best_ever[1]:  # Don't duplicate best_ever
                new_population.append(elite)
        
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
            
            fitness = eval_scale_cached(child)
            
            # Simulated annealing acceptance
            if new_population and fitness < new_population[-1][0]:
                delta = new_population[-1][0] - fitness
                accept_prob = math.exp(-delta / temperature) if temperature > 0 else 0
                if random.random() > accept_prob:
                    if random.random() < 0.3:
                        if config.fixed_notes:
                            num_ratios = config.fixed_notes - 1
                        else:
                            num_ratios = random.randint(config.min_notes - 1, config.max_notes - 1)
                        child = create_random_scale(candidates, num_ratios, candidate_weights)
                        fitness = eval_scale_cached(child)
            
            new_population.append((fitness, child))
        
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
    _, details = evaluate_scale(best_scale, metrics, octave_range, interval_cache, scale_metrics, edo)
    return sorted(best_scale), best_fitness, details


def format_scale(scale: List[Fraction], verbose: bool = False) -> str:
    """Format a scale for display."""
    lines = []
    for r in sorted(scale):
        if verbose:
            cents = 1200 * math.log2(float(r))
            lines.append(f"{r.numerator}/{r.denominator}  ({cents:.1f} cents, Tenney: {tenney_height(r):.2f})")
        else:
            lines.append(f"{r.numerator}/{r.denominator}")
    if verbose:
        lines.append(f"2/1  (1200.0 cents, Tenney: {tenney_height(Fraction(2,1)):.2f})")
    else:
        lines.append("2/1")
    return "\n".join(lines)


def print_interval_matrix(scale: List[Fraction], metrics: List[Tuple[str, float]] = None):
    """Print the interval matrix for a scale."""
    if metrics is None:
        metrics = [('tenney', 1.0)]
    
    full_scale = [Fraction(1, 1)] + sorted(scale) + [Fraction(2, 1)]
    
    # Header
    header = "        " + "  ".join(f"{r.numerator:>3}/{r.denominator:<3}" for r in full_scale)
    print(header)
    print("-" * len(header))
    
    for i, r1 in enumerate(full_scale):
        row = f"{r1.numerator:>3}/{r1.denominator:<3} |"
        for j, r2 in enumerate(full_scale):
            if j <= i:
                row += "   -    "
            else:
                interval = interval_between(r1, r2)
                score = combined_metric(interval, metrics, normalize=True)
                row += f" {score:5.2f}  "
        print(row)


def evaluate_named_scale(name: str, ratios: List[str], metrics: List[Tuple[str, float]] = None,
                         octave_range: int = 1, scale_metrics: List[Tuple[str, float]] = None,
                         edo: int = 0):
    """Evaluate and display a specific scale given as string ratios."""
    if metrics is None:
        metrics = [('tenney', 1.0)]
    if scale_metrics is None:
        scale_metrics = []
    
    scale = [Fraction(r) for r in ratios if r != "2/1"]
    fitness, details = evaluate_scale(scale, metrics, octave_range, None, scale_metrics, edo)
    
    # Build metric description
    if len(metrics) == 1:
        _, metric_desc = CONSONANCE_METRICS[metrics[0][0]]
    else:
        parts = [f"{name}:{weight}" for name, weight in metrics]
        metric_desc = "Combined: " + ", ".join(parts)
    
    if scale_metrics:
        scale_parts = []
        for mname, weight in scale_metrics:
            if mname == 'edo':
                scale_parts.append(f"edo{edo}:{weight}")
            else:
                scale_parts.append(f"{mname}:{weight}")
        metric_desc += " + scale[" + ", ".join(scale_parts) + "]"
    
    print(f"\n{'=' * 60}")
    print(f"Scale: {name}")
    print("=" * 60)
    print()
    print(format_scale(scale))
    print()
    print("--- Details ---")
    print(format_scale(scale, verbose=True))
    print()
    print(f"Metric: {metric_desc}")
    print(f"Octaves evaluated: {2 * octave_range + 1}")
    print(f"Total intervals: {details['num_intervals']}")
    print(f"Quality (0-1, higher=better): {details['quality']:.4f}")
    
    # Show EDO deviation if applicable
    if edo > 0 and 'edo_deviation' in details['metric_averages']:
        avg_dev = details['metric_averages']['edo_deviation']
        print(f"Average EDO deviation: {avg_dev:.1f} cents (from {edo}-EDO)")
    
    # Show individual metric scores
    if len(metrics) > 1 or scale_metrics:
        print("\nIndividual metric averages:")
        for metric_name, weight in metrics:
            avg = details['metric_averages'].get(metric_name, 0)
            print(f"  {metric_name}: {avg:.4f} (weight: {weight})")
        for metric_name, weight in scale_metrics:
            avg = details['metric_averages'].get(metric_name, 0)
            if metric_name == 'edo':
                print(f"  edo{edo}: {avg:.4f} (weight: {weight})")
            else:
                print(f"  {metric_name}: {avg:.4f} (weight: {weight})")


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary search for just intonation scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --notes 7 --metric tenney
  %(prog)s --notes 7 --metric "tenney:1,roughness:0.5"
  %(prog)s --notes 12 --metric "soft_tenney:1,primes:0.5" --ratio-limit 64
  %(prog)s --notes 12 --metric "tenney:1,edo12:2"  # neji scale
  %(prog)s --seeds my_scales.txt --notes 7  # use seed scales from file
  %(prog)s --compare --metric wilson

Seed file format (supports Scala .scl format):
  ! comment (Scala style)
  # comment (also supported)
  9/8
  5/4
  4/3
  3/2
  5/3
  15/8
  2            <- "2" = 2/1, removed automatically at end
  
  # blank lines or # comments separate scales
  8/7
  6/5
  ...

Defaults:
  --notes: 7 (heptatonic scale)
  --ratio-limit: 32 (allows up to 31-limit intervals)
  --generations: 500
  --population: 200
  --temp: 0.1
  --octaves: 1 (evaluate across 3 octaves)
  --seeds: none (unbiased random initialization)

Note counts include the tonic but not the octave repeat.
A 7-note scale has 6 ratios plus the implicit 1/1 tonic.
        """
    )
    parser.add_argument("--notes", type=int, default=None, 
                        help="Exact number of notes including tonic (default: 7)")
    parser.add_argument("--max-notes", type=int, default=7, 
                        help="Maximum notes in scale (default: 7)")
    parser.add_argument("--min-notes", type=int, default=3, 
                        help="Minimum notes in scale (default: 3)")
    parser.add_argument("--ratio-limit", type=int, default=32, 
                        help="Max numerator/denominator (default: 32)")
    parser.add_argument("--generations", type=int, default=500, 
                        help="Number of generations (default: 500)")
    parser.add_argument("--population", type=int, default=200, 
                        help="Population size (default: 200)")
    parser.add_argument("--runs", type=int, default=1, 
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--metric", type=str, default='tenney', 
                        help="Metric(s) to use. Single: 'tenney'. Combined: 'tenney:1,roughness:0.5'")
    parser.add_argument("--diversity", type=float, default=0.1, 
                        help="(deprecated, ignored)")
    parser.add_argument("--temp", type=float, default=0.1, 
                        help="Initial SA temperature (default: 0.1, higher=more exploration)")
    parser.add_argument("--octaves", type=int, default=1, 
                        help="Octave range (default: 1 = 3 octaves, 0=single, 2=5 octaves)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="File with seed scales to initialize population (default: none)")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare known scales instead of searching")
    parser.add_argument("--list-metrics", action="store_true", 
                        help="List available metrics and exit")
    args = parser.parse_args()
    
    if args.list_metrics:
        print("Available interval metrics (applied to each interval pair):")
        print()
        for name, (_, desc) in CONSONANCE_METRICS.items():
            print(f"  {name:12} - {desc}")
        print()
        print("Available scale metrics (applied to whole scale):")
        print()
        for name, (_, desc) in SCALE_METRICS.items():
            print(f"  {name:12} - {desc}")
        print(f"  {'edoN/nejiN':12} - EDO approximation (e.g., edo12, neji19) for neji scales")
        print()
        print("Combine metrics with: --metric 'metric1:weight1,metric2:weight2'")
        print("Example: --metric 'tenney:1,roughness:0.5,steps:0.3,primes:0.5'")
        print("Example (neji): --metric 'tenney:1,edo12:2' for 12-EDO approximation")
        print("Example (neji): --metric 'soft_tenney:1,neji19:1.5' for 19-EDO approximation")
        return
    
    # Parse metrics
    try:
        metrics, scale_metrics, edo = parse_metrics(args.metric)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if args.compare:
        # Compare some well-known scales
        scales_to_compare = [
            ("User's example (5-limit)", ["6/5", "5/4", "4/3", "3/2", "8/5", "5/3"]),
            ("Ptolemaic diatonic", ["9/8", "5/4", "4/3", "3/2", "5/3", "15/8"]),
            ("Natural harmonic (5-limit)", ["9/8", "5/4", "4/3", "3/2", "8/5", "15/8"]),
            ("Evolved septimal (7-limit)", ["10/9", "7/6", "5/4", "4/3", "3/2", "5/3", "7/4"]),
        ]
        
        for name, ratios in scales_to_compare:
            evaluate_named_scale(name, ratios, metrics, args.octaves, scale_metrics, edo)
        return
    
    config = SearchConfig(
        max_notes=args.max_notes,
        min_notes=args.min_notes,
        fixed_notes=args.notes,
        ratio_limit=args.ratio_limit,
        generations=args.generations,
        population_size=args.population,
        metrics=metrics,
        scale_metrics=scale_metrics,
        diversity_weight=args.diversity,
        initial_temp=args.temp,
        octave_range=args.octaves,
        edo=edo,
    )
    
    # Load seed scales if specified
    seed_scales = None
    if args.seeds:
        seed_scales = load_seed_scales(args.seeds)
        if seed_scales:
            print(f"Loaded {len(seed_scales)} seed scales from {args.seeds}")
    
    # Build metric description
    if len(metrics) == 1:
        _, metric_desc = CONSONANCE_METRICS[metrics[0][0]]
    else:
        parts = [f"{name}:{weight}" for name, weight in metrics]
        metric_desc = "Combined: " + ", ".join(parts)
    
    if scale_metrics:
        scale_parts = []
        for name, weight in scale_metrics:
            if name == 'edo':
                scale_parts.append(f"edo{edo}:{weight}")
            else:
                scale_parts.append(f"{name}:{weight}")
        metric_desc += " + scale[" + ", ".join(scale_parts) + "]"
    
    note_desc = f"exactly {config.fixed_notes}" if config.fixed_notes else f"{config.min_notes}-{config.max_notes}"
    octave_desc = f"{2 * config.octave_range + 1} octaves" if config.octave_range > 0 else "1 octave"
    
    print("=" * 60)
    print("Just Intonation Scale Search")
    print("=" * 60)
    print(f"Notes: {note_desc}")
    print(f"Ratio limit: {config.ratio_limit}")
    print(f"Octave range: {octave_desc}")
    print(f"Metric: {metric_desc}")
    print(f"Generations: {config.generations}")
    print("=" * 60)
    print()
    
    best_overall = None
    
    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {run + 1}/{args.runs} ---\n")
        
        scale, fitness, details = evolve(config, seed_scales=seed_scales)
        
        if best_overall is None or fitness > best_overall[1]:
            best_overall = (scale, fitness, details)
    
    scale, fitness, details = best_overall
    
    print("\n" + "=" * 60)
    print("BEST SCALE FOUND")
    print("=" * 60)
    print()
    print(format_scale(scale))
    print()
    print("--- Details ---")
    print(format_scale(scale, verbose=True))
    print()
    print(f"Metric: {metric_desc}")
    print(f"Intervals evaluated: {details['num_intervals']} (across {octave_desc})")
    print(f"Quality (0-1, higher=better): {details['quality']:.4f}")
    
    # Show individual metric scores
    if len(metrics) > 1:
        print("\nIndividual metric averages:")
        for metric_name, weight in metrics:
            _, desc = CONSONANCE_METRICS[metric_name]
            avg = details['metric_averages'][metric_name]
            print(f"  {metric_name}: {avg:.4f} (weight: {weight})")
    
    print()
    print("Interval Matrix (combined scores, single octave):")
    print()
    print_interval_matrix(scale, metrics)


if __name__ == "__main__":
    main()
