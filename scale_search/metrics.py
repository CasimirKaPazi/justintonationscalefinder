"""
Consonance metrics, scale metrics, normalization, and caching.

All interval metrics take a Fraction and return a float where lower = more consonant.
Scale metrics take a list of Fractions (the scale) and return a float.
"""

import heapq
import math
import re
from fractions import Fraction
from typing import List, Set, Tuple

from scale_search.util import prime_factors, generate_candidate_ratios

__all__ = [
    # Metric registries
    'CONSONANCE_METRICS', 'SCALE_METRICS',
    # Interval metrics
    'tenney_height', 'euler_gradus', 'harmonic_entropy', 'plomp_levelt_roughness',
    'wilson_complexity', 'barlow_harmonicity', 'prime_limit_penalty', 'odd_limit',
    # Scale metrics
    'step_size_penalty', 'prime_diversity_score', 'edo_penalty', 'edo_deviation',
    'scale_balance', 'scale_propriety', 'scale_variety',
    # Normalization
    'init_normalization', 'get_normalization',
    # Caching
    'combined_metric', 'cached_metric', 'clear_metric_cache',
    # Parsing and formatting
    'parse_metrics', 'format_metric_description',
    # Fingerprinting
    'scale_fingerprint', 'fingerprint_distance', 'fingerprint_diversity_score',
]

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
    Î“ = 1 + Î£(p - 1) * e for prime factorization Î p^e
    Applied to n*d of ratio n/d.
    Lower = more consonant. Favors powers of 2 and 3.
    """
    n = ratio.numerator * ratio.denominator
    factors = prime_factors(n)
    if not factors:
        return 1
    return 1 + sum((p - 1) * e for p, e in factors)


# Precomputed simple ratios for harmonic entropy (computed once at module load)
# Each entry is (log2(n/d), complexity=n*d) for coprime n/d with d<16 and n*d<=100
_ENTROPY_SIMPLE_RATIOS: List[Tuple[float, int]] = [
    (math.log2(n / d), n * d)
    for d in range(1, 16)
    for n in range(d, 2 * d + 1)
    if math.gcd(n, d) == 1 and n * d <= 100
]


def harmonic_entropy(ratio: Fraction, spread: float = 0.017) -> float:
    """
    Harmonic entropy based on Paul Erlich's model.
    Measures how "ambiguous" an interval is - intervals near multiple simple ratios
    have higher entropy than those clearly identified with one simple ratio.
    
    The spread parameter (in octaves) controls the uncertainty window.
    0.017 octaves â‰ˆ 20 cents, which is roughly a just-noticeable difference.
    
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
      - 0 cents (unison): penalty â‰ˆ 1.0
      - 30 cents: penalty â‰ˆ 0.3
      - 50 cents: penalty â‰ˆ 0.14
      - 80 cents: penalty â‰ˆ 0.04
    
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
    - Having 7 contributes 1/7 â‰ˆ 0.143
    - Having 11 contributes 1/11 â‰ˆ 0.091  
    - Having 13 contributes 1/13 â‰ˆ 0.077
    - Having 17 contributes 1/17 â‰ˆ 0.059
    - Having 19 contributes 1/19 â‰ˆ 0.053
    
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
    
    Uses cosine curve: penalty = (1 - cos(2Ï€ * distance / step_size)) / 2
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


def odd_limit(ratio: Fraction) -> float:
    """
    Odd-limit: the largest odd number in the ratio after removing factors of 2.
    
    Many theorists consider this more musically relevant than prime limit.
    For example:
    - 9/8 has odd-limit 9 (despite being 3-limit in terms of primes)
    - 5/4 has odd-limit 5
    - 3/2 has odd-limit 3
    - 15/8 has odd-limit 15
    
    Returns log2 of the odd-limit for better scaling with other metrics.
    """
    n, d = ratio.numerator, ratio.denominator
    
    # Remove factors of 2
    while n % 2 == 0:
        n //= 2
    while d % 2 == 0:
        d //= 2
    
    # The odd-limit is the larger of the two
    limit = max(n, d)
    
    # Return log2 for scaling (1->0, 3->1.58, 5->2.32, 7->2.81, 9->3.17, 15->3.91)
    return math.log2(limit) if limit > 0 else 0


def barlow_harmonicity(ratio: Fraction) -> float:
    """
    Clarence Barlow's indigestibility/harmonicity measure.
    
    Barlow's formula weights primes by 2*(p-1)^2/p, then sums for all prime factors.
    Lower values = more consonant.
    
    This gives different results than Tenney:
    - Penalizes prime 3 less than Tenney does relative to 2
    - Penalizes higher primes (7, 11, 13) more gradually
    
    Widely used in computer-assisted composition and microtonal theory.
    """
    def indigestibility(n):
        """Barlow's indigestibility of an integer."""
        if n <= 1:
            return 0
        # Each prime factor p with exponent e contributes e * 2*(p-1)^2/p
        return sum(e * 2 * (p - 1) ** 2 / p for p, e in prime_factors(n))
    
    return indigestibility(ratio.numerator) + indigestibility(ratio.denominator)


def scale_balance(scale: List[Fraction]) -> float:
    """
    Measure how evenly distributed notes are across the octave.
    
    Uses coefficient of variation (std dev / mean) of step sizes.
    Returns 0 for perfectly even scales, higher values for uneven scales.
    
    This is an alternative to step_size_penalty that measures overall
    distribution rather than just penalizing close notes.
    
    Examples:
    - 12-EDO: balance = 0 (all steps equal)
    - Major scale: balance â‰ˆ 0.33 (mix of whole and half steps)
    - Pentatonic: balance â‰ˆ 0.22 (fairly even)
    """
    if len(scale) < 1:
        return 0.0
    
    # Get all notes including implicit 1/1 and 2/1
    full_scale = [Fraction(1, 1)] + sorted(scale) + [Fraction(2, 1)]
    
    # Calculate step sizes in cents
    steps = []
    for i in range(len(full_scale) - 1):
        interval = full_scale[i + 1] / full_scale[i]
        cents = 1200 * math.log2(float(interval))
        steps.append(cents)
    
    if len(steps) < 2:
        return 0.0
    
    mean_step = sum(steps) / len(steps)
    if mean_step == 0:
        return 0.0
    
    variance = sum((s - mean_step) ** 2 for s in steps) / len(steps)
    std_dev = math.sqrt(variance)
    
    # Coefficient of variation (normalized std dev)
    return std_dev / mean_step


def scale_propriety(scale: List[Fraction]) -> float:
    """
    Rothenberg propriety: measures whether larger scale steps span larger intervals.
    
    A scale is "strictly proper" if every interval of n steps is smaller than
    every interval of n+1 steps. It's "proper" if they're non-overlapping.
    "Improper" scales have overlapping interval classes.
    
    Returns a penalty where:
    - 0 = strictly proper (ideal)
    - Small values = mostly proper with few ambiguities
    - Large values = highly improper
    
    Many common scales (major, pentatonic) are proper or strictly proper.
    Improper scales can feel unstable or ambiguous.
    """
    if len(scale) < 2:
        return 0.0
    
    # Full scale with 1/1 and 2/1
    full_scale = [Fraction(1, 1)] + sorted(scale) + [Fraction(2, 1)]
    n = len(full_scale) - 1  # Number of scale degrees (not counting octave)
    
    # Track only min and max interval size per step count (no list allocation)
    # intervals_range[k] = (min_cents, max_cents) for intervals spanning k steps
    intervals_range = {}
    
    for i in range(len(full_scale) - 1):
        for j in range(i + 1, len(full_scale)):
            steps = j - i
            interval = full_scale[j] / full_scale[i]
            # Don't octave reduce - we want actual interval sizes within the octave
            cents = 1200 * math.log2(float(interval))
            
            if steps in intervals_range:
                mn, mx = intervals_range[steps]
                if cents < mn:
                    mn = cents
                if cents > mx:
                    mx = cents
                intervals_range[steps] = (mn, mx)
            else:
                intervals_range[steps] = (cents, cents)
    
    # Check for propriety violations
    # Compare max of k-step intervals with min of (k+1)-step intervals
    violations = 0.0
    total_comparisons = 0
    
    step_counts = sorted(intervals_range.keys())
    for i in range(len(step_counts) - 1):
        k = step_counts[i]
        k_next = step_counts[i + 1]
        
        if k_next != k + 1:
            continue  # Skip if not consecutive step classes
        
        max_k = intervals_range[k][1]
        min_k_next = intervals_range[k_next][0]
        
        total_comparisons += 1
        
        # Violation if max of smaller step class >= min of larger step class
        if max_k >= min_k_next:
            # Degree of violation (how much overlap in cents, normalized)
            overlap = (max_k - min_k_next) / 100  # Normalize by ~semitone
            violations += overlap
    
    return violations / total_comparisons if total_comparisons > 0 else 0.0


def scale_variety(scale: List[Fraction]) -> float:
    """
    Measure interval variety using soft clustering via Gaussian kernel.
    
    For each pair of intervals, computes similarity = exp(-dÂ²/2ÏƒÂ²) where d is
    the distance in cents. Sums all similarities - lower sum means intervals
    are more spread out (higher variety).
    
    Returns 1 - (total_similarity / max_possible_similarity).
    
    The kernel width σ = 10 cents is motivated by the Just Noticeable Difference
    (JND) for melodic pitch - intervals within ~10 cents are perceptually similar.
    
    This is smooth and principled:
    - 3/2 (702¢) and 301/200 (707¢) contribute high similarity (low variety)
    - 3/2 (702¢) and 7/5 (583¢) contribute negligible similarity (high variety)
    - No hard thresholds - similarity falls off smoothly with distance
    
    After sorting, only nearby pairs need comparison since the Gaussian kernel
    decays to negligible values beyond ~5σ = 50 cents.
    
    Examples:
    - Hexany: ~0.98 (all intervals perceptually distinct)
    - Major scale: ~0.95 (some intervals cluster, e.g. multiple ~200¢)
    """
    if len(scale) < 1:
        return 1.0
    
    # Full scale with 1/1 and 2/1
    full_scale = [Fraction(1, 1)] + sorted(scale) + [Fraction(2, 1)]
    
    # Collect all intervals in cents (octave-reduced)
    interval_cents = []
    for i in range(len(full_scale)):
        for j in range(i + 1, len(full_scale)):
            interval = full_scale[j] / full_scale[i]
            # Octave reduce
            while interval >= 2:
                interval = interval / 2
            cents = 1200 * math.log2(float(interval)) if interval > 1 else 0
            interval_cents.append(cents)
    
    n = len(interval_cents)
    if n < 2:
        return 1.0
    
    # Gaussian kernel width based on JND (~10 cents)
    sigma = 10.0
    two_sigma_sq = 2 * sigma * sigma
    
    # Sort so we can skip distant pairs early.
    # At 5σ = 50 cents, similarity = exp(-25/2) ≈ 3.7e-6, negligible.
    cutoff = 5.0 * sigma
    
    interval_cents.sort()
    
    # Sum pairwise similarities (only nearby pairs contribute)
    total_similarity = 0.0
    for i in range(n):
        ci = interval_cents[i]
        for j in range(i + 1, n):
            dist = interval_cents[j] - ci  # Already sorted, so dist >= 0
            if dist > cutoff:
                break  # All subsequent j are even farther
            total_similarity += math.exp(-dist * dist / two_sigma_sq)
    
    # Maximum similarity if all intervals identical
    max_similarity = n * (n - 1) / 2
    
    # Variety = 1 - normalized similarity
    return 1.0 - (total_similarity / max_similarity)



CONSONANCE_METRICS = {
    'tenney': (tenney_height, "Tenney height: log2(n*d)"),
    'euler': (euler_gradus, "Euler's Gradus Suavitatis"),
    'entropy': (harmonic_entropy, "Harmonic entropy (Erlich-style)"),
    'roughness': (plomp_levelt_roughness, "Plomp-Levelt roughness model"),
    'wilson': (wilson_complexity, "Wilson harmonic complexity"),
    'soft_tenney': (prime_limit_penalty, "Softened Tenney (sqrt) - less harsh on higher primes"),
    'odd_limit': (odd_limit, "Odd-limit: largest odd factor (log2)"),
    'barlow': (barlow_harmonicity, "Barlow harmonicity/indigestibility"),
}

# Scale-level metrics (operate on whole scale, not individual intervals)
# Note: 'edo' is handled specially - use 'edo12', 'edo19', etc. to specify which EDO
SCALE_METRICS = {
    'steps': (step_size_penalty, "Step size penalty (penalizes clustered notes)"),
    'primes': (prime_diversity_score, "Prime diversity (penalizes lack of 7/11/13-limit)"),
    'balance': (scale_balance, "Step size balance (coefficient of variation)"),
    'propriety': (scale_propriety, "Rothenberg propriety (penalizes improper scales)"),
    'variety': (scale_variety, "Interval variety (perceptually distinct intervals)"),
    # 'edoN' handled specially in parse_metrics - generates neji scales
}

# Scale-level metric normalization ranges (these are inherent properties 
# of scales, not dependent on ratio_limit)
_SCALE_METRIC_NORMALIZATION = {
    'steps': (0.0, 0.5),        # 0 = no close steps, 0.5 = very clustered
    'primes': (0.0, 0.5),       # sum(1/p): 0=5-limit, 0.31=13-limit, 0.42=19-limit
    'balance': (0.0, 1.0),      # 0 = even steps, 1 = very uneven
    'propriety': (0.0, 1.0),    # 0 = proper, higher = more improper
    'variety': (0.0, 1.0),      # 0 = all intervals cluster, 1 = all perceptually distinct
    'edo': (0.0, 1.0),          # 0 = on EDO steps, 1 = at midpoints
}

# Active normalization ranges, populated by init_normalization()
# Combines dynamically computed interval metric ranges with static scale metric ranges
_normalization: dict = dict(_SCALE_METRIC_NORMALIZATION)
_normalization_limit: int = 0  # Which ratio_limit was used to compute current ranges


def init_normalization(ratio_limit: int, verbose: bool = False):
    """
    Compute normalization ranges for interval metrics by evaluating all candidate
    ratios within the given ratio_limit.
    
    This replaces the old hardcoded ranges which were far too narrow for most
    metrics (especially euler, wilson, barlow, odd_limit), causing the optimizer
    to clamp the vast majority of intervals to the same maximum penalty score
    and lose all discrimination between them.
    
    Must be called before any evaluation. Caches results per ratio_limit.
    """
    global _normalization, _normalization_limit
    
    if ratio_limit == _normalization_limit:
        return  # Already computed for this limit
    
    candidates, _ = generate_candidate_ratios(ratio_limit)
    
    for name in CONSONANCE_METRICS:
        fn, _ = CONSONANCE_METRICS[name]
        values = [fn(r) for r in candidates]
        if values:
            _normalization[name] = (min(values), max(values))
        else:
            _normalization[name] = (0.0, 1.0)
    
    # Scale metrics keep their static ranges
    _normalization.update(_SCALE_METRIC_NORMALIZATION)
    
    _normalization_limit = ratio_limit
    
    if verbose:
        print("Metric normalization ranges (computed from candidates):")
        for name in CONSONANCE_METRICS:
            mn, mx = _normalization[name]
            print(f"  {name:12s}: [{mn:.2f}, {mx:.2f}]")
        print()


def get_normalization(metric_name: str) -> Tuple[float, float]:
    """Get the normalization range for a metric."""
    return _normalization.get(metric_name, (0.0, 1.0))


def scale_fingerprint(details: dict) -> Tuple[float, ...]:
    """
    Compute a metric fingerprint for a scale based on its evaluation details.
    
    The fingerprint is a tuple of normalized metric scores (0-1 range).
    Scales with similar fingerprints are likely similar in character,
    even if they have different notes.
    
    This allows O(k) diversity comparison (k = number of metrics)
    instead of O(n) structural comparison.
    """
    fingerprint = []
    
    # Add interval metric averages (normalized)
    for metric_name, _ in details.get('metrics', []):
        raw = details['metric_averages'].get(metric_name, 0)
        min_val, max_val = get_normalization(metric_name)
        norm = (raw - min_val) / (max_val - min_val) if max_val > min_val else 0
        norm = max(0.0, min(1.0, norm))
        fingerprint.append(norm)
    
    # Add scale metric values (normalized)
    for metric_name, _ in details.get('scale_metrics', []):
        raw = details['metric_averages'].get(metric_name, 0)
        min_val, max_val = get_normalization(metric_name)
        norm = (raw - min_val) / (max_val - min_val) if max_val > min_val else 0
        norm = max(0.0, min(1.0, norm))
        fingerprint.append(norm)
    
    # Add quality score itself as a dimension
    fingerprint.append(details.get('quality', 0))
    
    return tuple(fingerprint)


def fingerprint_distance(fp1: Tuple[float, ...], fp2: Tuple[float, ...]) -> float:
    """Euclidean distance between two fingerprints."""
    if len(fp1) != len(fp2):
        return 1.0  # Max distance if incompatible
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(fp1, fp2)))


def fingerprint_diversity_score(fingerprint: Tuple[float, ...], 
                                population_fingerprints: List[Tuple[float, ...]],
                                k: int = 5) -> float:
    """
    Compute diversity score based on distance to k nearest neighbors in metric space.
    
    Higher score = more different from existing population.
    Returns average distance to k nearest neighbors (0-1 range, roughly).
    """
    if not population_fingerprints:
        return 1.0  # Maximum diversity if no population yet
    
    # Compute distances to all population members, keep only k smallest
    distances = [fingerprint_distance(fingerprint, fp) for fp in population_fingerprints]
    k_nearest = heapq.nsmallest(min(k, len(distances)), distances)
    
    if not k_nearest:
        return 1.0
    
    return sum(k_nearest) / len(k_nearest)



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
    
    # Default to barlow if no interval metrics specified
    if not interval_metrics:
        interval_metrics = [('barlow', 1.0)]
    
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
            min_val, max_val = get_normalization(metric_name)
            score = (score - min_val) / (max_val - min_val)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        total += weight * score
        total_weight += weight
    
    return total / total_weight if total_weight > 0 else 0.0




def format_metric_description(metrics: List[Tuple[str, float]], 
                               scale_metrics: List[Tuple[str, float]] = None,
                               edo: int = 0) -> str:
    """
    Build a human-readable description of the metric configuration.
    
    Examples:
        "Barlow harmonicity/indigestibility"
        "Combined: tenney:1.0, roughness:0.5 + scale[steps:0.3]"
    """
    if scale_metrics is None:
        scale_metrics = []
    
    if len(metrics) == 1:
        _, desc = CONSONANCE_METRICS[metrics[0][0]]
    else:
        parts = [f"{name}:{weight}" for name, weight in metrics]
        desc = "Combined: " + ", ".join(parts)
    
    if scale_metrics:
        scale_parts = []
        for name, weight in scale_metrics:
            if name == 'edo':
                scale_parts.append(f"edo{edo}:{weight}")
            else:
                scale_parts.append(f"{name}:{weight}")
        desc += " + scale[" + ", ".join(scale_parts) + "]"
    
    return desc

