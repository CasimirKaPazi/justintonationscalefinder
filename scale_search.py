#!/usr/bin/env python3
"""
Just Intonation Scale Search
Evolutionary optimization of microtonal scales using multiple consonance metrics.

Copyright (C) 2025

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
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
        combine_method: 'geometric' (how to combine interval scores)
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
    # Multi-octave evaluation
    octave_range: int = 1       # Octaves above/below to consider (0=single, 1=3 total, 2=5 total)
    # EDO approximation (for neji scales)
    edo: int = 0                # If > 0, penalize notes far from this EDO's steps
    # Note constraints
    required_notes: List[Fraction] = None  # Notes that must be in every scale
    excluded_notes: List[Fraction] = None  # Notes that must not appear
    # Interval combination method
    combine_method: str = 'geometric'  # 'geometric', 'worst', or 'mean'
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [('tenney', 1.0)]
        if self.scale_metrics is None:
            self.scale_metrics = []
        if self.required_notes is None:
            self.required_notes = []
        if self.excluded_notes is None:
            self.excluded_notes = []
        # Swap min/max if backwards
        if self.min_notes > self.max_notes:
            self.min_notes, self.max_notes = self.max_notes, self.min_notes
        # Validate: required notes must fit within scale size
        if self.fixed_notes and len(self.required_notes) > self.fixed_notes - 1:
            raise ValueError(f"Too many required notes ({len(self.required_notes)}) for {self.fixed_notes}-note scale")
        if len(self.required_notes) > self.max_notes - 1:
            raise ValueError(f"Too many required notes ({len(self.required_notes)}) for max {self.max_notes}-note scale")


def parse_ratios(ratio_str: str) -> List[Fraction]:
    """
    Parse a comma-separated string of ratios into Fractions.
    
    Examples:
        "3/2,5/4,7/4" -> [Fraction(3,2), Fraction(5,4), Fraction(7,4)]
        "3/2, 5/4" -> [Fraction(3,2), Fraction(5,4)]  (spaces ok)
        "" -> []
    """
    if not ratio_str or not ratio_str.strip():
        return []
    
    ratios = []
    for part in ratio_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '/' in part:
            n, d = part.split('/')
            ratios.append(Fraction(int(n.strip()), int(d.strip())))
        else:
            # Bare integer means n/1
            ratios.append(Fraction(int(part.strip()), 1))
    return ratios


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
        
        result = 0
        temp = n
        p = 2
        while p * p <= temp:
            while temp % p == 0:
                # Each prime factor p contributes 2*(p-1)^2/p
                result += 2 * (p - 1) ** 2 / p
                temp //= p
            p += 1
        if temp > 1:
            # Remaining prime factor
            result += 2 * (temp - 1) ** 2 / temp
        
        return result
    
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
    - Major scale: balance ≈ 0.33 (mix of whole and half steps)
    - Pentatonic: balance ≈ 0.22 (fairly even)
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
    
    # Calculate all intervals by step count
    # intervals_by_steps[k] = list of interval sizes (in cents) spanning k steps
    intervals_by_steps = {}
    
    for i in range(len(full_scale) - 1):
        for j in range(i + 1, len(full_scale)):
            steps = j - i
            interval = full_scale[j] / full_scale[i]
            # Don't octave reduce - we want actual interval sizes within the octave
            cents = 1200 * math.log2(float(interval))
            
            if steps not in intervals_by_steps:
                intervals_by_steps[steps] = []
            intervals_by_steps[steps].append(cents)
    
    # Check for propriety violations
    # Compare max of k-step intervals with min of (k+1)-step intervals
    violations = 0.0
    total_comparisons = 0
    
    step_counts = sorted(intervals_by_steps.keys())
    for i in range(len(step_counts) - 1):
        k = step_counts[i]
        k_next = step_counts[i + 1]
        
        if k_next != k + 1:
            continue  # Skip if not consecutive step classes
        
        max_k = max(intervals_by_steps[k])
        min_k_next = min(intervals_by_steps[k_next])
        
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
    
    For each pair of intervals, computes similarity = exp(-d²/2σ²) where d is
    the distance in cents. Sums all similarities - lower sum means intervals
    are more spread out (higher variety).
    
    Returns 1 - (total_similarity / max_possible_similarity).
    
    The kernel width σ = 10 cents is motivated by the Just Noticeable Difference
    (JND) for melodic pitch - intervals within ~10 cents are perceptually similar.
    
    This is smooth and principled:
    - 3/2 (702¢) and 301/200 (707¢) contribute high similarity (low variety)
    - 3/2 (702¢) and 7/5 (583¢) contribute negligible similarity (high variety)
    - No hard thresholds - similarity falls off smoothly with distance
    
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
    
    # Sum pairwise similarities
    total_similarity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(interval_cents[i] - interval_cents[j])
            similarity = math.exp(-dist * dist / (2 * sigma * sigma))
            total_similarity += similarity
    
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

# Approximate ranges for normalization (based on typical intervals)
# These help combine metrics that have very different scales
METRIC_NORMALIZATION = {
    'tenney': (1.0, 12.0),      # log2(2) to log2(64*64)
    'euler': (1.0, 20.0),       # simple to complex
    'entropy': (0.5, 4.0),      # low to high entropy
    'roughness': (0.0, 1.5),    # smooth to rough
    'wilson': (0.0, 12.0),      # simple to complex
    'soft_tenney': (1.0, 3.5),  # sqrt of tenney range
    'odd_limit': (0.0, 4.0),    # log2(1) to log2(15)
    'barlow': (0.0, 20.0),      # simple to complex (similar range to euler)
    'steps': (0.0, 0.5),        # 0 = no close steps, 0.5 = very clustered
    'primes': (0.0, 0.5),       # sum(1/p): 0=5-limit, 0.31=13-limit, 0.42=19-limit
    'balance': (0.0, 1.0),      # 0 = even steps, 1 = very uneven
    'propriety': (0.0, 1.0),    # 0 = proper, higher = more improper
    'variety': (0.0, 1.0),      # 0 = all intervals cluster, 1 = all perceptually distinct
    'edo': (0.0, 1.0),          # 0 = on EDO steps, 1 = at midpoints
}


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
        min_val, max_val = METRIC_NORMALIZATION.get(metric_name, (0, 1))
        norm = (raw - min_val) / (max_val - min_val) if max_val > min_val else 0
        norm = max(0.0, min(1.0, norm))
        fingerprint.append(norm)
    
    # Add scale metric values (normalized)
    for metric_name, _ in details.get('scale_metrics', []):
        raw = details['metric_averages'].get(metric_name, 0)
        min_val, max_val = METRIC_NORMALIZATION.get(metric_name, (0, 1))
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
    
    # Compute distances to all population members
    distances = [fingerprint_distance(fingerprint, fp) for fp in population_fingerprints]
    
    # Average distance to k nearest (excluding self if present)
    distances.sort()
    k_nearest = distances[:min(k, len(distances))]
    
    if not k_nearest:
        return 1.0
    
    return sum(k_nearest) / len(k_nearest)


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
        # Power mean: (1/n * Σ x^p)^(1/p)
        # Higher p emphasizes worse (higher) penalties more
        # p=1 is arithmetic mean, p=2 is RMS, p→∞ approaches max
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
        # -log(1-p) grows unboundedly as p→1
        # We compute mean log loss then squash back to [0,1)
        # Clamp away from 1 to avoid log(0)
        clamped = [min(0.999, s) for s in interval_scores]
        log_loss_raw = sum(-math.log(1 - s) for s in clamped) / len(clamped)
        # Squash to [0,1): penalty = 1 - exp(-log_loss)
        raw_interval_penalty = 1 - math.exp(-log_loss_raw)
    else:  # 'mean' or default
        # Arithmetic mean: bad intervals can be averaged out
        raw_interval_penalty = sum(interval_scores) / len(interval_scores)
    
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
                        weights: List[float] = None,
                        required: List[Fraction] = None,
                        excluded: Set[Fraction] = None) -> List[Fraction]:
    """
    Create a random scale with the given number of ratios.
    
    Args:
        candidates: List of possible ratios to choose from
        num_ratios: Target number of ratios in scale
        weights: Optional weights for biased sampling (toward simpler ratios)
        required: Notes that must be included in the scale
        excluded: Notes that must not be included
    """
    if required is None:
        required = []
    if excluded is None:
        excluded = set()
    
    # Start with required notes
    selected = list(required)
    selected_set = set(selected)
    
    # Filter candidates: remove required (already have them) and excluded
    available_candidates = []
    available_weights = []
    for i, c in enumerate(candidates):
        if c not in selected_set and c not in excluded:
            available_candidates.append(c)
            if weights is not None and len(weights) == len(candidates):
                available_weights.append(weights[i])
    
    # How many more do we need?
    remaining = num_ratios - len(selected)
    
    if remaining <= 0:
        return selected
    
    if available_weights and len(available_weights) == len(available_candidates):
        # Weighted sampling without replacement
        available_idx = list(range(len(available_candidates)))
        current_weights = list(available_weights)
        
        for _ in range(min(remaining, len(available_candidates))):
            if not available_idx:
                break
            # Normalize weights
            total = sum(current_weights)
            if total <= 0:
                idx_in_list = random.randrange(len(available_idx))
            else:
                r = random.random() * total
                cumsum = 0
                idx_in_list = 0
                for i, w in enumerate(current_weights):
                    cumsum += w
                    if r <= cumsum:
                        idx_in_list = i
                        break
            
            selected.append(available_candidates[available_idx[idx_in_list]])
            available_idx.pop(idx_in_list)
            current_weights.pop(idx_in_list)
    else:
        # Uniform sampling
        extras = random.sample(available_candidates, min(remaining, len(available_candidates)))
        selected.extend(extras)
    
    return selected


def load_seed_scales(filepath: str, with_names: bool = False):
    """
    Load seed scales from a file. Supports both simple format and Scala .scl format.
    
    Comment styles:
    - Lines starting with # are comments (can be used as scale names)
    - Lines starting with ! are comments (Scala format)
    - Blank lines separate scales (in simple format)
    
    Ratio formats:
    - n/d (e.g., 9/8, 5/4, 3/2)
    - n (e.g., 2 means 2/1)
    
    Notes:
    - 2/1 (or just 2) at the end of a scale is removed (it's implicit)
    - Ratios outside the octave (like 3/1, 9/4) are reduced to the octave
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
    
    Args:
        filepath: Path to seed file
        with_names: If True, return (scales, names) tuple
    
    Returns:
        If with_names=False: List of scales (each scale is a list of Fractions)
        If with_names=True: Tuple of (scales, names) where names[i] is name for scales[i] or None
    """
    scales = []
    names = []
    current_scale = []
    current_name = None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Handle comments and blank lines
                if line.startswith('#'):
                    # Save current scale if we have one
                    if current_scale:
                        if current_scale[-1] == Fraction(2, 1):
                            current_scale = current_scale[:-1]
                        if current_scale:
                            current_scale = sanitize_scale(current_scale)
                            if current_scale:
                                scales.append(current_scale)
                                names.append(current_name)
                        current_scale = []
                    # Extract name from comment (strip # and whitespace)
                    current_name = line[1:].strip() or None
                    continue
                
                if line.startswith('!'):
                    # Scala comment - could be filename which serves as name
                    if current_name is None and '.scl' in line.lower():
                        # Extract name from "! filename.scl"
                        name = line[1:].strip()
                        if name.lower().endswith('.scl'):
                            name = name[:-4]
                        current_name = name or None
                    continue
                
                if not line:
                    # Blank line - save current scale
                    if current_scale:
                        if current_scale[-1] == Fraction(2, 1):
                            current_scale = current_scale[:-1]
                        if current_scale:
                            current_scale = sanitize_scale(current_scale)
                            if current_scale:
                                scales.append(current_scale)
                                names.append(current_name)
                        current_scale = []
                        current_name = None
                    continue
                
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
                                # Reduce to octave (1 < r < 2)
                                while ratio >= 2:
                                    ratio = ratio / 2
                                if ratio > 1:  # Skip if it reduced to exactly 1
                                    current_scale.append(ratio)
                    else:
                        # Could be integer like "2" meaning 2/1
                        val = int(line)
                        if val >= 2 and (current_scale or val <= 4):
                            ratio = Fraction(val, 1)
                            while ratio >= 2:
                                ratio = ratio / 2
                            if ratio > 1:
                                current_scale.append(ratio)
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Don't forget the last scale
            if current_scale:
                if current_scale[-1] == Fraction(2, 1):
                    current_scale = current_scale[:-1]
                if current_scale:
                    current_scale = sanitize_scale(current_scale)
                    if current_scale:
                        scales.append(current_scale)
                        names.append(current_name)
    
    except FileNotFoundError:
        print(f"Warning: Seed file '{filepath}' not found.")
        if with_names:
            return [], []
        return []
    except Exception as e:
        print(f"Warning: Error reading seed file '{filepath}': {e}")
        if with_names:
            return [], []
        return []
    
    if with_names:
        return scales, names
    return scales


def find_matching_seed(scale: List[Fraction], seed_scales: List[List[Fraction]], 
                       seed_names: List[str] = None) -> Tuple[int, str]:
    """
    Check if a scale matches any seed scale.
    
    Args:
        scale: The scale to check
        seed_scales: List of seed scales
        seed_names: Optional list of names corresponding to seed_scales
    
    Returns:
        (index, name) if match found, (-1, None) otherwise
    """
    scale_set = frozenset((r.numerator, r.denominator) for r in scale)
    
    for i, seed in enumerate(seed_scales):
        seed_set = frozenset((r.numerator, r.denominator) for r in seed)
        if scale_set == seed_set:
            name = seed_names[i] if seed_names and i < len(seed_names) else None
            return i, name
    
    return -1, None


def reduce_to_octave(ratio: Fraction) -> Fraction:
    """Reduce a ratio to within one octave (1 <= r < 2)."""
    while ratio >= 2:
        ratio = ratio / 2
    while ratio < 1:
        ratio = ratio * 2
    return ratio


def sanitize_scale(scale: List[Fraction]) -> List[Fraction]:
    """
    Ensure all ratios in a scale are valid (1 < r < 2).
    
    - Reduces any ratio >= 2 to the octave
    - Removes 1/1 and 2/1 (these are implicit)
    - Removes duplicates
    - Sorts by pitch
    """
    result = []
    seen = set()
    for r in scale:
        # Reduce to octave
        while r >= 2:
            r = r / 2
        while r < 1:
            r = r * 2
        
        # Skip 1/1 and 2/1
        if r == Fraction(1, 1) or r == Fraction(2, 1):
            continue
        
        # Skip duplicates
        key = (r.numerator, r.denominator)
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    
    return sorted(result, key=lambda x: float(x))


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


def scale_invert(scale: List[Fraction], limit: int = None) -> List[Fraction]:
    """
    Invert a scale - mirror all intervals around the octave.
    
    For each ratio r, the inverted ratio is 2/r.
    Turns major into minor and vice versa.
    
    Example: Major scale [9/8, 5/4, 4/3, 3/2, 5/3, 15/8]
          -> Inverted:   [16/15, 6/5, 4/3, 3/2, 8/5, 16/9]
             (which is Phrygian mode)
    
    Args:
        scale: List of ratios (between 1 and 2, not including 1/1 or 2/1)
        limit: Optional ratio limit - if set, filters out ratios exceeding it
    
    Returns:
        Inverted scale, sorted by pitch
    """
    inverted = []
    for r in scale:
        inv = Fraction(2, 1) / r  # 2/r
        # Reduce to octave (should already be, but be safe)
        while inv >= 2:
            inv = inv / 2
        while inv < 1:
            inv = inv * 2
        
        # Skip 1/1 and 2/1 - these are implicit, not scale degrees
        if inv == Fraction(1, 1) or inv == Fraction(2, 1):
            continue
        
        if limit is None or (inv.numerator <= limit and inv.denominator <= limit):
            inverted.append(inv)
    
    return sorted(inverted, key=lambda r: float(r))


def scale_rotate(scale: List[Fraction], pivot_idx: int = None, limit: int = None) -> List[Fraction]:
    """
    Rotate a scale - choose a new tonic (mode change).
    
    Makes the note at pivot_idx become the new 1/1, and recalculates
    all other intervals relative to it.
    
    Example: Major scale [9/8, 5/4, 4/3, 3/2, 5/3, 15/8], pivot on 5/4
          -> Dorian-ish: intervals relative to 5/4
    
    Args:
        scale: List of ratios (between 1 and 2, not including 1/1 or 2/1)
        pivot_idx: Index of note to become new 1/1. If None, random choice.
        limit: Optional ratio limit - if set, filters out ratios exceeding it
    
    Returns:
        Rotated scale, sorted by pitch
    """
    if not scale:
        return scale
    
    if pivot_idx is None:
        pivot_idx = random.randrange(len(scale))
    
    pivot = scale[pivot_idx]
    
    # Build new scale: all intervals relative to pivot
    # The pivot itself becomes 1/1 (implicit, not included)
    # The original 1/1 becomes a note relative to pivot
    
    rotated = []
    
    # Original 1/1 relative to pivot = 1/pivot (then octave reduce)
    inv_pivot = Fraction(1, 1) / pivot
    while inv_pivot < 1:
        inv_pivot = inv_pivot * 2
    while inv_pivot >= 2:
        inv_pivot = inv_pivot / 2
    # Only add if not 1/1 or 2/1 and within limit
    if inv_pivot != Fraction(1, 1) and inv_pivot != Fraction(2, 1):
        if limit is None or (inv_pivot.numerator <= limit and inv_pivot.denominator <= limit):
            rotated.append(inv_pivot)
    
    # Each other note relative to pivot
    for i, note in enumerate(scale):
        if i == pivot_idx:
            continue  # This becomes 1/1, skip
        
        rel = note / pivot
        # Reduce to octave
        while rel >= 2:
            rel = rel / 2
        while rel < 1:
            rel = rel * 2
        
        # Skip 1/1 and 2/1
        if rel == Fraction(1, 1) or rel == Fraction(2, 1):
            continue
        
        if limit is None or (rel.numerator <= limit and rel.denominator <= limit):
            rotated.append(rel)
    
    return sorted(set(rotated), key=lambda r: float(r))


def mutate(scale: List[Fraction], candidates: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Mutate a scale using musically-meaningful operations:
    1. Transpose a note by a simple interval (fifth, third, seventh, etc.)
    2. Comma shift (subtle pitch adjustment by syntonic/septimal comma)
    3. Combine two notes from the scale (like La Monte Young's 1323/1024)
    4. Stack intervals La Monte Young style (7/4 × 7/4 × 3/2 × 3/2...)
    5. Mediant - create ratio "between" two scale notes
    6. Random replacement from candidates (fallback)
    
    Respects config.required_notes (won't remove) and config.excluded_notes (won't add).
    """
    scale = scale.copy()
    scale_set = set(scale)
    limit = config.ratio_limit
    required_set = set(config.required_notes) if config.required_notes else set()
    excluded_set = set(config.excluded_notes) if config.excluded_notes else set()
    
    # Indices of notes that can be mutated (not required)
    mutable_indices = [i for i, note in enumerate(scale) if note not in required_set]
    if not mutable_indices:
        return scale  # All notes are required, can't mutate
    
    def try_add_note(new_note, idx):
        """Helper: replace note at idx if new_note is valid and not excluded."""
        if (new_note and 
            new_note not in scale_set and 
            new_note not in excluded_set and
            new_note != Fraction(1,1) and 
            new_note != Fraction(2,1)):
            scale[idx] = new_note
            return True
        return False
    
    # Choose mutation type
    r = random.random()
    
    if r < 0.18 and mutable_indices:
        # Transpose by simple interval
        idx = random.choice(mutable_indices)
        if try_add_note(mutate_transpose(scale[idx], limit), idx):
            return scale
    
    if r < 0.30 and mutable_indices:
        # Comma shift (subtle pitch adjustment)
        idx = random.choice(mutable_indices)
        if try_add_note(mutate_comma_shift(scale[idx], limit), idx):
            return scale
    
    if r < 0.48 and len(scale) >= 2 and mutable_indices:
        # Combine two scale notes
        idx = random.choice(mutable_indices)
        note1 = scale[idx]
        others = [n for n in scale if n != note1]
        note2 = random.choice(others) if others else note1
        if try_add_note(mutate_combine(note1, note2, limit), idx):
            return scale
    
    if r < 0.62 and mutable_indices:
        # La Monte Young style interval stacking
        idx = random.choice(mutable_indices)
        if try_add_note(mutate_stack(scale, limit), idx):
            return scale
    
    if r < 0.74 and len(scale) >= 2 and mutable_indices:
        # Mediant of two scale notes
        idx = random.choice(mutable_indices)
        note1 = scale[idx]
        others = [n for n in scale if n != note1]
        note2 = random.choice(others) if others else note1
        if try_add_note(mutate_mediant(note1, note2, limit), idx):
            return scale
    
    # Fallback: random replacement from candidates (excluding excluded notes)
    available = [c for c in candidates if c not in scale_set and c not in excluded_set]
    if mutable_indices and available:
        idx = random.choice(mutable_indices)
        scale[idx] = random.choice(available)
    
    return scale


def crossover_uniform(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Uniform crossover: combine both parents and randomly sample.
    Simple but doesn't preserve structure.
    Respects required_notes (always included) and excluded_notes (filtered out).
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = set(config.excluded_notes) if config.excluded_notes else set()
    
    # Combine parents, filter excluded, ensure required
    combined = (set(parent1) | set(parent2)) - excluded
    combined = combined | required  # Add back any required notes
    combined = list(combined)
    
    # Convert note count to ratio count: N notes = N-1 ratios
    if config.fixed_notes:
        num_ratios = config.fixed_notes - 1
        if len(combined) < num_ratios:
            return parent1.copy()
        # Must keep required, sample rest
        optional = [n for n in combined if n not in required]
        num_optional = num_ratios - len(required)
        if num_optional <= 0:
            return list(required)[:num_ratios]
        sampled = list(required) + random.sample(optional, min(num_optional, len(optional)))
        return sampled
    
    max_ratios = config.max_notes - 1
    min_ratios = config.min_notes - 1
    
    if len(combined) <= max_ratios:
        return combined
    
    # Must keep required, sample rest
    optional = [n for n in combined if n not in required]
    num_ratios = random.randint(min_ratios, max_ratios)
    num_optional = num_ratios - len(required)
    if num_optional <= 0:
        return list(required)[:num_ratios]
    sampled = list(required) + random.sample(optional, min(num_optional, len(optional)))
    return sampled


def crossover_blend(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Blend crossover: alternate taking notes from each parent (sorted by pitch).
    Preserves some structural properties from both parents.
    Respects required_notes and excluded_notes.
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = set(config.excluded_notes) if config.excluded_notes else set()
    
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Interleave from both parents
    child = list(required)  # Start with required notes
    i, j = 0, 0
    use_p1 = random.random() < 0.5
    
    target_len = len(parent1) if config.fixed_notes else random.randint(
        config.min_notes - 1, config.max_notes - 1)
    
    seen = set(required)
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
        
        if note not in seen and note not in excluded:
            child.append(note)
            seen.add(note)
        use_p1 = not use_p1
    
    return child if child else parent1.copy()


def crossover_pivot(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Pivot crossover: choose a pivot pitch, take lower notes from one parent
    and higher notes from the other.
    Respects required_notes and excluded_notes.
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = set(config.excluded_notes) if config.excluded_notes else set()
    
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Choose pivot point (in cents)
    pivot = random.uniform(200, 1000)  # Somewhere in the middle of the octave
    
    # Take notes below pivot from parent1, above from parent2 (or vice versa)
    if random.random() < 0.5:
        p1_sorted, p2_sorted = p2_sorted, p1_sorted
    
    child = list(required)  # Start with required notes
    seen = set(required)
    
    for note in p1_sorted:
        cents = 1200 * math.log2(float(note))
        if cents < pivot and note not in seen and note not in excluded:
            child.append(note)
            seen.add(note)
    
    for note in p2_sorted:
        cents = 1200 * math.log2(float(note))
        if cents >= pivot and note not in seen and note not in excluded:
            child.append(note)
            seen.add(note)
    
    # Adjust to target length
    target_len = (config.fixed_notes - 1) if config.fixed_notes else random.randint(
        config.min_notes - 1, config.max_notes - 1)
    
    if len(child) > target_len:
        # Must keep required, sample rest
        optional = [n for n in child if n not in required]
        num_optional = target_len - len(required)
        if num_optional <= 0:
            child = list(required)[:target_len]
        else:
            child = list(required) + random.sample(optional, min(num_optional, len(optional)))
    elif len(child) < target_len:
        # Fill from combined parents (excluding excluded notes)
        all_notes = (set(parent1) | set(parent2)) - excluded
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
    Respects required_notes and excluded_notes.
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = set(config.excluded_notes) if config.excluded_notes else set()
    
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
        
        child_set = (set(region1) | set(region2) | required) - excluded
        child = list(child_set)
    else:
        child_set = (set(parent1) | set(parent2) | required) - excluded
        child = list(child_set)
    
    # Adjust to target length
    target_len = (config.fixed_notes - 1) if config.fixed_notes else random.randint(
        config.min_notes - 1, config.max_notes - 1)
    
    if len(child) > target_len:
        # Must keep required, sample rest
        optional = [n for n in child if n not in required]
        num_optional = target_len - len(required)
        if num_optional <= 0:
            child = list(required)[:target_len]
        else:
            child = list(required) + random.sample(optional, min(num_optional, len(optional)))
    elif len(child) < target_len:
        all_notes = (set(parent1) | set(parent2)) - excluded
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
    
    Respects required_notes (won't modify) and excluded_notes (won't add).
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = set(config.excluded_notes) if config.excluded_notes else set()
    
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
    
    # Note constraints
    required_notes = config.required_notes if config.required_notes else []
    excluded_notes = set(config.excluded_notes) if config.excluded_notes else set()
    
    if verbose:
        print(f"Generated {len(candidates)} candidate ratios")
        if required_notes:
            print(f"Required notes: {', '.join(str(r) for r in required_notes)}")
        if excluded_notes:
            print(f"Excluded notes: {', '.join(str(r) for r in excluded_notes)}")
        print()
    
    # Lazy interval cache - compute on demand instead of precomputing all pairs
    # This avoids O(n²) precomputation which becomes prohibitive for large ratio limits
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
        """Add scale to population if it meets size and constraint requirements."""
        # Sanitize: ensure all ratios are within octave, no duplicates
        scale = sanitize_scale(scale)
        
        if not scale or not check_constraints(scale):
            return
        
        if target_len is not None:
            if len(scale) == target_len:
                fitness = eval_scale_cached(scale)
                population.append((fitness, list(scale)))
            elif len(scale) > target_len:
                # Randomly sample to target length, but keep required notes
                optional = [n for n in scale if n not in required_notes]
                num_optional = target_len - len(required_notes)
                if num_optional >= 0 and len(optional) >= num_optional:
                    trimmed = list(required_notes) + random.sample(optional, num_optional)
                    fitness = eval_scale_cached(trimmed)
                    population.append((fitness, trimmed))
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


def format_scale(scale: List[Fraction], verbose: bool = False) -> str:
    """Format a scale for display."""
    # Sanitize first to ensure valid output
    scale = sanitize_scale(scale)
    
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
                         edo: int = 0, combine_method: str = 'geometric'):
    """Evaluate and display a specific scale given as string ratios."""
    if metrics is None:
        metrics = [('tenney', 1.0)]
    if scale_metrics is None:
        scale_metrics = []
    
    scale = [Fraction(r) for r in ratios if r != "2/1"]
    fitness, details = evaluate_scale(scale, metrics, octave_range, None, scale_metrics, edo, combine_method)
    
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
    parser.add_argument("--temp", type=float, default=0.1, 
                        help="Initial SA temperature (default: 0.1, higher=more exploration)")
    parser.add_argument("--octaves", type=int, default=1, 
                        help="Octave range (default: 1 = 3 octaves, 0=single, 2=5 octaves)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="File with seed scales to initialize population (default: none)")
    parser.add_argument("--include", type=str, default=None,
                        help="Required notes (comma-separated ratios, e.g., '3/2,5/4')")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Excluded notes (comma-separated ratios, e.g., '7/4,11/8')")
    parser.add_argument("--combine", type=str, default='geometric',
                        help="How to combine interval scores: geometric (default), worst, mean, power2, power3, logloss")
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
            evaluate_named_scale(name, ratios, metrics, args.octaves, scale_metrics, edo, args.combine)
        return
    
    # Parse required/excluded notes
    required_notes = parse_ratios(args.include) if args.include else []
    excluded_notes = parse_ratios(args.exclude) if args.exclude else []
    
    config = SearchConfig(
        max_notes=args.max_notes,
        min_notes=args.min_notes,
        fixed_notes=args.notes,
        ratio_limit=args.ratio_limit,
        generations=args.generations,
        population_size=args.population,
        metrics=metrics,
        scale_metrics=scale_metrics,
        initial_temp=args.temp,
        octave_range=args.octaves,
        edo=edo,
        required_notes=required_notes,
        excluded_notes=excluded_notes,
        combine_method=args.combine,
    )
    
    # Load seed scales if specified (with names for matching)
    seed_scales = None
    seed_names = None
    if args.seeds:
        seed_scales, seed_names = load_seed_scales(args.seeds, with_names=True)
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
    
    # Build combine method description
    combine_desc_map = {
        'geometric': 'geometric mean', 
        'worst': 'worst interval', 
        'mean': 'arithmetic mean',
        'logloss': 'log loss (heavily penalizes bad intervals)'
    }
    if config.combine_method.startswith('power'):
        try:
            p = int(config.combine_method[5:]) if len(config.combine_method) > 5 else 2
        except ValueError:
            p = 2
        combine_desc_str = f'power mean (p={p})'
    else:
        combine_desc_str = combine_desc_map.get(config.combine_method, config.combine_method)
    
    print("=" * 60)
    print("Just Intonation Scale Search")
    print("=" * 60)
    print(f"Notes: {note_desc}")
    print(f"Ratio limit: {config.ratio_limit}")
    print(f"Octave range: {octave_desc}")
    print(f"Metric: {metric_desc}")
    print(f"Interval combination: {combine_desc_str}")
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
    print("RESULTS")
    print("=" * 60)
    
    # Summary
    print(f"\nQuality: {details['quality']:.4f}")
    print(f"Intervals evaluated: {details['num_intervals']} (across {octave_desc})")
    
    # Check if this scale matches a seed
    if seed_scales:
        match_idx, match_name = find_matching_seed(scale, seed_scales, seed_names)
        if match_idx >= 0:
            if match_name:
                print(f"Matches seed #{match_idx + 1}: {match_name}")
            else:
                print(f"Matches seed #{match_idx + 1}")
    
    # Show individual metric scores (normalized to 0-1, as quality: higher=better)
    if len(metrics) > 1 or scale_metrics:
        print("\nMetric breakdown (0-1, higher=better):")
        for metric_name, weight in metrics:
            raw = details['metric_averages'].get(metric_name, 0)
            min_val, max_val = METRIC_NORMALIZATION.get(metric_name, (0, 1))
            normalized = (raw - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
            quality = 1.0 - normalized  # Invert: lower raw = higher quality
            print(f"  {metric_name}: {quality:.3f} (weight {weight})")
        
        for metric_name, weight in scale_metrics:
            raw = details['metric_averages'].get(metric_name, 0)
            min_val, max_val = METRIC_NORMALIZATION.get(metric_name, (0, 1))
            normalized = (raw - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
            # 'primes' and 'variety' are positive (higher=better), others are penalties
            if metric_name in ('primes', 'variety'):
                quality = normalized
            else:
                quality = 1.0 - normalized
            print(f"  {metric_name}: {quality:.3f} (weight {weight})")
    
    # Interval matrix
    print("\nInterval matrix (single octave):")
    print()
    print_interval_matrix(scale, metrics)
    
    # The scale itself - just ratios for easy copying
    print("\n" + "=" * 60)
    print("SCALE")
    print("=" * 60)
    print()
    print(format_scale(scale))


if __name__ == "__main__":
    main()
