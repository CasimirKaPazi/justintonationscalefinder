"""Utility functions for ratio manipulation and scale operations."""

import math
from fractions import Fraction
from typing import List, Set, Tuple

__all__ = [
    'prime_factors', 'parse_ratios', 'reduce_to_octave', 'sanitize_scale',
    'is_valid_ratio', 'interval_between', 'generate_candidate_ratios',
]


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
            ratios.append(Fraction(int(part.strip()), 1))
    return ratios


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
        while r >= 2:
            r = r / 2
        while r < 1:
            r = r * 2
        
        if r == Fraction(1, 1) or r == Fraction(2, 1):
            continue
        
        key = (r.numerator, r.denominator)
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    
    return sorted(result, key=lambda x: float(x))


def is_valid_ratio(ratio: Fraction, limit: int) -> bool:
    """Check if ratio is within the complexity limit."""
    return ratio.numerator <= limit and ratio.denominator <= limit


def interval_between(r1: Fraction, r2: Fraction) -> Fraction:
    """Get the interval between two ratios (as a ratio within one octave)."""
    if r1 > r2:
        r1, r2 = r2, r1
    interval = r2 / r1
    while interval >= 2:
        interval = interval / 2
    while interval < 1:
        interval = interval * 2
    return interval


def generate_candidate_ratios(limit: int) -> Tuple[List[Fraction], List[float]]:
    """
    Generate all ratios n/d where 1 < n/d < 2 and n,d <= limit, in lowest terms.
    Returns (ratios, weights) where weights are Zipf-biased toward simpler ratios.
    Weight = 1 / (n * d) gives strong preference to low-complexity ratios.
    """
    ratios = set()
    for d in range(1, limit + 1):
        for n in range(d + 1, 2 * d):
            if n <= limit:
                g = math.gcd(n, d)
                ratios.add(Fraction(n // g, d // g))
    
    sorted_ratios = sorted(ratios)
    weights = [1.0 / (r.numerator * r.denominator) for r in sorted_ratios]
    
    return sorted_ratios, weights
