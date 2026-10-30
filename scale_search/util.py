"""Utility functions for ratio manipulation and scale operations."""

import math
from fractions import Fraction
from typing import List, Set, Tuple

def snap_to_candidates(scale, candidates, candidate_cents=None,
                       tolerance: float = 20.0, max_shift: float = 60.0):
    """
    Replace each ratio with a ratio available under the search's complexity
    limit, measured in cents.

    Needed to use the Scala archive for seeding. Roughly half the archive is
    written in cents rather than ratios, and converting those faithfully yields
    things like 720009/679598 — exact enough to measure against, but useless as
    a starting point for a just-intonation search, since no mutation of it means
    anything. Snapping turns such a scale into the nearest thing the search can
    actually express, which is a useful operation in its own right: it asks what
    the just interpretation of an archive scale would be.

    Within `tolerance` cents the SIMPLEST candidate wins, not the closest. The
    nearest ratio to 12-EDO's 400-cent third under a limit of 32 is 29/23, which
    is a worse seed than 5/4 despite being 13 cents closer — the point of a seed
    is to start the search somewhere meaningful, not to reproduce the original
    exactly. Outside the tolerance the nearest candidate is used instead.

    Rational scales already inside the limit are returned unchanged, so this is
    harmless to apply to everything.

    Returns None if two notes land on the same candidate — the scale would
    quietly shrink — or if any note has to move further than max_shift cents,
    which means the limit cannot represent it at all.
    """
    import bisect as _bisect
    if candidate_cents is None:
        order = sorted(range(len(candidates)),
                       key=lambda i: float(candidates[i]))
        candidates = [candidates[i] for i in order]
        candidate_cents = [1200.0 * math.log2(float(c)) for c in candidates]

    out = []
    seen = set()
    for r in scale:
        c = 1200.0 * math.log2(float(r))

        lo = _bisect.bisect_left(candidate_cents, c - max_shift)
        hi = _bisect.bisect_right(candidate_cents, c + max_shift)
        if lo >= hi:
            return None

        near = [(abs(candidate_cents[j] - c), j) for j in range(lo, hi)]
        within = [(candidates[j].numerator * candidates[j].denominator, d, j)
                  for d, j in near if d <= tolerance]
        if within:
            best = candidates[min(within)[2]]      # Simplest inside tolerance
        else:
            best = candidates[min(near)[1]]        # Otherwise the closest

        if best in seen:
            return None
        seen.add(best)
        out.append(best)

    return sorted(out)


__all__ = ['snap_to_candidates', 
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


def sanitize_scale(scale: List[Fraction], preserve_order: bool = False) -> List[Fraction]:
    """
    Ensure all ratios in a scale are valid (1 < r < 2).
    
    - Reduces any ratio >= 2 to the octave
    - Removes 1/1 and 2/1 (these are implicit)
    - Removes duplicates (keeping the first occurrence)
    - Sorts by pitch, unless preserve_order is set
    
    preserve_order is for grouped (slot-order) genomes, where position carries
    meaning — a note's slot decides which sub-scales it belongs to, so sorting
    would scramble the structure. Note that dropping duplicates can shorten the
    genome; grouped callers repair the length afterwards.
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
    
    if preserve_order:
        return result
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
