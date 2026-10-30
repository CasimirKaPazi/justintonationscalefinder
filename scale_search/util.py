"""Utility functions for ratio manipulation and scale operations."""

import bisect
import math
from fractions import Fraction
from typing import List, Optional, Tuple

__all__ = [
    'prime_factors', 'parse_ratios', 'reduce_to_octave', 'sanitize_scale',
    'is_valid_ratio', 'interval_between', 'generate_candidate_ratios',
    'snap_to_candidates',
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

    Raises ValueError naming the offending entry for anything that isn't a
    positive ratio n/d or integer n. Cents are rejected here: this is used for
    --include and --exclude, which match notes exactly, and a cents value only
    ever approximates a ratio.
    """
    if not ratio_str or not ratio_str.strip():
        return []

    ratios = []
    for part in ratio_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '.' in part:
            raise ValueError(
                f"'{part}' looks like cents; give an exact ratio such as 3/2")
        try:
            if '/' in part:
                n, d = part.split('/')
                value = Fraction(int(n.strip()), int(d.strip()))
            else:
                value = Fraction(int(part), 1)
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"'{part}' is not a ratio (expected n/d or n)") from None
        if value <= 0:
            raise ValueError(f"'{part}' is not a positive ratio")
        ratios.append(value)
    return ratios


def reduce_to_octave(ratio: Fraction) -> Fraction:
    """Reduce a ratio to within one octave (1 <= r < 2)."""
    if ratio <= 0:
        # Without this guard the loop below would never terminate on 0.
        raise ValueError(f"cannot octave-reduce non-positive ratio {ratio}")
    while ratio >= 2:
        ratio = ratio / 2
    while ratio < 1:
        ratio = ratio * 2
    return ratio


def sanitize_scale(scale: List[Fraction], preserve_order: bool = False) -> List[Fraction]:
    """
    Ensure all ratios in a scale are valid (1 < r < 2).

    - Reduces every ratio to the octave
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
        r = reduce_to_octave(r)
        if r == 1 or r in seen:
            continue
        seen.add(r)
        result.append(r)

    if preserve_order:
        return result
    return sorted(result)


def is_valid_ratio(ratio: Fraction, limit: int) -> bool:
    """Check if ratio is within the complexity limit."""
    return ratio.numerator <= limit and ratio.denominator <= limit


def interval_between(r1: Fraction, r2: Fraction) -> Fraction:
    """Get the interval between two ratios (as a ratio within one octave)."""
    if r1 > r2:
        r1, r2 = r2, r1
    return reduce_to_octave(r2 / r1)


def generate_candidate_ratios(limit: int) -> Tuple[List[Fraction], List[float]]:
    """
    Generate all ratios n/d where 1 < n/d < 2 and n,d <= limit, in lowest terms.
    Returns (ratios, weights) where weights are Zipf-biased toward simpler ratios.
    Weight = 1 / (n * d) gives strong preference to low-complexity ratios.
    """
    ratios = set()
    for d in range(1, limit + 1):
        for n in range(d + 1, min(2 * d, limit + 1)):
            ratios.add(Fraction(n, d))  # Fraction reduces to lowest terms

    sorted_ratios = sorted(ratios)
    weights = [1.0 / (r.numerator * r.denominator) for r in sorted_ratios]

    return sorted_ratios, weights


def snap_to_candidates(scale: List[Fraction], candidates: List[Fraction],
                       candidate_cents: List[float] = None,
                       tolerance: float = 20.0,
                       max_shift: float = 60.0) -> Optional[List[Fraction]]:
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

    `candidates` must be sorted by pitch if `candidate_cents` is given.
    Rational scales already inside the limit are returned unchanged, so this is
    harmless to apply to everything.

    Returns None if two notes land on the same candidate — the scale would
    quietly shrink — or if any note has to move further than max_shift cents,
    which means the limit cannot represent it at all.
    """
    if candidate_cents is None:
        candidates = sorted(candidates)
        candidate_cents = [1200.0 * math.log2(float(c)) for c in candidates]

    out = []
    seen = set()
    for r in scale:
        c = 1200.0 * math.log2(float(r))

        lo = bisect.bisect_left(candidate_cents, c - max_shift)
        hi = bisect.bisect_right(candidate_cents, c + max_shift)
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
