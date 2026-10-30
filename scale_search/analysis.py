"""
Post-hoc analysis: what does a scale resemble?

Two questions, both answered in cents rather than by ratio identity. Exact
ratio matching is too brittle to be useful here — 9/8 and 10/9 are different
ratios but only 21 cents apart, and a scale built from one is recognisably the
same object as a scale built from the other.

    nearest_scales   how close is this to scales I already have?
    edo_fits         which equal division does this sit on, and how well?
"""

import math
from fractions import Fraction
from typing import List, Optional, Tuple

__all__ = [
    'scale_cents', 'compare_scales', 'nearest_scales', 'edo_fits',
]


def scale_cents(scale: List[Fraction]) -> List[float]:
    """Pitch classes of a scale in cents, including the tonic at 0."""
    out = [0.0]
    for r in scale:
        c = 1200.0 * math.log2(float(r))
        c %= 1200.0
        if c > 1e-9:
            out.append(c)
    return sorted(set(round(c, 6) for c in out))


def _rotations(cents: List[float]) -> List[Tuple[int, List[float]]]:
    """
    Every mode of a scale, as (which degree became the tonic, cents).

    A scale and its modes are the same set of pitch classes read from different
    starting points, so a result can be a known scale without matching it
    degree for degree.
    """
    out = []
    n = len(cents)
    for k in range(n):
        shifted = sorted((c - cents[k]) % 1200.0 for c in cents)
        out.append((k, shifted))
    return out


def compare_scales(a: List[float], b: List[float],
                   tolerance: float = 25.0) -> Optional[dict]:
    """
    Compare two scales given as cents, trying every mode of b.

    Equal note counts are paired degree by degree. Unequal counts are scored by
    coverage instead: how many of a's notes land within tolerance of some note
    of b. Returns None if nothing matches at all.

    'relation' says which way any containment runs: 'same' for equal sizes,
    'within' when every note of a lies in the larger b, 'contains' when every
    note of the smaller b lies in a, and 'partial' otherwise.
    """
    def circ(x, y):
        """Signed distance from y to x, the short way round the octave."""
        d = x - y
        return d - 1200.0 * round(d / 1200.0)

    best = None

    for degree, rot in _rotations(b):
        if len(a) == len(rot):
            devs = [x - y for x, y in zip(a, rot)]
            max_dev = max(abs(d) for d in devs)
            rms = math.sqrt(sum(d * d for d in devs) / len(devs))
            matched, total = len(a), len(a)
            relation = 'same'
        else:
            devs = [min((circ(x, y) for y in rot), key=abs) for x in a]
            within = [d for d in devs if abs(d) <= tolerance]
            if not within:
                continue
            matched, total = len(within), len(a)
            max_dev = max(abs(d) for d in within)
            rms = math.sqrt(sum(d * d for d in within) / len(within))
            b_covered = sum(1 for y in rot
                            if any(abs(circ(x, y)) <= tolerance for x in a))
            if len(a) < len(rot) and matched == len(a):
                relation = 'within'
            elif len(a) > len(rot) and b_covered == len(rot):
                relation = 'contains'
            else:
                relation = 'partial'

        cand = {
            'mode': degree,
            'max_dev': max_dev,
            'rms': rms,
            'matched': matched,
            'total': total,
            'relation': relation,
            'same_length': len(a) == len(rot),
            # Identical only if it is the same scale, not merely contained
            # in a larger one: 7 notes all landing inside a 12-note scale is
            # a subset relation, and calling it exact would hide that.
            'exact': (max_dev < 1e-6 and matched == total
                      and len(a) == len(rot)),
        }
        # Prefer full-length matches, then more coverage, then tighter fit
        key = (not cand['same_length'], -cand['matched'], cand['max_dev'])
        if best is None or key < (not best['same_length'], -best['matched'],
                                  best['max_dev']):
            best = cand

    return best


def nearest_scales(scale: List[Fraction], others: List[List[Fraction]],
                   names: List[str] = None, top: int = 5,
                   tolerance: float = 25.0,
                   sources: List[str] = None) -> List[dict]:
    """Rank known scales by how closely they match this one."""
    a = scale_cents(scale)
    results = []
    for i, other in enumerate(others):
        b = scale_cents(other)
        if not b:
            continue
        cmp = compare_scales(a, b, tolerance)
        if cmp is None:
            continue
        cmp['index'] = i
        cmp['name'] = (names[i] if names and i < len(names) and names[i]
                       else f"Scale #{i + 1}")
        cmp['notes'] = len(b)
        cmp['source'] = (sources[i] if sources and i < len(sources) else None)
        results.append(cmp)

    results.sort(key=lambda r: (not r['same_length'], -r['matched'], r['max_dev']))
    return results[:top]


def edo_fits(scale: List[Fraction], min_edo: int = 5, max_edo: int = 72,
             top: int = 5) -> List[dict]:
    """
    Rank equal divisions by how well they approximate this scale.

    An EDO only counts as a fit if every note lands on a distinct step: a
    12-note scale cannot sit in 5-EDO, and letting two notes collapse onto one
    step would otherwise report a spurious near-perfect match.

    Ranked by worst-note deviation *relative to the step size*, not by absolute
    cents. Absolute error always favours large divisions — 53-EDO beats 12-EDO
    on any scale simply by having finer steps — which makes the ranking say
    nothing except "more steps is better". Relative error asks instead whether
    the division actually explains the scale: a random scale scores about 0.5
    in any EDO, so a low score means the notes genuinely sit on the grid.
    """
    cents = scale_cents(scale)
    fits = []

    for edo in range(min_edo, max_edo + 1):
        step = 1200.0 / edo
        steps = [round(c / step) % edo for c in cents]
        if len(set(steps)) != len(steps):
            continue  # Two notes would share a step

        devs = [c - (s * step) for c, s in zip(cents, steps)]
        devs = [d - 1200 * round(d / 1200) for d in devs]
        max_dev = max(abs(d) for d in devs)
        rms = math.sqrt(sum(d * d for d in devs) / len(devs))
        fits.append({
            'edo': edo,
            'max_dev': max_dev,
            'rms': rms,
            'rel': max_dev / (step / 2),   # 0 = on the grid, 1 = worst possible
            'steps': steps,
            'step_size': step,
        })

    # Smaller EDO wins ties, being the simpler explanation of the same fit
    fits.sort(key=lambda f: (round(f['rel'], 6), f['edo']))
    return fits[:top]
