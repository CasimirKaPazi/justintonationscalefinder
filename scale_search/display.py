"""Display, formatting, and seed scale I/O."""

import math
from fractions import Fraction
from typing import List, Optional, Tuple

from scale_search.util import sanitize_scale, interval_between
from scale_search.metrics import (
    combined_metric, quantile_normalize, format_metric_description, scale_metric_quality,
)
from scale_search.evolution import evaluate_scale

__all__ = [
    'load_seed_scales', 'find_matching_seed',
    'format_scale', 'print_interval_matrix',
    'evaluate_named_scale', 'print_scale_details',
    'metric_breakdown', 'parse_pitches',
]


def _parse_seed_line(line: str) -> Optional[Fraction]:
    """
    One pitch from a seed-file line: the first token as n/d or n.

    Only the first token is read, so a trailing comment ("9/8  major second")
    no longer makes the line silently disappear. Cents (a decimal point) are
    skipped, since a seed is meant to be an exact ratio.
    """
    token = line.split()[0]
    if '.' in token:
        return None
    try:
        if '/' in token:
            n, d = token.split('/')
            ratio = Fraction(int(n), int(d))
        else:
            ratio = Fraction(int(token), 1)
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if ratio > 0 else None


def load_seed_scales(filepath: str, with_names: bool = False,
                     preserve_order: bool = False):
    """
    Load seed scales from a file. Supports both simple format and Scala .scl format.

    Comment styles:
    - Lines starting with # are comments (can be used as scale names)
    - Lines starting with ! are comments (Scala format)
    - Blank lines separate scales (in simple format)

    Ratio formats:
    - n/d (e.g., 9/8, 5/4, 3/2)
    - n (e.g., 2 means 2/1)
    - anything after the first token on a line is ignored

    Notes:
    - 2/1 (or just 2) at the end of a scale is removed (it's implicit)
    - Ratios outside the octave (like 3/1, 9/4) are reduced to the octave
    - Cents values (containing '.') are skipped
    - A file ending in .scl is read with the dedicated Scala parser, which
      handles the count line, cents and the description properly

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

    Args:
        filepath: Path to seed file
        with_names: If True, return (scales, names) tuple
        preserve_order: Keep each scale's notes in file order instead of
            sorting them by pitch, for grouped seeds written in slot order

    Returns:
        If with_names=False: List of scales (each scale is a list of Fractions)
        If with_names=True: Tuple of (scales, names) where names[i] is name for scales[i] or None
    """
    if filepath.lower().endswith('.scl'):
        # Imported here: scala.py imports this module for its own seed-file case
        from scale_search.scala import parse_scl
        s = parse_scl(filepath)
        scales = [sanitize_scale(s.ratios)] if s and s.ratios else []
        names = [s.name] if scales else []
        return (scales, names) if with_names else scales

    scales = []
    names = []
    current_scale = []
    current_name = None

    def flush_scale():
        """Save current_scale if non-empty, reset state."""
        nonlocal current_scale, current_name
        if current_scale:
            if current_scale[-1] == Fraction(2, 1):
                current_scale = current_scale[:-1]
            if current_scale:
                current_scale = sanitize_scale(current_scale, preserve_order)
                if current_scale:
                    scales.append(current_scale)
                    names.append(current_name)
        current_scale = []
        current_name = None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()

                if not line:
                    # Blank line = delimiter: flush scale and reset name
                    flush_scale()
                    continue

                if line.startswith('#'):
                    # # comment: also acts as delimiter (backward compat), then sets name
                    flush_scale()
                    current_name = line[1:].strip() or None
                    continue

                if line.startswith('!'):
                    # ! comment (Scala style): first non-empty one becomes the name
                    # If mid-scale, skip (e.g. the "!" separator line in .scl format)
                    if current_scale:
                        continue
                    comment_text = line[1:].strip()
                    if comment_text and current_name is None:
                        if comment_text.lower().endswith('.scl'):
                            comment_text = comment_text[:-4]
                        current_name = comment_text
                    continue

                ratio = _parse_seed_line(line)
                if ratio is None:
                    continue
                bare_integer = '/' not in line.split()[0]
                if bare_integer and not current_scale and ratio > 4:
                    # A bare integer above 4 opening a block is the note-count
                    # line of an embedded .scl block, not a pitch. (2, 3 and 4
                    # are ambiguous and, as always, are read as pitches.)
                    continue
                if ratio > 1:
                    current_scale.append(ratio)  # Octave-reduced in flush_scale

            # Don't forget the last scale
            flush_scale()

    except FileNotFoundError:
        print(f"Warning: Seed file '{filepath}' not found.")
        return ([], []) if with_names else []
    except OSError as e:
        print(f"Warning: Error reading seed file '{filepath}': {e}")
        return ([], []) if with_names else []

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


def parse_pitches(text: str) -> List[Fraction]:
    """
    Parse a comma-separated list of pitches given as ratios or cents.

    Used for --compare, where checking a scale written in cents (an equal
    temperament, a measured tuning) is as reasonable as checking one written
    in ratios. Cents become the ratio they denote, to well under a thousandth
    of a cent. Raises ValueError naming the entry it cannot read.
    """
    from scale_search.scala import _parse_pitch  # Same rules as .scl files
    out = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        p = _parse_pitch(part)
        if p is None:
            raise ValueError(f"'{part}' is not a ratio (n/d or n) or a cents value")
        out.append(p)
    return out


def format_scale(scale: List[Fraction], verbose: bool = False) -> str:
    """Format a scale for display."""
    lines = []
    for r in sanitize_scale(scale):
        if verbose:
            cents = 1200 * math.log2(float(r))
            lines.append(f"{r.numerator}/{r.denominator}  ({cents:.1f} cents)")
        else:
            lines.append(f"{r.numerator}/{r.denominator}")
    lines.append("2/1  (1200.0 cents)" if verbose else "2/1")
    return "\n".join(lines)


def print_interval_matrix(scale: List[Fraction], metrics: List[Tuple[str, float]] = None):
    """
    Print the interval matrix for a scale: the combined penalty of each
    interval, 0 = most consonant, as the search scores it.
    """
    if metrics is None:
        metrics = [('barlow', 1.0)]

    full_scale = [Fraction(1, 1)] + sanitize_scale(scale) + [Fraction(2, 1)]

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


def metric_breakdown(details: dict, metrics: List[Tuple[str, float]],
                     scale_metrics: List[Tuple[str, float]],
                     edo: int = 0) -> List[Tuple[str, float, float]]:
    """
    Per-metric quality (0-1, higher = better) for display, as
    (label, quality, weight) triples.

    Interval metrics are shown the way the search scores them: each interval
    is ranked among all candidate ratios under the ratio limit (0 = best
    possible, 1 = worst), and the quality is one minus the average rank over
    every interval evaluated, across all octaves (each interval class counted
    as often as it occurs). The breakdown used to place
    the metric's average raw value on a straight line between the best and
    worst candidate instead, which for skewed metrics told a different story
    than the one the search acted on — Tenney looked like a weak point of the
    Ptolemaic major scale (0.74) while the search saw it as nearly perfect.

    Scale metrics have no candidate population to rank against, so they are
    shown as the search scores them, by scale_metric_quality.

    The quality shown is the metric's own, whatever the sign of its weight —
    a negatively weighted metric is one the search was asked to make worse.
    """
    values = details.get('metric_values', {})
    rows = []
    for name, weight in metrics:
        per_interval = values.get(name)
        if per_interval:
            counts = details.get('interval_counts') or [1] * len(per_interval)
            rank = (sum(c * quantile_normalize(v, name) for v, c in zip(per_interval, counts))
                    / sum(counts))
        else:
            rank = quantile_normalize(details['metric_averages'].get(name, 0), name)
        rows.append((name, 1.0 - rank, weight))
    for name, weight in scale_metrics:
        quality = scale_metric_quality(name, details['metric_averages'].get(name, 0))
        label = f"edo{edo}" if name == 'edo' else name
        rows.append((label, quality, weight))
    return rows


def evaluate_named_scale(scale_name: str, ratios: List, metrics: List[Tuple[str, float]] = None,
                         octave_range: int = 1, scale_metrics: List[Tuple[str, float]] = None,
                         edo: int = 0, combine_method: str = 'geometric') -> Tuple[float, dict, List]:
    """Evaluate a specific scale given as ratios (Fractions or strings).

    Note: call init_normalization(ratio_limit) before using this function
    to ensure metric normalization ranges are set for the relevant ratio limit.

    Returns:
        (quality, details, scale) where scale is a sorted list of Fractions
        with 1/1 and 2/1 removed.
    """
    if metrics is None:
        metrics = [('barlow', 1.0)]
    if scale_metrics is None:
        scale_metrics = []

    # sanitize_scale drops 2/1 however it is written ("2/1", "2", " 2/1 ")
    scale = sanitize_scale([Fraction(r) for r in ratios])
    fitness, details = evaluate_scale(scale, metrics, octave_range, None, scale_metrics, edo, combine_method)

    return fitness, details, scale


def print_scale_details(scale_name: str, scale: List[Fraction], details: dict,
                        metrics: List[Tuple[str, float]] = None,
                        scale_metrics: List[Tuple[str, float]] = None,
                        edo: int = 0):
    """Print detailed evaluation of a scale."""
    if metrics is None:
        metrics = [('barlow', 1.0)]
    if scale_metrics is None:
        scale_metrics = []

    # Build metric description
    metric_desc = format_metric_description(metrics, scale_metrics, edo)
    octave_range = details.get('octave_range', 1)

    print(f"\n{'=' * 60}")
    print(f"Scale: {scale_name}")
    print("=" * 60)
    print()
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

    # Same breakdown as after a search, so the numbers can be compared
    if len(metrics) > 1 or scale_metrics:
        print("\nMetric breakdown (0-1, higher=better):")
        for label, quality, weight in metric_breakdown(details, metrics, scale_metrics, edo):
            print(f"  {label}: {quality:.3f} (weight {weight})")
