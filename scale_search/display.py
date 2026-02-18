"""Display, formatting, and seed scale I/O."""

import math
from fractions import Fraction
from typing import List, Tuple, Optional

from scale_search.util import sanitize_scale, interval_between
from scale_search.metrics import (
    CONSONANCE_METRICS, SCALE_METRICS,
    combined_metric, get_normalization, format_metric_description,
)
from scale_search.evolution import evaluate_scale

__all__ = [
    'load_seed_scales', 'find_matching_seed',
    'format_scale', 'print_interval_matrix',
    'evaluate_named_scale', 'print_scale_details',
]

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
    
    def flush_scale():
        """Save current_scale if non-empty, reset state."""
        nonlocal current_scale, current_name
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
    
    try:
        with open(filepath, 'r') as f:
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
            flush_scale()
    
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


def format_scale(scale: List[Fraction], verbose: bool = False) -> str:
    """Format a scale for display."""
    # Sanitize first to ensure valid output
    scale = sanitize_scale(scale)
    
    lines = []
    for r in sorted(scale):
        if verbose:
            cents = 1200 * math.log2(float(r))
            lines.append(f"{r.numerator}/{r.denominator}  ({cents:.1f} cents)")
        else:
            lines.append(f"{r.numerator}/{r.denominator}")
    if verbose:
        lines.append(f"2/1  (1200.0 cents)")
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


def evaluate_named_scale(scale_name: str, ratios: List[str], metrics: List[Tuple[str, float]] = None,
                         octave_range: int = 1, scale_metrics: List[Tuple[str, float]] = None,
                         edo: int = 0, combine_method: str = 'geometric') -> Tuple[float, dict, List]:
    """Evaluate a specific scale given as string ratios and return results.
    
    Note: call init_normalization(ratio_limit) before using this function
    to ensure metric normalization ranges are set for the relevant ratio limit.
    
    Returns:
        (quality, details, scale) where scale is a list of Fractions.
    """
    if metrics is None:
        metrics = [('tenney', 1.0)]
    if scale_metrics is None:
        scale_metrics = []
    
    scale = [Fraction(r) for r in ratios if r != "2/1"]
    fitness, details = evaluate_scale(scale, metrics, octave_range, None, scale_metrics, edo, combine_method)
    
    return fitness, details, scale


def print_scale_details(scale_name: str, scale: List[Fraction], details: dict,
                        metrics: List[Tuple[str, float]] = None,
                        scale_metrics: List[Tuple[str, float]] = None,
                        edo: int = 0):
    """Print detailed evaluation of a scale."""
    if metrics is None:
        metrics = [('tenney', 1.0)]
    if scale_metrics is None:
        scale_metrics = []
    
    # Build metric description
    metric_desc = format_metric_description(metrics, scale_metrics, edo)
    octave_range = details.get('octave_range', 1)
    
    print(f"\n{'=' * 60}")
    print(f"Scale: {scale_name}")
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


