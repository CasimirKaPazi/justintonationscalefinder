"""CLI entry point for scale search. Run with: python -m scale_search"""

import random
import argparse

from scale_search.config import SearchConfig
from scale_search.util import parse_ratios
from scale_search.metrics import (
    CONSONANCE_METRICS, SCALE_METRICS,
    parse_metrics, get_normalization, init_normalization,
    format_metric_description,
)
from scale_search.evolution import evolve
from scale_search.display import (
    load_seed_scales, find_matching_seed,
    format_scale, print_interval_matrix, evaluate_named_scale,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary search for just intonation scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --notes 7 --metric barlow
  %(prog)s --notes 7 --metric "barlow:1,roughness:0.5"
  %(prog)s --notes 12 --metric "soft_tenney:1,primes:0.5" --ratio-limit 64
  %(prog)s --notes 12 --metric "tenney:1,edo12:2"  # neji scale
  %(prog)s --seeds my_scales.txt --notes 7  # use seed scales from file
  %(prog)s --compare --metric barlow
  %(prog)s --notes 7 --quiet --random-seed 42  # reproducible, script-friendly

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
  --ratio-limit: 64 (allows up to 63-limit intervals)
  --generations: 500
  --population: 300
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
    parser.add_argument("--ratio-limit", type=int, default=64, 
                        help="Max numerator/denominator (default: 64)")
    parser.add_argument("--generations", type=int, default=500, 
                        help="Number of generations (default: 500)")
    parser.add_argument("--population", type=int, default=300, 
                        help="Population size (default: 300)")
    parser.add_argument("--runs", type=int, default=1, 
                        help="Number of independent runs (default: 1)")
    parser.add_argument("--metric", type=str, default='barlow', 
                        help="Metric(s) to use. Single: 'barlow'. Combined: 'barlow:1,roughness:0.5'")
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
    parser.add_argument("--random-seed", type=int, default=None, metavar="N",
                        help="Random seed for reproducible results")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output (only show results)")
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
    
    # Set random seed for reproducibility
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    # Compute normalization ranges for the given ratio limit
    init_normalization(args.ratio_limit)
    
    if args.compare:
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
    
    verbose = not args.quiet
    
    # Load seed scales if specified (with names for matching)
    seed_scales = None
    seed_names = None
    if args.seeds:
        seed_scales, seed_names = load_seed_scales(args.seeds, with_names=True)
        if seed_scales and verbose:
            print(f"Loaded {len(seed_scales)} seed scales from {args.seeds}")
    
    # Build descriptions
    metric_desc = format_metric_description(metrics, scale_metrics, edo)
    note_desc = f"exactly {config.fixed_notes}" if config.fixed_notes is not None else f"{config.min_notes}-{config.max_notes}"
    octave_desc = f"{2 * config.octave_range + 1} octaves" if config.octave_range > 0 else "1 octave"
    
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
    
    if verbose:
        print("=" * 60)
        print("Just Intonation Scale Search")
        print("=" * 60)
        print(f"Notes: {note_desc}")
        print(f"Ratio limit: {config.ratio_limit}")
        print(f"Octave range: {octave_desc}")
        print(f"Metric: {metric_desc}")
        print(f"Interval combination: {combine_desc_str}")
        print(f"Generations: {config.generations}")
        if args.random_seed is not None:
            print(f"Random seed: {args.random_seed}")
        print("=" * 60)
        print()
    
    best_overall = None
    
    for run in range(args.runs):
        if args.runs > 1 and verbose:
            print(f"\n--- Run {run + 1}/{args.runs} ---\n")
        
        scale, fitness, details = evolve(config, verbose=verbose, seed_scales=seed_scales)
        
        if best_overall is None or fitness > best_overall[1]:
            best_overall = (scale, fitness, details)
    
    scale, fitness, details = best_overall
    
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
    
    # Summary
    print(f"\nQuality: {details['quality']:.4f}")
    if verbose:
        print(f"Intervals evaluated: {details['num_intervals']} (across {octave_desc})")
    
    # Check if this scale matches a seed
    if seed_scales:
        match_idx, match_name = find_matching_seed(scale, seed_scales, seed_names)
        if match_idx >= 0:
            if match_name:
                print(f"Matches seed #{match_idx + 1}: {match_name}")
            else:
                print(f"Matches seed #{match_idx + 1}")
    
    if verbose:
        # Show individual metric scores
        if len(metrics) > 1 or scale_metrics:
            print("\nMetric breakdown (0-1, higher=better):")
            for metric_name, weight in metrics:
                raw = details['metric_averages'].get(metric_name, 0)
                min_val, max_val = get_normalization(metric_name)
                normalized = (raw - min_val) / (max_val - min_val)
                normalized = max(0.0, min(1.0, normalized))
                quality = 1.0 - normalized
                print(f"  {metric_name}: {quality:.3f} (weight {weight})")
            
            for metric_name, weight in scale_metrics:
                raw = details['metric_averages'].get(metric_name, 0)
                min_val, max_val = get_normalization(metric_name)
                normalized = (raw - min_val) / (max_val - min_val)
                normalized = max(0.0, min(1.0, normalized))
                if metric_name in ('primes', 'variety'):
                    quality = normalized
                else:
                    quality = 1.0 - normalized
                print(f"  {metric_name}: {quality:.3f} (weight {weight})")
        
        # Interval matrix
        print("\nInterval matrix (single octave):")
        print()
        print_interval_matrix(scale, metrics)
    
    # The scale itself
    if verbose:
        print("\n" + "=" * 60)
        print("SCALE")
        print("=" * 60)
    print()
    print(format_scale(scale))


if __name__ == "__main__":
    main()
