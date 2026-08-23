"""CLI entry point for scale search. Run with: python -m scale_search"""

import json
import math
import random
import argparse

from scale_search.config import SearchConfig
from scale_search.util import parse_ratios
from scale_search.metrics import (
    CONSONANCE_METRICS, SCALE_METRICS,
    parse_metrics, get_normalization, init_normalization,
    format_metric_description,
)
from scale_search.evolution import evolve, slots_to_pitches
from scale_search.groups import parse_group_spec, describe_structures
from scale_search.display import (
    load_seed_scales, find_matching_seed,
    format_scale, print_interval_matrix,
    evaluate_named_scale, print_scale_details,
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
  %(prog)s --compare "9/8,5/4,4/3,3/2,5/3,15/8" --metric barlow
  %(prog)s --compare --seeds my_scales.txt --metric barlow
  %(prog)s --notes 7 --quiet --random-seed 42  # reproducible, script-friendly
  %(prog)s --notes 7 --json  # machine-readable output
  %(prog)s --runs 3 --notes 7  # 3 runs, best of each seeds the next

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
                        help="Exact number of notes including tonic (default: 7). "
                             "Ignored with --groups, which sets its own note count.")
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
                        help="Number of runs (default: 1). Best scales from each run seed the next.")
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
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (implies --quiet)")
    parser.add_argument("--compare", nargs='?', const=True, default=False, metavar="RATIOS",
                        help="Compare scales. With no argument: use --seeds file. "
                             "With ratios: evaluate that scale (e.g., '9/8,5/4,4/3,3/2,5/3,15/8')")
    parser.add_argument("--groups", type=str, default=None, metavar="SPEC",
                        help="Optimize sub-scales instead of one flat scale. "
                             "SPEC is a polytope (dodecahedron, 24cell, cube, "
                             "icosahedron, octahedron, tetrahedron), "
                             "'cyclic:N,k[,step]', or a group file. "
                             "Sets the note count; --notes is ignored.")
    parser.add_argument("--global-weight", type=float, default=0.3, metavar="W",
                        help="With --groups: blend sub-scale quality with whole-scale "
                             "quality as local^(1-W) * global^W (default: 0.3). "
                             "0 = sub-scales only, 1 = whole scale only.")
    parser.add_argument("--list-groups", action="store_true",
                        help="List available group structures and exit")
    parser.add_argument("--export-groups", type=str, default=None, metavar="FILE",
                        help="Write the structure named by --groups to FILE as a "
                             "group file and exit, for inspection or hand-editing")
    parser.add_argument("--list-metrics", action="store_true", 
                        help="List available metrics and exit")
    args = parser.parse_args()
    
    if args.list_groups:
        print(describe_structures())
        return
    
    if args.export_groups:
        if not args.groups:
            print("Error: --export-groups needs --groups to say what to export.")
            return
        try:
            structure = parse_group_spec(args.groups)
        except (ValueError, OSError) as e:
            print(f"Error: {e}")
            return
        try:
            with open(args.export_groups, 'w') as f:
                f.write(f"# {structure.name}: {structure.description}\n")
                f.write(f"# {structure.summary()}\n")
                f.write("# one sub-scale per line, as slot indices\n")
                for g in structure.groups:
                    f.write(' '.join(str(i) for i in g) + '\n')
        except OSError as e:
            print(f"Error writing {args.export_groups}: {e}")
            return
        print(f"Wrote {len(structure.groups)} sub-scales "
              f"({structure.num_slots} slots) to {args.export_groups}")
        return
    
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
    
    # JSON implies quiet
    if args.json:
        args.quiet = True
    
    if args.compare is not False:
        _handle_compare(args, metrics, scale_metrics, edo)
        return
    
    # Parse required/excluded notes
    required_notes = parse_ratios(args.include) if args.include else []
    excluded_notes = parse_ratios(args.exclude) if args.exclude else []
    
    # Parse group structure, if any
    structure = None
    if args.groups:
        try:
            structure = parse_group_spec(args.groups)
        except (ValueError, OSError) as e:
            print(f"Error: {e}")
            return
        if args.notes is not None and args.notes != structure.num_slots:
            print(f"Note: --notes {args.notes} ignored; {structure.name} "
                  f"has {structure.num_slots} slots.")

    try:
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
            groups=structure.groups if structure else None,
            num_slots=structure.num_slots if structure else None,
            group_name=structure.name if structure else None,
            global_weight=args.global_weight,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
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
        if structure:
            print(f"Groups: {structure.name} ({structure.description})")
            print(f"  {structure.summary()}")
            print(f"  global weight: {config.global_weight} "
                  f"(0 = sub-scales only, 1 = whole scale only)")
        else:
            print(f"Notes: {note_desc}")
        print(f"Ratio limit: {config.ratio_limit}")
        print(f"Octave range: {octave_desc}")
        print(f"Metric: {metric_desc}")
        print(f"Interval combination: {combine_desc_str}")
        print(f"Generations: {config.generations}")
        if args.runs > 1:
            print(f"Runs: {args.runs} (cross-seeding)")
        if args.random_seed is not None:
            print(f"Random seed: {args.random_seed}")
        print("=" * 60)
        print()
    
    best_overall = None
    discovered_scales = []  # Cross-seeding: accumulate good scales across runs
    
    for run in range(args.runs):
        if args.runs > 1 and verbose:
            print(f"\n--- Run {run + 1}/{args.runs} ---\n")
        
        # Build effective seed list: original seeds + scales found in previous runs
        effective_seeds = list(seed_scales) if seed_scales else []
        effective_seeds.extend(discovered_scales)
        
        scale, fitness, details = evolve(config, verbose=verbose,
                                         seed_scales=effective_seeds if effective_seeds else None)
        
        # Add this run's best to discovered scales for cross-seeding
        discovered_scales.append(list(scale))
        
        if best_overall is None or fitness > best_overall[1]:
            best_overall = (scale, fitness, details)
    
    scale, fitness, details = best_overall
    
    # JSON output
    if args.json:
        _output_json(scale, details, metrics, scale_metrics, edo, octave_desc,
                     seed_scales, seed_names)
        return
    
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
    
    # Sub-scale breakdown
    if structure:
        _print_group_results(scale, details, structure, verbose)

    # The scale itself
    if verbose:
        print("\n" + "=" * 60)
        print("SCALE")
        print("=" * 60)
    print()
    print(format_scale(scale))

    if structure:
        # Slot order is what reconstructs the geometry; the sorted listing
        # above is what goes into a tuning file.
        print()
        print("# slot order (slot 0 = 1/1)")
        for i, r in enumerate(slots_to_pitches(scale)):
            print(f"#   {i:>2d}  {r.numerator}/{r.denominator}")


def _print_group_results(scale, details, structure, verbose):
    """Print the per-sub-scale breakdown for a grouped search."""
    group_details = details.get('groups')
    if not group_details:
        return

    full_slots = slots_to_pitches(scale)

    print()
    print(f"Local (sub-scale) quality:  {details['local_quality']:.4f}")
    print(f"Global (whole-scale) quality: {details['global_quality']:.4f}")
    print()
    print("=" * 60)
    print(f"SUB-SCALES ({len(group_details)})")
    print("=" * 60)
    print()

    # Which sub-scales share notes with which — the moves a player can make.
    neighbours = {i: [] for i in range(len(group_details))}
    for i, j, shared in structure.adjacency():
        if shared >= 2:  # A single shared note is a weak pivot; 2+ is a real move
            neighbours[i].append(j)
            neighbours[j].append(i)

    ranked = sorted(range(len(group_details)),
                    key=lambda i: group_details[i]['quality'], reverse=True)

    # Notes are shown as they sit in the parent scale, not transposed to the
    # sub-scale's own tonic: you stay in one tuning and shift between
    # sub-scales, so these are the pitches actually played, and a note shared
    # by two sub-scales reads as the same ratio in both rows.
    slot_w = max(len(','.join(str(x) for x in g['group'])) for g in group_details)
    slot_w = min(max(slot_w, 5), 24)

    if verbose:
        adj_w = max((len(','.join(str(x) for x in neighbours[i]))
                     for i in range(len(group_details))), default=0)
        adj_w = min(max(adj_w, 8), 20)
        print(f"  {'#':>3s}  {'Quality':>7s}  {'Slots':<{slot_w}s}  "
              f"{'Adjacent':<{adj_w}s}  Notes (as played)")
        print(f"  {'-'*3}  {'-'*7}  {'-'*slot_w}  {'-'*adj_w}  {'-'*40}")
    else:
        print(f"  {'#':>3s}  {'Quality':>7s}  {'Slots':<{slot_w}s}  Notes (as played)")
        print(f"  {'-'*3}  {'-'*7}  {'-'*slot_w}  {'-'*40}")

    for i in ranked:
        g = group_details[i]
        slots = ','.join(str(x) for x in g['group'])
        if len(slots) > slot_w:
            slots = slots[:slot_w - 3] + '...'
        played = sorted(full_slots[k] for k in g['group'])
        notes = ', '.join(f"{r.numerator}/{r.denominator}" for r in played)
        if verbose:
            adj = ','.join(str(x) for x in neighbours[i])
            if len(adj) > adj_w:
                adj = adj[:adj_w - 3] + '...'
            print(f"  {i:>3d}  {g['quality']:>7.4f}  {slots:<{slot_w}s}  "
                  f"{adj:<{adj_w}s}  {notes}")
        else:
            print(f"  {i:>3d}  {g['quality']:>7.4f}  {slots:<{slot_w}s}  {notes}")

    if verbose:
        print()
        # Shapes: sub-scales transposed to their own tonic. Reported as a count
        # rather than per row, since in practice they are all distinct and a
        # column of them would say the same thing on every line.
        shapes = {tuple(g['scale']) for g in group_details}
        print(f"Distinct sub-scale shapes: {len(shapes)} of {len(group_details)}")

        adj_all = structure.adjacency()
        if adj_all:
            shares = {}
            for _, _, shared in adj_all:
                shares[shared] = shares.get(shared, 0) + 1
            desc = ', '.join(f"{c} pairs share {n} notes"
                             for n, c in sorted(shares.items(), reverse=True))
            print(f"Adjacency: {desc}")


def _handle_compare(args, metrics, scale_metrics, edo):
    """Handle --compare mode: evaluate scales and display ranked results."""
    verbose = not args.quiet
    
    # Parse filters
    fixed_notes = args.notes
    required_notes = parse_ratios(args.include) if args.include else []
    excluded_notes = set(parse_ratios(args.exclude)) if args.exclude else set()
    
    # Build list of (name, ratios_str_list) to evaluate
    to_evaluate = []
    
    if args.compare is True:
        # No ratios given — use seed file or fallback to built-in examples
        if args.seeds:
            scales, names = load_seed_scales(args.seeds, with_names=True)
            if not scales:
                print(f"No scales found in {args.seeds}")
                return
            for i, (scale_ratios, name) in enumerate(zip(scales, names)):
                label = name or f"Scale #{i + 1}"
                ratios_str = [f"{r.numerator}/{r.denominator}" for r in scale_ratios]
                to_evaluate.append((label, ratios_str))
        else:
            to_evaluate = [
                ("Ptolemaic diatonic", ["9/8", "5/4", "4/3", "3/2", "5/3", "15/8"]),
                ("Natural harmonic (5-limit)", ["9/8", "5/4", "4/3", "3/2", "8/5", "15/8"]),
                ("Septimal (7-limit)", ["10/9", "7/6", "5/4", "4/3", "3/2", "5/3", "7/4"]),
            ]
    else:
        to_evaluate.append(("Custom scale", args.compare.split(',')))
    
    # Filter scales by note count and constraints
    from fractions import Fraction
    has_filters = (fixed_notes is not None or required_notes or excluded_notes)
    filtered = []
    skipped = 0
    for name, ratios_str in to_evaluate:
        scale = [Fraction(r) for r in ratios_str if r.strip() != "2/1"]
        num_notes = len(scale) + 1  # +1 for tonic
        
        if fixed_notes is not None and num_notes != fixed_notes:
            skipped += 1
            continue
        if required_notes:
            scale_set = set(scale)
            if not all(r in scale_set for r in required_notes):
                skipped += 1
                continue
        if excluded_notes:
            scale_set = set(scale)
            if any(e in scale_set for e in excluded_notes):
                skipped += 1
                continue
        
        filtered.append((name, ratios_str))
    
    if not filtered:
        filter_parts = []
        if fixed_notes is not None:
            filter_parts.append(f"--notes {fixed_notes}")
        if required_notes:
            filter_parts.append(f"--include {args.include}")
        if excluded_notes:
            filter_parts.append(f"--exclude {args.exclude}")
        print(f"No scales match filters ({', '.join(filter_parts)}). "
              f"{skipped} scale{'s' if skipped != 1 else ''} filtered out.")
        return
    
    if skipped > 0 and verbose:
        print(f"({skipped} scale{'s' if skipped != 1 else ''} filtered out)")
    
    # Evaluate matching scales
    results = []
    for name, ratios_str in filtered:
        fitness, details, scale = evaluate_named_scale(
            name, ratios_str, metrics, args.octaves,
            scale_metrics, edo, args.combine)
        results.append((name, details['quality'], details, scale))
    
    # Sort by quality descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print detailed output for each scale (in ranked order) if verbose
    if verbose and len(results) > 1:
        for name, quality, details, scale in results:
            print_scale_details(name, scale, details, metrics, scale_metrics, edo)
    
    # Always print ranked summary
    if len(results) == 1:
        # Single scale — just show the result
        name, quality, details, scale = results[0]
        if verbose:
            print_scale_details(name, scale, details, metrics, scale_metrics, edo)
        else:
            print(f"{name}: quality = {quality:.4f}")
            print(", ".join(f"{r.numerator}/{r.denominator}" for r in sorted(scale)))
    else:
        # Multiple scales — ranked summary table
        print(f"\n{'=' * 60}")
        print("RANKING")
        print("=" * 60)
        
        # Find max name length for alignment
        max_name = max(len(r[0]) for r in results)
        max_name = min(max_name, 40)  # Cap for very long names
        
        print(f"\n  {'#':>2}  {'Quality':>7}  {'Notes':>5}  {'Name':<{max_name}}  Scale")
        print(f"  {'—'*2}  {'—'*7}  {'—'*5}  {'—'*max_name}  {'—'*30}")
        
        for rank, (name, quality, details, scale) in enumerate(results, 1):
            truncated_name = name[:max_name]
            notes = len(scale) + 1  # +1 for tonic
            ratios = ", ".join(f"{r.numerator}/{r.denominator}" for r in sorted(scale))
            # Truncate ratios if very long
            if len(ratios) > 60:
                ratios = ratios[:57] + "..."
            print(f"  {rank:>2}  {quality:>7.4f}  {notes:>5}  {truncated_name:<{max_name}}  {ratios}")
        
        print()


def _output_json(scale, details, metrics, scale_metrics, edo, octave_desc,
                 seed_scales=None, seed_names=None):
    """Output results as JSON."""
    from fractions import Fraction
    
    scale_sorted = sorted(scale)
    
    result = {
        "quality": round(details['quality'], 6),
        "num_intervals": details['num_intervals'],
        "octave_range": details['octave_range'],
        "scale": [f"{r.numerator}/{r.denominator}" for r in scale_sorted] + ["2/1"],
        "cents": [round(1200 * math.log2(float(r)), 1) for r in scale_sorted] + [1200.0],
    }
    
    # Metric breakdown
    if len(metrics) > 1 or scale_metrics:
        breakdown = {}
        for metric_name, weight in metrics:
            raw = details['metric_averages'].get(metric_name, 0)
            min_val, max_val = get_normalization(metric_name)
            normalized = (raw - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
            quality = 1.0 - normalized
            breakdown[metric_name] = {"quality": round(quality, 4), "weight": weight}
        
        for metric_name, weight in scale_metrics:
            raw = details['metric_averages'].get(metric_name, 0)
            min_val, max_val = get_normalization(metric_name)
            normalized = (raw - min_val) / (max_val - min_val)
            normalized = max(0.0, min(1.0, normalized))
            if metric_name in ('primes', 'variety'):
                quality = normalized
            else:
                quality = 1.0 - normalized
            label = f"edo{edo}" if metric_name == 'edo' else metric_name
            breakdown[label] = {"quality": round(quality, 4), "weight": weight}
        
        result["metrics"] = breakdown
    
    # Seed match
    if seed_scales:
        match_idx, match_name = find_matching_seed(scale, seed_scales, seed_names)
        if match_idx >= 0:
            result["seed_match"] = {"index": match_idx + 1}
            if match_name:
                result["seed_match"]["name"] = match_name
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
