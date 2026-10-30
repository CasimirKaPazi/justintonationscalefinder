"""CLI entry point for scale search. Run with: python -m scale_search"""

import json
import math
import random
import argparse

from scale_search.config import SearchConfig, validate_combine_method, COMBINE_METHODS
import math
import os
from scale_search.util import (parse_ratios, generate_candidate_ratios,
                               snap_to_candidates)
from scale_search.metrics import (
    CONSONANCE_METRICS, SCALE_METRICS,
    parse_metrics, get_normalization, init_normalization,
    format_metric_description,
)
from scale_search.evolution import evolve, slots_to_pitches, evaluate_scale
from scale_search.groups import parse_group_spec, describe_structures
from scale_search.analysis import nearest_scales, edo_fits
from scale_search.scala import load_reference_scales
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
                        help="How to combine interval scores: geometric (default), "
                             "worst, mean, power2, power3, logloss. Also used to combine "
                             "sub-scale scores with --groups.")
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
    parser.add_argument("--ordered-slots", action="store_true",
                        help="With --groups: require slot i to hold the i-th lowest "
                             "pitch. Use when slot numbers mean something (chromatic "
                             "steps, scale degrees); leave off for polytopes, where "
                             "the numbering is arbitrary.")
    parser.add_argument("--analyze", action="store_true",
                        help="After searching, report which scales in --seeds the "
                             "result resembles (in cents, modes included) and which "
                             "equal divisions it sits on")
    parser.add_argument("--max-seeds", type=int, default=50, metavar="N",
                        help="With --seeds pointing at a Scala archive: how many of "
                             "the best-scoring matching scales to seed with "
                             "(default: 50)")
    parser.add_argument("--reference", type=str, default=None, metavar="PATH",
                        help="Library for --analyze to compare against: a directory "
                             "of .scl files (e.g. the Scala archive), a single .scl "
                             "file, or a seed file. Defaults to --seeds.")
    parser.add_argument("--include-non-octave", action="store_true",
                        help="With --reference: also load scales that don't repeat "
                             "at the octave (Bohlen-Pierce and similar). Excluded by "
                             "default, since comparison is under octave equivalence.")
    parser.add_argument("--edo-range", type=str, default="5:72", metavar="MIN:MAX",
                        help="EDO range searched by --analyze (default: 5:72)")
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
    
    try:
        validate_combine_method(args.combine)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
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
            ordered_slots=args.ordered_slots,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    verbose = not args.quiet
    
    # Load seed scales if specified (with names for matching)
    seed_scales = None
    seed_names = None
    if args.seeds:
        if os.path.isdir(args.seeds) or args.seeds.lower().endswith('.scl'):
            seed_scales, seed_names = _seeds_from_archive(args.seeds, config,
                                                          args, verbose)
        else:
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
    if args.analyze:
        _print_analysis(scale, seed_scales, seed_names, args)

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
        _print_incompatible_pairs(scale, structure, verbose)

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


def _print_incompatible_pairs(scale, structure, verbose):
    """
    State the pairwise rule for combining notes, when there is one.

    Printed only when the pairwise view is complete — when every maximal set of
    mutually compatible notes is itself a sub-scale, so "obey these pair rules"
    and "stay inside a sub-scale" say the same thing. For structures where it
    isn't (cube, dodecahedron, 24-cell by cells) a pair list would be a
    necessary condition masquerading as the rule, so nothing is printed and the
    sub-scale listing stands on its own.

    Whichever of the forbidden and permitted relations is shorter is the one
    shown: the dodecahedron's edges forbid 160 pairs but permit only 30, and
    "only play X with ..." is the same information at a fifth the length.
    """
    if not structure.exclusions_are_exact():
        return

    forbidden = structure.incompatible_pairs()
    if not forbidden:
        return

    n = structure.num_slots
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    permitted = [p for p in all_pairs if p not in set(forbidden)]

    invert = len(permitted) < len(forbidden)
    relation = permitted if invert else forbidden

    full_slots = slots_to_pitches(scale)
    # Order by pitch so the table reads like the scale does; each relation is
    # stated once, under whichever of its two notes comes first.
    order = sorted(range(n), key=lambda i: full_slots[i])
    rank = {slot: pos for pos, slot in enumerate(order)}

    partners = {i: [] for i in range(n)}
    if invert:
        # A permission list must be complete for every note: "4/3 only with X"
        # is false if 4/3's other permitted partners were folded into their own
        # rows. So each note carries its full set, and every relation appears
        # under both of its notes.
        for a, b in relation:
            partners[a].append(b)
            partners[b].append(a)
    else:
        # A prohibition is symmetric, so stating each pair once is complete —
        # this is a list of forbidden pairs grouped by their first note, not a
        # per-note table, and it is labelled as such below.
        for a, b in relation:
            lo, hi = (a, b) if rank[a] < rank[b] else (b, a)
            partners[lo].append(hi)

    rows = []
    for slot in order:
        if partners[slot]:
            others = sorted(partners[slot], key=lambda x: rank[x])
            rows.append((slot, others))
    if not rows:
        return

    def name(i):
        r = full_slots[i]
        return f"{r.numerator}/{r.denominator}"

    print()
    print("=" * 60)
    print("ONLY COMBINE" if invert else "DO NOT COMBINE")
    print("=" * 60)
    print()
    if invert:
        print("Each note may only sound together with the notes beside it.")
    else:
        print("These pairs never occur together, grouped by their first note.")
        print("Any pair not listed here is fine.")
    print("Obeying this is equivalent to staying within a sub-scale.")
    print()

    width = max(len(name(slot)) for slot, _ in rows)
    joiner = "  only with  " if invert else "  never with  "
    for slot, others in rows:
        print(f"  {name(slot):>{width}s}{joiner}{', '.join(name(o) for o in others)}")

    if verbose:
        lead = "# Only play" if invert else "# Don't play"
        parts = '; '.join(f"{name(slot)} with {', '.join(name(o) for o in others)}"
                          for slot, others in rows)
        print()
        print(f"{lead}: {parts}")


def _seeds_from_archive(path, config, args, verbose):
    """
    Draw seeds from a Scala archive directory or .scl file.

    An archive holds thousands of scales and a population holds tens, so this
    has to select rather than load. Three steps, in order of how much they
    discard:

      1. Keep scales of a usable size. Exact matches are preferred, but the
         search has always accepted wrong-sized seeds and trimmed or extended
         them, so near misses are passed through for it to resize as usual.
         Scales far from the target are dropped: extending a 5-note scale to 24
         means inventing 19 notes, which is a random scale with five anchors,
         and scoring thousands of such candidates costs more than it returns.
      2. Snap every pitch to a ratio the search can express. Much of the
         archive is written in cents, which converts faithfully to ratios like
         720009/679598 — fine to measure against, meaningless to mutate. This
         also rescues rational scales whose ratios exceed the complexity limit.
      3. Score what survives and keep the best, since a seed is only worth a
         population slot if it beats what random initialisation would give.
    """
    scales, names, files, stats = load_reference_scales(
        path, octave_only=not args.include_non_octave)
    if not scales:
        if verbose:
            print(f"No usable scales found in {path}")
        return None, None

    if config.fixed_notes is not None:
        target_lo = target_hi = config.fixed_notes
    else:
        target_lo, target_hi = config.min_notes, config.max_notes
    # How far from the target a scale may sit and still be worth resizing
    slack = max(2, target_hi // 4)
    band_lo, band_hi = max(2, target_lo - slack), target_hi + slack
    size_desc = (f"{target_lo} notes" if target_lo == target_hi
                 else f"{target_lo}-{target_hi} notes")

    candidates, _ = generate_candidate_ratios(config.ratio_limit)
    candidates = sorted(candidates, key=float)
    cand_cents = [1200.0 * math.log2(float(c)) for c in candidates]

    kept, wrong_size, unsnappable, failed_constraints = [], 0, 0, 0
    n_exact = 0
    seen = set()
    for ratios, name, fn in zip(scales, names, files):
        notes = len(ratios) + 1
        if not (band_lo <= notes <= band_hi):
            wrong_size += 1
            continue
        snapped = snap_to_candidates(ratios, candidates, cand_cents)
        if snapped is None:
            unsnappable += 1
            continue
        if not config.check_constraints(snapped):
            failed_constraints += 1
            continue
        key = tuple((r.numerator, r.denominator) for r in snapped)
        if key in seen:
            continue
        seen.add(key)
        exact = target_lo <= notes <= target_hi
        n_exact += exact
        kept.append((snapped, name, fn, exact))

    if not kept:
        if verbose:
            print(f"No scales in {path} usable as seeds "
                  f"({stats['loaded']} loaded, none with {size_desc} "
                  f"that fit ratio limit {config.ratio_limit}).")
        return None, None

    scored = []
    interval_cache = {}
    for snapped, name, fn, exact in kept:
        fitness, _ = evaluate_scale(snapped, config.metrics,
                                    config.octave_range, interval_cache,
                                    config.scale_metrics, config.edo,
                                    config.combine_method)
        scored.append((fitness, snapped, name, fn, exact))
    # Right-sized scales first: a resized seed has had notes invented or
    # discarded at random, so it carries less of the original scale than its
    # score suggests.
    scored.sort(key=lambda x: (not x[4], -x[0]))
    # Seeds are added to the population before it is filled out with random
    # scales, so asking for more than the population size would silently grow
    # the population past what was requested and leave no room for fresh blood.
    cap = max(1, min(args.max_seeds, config.population_size))
    chosen = scored[:cap]

    if verbose:
        capped = (" (capped by --population)"
                  if args.max_seeds > config.population_size
                  and len(kept) > cap else "")
        print(f"Loaded {stats['loaded']} scales from {path}; "
              f"{len(kept)} usable as seeds, keeping best {len(chosen)}{capped}.")
        n_resized = len(chosen) - sum(1 for c in chosen if c[4])
        if n_resized:
            print(f"  {len(chosen) - n_resized} at {size_desc}, "
                  f"{n_resized} to be resized by the search "
                  f"(from {band_lo}-{band_hi} notes)")
        reasons = []
        if wrong_size:
            reasons.append(f"{wrong_size} outside {band_lo}-{band_hi} notes")
        if unsnappable:
            reasons.append(f"{unsnappable} outside ratio limit "
                           f"{config.ratio_limit}")
        if failed_constraints:
            reasons.append(f"{failed_constraints} failed include/exclude")
        if reasons:
            print(f"  Skipped: {', '.join(reasons)}")
        best = chosen[0]
        label = best[3] or best[2]
        print(f"  Best seed: {label} (quality {best[0]:.4f})")

    return [c[1] for c in chosen], [c[3] or c[2] for c in chosen]


def _print_analysis(scale, seed_scales, seed_names, args):
    """Report what a result resembles: known scales, then equal divisions."""
    try:
        lo, _, hi = args.edo_range.partition(':')
        min_edo, max_edo = int(lo), int(hi)
        if min_edo < 2 or max_edo < min_edo:
            raise ValueError
    except ValueError:
        print(f"Error: --edo-range must be MIN:MAX with 2 <= MIN <= MAX, "
              f"got '{args.edo_range}'")
        return

    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if args.reference:
        ref_scales, ref_names, ref_files, stats = load_reference_scales(
            args.reference, octave_only=not args.include_non_octave)
        source = args.reference
        skipped = []
        if stats['unparsed']:
            skipped.append(f"{stats['unparsed']} unreadable")
        if stats['non_octave']:
            skipped.append(f"{stats['non_octave']} non-octave")
        if stats['too_large']:
            skipped.append(f"{stats['too_large']} too large")
        note = f" ({', '.join(skipped)} skipped)" if skipped else ""
        loaded_note = f"  [{stats['loaded']} scales{note}]"
    else:
        ref_scales, ref_names = seed_scales, seed_names
        ref_files = None
        source = args.seeds
        loaded_note = ""

    if ref_scales:
        matches = nearest_scales(scale, ref_scales, ref_names, top=5,
                                 sources=ref_files)
        print()
        print(f"Closest scales in {source}:{loaded_note}")
        print()
        if not matches:
            print("  Nothing comparable.")
        else:
            # The file column goes before the description: filenames are the
            # archive's identifier and are short and uniform, while
            # descriptions vary wildly in length, so putting the variable
            # column last keeps the table aligned.
            show_files = any(m.get('source') for m in matches)
            fw = (max(len(m['source'] or '') for m in matches)
                  if show_files else 0)
            fw = min(max(fw, 4), 30)

            header = f"  {'Max dev':>8s}  {'RMS':>6s}  {'Match':>6s}  "
            rule = f"  {'-'*8}  {'-'*6}  {'-'*6}  "
            if show_files:
                header += f"{'File':<{fw}s}  "
                rule += f"{'-'*fw}  "
            print(header + "Name")
            print(rule + '-' * 30)

            for m in matches:
                tags = []
                if m['same_length']:
                    if m['exact']:
                        tags.append('identical')
                    # The mode matters even for an identical match: it says the
                    # result is that scale started from a different degree.
                    if m['mode']:
                        tags.append(f"mode {m['mode']}")
                else:
                    tags.append(f"within {m['notes']}-note scale")
                    if m['mode']:
                        tags.append(f"mode {m['mode']}")
                tag = f"  ({', '.join(tags)})" if tags else ""

                row = (f"  {m['max_dev']:7.1f}c  {m['rms']:5.1f}c  "
                       f"{m['matched']:>2d}/{m['total']:<3d}  ")
                if show_files:
                    src = m['source'] or ''
                    if len(src) > fw:
                        src = '...' + src[-(fw - 3):]
                    row += f"{src:<{fw}s}  "
                print(row + m['name'][:40] + tag)
    else:
        print()
        print("Nothing to compare against — give --reference PATH (a directory "
              "of .scl\nfiles, an .scl file, or a seed file) or --seeds FILE.")

    fits = edo_fits(scale, min_edo, max_edo, top=5)
    print()
    print("Best equal divisions:")
    print()
    if not fits:
        print(f"  None in {min_edo}-{max_edo}: every division would put two notes "
              f"on the same step.")
    else:
        print(f"  {'EDO':>4s}  {'Fit':>5s}  {'Max dev':>8s}  Steps")
        print(f"  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*38}")
        for f in fits:
            steps = ' '.join(str(x) for x in f['steps'])
            if len(steps) > 38:
                steps = steps[:35] + '...'
            print(f"  {f['edo']:>4d}  {f['rel']:5.2f}  {f['max_dev']:7.1f}c  {steps}")
        print()
        print("  Fit is the worst note's error as a fraction of half a step:")
        print("  0 = exactly on the grid, ~0.5 = no better than chance. Ranking on")
        print("  this rather than on cents keeps large divisions from winning by")
        print("  default just for having finer steps.")


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
    
    # With a single scale, --analyze answers "what is this?" -- but only if the
    # comparison set isn't itself the seed file, where every scale would
    # trivially match itself.
    if args.analyze and len(results) == 1 and args.compare is not True:
        seeds, seed_names = ((None, None) if not args.seeds
                             else load_seed_scales(args.seeds, with_names=True))
        _print_analysis(results[0][3], seeds, seed_names, args)
    
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
