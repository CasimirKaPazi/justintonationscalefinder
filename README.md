# Evolutionary Search for Just Intonation Scales

Uses an evolutionary algorithm to search for microtonal just intonation scales by various music theoretic constraints.

This was created with Claude (Anthropic).

## Features

- **Multiple consonance metrics**: Tenney height, Euler gradus, harmonic entropy, Plomp-Levelt roughness, Wilson complexity, Barlow harmonicity, odd-limit
- **Scale-level metrics**: Step balance, Rothenberg propriety, prime diversity, interval variety, EDO approximation
- **Interval combination methods**: Geometric mean (default), worst-case, arithmetic mean, power mean, log loss
- **Evolutionary optimization**: Genetic algorithm with simulated annealing, adaptive mutation, and fingerprint-based diversity
- **Flexible constraints**: Required/excluded notes, variable note counts, ratio complexity limits
- **Sub-scale groups**: Optimize a large scale that decomposes into playable sub-scales, using regular polytopes (dodecahedron, 24-cell, ...), cyclic windows, or a custom group file
- **Seed scales**: Initialize from existing scales (Scala .scl format supported)
- **Multi-run cross-seeding**: Best scales from each run seed the next for better exploration
- **Multiple output formats**: Human-readable (default), quiet mode, JSON for scripting
- **Reproducibility**: Optional random seed for deterministic results

## Installation

Requires Python 3.7+. No external dependencies.

```bash
python scale_search.py --help
```

## Quick Start

```bash
# Display the help to see all options
python scale_search.py --help

# Find a 7-note scale optimizing for simple ratios
python scale_search.py --notes 7

# List all available metrics
python scale_search.py --list-metrics

# Combine multiple metrics
python scale_search.py --metric "wilson:1,barlow:1,propriety:1,variety:0.1,entropy:0.5" --combine geometric --notes 7

# Require specific intervals
python scale_search.py --notes 7 --include "3/2,4/3"

# 19-EDO neji (near-equivalent just intonation approximation of 19-EDO)
python scale_search.py --metric "soft_tenney:1,edo19:2" --notes 19

# Reproducible, script-friendly output
python scale_search.py --notes 7 --quiet --random-seed 42

# Machine-readable JSON output
python scale_search.py --notes 7 --json --random-seed 42

# Multiple runs with cross-seeding (best scales from each run seed the next)
python scale_search.py --notes 7 --runs 3

# Compare a specific scale
python scale_search.py --compare "9/8,5/4,4/3,3/2,5/3,15/8" --metric barlow

# Compare all scales in a seed file
python scale_search.py --compare --seeds seeds.txt --metric barlow

# Sub-scale search: 20 notes on a dodecahedron, 12 pentatonic faces
python scale_search.py --groups dodecahedron --metric "soft_tenney:1,propriety:0.5"

# List the available group structures
python scale_search.py --list-groups

# Elaborate search with various settings
python scale_search.py --metric "wilson:1,barlow:1,propriety:1,variety:0.1,entropy:0.5,roughness:0.5" --population 500 --octaves 2 --min-notes 12 --max-notes 16 --include "7/6" --seeds seeds.txt
```

## Metrics

### Interval Metrics
| Metric | Description |
|--------|-------------|
| `tenney` | Tenney height: log₂(n×d) |
| `soft_tenney` | √(Tenney) — gentler on higher primes |
| `euler` | Euler's Gradus Suavitatis |
| `barlow` | Barlow harmonicity/indigestibility (default) |
| `odd_limit` | Largest odd factor (log₂) |
| `roughness` | Plomp-Levelt sensory dissonance |
| `entropy` | Harmonic entropy (Erlich) |
| `wilson` | Wilson harmonic complexity |

### Scale Metrics
| Metric | Description |
|--------|-------------|
| `steps` | Penalizes clustered notes |
| `balance` | Step size evenness (coefficient of variation) |
| `propriety` | Rothenberg propriety (interval class overlap) |
| `primes` | Rewards prime diversity (better use variety) |
| `variety` | Interval variety |
| `edoN` | Proximity to N-EDO (for neji scales), e.g. `edo12` |

## Interval Combination Methods

The `--combine` option controls how individual interval scores are combined (in order of strictness):

| Method | Description |
|--------|-------------|
| `mean` | Arithmetic mean. Bad intervals can be averaged out by good ones. Most lenient. |
| `geometric` | Geometric mean (default). One bad interval pulls down the whole score. Best for ensuring all intervals work together. |
| `power2` | Power mean (p=2, RMS). Emphasizes bad intervals more than arithmetic mean. |
| `power3` | Power mean (p=3). Even stricter than p=2. |
| `logloss` | Log loss. Heavily penalizes near-dissonant intervals — a 0.9 penalty hurts much more than two 0.45s. |
| `worst` | Minimax. Optimizes the worst interval. Most demanding — no bad intervals tolerated. Diverges the most from the others. |

## Sub-Scale Groups

Normally the search produces one flat scale meant to be used all at once. With
`--groups` it instead produces a larger scale that decomposes into smaller
playable sub-scales, so a musician stays on one sub-scale at a time and shifts
between adjacent ones that share notes.

Polytope structures take the vertices of a regular polytope as notes and its
facets as sub-scales. On a dodecahedron, 20 vertices become 20 notes and each
of the 12 pentagonal faces is a 5-note scale; adjacent faces share the 2 notes
of their common edge.

```bash
python scale_search.py --groups dodecahedron \
    --metric "soft_tenney:1,propriety:0.5,variety:0.3" \
    --combine logloss --global-weight 0.3
```

| Spec | Notes | Sub-scales | Adjacency |
|------|-------|------------|-----------|
| `tetrahedron` | 4 | 4 of 3 | 6 pairs share 2 |
| `octahedron` | 6 | 8 of 3 | 12 pairs share 2 |
| `cube` | 8 | 6 of 4 | 12 pairs share 2 |
| `icosahedron` | 12 | 20 of 3 | 30 pairs share 2 |
| `dodecahedron` | 20 | 12 of 5 | 30 pairs share 2 |
| `16cell` | 8 | 16 of 4 | 32 pairs share 3 |
| `24cell` | 24 | 24 of 6 | 96 pairs share 3 |
| `cyclic:N,k[,step]` | N | k-slot windows every `step` | k−step |
| *filename* | inferred | one per line | as given |

`--list-groups` prints this table. `--groups` sets the note count, so `--notes`
is ignored.

### Face-lattice ranks

The default groups by facets, but any rank of the face lattice can be used, and
ranks can be combined with `+` so one tuning carries sub-scales of more than one
size at once:

```bash
# 8 notes, 16 tetrahedral sub-scales
python scale_search.py --groups 16cell

# the same 8 notes, but its 32 triangular faces as sub-scales
python scale_search.py --groups 16cell:faces

# both levels at once: 16 four-note and 32 three-note sub-scales
python scale_search.py --groups 16cell:cells+faces
```

| Spec | Notes | Sub-scales | Sizes |
|------|-------|------------|-------|
| `16cell` | 8 | 16 | 4 |
| `16cell:faces` | 8 | 32 | 3 |
| `16cell:cells+faces` | 8 | 48 | 3 + 4 |
| `16cell:edges` | 8 | 24 | 2 |
| `24cell` | 24 | 24 | 6 |
| `24cell:faces` | 24 | 96 | 3 |
| `24cell:cells+faces` | 24 | 120 | 3 + 6 |
| `dodecahedron:edges` | 20 | 30 | 2 |

Rank names are `cells` (4D only), `faces`, and `edges`. Every sub-scale is
scored on its own terms and they are combined equally, so the smaller ones
aren't penalized for being smaller. Edges as sub-scales are single intervals —
a way of asking that every edge of the polytope be consonant.

Every rank is checked against the published combinatorics at build time, so a
geometry bug fails loudly rather than producing a structure that runs and means
nothing.

If you'd rather work with an explicit file — to inspect a structure, trim it, or
hand-edit it — `--export-groups` writes any of them out:

```bash
python scale_search.py --groups 16cell:cells+faces --export-groups 16cell.txt
python scale_search.py --groups 16cell.txt --metric soft_tenney
```

**Group files** list one sub-scale per line as slot indices, `#` for comments:

```
# two overlapping tetrachords plus a linking triad
0 1 2 3
3 4 5 6
0 2 4 6
```

Every slot must appear in at least one sub-scale, or its note would be
unconstrained; this is checked and reported.

### How grouped scoring works

Each sub-scale is transposed to its own tonic before scoring, so a metric means
the same thing locally as it does globally — a group sitting high in the parent
scale isn't judged as a fragment. Sub-scale scores are then combined with the
same `--combine` method used for intervals, so `logloss` or `worst` prevents one
unplayable sub-scale from hiding behind good ones.

`--global-weight W` (default 0.3) blends sub-scale quality with whole-scale
quality:

```
final = local^(1-W) * global^W
```

At `W=0` only the sub-scales matter, and the parent scale can end up incoherent
— on a test run the 20 sub-scales scored 0.88 while the full 12-note set scored
0.19, meaning each face was playable but moving across them was not. At `W=1`
this reduces to ordinary whole-scale search and the grouping only fixes the note
count. The default sits nearer the local end.

Note that the geometry already forces the sub-scales to differ from each other:
each note sits in several facets and they cannot all be congruent. In testing,
distinct sub-scale shapes stayed at maximum across the whole range of `W`. If
you want sub-scales actively pushed apart rather than merely non-identical, that
would need a term rewarding dissimilarity between groups, which is not
implemented.

### Reading the output

The sub-scale table lists each group's notes **as they sit in the parent scale**,
not transposed to the group's own tonic. You stay in one tuning and shift between
sub-scales, so these are the pitches actually played, and a note shared by two
adjacent sub-scales reads as the same ratio in both rows. Verbose output adds an
`Adjacent` column listing the sub-scales that share at least two notes — the
moves available from where you are.

Scoring still transposes each group to its own tonic internally, since a group
that doesn't contain slot 0 would otherwise be judged as a fragment. The
transposed form is kept in `details['groups'][i]['scale']` if you want it
programmatically, but it isn't printed: in testing every sub-scale came out a
distinct shape even up to mode, on every structure, so a per-row shape column
said the same thing on every line. The count is reported instead, as
`Distinct sub-scale shapes`.

Slot 0 always holds the tonic 1/1. On a vertex-transitive polytope that is an
arbitrary symmetry break: facets containing slot 0 get simple ratios and tend to
score better than the rest. Output includes a `# slot order` block alongside the
sorted scale — the sorted listing is what goes into a tuning file, the slot order
is what reconstructs the geometry.

## Seed File

Specify a seed file with `--seeds`

Format:

```
! Scale Name
16/15
6/5
4/3
3/2
8/5
9/5
2/1

! Another Scale
21/20
7/6
21/16
7/5
3/2
7/4
2/1
```

Scala .scl files are also supported.

Scales that are outside of the specified note range get extended or trimmed. This way a e.g. 12 notes search can be informed by e.g. 5 note or 31 note scales.

The file `seeds.txt` provides over 200 scales for up to 31 notes.

## Output

The tool outputs:

- The best scale found (ratios)
- Quality score (0-1, higher = better)
- Interval matrix showing pairwise consonance
- Metric breakdown when using multiple metrics
- Match notification if result equals a seed scale
- Elapsed time and ETA during search

Use `--quiet` to suppress everything except the quality score and scale ratios.
Use `--json` for machine-readable output (includes quality, scale ratios, cents, and metric breakdown).

## Algorithm

1. Generate candidate ratios up to complexity limit
2. Initialize population from seeds + random scales
3. Evolve using crossover, mutation, and local search
4. Apply simulated annealing with adaptive reheating
5. Maintain diversity via fingerprint-based selection
6. Polish best solution with hill climbing

## Tips

- Increasing `min_notes`, `ratio_limit`, `octaves` and `generations` slows down search quite a lot.
- It saves computation to do a fast search first with low values of octaves and generations and then
 refine with higher ones, using the previous result in seeds.
- Using the seeds.txt file reduces search time. Most settings converge on the same scales anyway.
- The `entropy` and `roughness` metrics are also slow, and usually you don't need more than one of those two, as they are quite similar.
- Using `entropy` or `roughness` also tends to cluster the notes in a few places. To balance this out you can use `balance`. But I recommend starting with a low value (e.g. `balance:0.05`) and increase when needed.
- Using the `primes` metric can lead to more interesting scales, but is better used with low weight (e.g. `primes:0.3`). Even better to use `variety`.
- Below 7 notes, most combinations of metrics converge on the same scales.
- Including and excluding notes can be used to steer the result towards an intended flavor.
- Metric can also be used with negative value.
- You can use `--compare` in conjunction with `--notes` or `--include` to only compare scales that have the specified note count or notes.
- With `--groups` that have higher note count, use a strict `--combine` (`logloss` or `worst`). Sub-scale scores are combined the same way as intervals, so with `mean` one unplayable sub-scale disappears into the average.

## License

GPL-3.0

## Development

I won't develop it further and won't merge pull requests, just because I don't want the work of maintaining this project. If you want to contribute, then just fork the repo and publish your changes.
