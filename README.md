# Evolutionary Search for Just Intonation Scales

Uses an evolutionary algorithm to search for microtonal just intonation scales by various music theoretic constraints.

This was created with Claude (Anthropic).

## Features

- **Multiple consonance metrics**: Tenney height, Euler gradus, harmonic entropy, Plomp-Levelt roughness, Wilson complexity, Barlow harmonicity, odd-limit
- **Scale-level metrics**: Step balance, Rothenberg propriety, prime diversity, interval variety, EDO approximation
- **Interval combination methods**: Geometric mean (default), worst-case, arithmetic mean, power mean, log loss
- **Evolutionary optimization**: Genetic algorithm with simulated annealing, adaptive mutation, and fingerprint-based diversity
- **Flexible constraints**: Required/excluded notes, variable note counts, ratio complexity limits
- **Sub-scale groups**: Optimize a large scale that decomposes into playable sub-scales, using regular polytopes (dodecahedron, 24-cell, ...), cyclic windows, or a custom group file, with separate metrics for the sub-scales and the whole scale if wanted
- **Seed scales**: Initialize from existing scales (Scala .scl format supported), combine several sources, export a ranked seed file, and save results for later use
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
| `propriety` | Rothenberg propriety (interval class overlap, all modes) |
| `primes` | Rewards using primes above 5 (7, 11, 13, ...), lower ones more |
| `variety` | Interval variety |
| `edoN` | Proximity to N-EDO (for neji scales), e.g. `edo12` |

#### How propriety behaves

`propriety` measures by how much interval classes overlap: whether some
k-step interval is larger than some (k+1)-step interval, taken over all modes.
The overlap is measured in semitones and scored as `1 / (1 + overlap)`, so a
proper scale scores 1, a scale overlapping by 0.1 semitones 0.91, and by one
semitone 0.5.

Unlike the other scale metrics, propriety reaches its best value exactly, so it
acts more like a constraint than a trade-off. Whenever a proper scale can be
had cheaply, the search makes the scale proper, almost regardless of the
weight. In tests with `barlow`:

- At 7 notes it changed nothing: the most consonant 7-note scales are proper
  anyway.
- At 12 notes, weights 0.01, 0.1 and 1 all gave the same result. Without
  propriety, the search placed notes a comma apart (10/9 next to 9/8, 16/9
  next to 9/5) and left the tritone empty. With it, the result was essentially
  the 5-limit chromatic scale, with 45/32 or 64/45 as the tritone, at almost
  no cost in consonance (0.9957 → 0.9955).
- At 17 notes the weight began to matter: 0.01 left a slight overlap, 0.1
  removed it.

So the scale gets proper by placing notes evenly, often with one more complex
ratio filling a gap. In grouped search, where the sub-scales cannot all be
proper at once, propriety costs consonance instead. On the dodecahedron,
`soft_tenney:1,propriety:0.5,variety:0.3` with `--combine logloss` raised the
share of proper faces from 6% to 44% and lowered average sub-scale consonance
from 0.991 to 0.980.

Propriety used to be placed on a fixed range of 0 to 1 semitone of overlap and
clamped. A random 12-note scale overlaps by about 3 semitones, so nearly every
improper scale sat at the floor, and a sub-scale reaching one semitone lost
almost all of its score at once. In the dodecahedron test above, this cut
sub-scale consonance to 0.953 for a smaller gain in properness (42% of faces),
because the search sacrificed consonance everywhere to keep sub-scales off the
floor. With a negative weight, the inverted metric now keeps rewarding more
overlap where it used to stop at one semitone, so `propriety:-1` packs notes
together strongly.

### Weights

Each metric takes a weight, `name:weight`; without one it counts 1. A
**negative weight inverts** the metric: `variety:-0.5` rewards *low* variety,
`tenney:-1` prefers complex ratios. Internally the metric's quality *q* counts
as *1 − q* with weight |*w*|, which keeps every score between 0 and 1.

Propriety compares interval classes across all modes, including intervals that
cross the octave. The Pythagorean diatonic is improper because its 4-step
diminished fifth B–F (588 cents) is smaller than its 3-step augmented fourth F–B
(612 cents), and B–F only exists by wrapping past the octave.

#### What a weight costs

Weights are not comparable between scale metrics: the same weight can cost
nothing for one metric and a lot for another. The table shows how much
consonance each scale metric costs at a given weight when added to `barlow`,
as the drop in Barlow quality against Barlow alone (7 notes / 12 notes,
average of 4 searches at 120 generations and population 150).

| Added metric | 0.01 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|
| `steps` | 0 / 0.0003 | 0 / 0.0010 | 0 / 0.0016 | 0 / 0.0015 |
| `balance` | 0 / 0.0006 | 0.0065 / 0.0010 | 0.0064 / 0.0063 | 0.0153 / 0.0194 |
| `variety` | 0 / 0 | 0.0017 / 0.0015 | 0.0032 / 0.0020 | 0.0052 / 0.0054 |
| `primes` | 0 / 0 | 0.0025 / 0.0029 | 0.0076 / 0.0073 | 0.0152 / 0.0177 |
| `edo12` | 0.0005 / 0.0001 | 0.0005 / 0.0020 | 0.0005 / 0.0091 | 0.0049 / 0.0150 |
| `propriety` | 0 / 0.0002 | 0 / 0.0002 | 0 / 0.0002 | 0 / 0.0002 |

For scale: with Barlow alone the best 7-note scale scores 0.9978 (Ptolemy's
intense diatonic), the Pythagorean diatonic 0.9972, the septimal scale
8/7, 7/6, 4/3, 3/2, 14/9, 7/4 0.9951, and harmonics 8–14 0.9904. A cost of
0.005 is therefore about the step from the best 5-limit scale to a septimal
one; 0.02 is far more than that.

Reading the table:

- `steps` and `propriety` are cheap where the most consonant scale already
  satisfies them, as at 7 notes. At 12 notes propriety acts as a switch rather
  than a weight (see [How propriety behaves](#how-propriety-behaves)).
- `variety` and `balance` grow steadily with the weight; `balance` costs
  three to four times as much as `variety` at the same weight.
- `edo12` costs little at 7 notes, where 5-limit scales lie close to 12-EDO
  anyway, and much more at 12 notes.
- `primes` grows steadily too, at about the cost of `balance`. At 7 notes,
  weight 0.1 brought in about two primes above 5, 0.3 about four, 1 about
  five, and 3 all primes up to 29. It used to score a 5-limit scale 0, the
  lowest possible value, which made even weight 0.01 cost more than `steps`
  at weight 1 and made 0.1 already collect every prime up to 29. It now
  scores 5-limit 0.9, rising with the sum of 1/p over the primes above 5 used:
  7-limit 0.93, 7 and 11 0.95, 7 to 19 0.98, 7 to 29 1. With a negative
  weight, which prefers fewer primes, the inverted score runs from 0.1 for
  5-limit down to the floor, so `primes:-1` pushes hard towards 5-limit.

The numbers depend on the interval metric, the note count and the other
settings, so treat them as orders of magnitude.

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

A line given more than once counts more than once, which is how to weight a
sub-scale: in `12tone-full.txt` the 36 chords otherwise outvote the 12 keys,
and repeating each key line three times puts them on an equal footing.

**Slot numbering.** By default the search may put any pitch in any slot, which
is right for polytopes, where the numbering is an arbitrary labelling of
vertices. But when slot numbers carry their own meaning — chromatic steps, scale
degrees — the search will otherwise satisfy the overlap pattern with unrelated
notes: on the 12-key file below it produced a set where "C major" came out
Phrygian. `--ordered-slots` requires slot *i* to hold the *i*-th lowest pitch,
which fixes that.

### Included group files

`groups/` holds three files describing the common-practice 12-note vocabulary.
All want `--ordered-slots`, since their slots are chromatic steps:

| File | Sub-scales | Asks for |
|------|-----------|----------|
| `12tone-keys.txt` | 12 of 7 | a 12-note scale in which all 12 major keys work |
| `12tone-triads.txt` | 24 of 3 | one in which all 24 major and minor triads are consonant |
| `12tone-full.txt` | 48, mixed | keys, triads and dominant sevenths together |

```bash
python scale_search.py --groups groups/12tone-keys.txt --ordered-slots \
    --metric "soft_tenney:1,barlow:1,propriety:0.5" --combine logloss
```

In 12-TET all twelve keys are identical by construction; in just intonation they
cannot be, so these search for the best available compromise — effectively a
well-temperament in JI. The per-key scores show where the compromise falls.

### How grouped scoring works

Each sub-scale is transposed to its own tonic before scoring, so a metric means
the same thing locally as it does globally — a group sitting high in the parent
scale isn't judged as a fragment. A sub-scale has no tonic of its own, so it is
scored the same whichever of its notes it starts from: every interval and its
octave complement (5/4 and 8/5) count equally. A whole scale is scored from its
tonic: over 3 octaves, intervals that don't cross the tonic count twice as
often as those that do. Sub-scales used to be scored that way too, starting
from their lowest note above the parent's 1/1, so where the parent's tonic fell
decided which mode each sub-scale was judged in; the modes of one sub-scale
could differ by up to 0.046. Sub-scale scores are then combined with the
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

### Separate metrics for the whole scale

By default the sub-scales and the whole scale are scored with the same
`--metric`. `--global-metric` gives the whole scale its own metrics, in the
same format, while `--metric` then applies to the sub-scales only:

```bash
python scale_search.py --groups 24cell --metric soft_tenney \
    --global-metric "edo24:2,steps:1"
```

This matters for properties that only make sense for the whole tuning. An EDO
term placed in `--metric` is applied to each sub-scale separately, and nothing
there stops notes from different cells landing on the same EDO step: in
testing, `--metric soft_tenney:1,edo24:1` on the 24-cell left 2–3 of the 24
steps empty in every run, with notes doubled up on others. With the EDO term
moved into `--global-metric "edo24:2,steps:1"` every run occupied all 24 steps,
at a slightly larger average deviation (about 3 cents instead of 2).

`--global-metric` may consist of scale metrics alone, as above; then no
intervals of the whole scale are scored. It is still blended in by
`--global-weight`. In the output, the metric breakdown reports the whole scale
with these metrics, while the interval matrix and the sub-scale table use
`--metric`.

### Sub-scales around the tonic

The sub-scales containing 1/1 tend to come out somewhat more consonant than the
others, although on a regular polytope every vertex is equivalent. It is a
tendency, not a rule. On the 24-cell (`24cell:cells`, metric
`barlow,euler,wilson,soft_tenney`, `--combine logloss`, `--global-metric edo24`,
`--global-weight 0.1`), six short searches gave the cells containing 1/1 an
average advantage of 0.010–0.025. Looking at every vertex in turn, though, the
cells around a vertex differ from the rest by up to ±0.03, and the vertex 1/1
ranked between 1st and 7th of 24. A single run can therefore show the cells
around 1/1 below average too. Vertices with simpler ratios tend to have better
cells around them.

The cause is the ratio limit, which is measured from 1/1. When the sub-scales
cannot all be equally good, the most consonant region of the scale settles
where the limit leaves the most room: around 1/1, whose neighbours have small
numerators and denominators. A higher `--ratio-limit` loosens this.

Neither the slot nor the scoring favours 1/1. Sub-scales are scored the same in
every mode (see [How grouped scoring works](#how-grouped-scoring-works)), and
moving the tonic to another vertex would only renumber the slots. Things that
did not remove the tendency: `--combine worst`; a mutation that places notes a
simple interval from their sub-scale neighbours rather than from 1/1, which
improved grouped results slightly (better in 11 of 15 test runs, worse in 4);
and scoring sub-scales the same in every mode, after which the advantage was
somewhat larger (0.019 on average in the six runs above, against 0.005 with the
earlier mode-dependent scoring).

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

### Pairwise rules

For some structures the whole grouping reduces to a rule about pairs of notes,
which is far easier to play from than a list of sub-scales. When that is the
case, the output says so:

```
DO NOT COMBINE

These pairs never occur together, grouped by their first note.
Any pair not listed here is fine.
Obeying this is equivalent to staying within a sub-scale.

  1/1  never with  13/12
  5/4  never with  4/3
  3/2  never with  5/3

# Don't play: 1/1 with 13/12; 5/4 with 4/3; 3/2 with 5/3
```

This is printed **only when the pairwise view is complete** — when every maximal
set of mutually compatible notes is itself a sub-scale, so obeying the pair rule
and staying inside a sub-scale are the same thing. Otherwise nothing is printed,
because a pair list would then be a necessary condition wearing the costume of a
rule: on the dodecahedron, 20 note sets avoid every forbidden pair without being
one of the 12 faces.

| Structure | Pairwise rule? |
|---|---|
| `octahedron` | yes — 3 forbidden pairs |
| `16cell`, `16cell:cells+faces` | yes — 4 forbidden pairs |
| `icosahedron` | yes — as permissions |
| `24cell:faces`, `dodecahedron:edges` | yes — as permissions |
| `cube`, `dodecahedron`, `24cell`, `tetrahedron` | no — nothing printed |

Whichever of the two relations is shorter is the one shown. The dodecahedron's
edges forbid 160 pairs but permit only 30, so it prints as `only with` instead.
Note the asymmetry in how the two are listed: a prohibition is symmetric, so
each forbidden pair is stated once and grouped under its first note, but a
permission list has to be complete for every note or it would be false — so
there each note carries its full set and every relation appears twice.

Slot 0 always holds the tonic 1/1. On a vertex-transitive polytope that is an
arbitrary symmetry break: facets containing slot 0 get simple ratios and tend to
score better than the rest. Output includes a `# slot order` block alongside the
sorted scale — the sorted listing is what goes into a tuning file, the slot order
is what reconstructs the geometry.

## Analysis

`--analyze` reports what a result resembles, after the search finishes:

```bash
python scale_search.py --notes 7 --metric "barlow:1,propriety:1" \
    --seeds known.txt --analyze
```

```
Closest scales in known.txt:

   Max dev     RMS   Match  Name
      0.0c    0.0c   7/7    Natural minor        (identical, mode 2)
     21.5c    8.1c   7/7    Ptolemaic diatonic
      0.0c    0.0c   7/7    5-limit chromatic    (within 12-note scale)

Best equal divisions:

   EDO    Fit   Max dev  Steps
    53   0.13      1.5c  0 8 17 22 31 39 48
    12   0.35     17.6c  0 2 4 5 7 9 11
```

Comparison is in cents, not by ratio identity: 9/8 and 10/9 are different
ratios but only 21 cents apart, and a scale built from one is recognisably the
same object as one built from the other — the example above says exactly that,
the result differing from the Ptolemaic diatonic by one syntonic comma. Modes
are checked too, so a result is still recognised as a known scale when it
starts from a different degree.

Scales of a different size are reported as containment rather than identity,
with the `Match` column showing how many of the result's notes landed on the
reference scale: `within N-note scale` when the result lies inside a larger
reference, `contains N-note scale` when a smaller reference lies inside the
result, and `partial` otherwise.

`--edo-range MIN:MAX` sets the divisions searched (default `5:72`). An equal
division only counts if every note lands on a **distinct** step, so a 12-note
scale is never reported as fitting 5-EDO. **Fit** is the worst note's error as
a fraction of half a step: 0 means exactly on the grid, about 0.5 means no
better than chance. Ranking on that rather than on raw cents matters — absolute
error always favours large divisions, so 53-EDO would beat 12-EDO on any scale
whatsoever and the ranking would tell you nothing.

### Comparing against the Scala archive

`--reference PATH` sets what `--analyze` compares against, and accepts a
directory of `.scl` files, a single `.scl` file, or a seed file. It defaults to
`--seeds`. Pointed at the Scala archive it searches the whole thing:

```bash
python scale_search.py --notes 7 --metric "barlow:1,propriety:1" \
    --analyze --reference ~/scala/scl
```

```
Closest scales in ~/scala/scl:  [4483 scales (17 non-octave, 4 too large skipped)]

   Max dev     RMS   Match  File                 Name
  --------  ------  ------  -------------------  ------------------------------
      0.0c    0.0c   7/7    ptolemy_diat.scl     Ptolemy's intense diatonic  (identical)
     15.6c    9.2c   7/7    12edo.scl            12-tone equal temperament  (within 12-note scale)
```

Around two seconds for four and a half thousand files, parsing included.

The filename is shown separately from the description, since in the archive it
is the identifier you would use to find the scale again and descriptions are
neither unique nor always present. Paths stay relative to the directory given,
so nested layouts remain legible. The column is omitted when the source is a
seed file, where every scale would carry the same filename.

The `.scl` format needs a real parser rather than the seed-file reader, which
gets three things wrong on it. The second meaningful line is a **note count**,
not a pitch — read as a pitch, a three-note scale contributes a spurious 3/1.
Pitches may be written in **cents**, marked by a decimal point, which the seed
reader skips, silently emptying the many archive files written that way. And the
scale's **name** is the description line, not a comment.

Two kinds of file are set aside, both reported in the count:

- **Non-octave scales** (Bohlen-Pierce and similar). Everything here compares
  pitch classes under octave equivalence, so octave-reducing a tritave scale and
  then reporting it as a close match would mislead rather than merely
  approximate. `--include-non-octave` loads them anyway.
- **Very large scales**, which cost time proportional to the square of their size
  and say little when matched against seven notes.

Cents are converted to ratios with a denominator limit of 10^6, an error far
below a thousandth of a cent — a `.scl` file holding 12-EDO in cents round-trips
to a perfect 12-EDO fit.

`--analyze` also works on a single `--compare` scale, which answers "what is
this scale?". `--compare` accepts cents as well as ratios, so an equal
temperament or a measured tuning can be identified the same way:

```bash
python scale_search.py --compare "9/8,5/4,4/3,3/2,5/3,15/8" \
    --reference ~/scala/scl --analyze
python scale_search.py --compare "100.0,200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1100.0" \
    --reference ~/scala/scl --analyze
```

`--compare --seeds` ranks every scale in a seed file, a `.scl` file or a whole
archive; with an archive, `--notes` narrows the set before anything is
evaluated.

### Seeding from the Scala archive

The [Scala Archive](https://www.huygens-fokker.org/microtonality/scales.html) inclues over 5350 scales.

`--seeds` accepts the same sources as `--reference`, so the archive can supply
starting points as well as comparisons:

```bash
python scale_search.py --notes 7 --metric "barlow:1,propriety:1" \
    --seeds ~/scala/scl --ratio-limit 32
```

```
Loaded 4500 scales from ~/scala/scl; 1792 usable as seeds, keeping best 150.
  Skipped: 2313 outside 5-9 notes, 392 outside ratio limit 32
  Best seed: <file>.scl (quality 0.9748)
...
Seeded 150 of 736 seed-derived scales (best first); the other 150 start random.
```

(The counts are from a run on a 4500-file test archive; they depend on the
archive and the settings. When some of the kept scales have the wrong size, a
further line says how many the search will resize.)

An archive holds thousands of scales and a population holds tens, so this
selects rather than loads, in three steps:

1. **Size.** The search has always accepted wrong-sized seeds and trimmed or
   extended them, so near misses are passed through for it to resize as usual;
   right-sized scales are simply ranked ahead of them, since a resized seed has
   had notes invented or discarded at random and carries less of the original
   than its score suggests. Scales far from the target are dropped — extending a
   5-note scale to 24 means inventing 19 notes, which is a random scale with
   five anchors, and scoring thousands of such candidates costs more than it
   returns. The accepted band is the target plus or minus a quarter of it, and
   the counts are reported both ways.
2. **Snapping.** Every pitch is moved to the nearest ratio the search can
   express. Roughly half the archive is written in cents, which converts
   faithfully to ratios like `720009/679598` — exact enough to measure against,
   but meaningless to mutate. Within 20 cents the *simplest* candidate wins
   rather than the closest: the nearest ratio to a 400-cent third under a limit
   of 32 is 29/23, which is a worse seed than 5/4 despite being 13 cents nearer.
   A cents-written 12-EDO file snaps to `15/14, 9/8, 6/5, 5/4, 4/3, 7/5, 3/2,
   8/5, 5/3, 9/5, 15/8`. Pitches already written as ratios inside the limit
   are kept as written, so a rational scale passes through unchanged.
3. **Scoring.** What survives is evaluated and the best kept, since a seed only
   earns a population slot if it beats random initialisation.

How many are kept is `--max-seeds`, which applies to every kind of seed source
and is described under [Seed File](#seed-file).

In testing the gain was small — about 0.0003 on a 12-note search — so treat this
as a way to start somewhere musically real rather than as a way to get better
scores.

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

`--seeds` also takes a single `.scl` file or a directory of them (see
[Seeding from the Scala archive](#seeding-from-the-scala-archive)). Only the
first item on a line is read, so a line may carry a comment after the pitch.

A line starting with `#` also names the scale that follows. When a scale has
a `#` name, `!` lines before its pitches are ignored, which the files written
by `--export-seeds` and `--save-result` use for details:

```
# Ptolemy intense diatonic
! 7 notes, quality 0.9978, from seeds.txt
9/8           203.9
5/4           386.3
...
```

`--seeds` may be given several times, mixing seed files and archives; the
sources are combined. Archives are selected from as described above, each on
its own, and the seed limit below applies to the combined set. `--compare` and
`--analyze` read all sources too.

Seeds are adapted to the search rather than used as they are:

- Scales outside the note range are trimmed or extended, both with `--notes`
  and with `--min-notes`/`--max-notes`. This way a 12-note search can be
  informed by 5-note or 31-note scales.
- Excluded notes are dropped and missing required notes added, so `--include`
  doesn't silently disqualify every seed that lacks the note.
- Each seed also contributes its inversion and up to three rotations.

Seeds and their variants fill **at most half the population** by default, best
first; the rest starts random. `--max-seeds N` changes that number, up to the
whole population. Without a limit, a 200-scale seed file used to produce about
1000 seed-derived individuals for a population of 300, so the first generation
held no random scales at all.

The file `seeds.txt` provides over 200 scales for up to 31 notes.

### Exporting a combined seed file

`--export-seeds FILE` merges every `--seeds` source into one ranked seed file
instead of searching:

```bash
python scale_search.py --seeds seeds.txt --seeds ~/scala/scl \
    --min-notes 5 --max-notes 7 --metric tenney --top 20 \
    --export-seeds seeds-5to7-tenney.txt
```

What happens to the scales:

- **Filter.** Without a note count, scales of every size are exported. With
  `--notes`, `--min-notes`, `--max-notes` (either alone sets one bound) or
  `--groups` (its slot count), only those sizes are kept; unlike seeding,
  there is no band of near misses. The scale must also contain every
  `--include` note and no `--exclude` note, the same filter as `--compare`.
- **Snapping.** Archive scales are snapped to ratios under `--ratio-limit` as
  for seeding, so the file holds what the search would actually start from.
  When snapping moved a note by more than a tenth of a cent, the detail line
  says by how much at most. Scales from seed files are copied as written.
  Large scales often cannot be snapped, because two of their notes would land
  on the same ratio; they are counted as not expressible, and a higher
  `--ratio-limit` keeps more of them.
- **Ranking.** Each scale is scored on its own with `--metric`, `--octaves` and
  `--combine`. With `--groups` too, scales are scored as flat scales, as when
  an archive is used for seeding.
- **Modes.** Of a scale and its modes (the same scale started from another
  degree) only the best-scoring one is kept, and exact duplicates only once.
  The search adds rotations of every seed itself.
- **Top N.** `--top N` keeps the best N scales of each note count. Without
  it, all scales that pass the filter are written.

The file lists scales by note count, best first. A header records the
sources, the ranking settings and how many scales of each size were kept.
Archive scales are named `file.scl: description`. The output file may not be
one of the `--seeds` sources.

On a 4500-file test archive plus a small seed file, the export above took about
a second.

### Saving results

`--save-result FILE` appends the best scale of a search to FILE in the same
format, with the date, note count or group structure and quality as its name,
and the full command line on a detail line. The file is created if needed.
Results collect there and can be compared with `--seeds FILE --compare` or
used as seeds later. Rename the `#` line by hand to give a result a name of its
own.

Grouped results are written in slot order (slot 1 first; slot 0 is the tonic
1/1 and is not listed), because that order decides which notes form which
sub-scales. With `--groups`, a seed file is read in its own order rather than
sorted by pitch, so a saved result seeds exactly the same assignment. As a
check, re-seeding a saved icosahedron result with `--generations 0`
reproduced its quality exactly.

## Output

The tool outputs:

- The best scale found (ratios)
- Quality score (0-1, higher = better)
- Interval matrix showing pairwise consonance
- Metric breakdown when using multiple metrics: for interval metrics, how each interval
  ranks among all candidate ratios, averaged over every interval scored (the same
  numbers the search works with); for scale metrics, their value on a fixed 0-1 range
- Match notification if result equals a seed scale
- Elapsed time and ETA during search

Use `--quiet` to suppress everything except the quality score and scale ratios.
Use `--json` for machine-readable output: quality, scale ratios, cents and the
metric breakdown, plus slot order and sub-scales with `--groups` and the
analysis with `--analyze`. `--compare --json` gives the ranking.

Invalid options are reported before the search starts, on stderr, with exit
code 2.

### Saving the result for other programs

`--output-scl FILE` writes the best scale as a Scala `.scl` file, in pitch
order, for Scale Workshop, Scala, synthesizers and tuning apps. The
description line gives the date, note count or group structure and quality,
and a comment line the full command. An existing file is overwritten.

With `--groups` the file also lists the sub-scales as scale degrees, counted
in the file's own order (0 = 1/1, 1 = the lowest note above it, and so on),
which is what is needed to find them on an instrument tuned to the file:

```
! Sub-scales as scale degrees (0 = 1/1):
!   1: 0 3 5 7
!   2: 0 1 3 6
...
```

## Algorithm

1. Generate candidate ratios up to complexity limit
2. Initialize population from seeds + random scales
3. Evolve using crossover, mutation, and local search. Mutations move a note
   by a simple interval or a comma, combine or stack existing notes, take a
   mediant, place the note a simple interval from another note (in grouped
   search, one sharing a sub-scale with it), or pick a random candidate
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

## Testing

```bash
python tests/run_tests.py
```

runs a set of short searches and comparisons with a fixed random seed and
compares their results with stored copies in `tests/expected/`, checks that
invalid options are rejected with a message, and checks that files written by
`--save-result`, `--export-seeds` and `--output-scl` read back to the same
scales. It takes under a minute and uses small test files in `tests/data/`.

A changed result is not necessarily a bug: a change to the algorithm or a
metric is expected to change results. Once the new results are understood to
be right, `python tests/run_tests.py --update` stores them as the new expected
results. Words after the command run only the tests whose names contain them,
e.g. `python tests/run_tests.py group`.

The stored results were produced with Python 3.13. Another Python version may
give different but equally valid results from the same random seed; then run
`--update` once on a version of the code known to be good, and compare later
changes against that.

## License

GPL-3.0

## Development

I won't develop it further and won't merge pull requests, just because I don't want the work of maintaining this project. If you want to contribute, then just fork the repo and publish your changes.
