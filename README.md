# Evolutionary Search for Just Intonation Scales

Uses an evolutionary algorithm to search for microtonal just intonation scales by various music theoretic constraints.

This was created with Claude (Anthropic).

## Features

- **Multiple consonance metrics**: Tenney height, Euler gradus, harmonic entropy, Plomp-Levelt roughness, Wilson complexity, Barlow harmonicity, odd-limit
- **Scale-level metrics**: Step balance, Rothenberg propriety, prime diversity, EDO approximation
- **Interval combination methods**: Geometric mean (default), worst-case, arithmetic mean
- **Evolutionary optimization**: Genetic algorithm with simulated annealing, adaptive mutation, and fingerprint-based diversity
- **Flexible constraints**: Required/excluded notes, variable note counts, ratio complexity limits
- **Seed scales**: Initialize from existing scales (Scala .scl format supported)

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
| `barlow` | Barlow harmonicity/indigestibility |
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

- The best scale found (ratios and cents)
- Quality score (0-1, higher = better)
- Interval matrix showing pairwise consonance
- Match notification if result equals a seed scale

## Algorithm

1. Generate candidate ratios up to complexity limit
2. Initialize population from seeds + random scales
3. Evolve using crossover, mutation, and local search
4. Apply simulated annealing with adaptive reheating
5. Maintain diversity via fingerprint-based selection
6. Polish best solution with hill climbing

## Tips

- Increasing `min_notes`, `ratio_limit`, `octaves` and `generations` slows down search quite a lot.
- Using the seeds.txt file reduces search time. Most settings converge on the same scales anyway.
- The `entropy` and `roughness` metrics are also slow, and usually you don't need more than one of those two, as they are quite similar.
- Using `entropy` or `roughness` also tends to cluster the notes in a few places. To balance this out you can use `balance`. But I recommend starting with a low value (e.g. `balance:0.05`) and increase when needed.
- Using the `primes` metric can lead to more interesting scales, but is better used with low weight (e.g. `primes:0.3`). Even better to use `variety`.
- Below 7 notes, most combinations of metrics converge on the same scales.
- Including and excluding notes can be used to steer the result towards an intended flavor.
- Metric can also be used with negative value.

## License

GPL-3.0

## Development

I won't develop it further and won't merge pull requests, just because I don't want the work of maintaining this project. If you want to contribute, then just fork the repo and publish your changes.
