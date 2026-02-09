# Just Intonation Scale Evolution

Evolutionary search for microtonal scales using multiple consonance metrics.

## Features

- **Multiple consonance metrics**: Tenney height, Euler gradus, harmonic entropy, Plomp-Levelt roughness, Wilson complexity, Barlow harmonicity, odd-limit
- **Scale-level metrics**: Step balance, Rothenberg propriety, prime diversity, EDO approximation
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
python scale_search.py --metric "soft_tenney:1,roughness:0.5,balance:0.2" --notes 7

# Require specific intervals
python scale_search.py --notes 7 --include "3/2,4/3"

# 19-EDO neji (near-equivalent just intonation approximation of 19-EDO)
python scale_search.py --metric "soft_tenney:1,edo19:2" --notes 19

# Elaborate search with various settings
python scale_search.py --metric "euler:1,wilson:1,odd_limit:1,propriety:1,barlow:1,entropy:1,balance:0.2" --population 500 --octaves 2 --ratio-limit 64 --min-notes 12 --max-notes 16 --include "7/6" --seeds seeds.txt
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
| `primes` | Rewards prime diversity (7/11/13-limit) |
| `edo12` | Proximity to 12-EDO (for neji scales) |

## Seed File Format

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

Increasing *min_notes*, *ratio_limit*, *octaves* and *generations* slow down search qutie a lot.

The *entropy* and *roughness* metrics are also slow, and usually you don't need more than one of those two, as they are quite similar.

Using the *primes* metric can lead to more interestig scales, but is better used with low weight (e.g. primes:0.3).

Below 7 notes, most combinations of metrics converge on the same scales.

Including and excluding notes can be used to steer the result towards an intended flavor.

## License

GPL-3.0

Created with Claude (Anthropic)
