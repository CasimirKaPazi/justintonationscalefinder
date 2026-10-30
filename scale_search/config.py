"""Search configuration."""

import difflib
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Set, Tuple

from scale_search.util import reduce_to_octave

__all__ = ['SearchConfig', 'COMBINE_METHODS', 'validate_combine_method', 'normalize_notes']


COMBINE_METHODS = ['geometric', 'worst', 'mean', 'logloss', 'power2', 'power3']


def validate_combine_method(name: str) -> str:
    """
    Check a combine method name, raising ValueError on anything unrecognized.

    Unknown names used to fall through to the arithmetic mean, so a typo like
    'geometic' silently ran a different and much more forgiving combination
    than the one asked for, with nothing in the output to show it.
    """
    if name in COMBINE_METHODS:
        return name
    if name.startswith('power'):
        suffix = name[5:]
        if suffix == '' or (suffix.isdigit() and 1 <= int(suffix) <= 10):
            return name
    close = difflib.get_close_matches(name, COMBINE_METHODS, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    raise ValueError(
        f"Unknown combine method '{name}'.{hint} "
        f"Valid: {', '.join(COMBINE_METHODS)}, or powerN for N in 1-10.")


def normalize_notes(notes: Iterable[Fraction], what: str) -> List[Fraction]:
    """
    Octave-reduce a list of required or excluded notes and drop duplicates.

    Scale notes are always held octave-reduced, so an unreduced constraint such
    as 9/4 could never be matched: as a required note it made every candidate
    fail, and in grouped mode it was inserted next to its own octave-reduced
    form, putting the same pitch class in the genome twice.
    """
    out = []
    seen = set()
    for r in notes:
        r = Fraction(r)
        if r <= 0:
            raise ValueError(f"{what}: {r} is not a positive ratio")
        r = reduce_to_octave(r)
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


@dataclass
class SearchConfig:
    """
    Configuration for scale search.

    Note: 'notes' counts pitch classes INCLUDING the tonic 1/1, but NOT the
    octave repeat 2/1. So a 7-note scale has 7 pitch classes (1/1 plus 6 ratios),
    with 2/1 as the implicit octave boundary.

    Defaults:
        max_notes: 7 (heptatonic)
        min_notes: 3 (minimum viable scale)
        ratio_limit: 64 (allows up to 63-limit intervals)
        population_size: 300
        generations: 500
        mutation_rate: 0.3
        tournament_size: 5
        elite_count: 2
        initial_temp: 0.1
        final_temp: 0.001
        octave_range: 1 (evaluate across 3 octaves)
        combine_method: 'geometric' (how to combine interval scores)
        max_seed_individuals: half of population_size
    """
    max_notes: int = 7
    min_notes: int = 3
    fixed_notes: Optional[int] = None
    ratio_limit: int = 64
    population_size: int = 300
    generations: int = 500
    mutation_rate: float = 0.3
    tournament_size: int = 5
    elite_count: int = 2
    metrics: List[Tuple[str, float]] = None
    scale_metrics: List[Tuple[str, float]] = None
    initial_temp: float = 0.1
    final_temp: float = 0.001
    octave_range: int = 1
    edo: int = 0
    required_notes: List[Fraction] = None
    excluded_notes: Set[Fraction] = None
    combine_method: str = 'geometric'

    # How many members of the initial population may come from seed scales
    # (each seed contributes itself plus its inversion and some rotations).
    # The rest is filled with random scales. None means half the population.
    max_seed_individuals: Optional[int] = None

    # Grouped (sub-scale) search. When `groups` is set the genome is held in
    # SLOT ORDER rather than pitch order: index i is slot i+1, and slot 0 is
    # the implicit tonic 1/1. Each entry of `groups` is a tuple of slot indices
    # forming one sub-scale. `num_slots` is the total note count including the
    # tonic, so the genome has num_slots - 1 entries.
    groups: List[Tuple[int, ...]] = None
    num_slots: Optional[int] = None
    group_name: Optional[str] = None
    # When True, slot i must hold the i-th lowest pitch. Group structures whose
    # slot numbering carries its own meaning -- chromatic step, scale degree --
    # need this, or the search will permute the assignment and satisfy the
    # overlap pattern with unrelated notes. Polytope slots have no such meaning
    # and are left free.
    ordered_slots: bool = False
    # Blend between sub-scale quality and whole-scale quality:
    # final = local^(1 - global_weight) * global^global_weight
    global_weight: float = 0.3
    # Metrics for the whole scale in grouped search. None means the whole
    # scale is scored with the same metrics as the sub-scales.
    global_metrics: Optional[List[Tuple[str, float]]] = None
    global_scale_metrics: Optional[List[Tuple[str, float]]] = None
    global_edo: int = 0

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [('barlow', 1.0)]
        if self.scale_metrics is None:
            self.scale_metrics = []

        self.required_notes = normalize_notes(self.required_notes or [],
                                               'required note')
        # 1/1 is always present, so requiring it is a no-op
        self.required_notes = [r for r in self.required_notes if r != 1]
        excluded = normalize_notes(self.excluded_notes or [], 'excluded note')
        if Fraction(1) in excluded:
            raise ValueError("cannot exclude 1/1 (or 2/1): the tonic is always part of the scale")
        self.excluded_notes = set(excluded)
        clash = sorted(set(self.required_notes) & self.excluded_notes)
        if clash:
            raise ValueError(
                f"{', '.join(str(r) for r in clash)} both required and excluded")

        self._validate_ranges()

        # Group structure fixes the note count: one note per slot.
        if self.groups:
            self.groups = [tuple(g) for g in self.groups]
            if self.num_slots is None:
                self.num_slots = max(max(g) for g in self.groups) + 1
            bad = sorted({i for g in self.groups for i in g
                          if not 0 <= i < self.num_slots})
            if bad:
                raise ValueError(
                    f"group slot indices {bad} outside 0-{self.num_slots - 1}")
            self.fixed_notes = self.num_slots
            if not (0.0 <= self.global_weight <= 1.0):
                raise ValueError(
                    f"global_weight must be between 0 and 1, got {self.global_weight}")
        elif self.global_metrics is not None or self.global_scale_metrics is not None:
            raise ValueError("global metrics only apply to grouped search (--groups)")

        self.combine_method = validate_combine_method(self.combine_method)

        if self.min_notes > self.max_notes:
            self.min_notes, self.max_notes = self.max_notes, self.min_notes
        # When fixed_notes is set, ensure min/max are consistent
        if self.fixed_notes is not None:
            self.min_notes = self.fixed_notes
            self.max_notes = self.fixed_notes
        if len(self.required_notes) > self.max_notes - 1:
            raise ValueError(
                f"Too many required notes ({len(self.required_notes)}) for a "
                f"{self.max_notes}-note scale, which has room for "
                f"{self.max_notes - 1} besides the tonic")

        if self.max_seed_individuals is None:
            self.max_seed_individuals = self.population_size // 2
        self.max_seed_individuals = max(0, min(self.max_seed_individuals,
                                               self.population_size))

    def _validate_ranges(self):
        """
        Reject parameter values that would crash the search or quietly make it
        meaningless, with a message that names the parameter.
        """
        problems = []
        if self.ratio_limit < 3:
            problems.append(f"ratio_limit must be at least 3 (got {self.ratio_limit}); "
                            f"below that no ratio lies strictly inside the octave")
        if self.fixed_notes is not None and self.fixed_notes < 2:
            problems.append(f"note count must be at least 2 (got {self.fixed_notes})")
        if self.fixed_notes is None and not self.groups and self.min_notes < 2:
            problems.append(f"min_notes must be at least 2 (got {self.min_notes})")
        if self.population_size < 2:
            problems.append(f"population size must be at least 2 (got {self.population_size})")
        if self.generations < 0:
            problems.append(f"generations cannot be negative (got {self.generations})")
        if self.initial_temp <= 0 or self.final_temp <= 0:
            problems.append("temperatures must be positive")
        if self.octave_range < 0:
            problems.append(f"octave range cannot be negative (got {self.octave_range})")
        if not 0.0 <= self.mutation_rate <= 1.0:
            problems.append(f"mutation_rate must be between 0 and 1 (got {self.mutation_rate})")
        if problems:
            raise ValueError('; '.join(problems))
        # Tournament selection samples without replacement, so it cannot ask
        # for more contestants than there are individuals.
        self.tournament_size = max(1, min(self.tournament_size, self.population_size))

    def target_num_ratios(self) -> int:
        """
        Return the target number of ratios for a scale.

        If fixed_notes is set, returns fixed_notes - 1.
        Otherwise, returns a random value in [min_notes - 1, max_notes - 1].
        """
        if self.fixed_notes is not None:
            return self.fixed_notes - 1
        return random.randint(self.min_notes - 1, self.max_notes - 1)

    def check_constraints(self, scale) -> bool:
        """
        Check if a scale satisfies required/excluded note constraints.

        Args:
            scale: List or set of Fraction ratios (not including 1/1 or 2/1)

        Returns:
            True if all required notes are present and no excluded notes are present.
        """
        scale_set = set(scale)
        if self.required_notes and not all(r in scale_set for r in self.required_notes):
            return False
        if self.excluded_notes and any(e in scale_set for e in self.excluded_notes):
            return False
        return True
