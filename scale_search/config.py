"""Search configuration."""

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Set, Tuple


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
    """
    max_notes: int = 7
    min_notes: int = 3
    fixed_notes: int = None
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
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [('barlow', 1.0)]
        if self.scale_metrics is None:
            self.scale_metrics = []
        if self.required_notes is None:
            self.required_notes = []
        if self.excluded_notes is None:
            self.excluded_notes = set()
        else:
            self.excluded_notes = set(self.excluded_notes)
        if self.min_notes > self.max_notes:
            self.min_notes, self.max_notes = self.max_notes, self.min_notes
        # When fixed_notes is set, ensure min/max are consistent
        if self.fixed_notes is not None:
            self.min_notes = self.fixed_notes
            self.max_notes = self.fixed_notes
        if self.fixed_notes is not None and len(self.required_notes) > self.fixed_notes - 1:
            raise ValueError(f"Too many required notes ({len(self.required_notes)}) for {self.fixed_notes}-note scale")
        if len(self.required_notes) > self.max_notes - 1:
            raise ValueError(f"Too many required notes ({len(self.required_notes)}) for max {self.max_notes}-note scale")
    
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
