"""Genetic operators: mutation, crossover, and scale transformations."""

import math
import random
from fractions import Fraction
from typing import List, Set

from scale_search.config import SearchConfig
from scale_search.util import reduce_to_octave, is_valid_ratio

SIMPLE_INTERVALS = [
    Fraction(3, 2),   # Perfect fifth
    Fraction(4, 3),   # Perfect fourth
    Fraction(5, 4),   # Major third
    Fraction(6, 5),   # Minor third
    Fraction(5, 3),   # Major sixth
    Fraction(8, 5),   # Minor sixth
    Fraction(7, 4),   # Harmonic seventh
    Fraction(7, 6),   # Septimal minor third
    Fraction(9, 8),   # Major second
    Fraction(10, 9),  # Minor whole tone
    Fraction(8, 7),   # Septimal major second
    Fraction(11, 8),  # Undecimal tritone
    Fraction(11, 9),  # Undecimal neutral third
    Fraction(13, 8),  # Tridecimal neutral sixth
]

# Musical commas for subtle pitch shifts
COMMAS = [
    Fraction(81, 80),   # Syntonic comma (21.5 cents) - 5-limit
    Fraction(64, 63),   # Septimal comma (27.3 cents) - 7-limit
    Fraction(33, 32),   # Undecimal comma (53.3 cents) - 11-limit
    Fraction(128, 125), # Diesis (41.1 cents) - 5-limit
    Fraction(36, 35),   # Septimal diesis (48.8 cents) - 7-limit
    Fraction(49, 48),   # Septimal sixth-tone (35.7 cents) - 7-limit
    Fraction(225, 224), # Septimal kleisma (7.7 cents) - very small
    Fraction(126, 125), # Septimal semicomma (13.8 cents)
]


def mutate_transpose(note: Fraction, limit: int) -> Fraction:
    """Transpose a note by a simple interval (up or down)."""
    interval = random.choice(SIMPLE_INTERVALS)
    if random.random() < 0.5:
        new_note = note * interval
    else:
        new_note = note / interval
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate_comma_shift(note: Fraction, limit: int) -> Fraction:
    """Shift a note by a musical comma (subtle pitch adjustment)."""
    comma = random.choice(COMMAS)
    if random.random() < 0.5:
        new_note = note * comma
    else:
        new_note = note / comma
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate_combine(note1: Fraction, note2: Fraction, limit: int) -> Fraction:
    """Combine two notes by multiplication or division."""
    if random.random() < 0.5:
        new_note = note1 * note2
    else:
        new_note = note1 / note2
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def mutate_stack(scale: List[Fraction], limit: int) -> Fraction:
    """
    La Monte Young style: stack intervals to create complex ratios.
    E.g., 7/4 Ã— 7/4 Ã— 3/2 Ã— 3/2 Ã— 3/2 = 1323/1024
    """
    # Choose 2-5 intervals to stack (from scale or simple intervals)
    num_stacks = random.randint(2, 5)
    pool = list(scale) + SIMPLE_INTERVALS[:8]
    
    result = Fraction(1, 1)
    for _ in range(num_stacks):
        interval = random.choice(pool)
        if random.random() < 0.8:  # Usually multiply
            result = result * interval
        else:
            result = result / interval
    
    result = reduce_to_octave(result)
    if is_valid_ratio(result, limit):
        return result
    return None


def mutate_mediant(note1: Fraction, note2: Fraction, limit: int) -> Fraction:
    """
    Create mediant of two ratios: (a+c)/(b+d) for a/b and c/d.
    This creates ratios "between" two existing ones in the Stern-Brocot tree.
    """
    a, b = note1.numerator, note1.denominator
    c, d = note2.numerator, note2.denominator
    new_note = Fraction(a + c, b + d)
    new_note = reduce_to_octave(new_note)
    if is_valid_ratio(new_note, limit):
        return new_note
    return None


def scale_invert(scale: List[Fraction], limit: int = None) -> List[Fraction]:
    """
    Invert a scale - mirror all intervals around the octave.
    
    For each ratio r, the inverted ratio is 2/r.
    Turns major into minor and vice versa.
    
    Example: Major scale [9/8, 5/4, 4/3, 3/2, 5/3, 15/8]
          -> Inverted:   [16/15, 6/5, 4/3, 3/2, 8/5, 16/9]
             (which is Phrygian mode)
    
    Args:
        scale: List of ratios (between 1 and 2, not including 1/1 or 2/1)
        limit: Optional ratio limit - if set, filters out ratios exceeding it
    
    Returns:
        Inverted scale, sorted by pitch
    """
    inverted = []
    for r in scale:
        inv = Fraction(2, 1) / r  # 2/r
        # Reduce to octave (should already be, but be safe)
        while inv >= 2:
            inv = inv / 2
        while inv < 1:
            inv = inv * 2
        
        # Skip 1/1 and 2/1 - these are implicit, not scale degrees
        if inv == Fraction(1, 1) or inv == Fraction(2, 1):
            continue
        
        if limit is None or (inv.numerator <= limit and inv.denominator <= limit):
            inverted.append(inv)
    
    return sorted(inverted, key=lambda r: float(r))


def scale_rotate(scale: List[Fraction], pivot_idx: int = None, limit: int = None) -> List[Fraction]:
    """
    Rotate a scale - choose a new tonic (mode change).
    
    Makes the note at pivot_idx become the new 1/1, and recalculates
    all other intervals relative to it.
    
    Example: Major scale [9/8, 5/4, 4/3, 3/2, 5/3, 15/8], pivot on 5/4
          -> Dorian-ish: intervals relative to 5/4
    
    Args:
        scale: List of ratios (between 1 and 2, not including 1/1 or 2/1)
        pivot_idx: Index of note to become new 1/1. If None, random choice.
        limit: Optional ratio limit - if set, filters out ratios exceeding it
    
    Returns:
        Rotated scale, sorted by pitch
    """
    if not scale:
        return scale
    
    if pivot_idx is None:
        pivot_idx = random.randrange(len(scale))
    
    pivot = scale[pivot_idx]
    
    # Build new scale: all intervals relative to pivot
    # The pivot itself becomes 1/1 (implicit, not included)
    # The original 1/1 becomes a note relative to pivot
    
    rotated = []
    
    # Original 1/1 relative to pivot = 1/pivot (then octave reduce)
    inv_pivot = Fraction(1, 1) / pivot
    while inv_pivot < 1:
        inv_pivot = inv_pivot * 2
    while inv_pivot >= 2:
        inv_pivot = inv_pivot / 2
    # Only add if not 1/1 or 2/1 and within limit
    if inv_pivot != Fraction(1, 1) and inv_pivot != Fraction(2, 1):
        if limit is None or (inv_pivot.numerator <= limit and inv_pivot.denominator <= limit):
            rotated.append(inv_pivot)
    
    # Each other note relative to pivot
    for i, note in enumerate(scale):
        if i == pivot_idx:
            continue  # This becomes 1/1, skip
        
        rel = note / pivot
        # Reduce to octave
        while rel >= 2:
            rel = rel / 2
        while rel < 1:
            rel = rel * 2
        
        # Skip 1/1 and 2/1
        if rel == Fraction(1, 1) or rel == Fraction(2, 1):
            continue
        
        if limit is None or (rel.numerator <= limit and rel.denominator <= limit):
            rotated.append(rel)
    
    return sorted(set(rotated), key=lambda r: float(r))

def mutate(scale: List[Fraction], candidates: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Mutate a scale using musically-meaningful operations.
    
    Selects a mutation type by weighted random roll, tries it, and falls through
    to the next type if it fails (e.g., result exceeds ratio limit). The fallback
    chain ensures we almost always produce a mutation.
    
    Operations (in fallback order):
    1. Transpose a note by a simple interval (fifth, third, seventh, etc.)
    2. Comma shift (subtle pitch adjustment by syntonic/septimal comma)
    3. Combine two notes from the scale (like La Monte Young's 1323/1024)
    4. Stack intervals La Monte Young style (7/4 × 7/4 × 3/2 × 3/2...)
    5. Mediant - create ratio "between" two scale notes
    6. Random replacement from candidates (final fallback)
    
    Respects config.required_notes (won't remove) and config.excluded_notes (won't add).
    """
    scale = scale.copy()
    scale_set = set(scale)
    limit = config.ratio_limit
    required_set = set(config.required_notes) if config.required_notes else set()
    excluded_set = config.excluded_notes
    
    # Indices of notes that can be mutated (not required)
    mutable_indices = [i for i, note in enumerate(scale) if note not in required_set]
    if not mutable_indices:
        return scale  # All notes are required, can't mutate
    
    def try_replace(new_note, idx):
        """Replace note at idx if new_note is valid and not excluded."""
        if (new_note and 
            new_note not in scale_set and 
            new_note not in excluded_set and
            new_note != Fraction(1,1) and 
            new_note != Fraction(2,1)):
            scale[idx] = new_note
            return True
        return False
    
    # Build fallback chain based on initial random roll.
    # Each mutation type has a probability threshold; we try the selected type
    # and all subsequent types until one succeeds.
    r = random.random()
    
    # Determine starting point in fallback chain
    if r < 0.18:
        start = 0   # Start with transpose
    elif r < 0.30:
        start = 1   # Start with comma shift
    elif r < 0.48:
        start = 2   # Start with combine
    elif r < 0.62:
        start = 3   # Start with stack
    elif r < 0.74:
        start = 4   # Start with mediant
    else:
        start = 5   # Skip to random replacement
    
    # Try each operation from starting point onwards
    for op in range(start, 6):
        idx = random.choice(mutable_indices)
        
        if op == 0:
            # Transpose by simple interval
            if try_replace(mutate_transpose(scale[idx], limit), idx):
                return scale
        
        elif op == 1:
            # Comma shift (subtle pitch adjustment)
            if try_replace(mutate_comma_shift(scale[idx], limit), idx):
                return scale
        
        elif op == 2 and len(scale) >= 2:
            # Combine two scale notes
            note1 = scale[idx]
            others = [n for n in scale if n != note1]
            note2 = random.choice(others) if others else note1
            if try_replace(mutate_combine(note1, note2, limit), idx):
                return scale
        
        elif op == 3:
            # La Monte Young style interval stacking
            if try_replace(mutate_stack(scale, limit), idx):
                return scale
        
        elif op == 4 and len(scale) >= 2:
            # Mediant of two scale notes
            note1 = scale[idx]
            others = [n for n in scale if n != note1]
            note2 = random.choice(others) if others else note1
            if try_replace(mutate_mediant(note1, note2, limit), idx):
                return scale
        
        elif op == 5:
            # Final fallback: random replacement from candidates
            available = [c for c in candidates if c not in scale_set and c not in excluded_set]
            if available:
                scale[idx] = random.choice(available)
                return scale
    
    return scale


def crossover_uniform(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Uniform crossover: combine both parents and randomly sample.
    Simple but doesn't preserve structure.
    Respects required_notes (always included) and excluded_notes (filtered out).
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = config.excluded_notes
    
    # Combine parents, filter excluded, ensure required
    combined = (set(parent1) | set(parent2)) - excluded
    combined = combined | required  # Add back any required notes
    combined = list(combined)
    
    num_ratios = config.target_num_ratios()
    
    if len(combined) < num_ratios:
        if config.fixed_notes is not None:
            return parent1.copy()  # Can't meet fixed requirement
        # Variable notes: just return what we have if it meets minimum
        if len(combined) >= config.min_notes - 1:
            return combined
        return parent1.copy()
    
    if len(combined) == num_ratios:
        return combined
    
    # More notes than target: keep required, sample rest
    optional = [n for n in combined if n not in required]
    num_optional = num_ratios - len(required)
    if num_optional <= 0:
        return list(required)[:num_ratios]
    return list(required) + random.sample(optional, min(num_optional, len(optional)))


def crossover_blend(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Blend crossover: alternate taking notes from each parent (sorted by pitch).
    Preserves some structural properties from both parents.
    Respects required_notes and excluded_notes.
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = config.excluded_notes
    
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Interleave from both parents
    child = list(required)  # Start with required notes
    i, j = 0, 0
    use_p1 = random.random() < 0.5
    
    target_len = config.target_num_ratios()
    
    seen = set(required)
    while len(child) < target_len and (i < len(p1_sorted) or j < len(p2_sorted)):
        if use_p1 and i < len(p1_sorted):
            note = p1_sorted[i]
            i += 1
        elif j < len(p2_sorted):
            note = p2_sorted[j]
            j += 1
        elif i < len(p1_sorted):
            note = p1_sorted[i]
            i += 1
        else:
            break
        
        if note not in seen and note not in excluded:
            child.append(note)
            seen.add(note)
        use_p1 = not use_p1
    
    return child if child else parent1.copy()


def crossover_pivot(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Pivot crossover: choose a pivot pitch, take lower notes from one parent
    and higher notes from the other.
    Respects required_notes and excluded_notes.
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = config.excluded_notes
    
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Choose pivot point (in cents)
    pivot = random.uniform(200, 1000)  # Somewhere in the middle of the octave
    
    # Take notes below pivot from parent1, above from parent2 (or vice versa)
    if random.random() < 0.5:
        p1_sorted, p2_sorted = p2_sorted, p1_sorted
    
    child = list(required)  # Start with required notes
    seen = set(required)
    
    for note in p1_sorted:
        cents = 1200 * math.log2(float(note))
        if cents < pivot and note not in seen and note not in excluded:
            child.append(note)
            seen.add(note)
    
    for note in p2_sorted:
        cents = 1200 * math.log2(float(note))
        if cents >= pivot and note not in seen and note not in excluded:
            child.append(note)
            seen.add(note)
    
    # Adjust to target length
    target_len = config.target_num_ratios()
    
    if len(child) > target_len:
        # Must keep required, sample rest
        optional = [n for n in child if n not in required]
        num_optional = target_len - len(required)
        if num_optional <= 0:
            child = list(required)[:target_len]
        else:
            child = list(required) + random.sample(optional, min(num_optional, len(optional)))
    elif len(child) < target_len:
        # Fill from combined parents (excluding excluded notes)
        all_notes = (set(parent1) | set(parent2)) - excluded
        available = [n for n in all_notes if n not in seen]
        while len(child) < target_len and available:
            note = random.choice(available)
            child.append(note)
            available.remove(note)
    
    return child if child else parent1.copy()


def crossover_interval_preserving(parent1: List[Fraction], parent2: List[Fraction], 
                                   config: SearchConfig) -> List[Fraction]:
    """
    Interval-preserving crossover: try to preserve good interval relationships
    from both parents by keeping contiguous pitch regions.
    Respects required_notes and excluded_notes.
    """
    required = set(config.required_notes) if config.required_notes else set()
    excluded = config.excluded_notes
    
    p1_sorted = sorted(parent1, key=lambda r: float(r))
    p2_sorted = sorted(parent2, key=lambda r: float(r))
    
    # Choose how many contiguous notes to take from each parent
    if len(p1_sorted) >= 2 and len(p2_sorted) >= 2:
        # Take a contiguous region from parent1
        start1 = random.randint(0, len(p1_sorted) - 1)
        end1 = random.randint(start1, len(p1_sorted))
        region1 = p1_sorted[start1:end1]
        
        # Take complementary region from parent2 (notes not overlapping in pitch)
        if region1:
            min_cents = 1200 * math.log2(float(min(region1)))
            max_cents = 1200 * math.log2(float(max(region1)))
            region2 = [n for n in p2_sorted 
                      if 1200 * math.log2(float(n)) < min_cents - 50 
                      or 1200 * math.log2(float(n)) > max_cents + 50]
        else:
            region2 = p2_sorted
        
        child_set = (set(region1) | set(region2) | required) - excluded
        child = list(child_set)
    else:
        child_set = (set(parent1) | set(parent2) | required) - excluded
        child = list(child_set)
    
    # Adjust to target length
    target_len = config.target_num_ratios()
    
    if len(child) > target_len:
        # Must keep required, sample rest
        optional = [n for n in child if n not in required]
        num_optional = target_len - len(required)
        if num_optional <= 0:
            child = list(required)[:target_len]
        else:
            child = list(required) + random.sample(optional, min(num_optional, len(optional)))
    elif len(child) < target_len:
        all_notes = (set(parent1) | set(parent2)) - excluded
        available = [n for n in all_notes if n not in child]
        while len(child) < target_len and available:
            note = random.choice(available)
            child.append(note)
            available.remove(note)
    
    return child if child else parent1.copy()


def crossover(parent1: List[Fraction], parent2: List[Fraction], config: SearchConfig) -> List[Fraction]:
    """
    Combine two scales using a randomly chosen crossover operator.
    """
    r = random.random()
    if r < 0.3:
        return crossover_uniform(parent1, parent2, config)
    elif r < 0.5:
        return crossover_blend(parent1, parent2, config)
    elif r < 0.7:
        return crossover_pivot(parent1, parent2, config)
    else:
        return crossover_interval_preserving(parent1, parent2, config)



def create_random_scale(candidates: List[Fraction], num_ratios: int, 
                        weights: List[float] = None,
                        required: List[Fraction] = None,
                        excluded: Set[Fraction] = None) -> List[Fraction]:
    """
    Create a random scale with the given number of ratios.
    
    Args:
        candidates: List of possible ratios to choose from
        num_ratios: Target number of ratios in scale
        weights: Optional weights for biased sampling (toward simpler ratios)
        required: Notes that must be included in the scale
        excluded: Notes that must not be included
    """
    if required is None:
        required = []
    if excluded is None:
        excluded = set()
    
    # Start with required notes
    selected = list(required)
    selected_set = set(selected)
    
    # Filter candidates: remove required (already have them) and excluded
    available_candidates = []
    available_weights = []
    for i, c in enumerate(candidates):
        if c not in selected_set and c not in excluded:
            available_candidates.append(c)
            if weights is not None and len(weights) == len(candidates):
                available_weights.append(weights[i])
    
    # How many more do we need?
    remaining = num_ratios - len(selected)
    
    if remaining <= 0:
        return selected
    
    if available_weights and len(available_weights) == len(available_candidates):
        # Weighted sampling without replacement
        available_idx = list(range(len(available_candidates)))
        current_weights = list(available_weights)
        
        for _ in range(min(remaining, len(available_candidates))):
            if not available_idx:
                break
            # Normalize weights
            total = sum(current_weights)
            if total <= 0:
                idx_in_list = random.randrange(len(available_idx))
            else:
                r = random.random() * total
                cumsum = 0
                idx_in_list = 0
                for i, w in enumerate(current_weights):
                    cumsum += w
                    if r <= cumsum:
                        idx_in_list = i
                        break
            
            selected.append(available_candidates[available_idx[idx_in_list]])
            available_idx.pop(idx_in_list)
            current_weights.pop(idx_in_list)
    else:
        # Uniform sampling
        extras = random.sample(available_candidates, min(remaining, len(available_candidates)))
        selected.extend(extras)
    
    return selected

