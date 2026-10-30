"""
Scala (.scl) file and archive support.

The Scala archive is thousands of individual .scl files, which is a different
shape from this tool's own seed files and a different format. Three things in
particular need handling that the seed-file reader gets wrong:

  - The second meaningful line is a NOTE COUNT, not a pitch. Read as a pitch,
    a scale of 3 notes would contribute a spurious 3/1.
  - Pitches may be given in cents, marked by a decimal point. The seed reader
    skips those, which silently empties the many archive files written that way.
  - The scale's name is the description line, not a comment.

Format, per the Scala documentation:

    ! name.scl          lines starting with ! are comments
    !
    Description text    first non-comment line
     7                  second: number of pitches that follow
    !
     9/8                a ratio,
     203.910            or cents, distinguished by the decimal point
     2                  a bare integer means n/1
     ...

The final pitch is the repeat interval, usually 2/1.
"""

import os
from fractions import Fraction
from typing import List, Optional, Tuple

__all__ = [
    'ScalaScale', 'parse_scl', 'load_scala_archive', 'load_reference_scales',
]

# Cents are converted to ratios so the rest of the pipeline stays uniform.
# At this denominator the error is far below a thousandth of a cent, which is
# nowhere near the tolerances any of this is compared at.
_CENTS_DENOM_LIMIT = 1000000


class ScalaScale:
    """One parsed .scl file."""

    def __init__(self, name: str, ratios: List[Fraction],
                 repeat: Optional[Fraction], path: str = ""):
        self.name = name
        self.ratios = ratios          # Pitches above the tonic, repeat excluded
        self.repeat = repeat          # Formal repeat interval, usually 2/1
        self.path = path

    @property
    def is_octave(self) -> bool:
        return self.repeat == Fraction(2, 1)

    def __len__(self):
        return len(self.ratios) + 1   # Including the implicit 1/1


def _parse_pitch(token: str) -> Optional[Fraction]:
    """
    One pitch value. A decimal point means cents, a slash means a ratio, and a
    bare integer means n/1. Anything after the value is ignored, as the format
    allows trailing commentary on a pitch line.
    """
    token = token.strip()
    if not token:
        return None
    token = token.split()[0]

    try:
        if '.' in token:
            cents = float(token)
            if cents < 0:
                return None
            return Fraction(2.0 ** (cents / 1200.0)).limit_denominator(
                _CENTS_DENOM_LIMIT)
        if '/' in token:
            num, _, den = token.partition('/')
            n, d = int(num), int(den)
            if n <= 0 or d <= 0:
                return None
            return Fraction(n, d)
        n = int(token)
        if n <= 0:
            return None
        return Fraction(n, 1)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def parse_scl(path: str) -> Optional[ScalaScale]:
    """
    Parse one .scl file. Returns None if it yields no usable pitches.

    Archive files are old and inconsistently encoded, so bytes that aren't
    valid UTF-8 are replaced rather than raising -- a stray character in a
    description shouldn't cost us the scale.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read().splitlines()
    except OSError:
        return None

    meaningful = [ln for ln in raw if not ln.lstrip().startswith('!')]
    if len(meaningful) < 2:
        return None

    description = meaningful[0].strip()

    # The declared count is a cross-check, not a parser instruction: files with
    # a wrong count are common enough that trusting it loses real scales.
    try:
        declared = int(meaningful[1].strip().split()[0])
    except (ValueError, IndexError):
        declared = None

    pitches = []
    for line in meaningful[2:]:
        p = _parse_pitch(line)
        if p is not None:
            pitches.append(p)

    if declared is not None and len(pitches) > declared:
        pitches = pitches[:declared]
    # The tonic is implicit in Scala files, but some list it anyway as 1/1 or
    # 0.0. Kept, it would count as an extra note, and seeding would snap it to
    # whatever comma-sized ratio lies nearest.
    pitches = [p for p in pitches if p != 1]
    if not pitches:
        return None

    repeat = pitches[-1]
    ratios = pitches[:-1]

    name = description or os.path.splitext(os.path.basename(path))[0]
    return ScalaScale(name, ratios, repeat, path)


def load_scala_archive(directory: str, octave_only: bool = True,
                       max_notes: int = 100) -> Tuple[List[List[Fraction]],
                                                      List[str], List[str], dict]:
    """
    Load every .scl file under a directory, recursively.

    Returns (scales, names, files, stats). The filename is carried separately
    from the description because in the archive it is the identifier you would
    use to find the scale again, and descriptions are neither unique nor always
    present. Paths are kept relative to the directory given, so nested archive
    layouts stay legible.

    Two kinds of file are set aside rather than loaded, both reported in stats:

      - Non-octave scales (Bohlen-Pierce and friends). Everything here compares
        pitch classes under octave equivalence, so octave-reducing a tritave
        scale and calling the result a match would be misleading rather than
        merely approximate. Pass octave_only=False to include them anyway.
      - Very large scales, which cost time proportional to the square of their
        size during comparison and tell you little when matched against seven
        notes.
    """
    scales, names, sources = [], [], []
    stats = {'files': 0, 'loaded': 0, 'unparsed': 0,
             'non_octave': 0, 'too_large': 0}

    for root, _, files in os.walk(directory):
        for fn in sorted(files):
            if not fn.lower().endswith('.scl'):
                continue
            stats['files'] += 1
            full = os.path.join(root, fn)
            s = parse_scl(full)
            if s is None or not s.ratios:
                stats['unparsed'] += 1
                continue
            if octave_only and not s.is_octave:
                stats['non_octave'] += 1
                continue
            if len(s) > max_notes:
                stats['too_large'] += 1
                continue
            scales.append(s.ratios)
            names.append(s.name)
            sources.append(os.path.relpath(full, directory))
            stats['loaded'] += 1

    return scales, names, sources, stats


def load_reference_scales(path: str, octave_only: bool = True,
                          max_notes: int = 100):
    """
    Load a comparison library from whatever the path points at: a directory of
    .scl files, a single .scl file, or one of this tool's own seed files.
    """
    from scale_search.display import load_seed_scales

    if os.path.isdir(path):
        return load_scala_archive(path, octave_only, max_notes)

    if path.lower().endswith('.scl'):
        s = parse_scl(path)
        if s is None or not s.ratios:
            return [], [], [], {'files': 1, 'loaded': 0, 'unparsed': 1,
                                'non_octave': 0, 'too_large': 0}
        return ([s.ratios], [s.name], [os.path.basename(path)],
                {'files': 1, 'loaded': 1, 'unparsed': 0,
                 'non_octave': 0, 'too_large': 0})

    # A seed file holds every scale in one file, so a per-scale filename would
    # be the same string on every row; None suppresses the column.
    scales, names = load_seed_scales(path, with_names=True)
    return (scales, names, [None] * len(scales),
            {'files': 1, 'loaded': len(scales), 'unparsed': 0,
             'non_octave': 0, 'too_large': 0})
