"""
Sub-scale group structures.

A "group structure" assigns N notes to N slots and defines a set of sub-scales,
each a subset of those slots. The idea: build one large scale that decomposes
into smaller playable scales, so a musician moves between adjacent sub-scales
sharing some notes rather than using all N notes at once.

Polytope structures take the vertices of a regular polytope as slots and its
facets as sub-scales. On a dodecahedron, 20 vertices become 20 notes and each
of the 12 pentagonal faces is a 5-note sub-scale; adjacent faces share the 2
notes of their common edge.

Structures available:
    tetrahedron   4 slots, 4 sub-scales of 3
    octahedron    6 slots, 8 sub-scales of 3
    cube          8 slots, 6 sub-scales of 4
    icosahedron  12 slots, 20 sub-scales of 3
    dodecahedron 20 slots, 12 sub-scales of 5
    24cell       24 slots, 24 sub-scales of 6
    cyclic:N,k[,step]  N slots, sub-scales of k consecutive slots (wrapping)
    <filename>   one sub-scale per line, slot indices, '#' for comments

Slot numbering is an arbitrary but fixed labelling of the vertices. Slot 0
always holds the tonic 1/1.
"""

import math
import os
from itertools import combinations
from typing import List, Tuple

__all__ = [
    'BUILTIN_STRUCTURES', 'GroupStructure', 'RANK_NAMES', 'build_polytope',
    'parse_group_spec', 'describe_structures',
]

_PHI = (1 + math.sqrt(5)) / 2
_EPS = 1e-7


class GroupStructure:
    """A slot count plus the sub-scales defined over those slots."""

    def __init__(self, name: str, num_slots: int, groups: List[Tuple[int, ...]],
                 description: str = ""):
        self.name = name
        self.num_slots = num_slots
        self.groups = [tuple(g) for g in groups]
        self.description = description

    @property
    def group_size(self):
        """Size of sub-scales, or None if they differ."""
        sizes = {len(g) for g in self.groups}
        return sizes.pop() if len(sizes) == 1 else None

    def validate(self):
        """Check slot indices are in range and every slot is used."""
        for g in self.groups:
            for idx in g:
                if not (0 <= idx < self.num_slots):
                    raise ValueError(
                        f"{self.name}: slot index {idx} out of range "
                        f"(structure has {self.num_slots} slots, 0-{self.num_slots - 1})")
            if len(set(g)) != len(g):
                raise ValueError(f"{self.name}: sub-scale {g} repeats a slot")
        covered = set()
        for g in self.groups:
            covered.update(g)
        missing = sorted(set(range(self.num_slots)) - covered)
        if missing:
            raise ValueError(
                f"{self.name}: slots {missing} appear in no sub-scale — "
                f"those notes would be unconstrained")
        return self

    def adjacency(self) -> List[Tuple[int, int, int]]:
        """Return (i, j, shared) for each pair of sub-scales sharing >=1 slot."""
        result = []
        for i in range(len(self.groups)):
            for j in range(i + 1, len(self.groups)):
                shared = len(set(self.groups[i]) & set(self.groups[j]))
                if shared:
                    result.append((i, j, shared))
        return result

    def incompatible_pairs(self) -> List[Tuple[int, int]]:
        """
        Slot pairs that never share a sub-scale.

        Often a shorter description of a structure than the sub-scale list:
        the octahedron's 8 faces are equivalent to just 3 forbidden pairs (its
        antipodal vertices).
        """
        together = set()
        for g in self.groups:
            for pair in combinations(sorted(g), 2):
                together.add(pair)
        return [p for p in combinations(range(self.num_slots), 2)
                if p not in together]

    def _cooccurrence(self):
        """Adjacency sets of the graph 'these two slots share a sub-scale'."""
        adj = {i: set() for i in range(self.num_slots)}
        for g in self.groups:
            for a, b in combinations(sorted(g), 2):
                adj[a].add(b)
                adj[b].add(a)
        return adj

    def maximal_cliques(self) -> List[Tuple[int, ...]]:
        """Maximal sets of mutually compatible slots (Bron-Kerbosch, pivoted)."""
        adj = self._cooccurrence()
        out = []

        def expand(r, p, x):
            if not p and not x:
                out.append(tuple(sorted(r)))
                return
            pivot = max(p | x, key=lambda u: len(adj[u]))
            for v in list(p - adj[pivot]):
                expand(r | {v}, p & adj[v], x & adj[v])
                p = p - {v}
                x = x | {v}

        expand(set(), set(range(self.num_slots)), set())
        return sorted(out)

    def exclusions_are_exact(self) -> bool:
        """
        Whether the forbidden pairs fully determine the sub-scales.

        True when every maximal set of mutually compatible slots is itself a
        sub-scale — then "avoid these pairs" and "play these sub-scales" say
        the same thing. False when some combination avoids every forbidden pair
        without being a sanctioned sub-scale, which happens whenever sub-scales
        are nested (a 16-cell triangle sits inside a tetrahedron, so the
        triangles impose no pair restriction the tetrahedra don't already).
        """
        groups = {tuple(sorted(g)) for g in self.groups}
        return all(c in groups for c in self.maximal_cliques())

    def summary(self) -> str:
        size = self.group_size
        size_desc = f"{size}" if size is not None else "mixed size"
        return (f"{self.num_slots} notes, {len(self.groups)} sub-scales of {size_desc}")


# =============================================================================
# POLYTOPE VERTEX COORDINATES
# =============================================================================

def _cyclic_perms(a, b, c):
    """The three cyclic permutations of a coordinate triple."""
    return [(a, b, c), (b, c, a), (c, a, b)]


def _vertices_tetrahedron():
    return [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def _vertices_cube():
    return [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)]


def _vertices_octahedron():
    return [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _vertices_icosahedron():
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts.extend(_cyclic_perms(0.0, s1 * 1.0, s2 * _PHI))
    return verts


def _vertices_dodecahedron():
    verts = [(x, y, z) for x in (1.0, -1.0) for y in (1.0, -1.0) for z in (1.0, -1.0)]
    inv_phi = 1.0 / _PHI
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts.extend(_cyclic_perms(0.0, s1 * inv_phi, s2 * _PHI))
    return verts


def _vertices_24cell():
    """All permutations of (+-1, +-1, 0, 0) — 24 vertices."""
    verts = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i] = si
                    v[j] = sj
                    verts.append(tuple(v))
    return verts


def _vertices_16cell():
    """The 8 points (+-1, 0, 0, 0) and permutations — the 4D cross-polytope."""
    verts = []
    for i in range(4):
        for sign in (1, -1):
            v = [0, 0, 0, 0]
            v[i] = sign
            verts.append(tuple(v))
    return verts


def _normals_16cell():
    """Cell normals of the 16-cell: the 16 points (+-1,+-1,+-1,+-1)."""
    return [(a, b, c, d)
            for a in (1, -1) for b in (1, -1)
            for c in (1, -1) for d in (1, -1)]


def _normals_24cell():
    """
    Cell normals of the 24-cell: the vertices of its dual, which is another
    24-cell — the 16 points (+-1,+-1,+-1,+-1) and the 8 points (+-2,0,0,0)
    and permutations.
    """
    normals = [(a, b, c, d)
               for a in (1, -1) for b in (1, -1)
               for c in (1, -1) for d in (1, -1)]
    for i in range(4):
        for sign in (2, -2):
            v = [0, 0, 0, 0]
            v[i] = sign
            normals.append(tuple(v))
    return normals


# =============================================================================
# FACET COMPUTATION
# =============================================================================

def _facets_3d(vertices) -> List[Tuple[int, ...]]:
    """
    Find convex hull facets of a 3D point set by brute force over vertex triples.

    For each triple, the plane through it is a supporting plane of the hull if
    every vertex lies on one side. The facet is then all vertices on that plane.
    Derived from the vertex set directly rather than from dual coordinates,
    which avoids depending on a particular cyclic-permutation convention lining
    up between a solid and its dual.
    """
    n = len(vertices)
    facets = set()

    for i, j, k in combinations(range(n), 3):
        a, b, c = vertices[i], vertices[j], vertices[k]
        # Normal via cross product of two edge vectors
        u = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        v = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        nx = u[1] * v[2] - u[2] * v[1]
        ny = u[2] * v[0] - u[0] * v[2]
        nz = u[0] * v[1] - u[1] * v[0]
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm < _EPS:
            continue  # Collinear triple
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        offset = nx * a[0] + ny * a[1] + nz * a[2]

        dots = [nx * p[0] + ny * p[1] + nz * p[2] for p in vertices]
        above = any(d > offset + _EPS for d in dots)
        below = any(d < offset - _EPS for d in dots)
        if above and below:
            continue  # Plane cuts through the solid, not a supporting plane

        facet = tuple(sorted(idx for idx, d in enumerate(dots)
                             if abs(d - offset) < _EPS))
        if len(facet) >= 3:
            facets.add(facet)

    return sorted(facets)


def _facets_by_normals(vertices, normals) -> List[Tuple[int, ...]]:
    """
    Facets as the vertices maximizing the dot product with each given normal.

    Used for the 4D polytopes. Brute-force supporting-hyperplane search works
    in 3D, but in 4D it needs vertex quadruples rather than triples, which is
    both slower and more numerically delicate — so the known facet normals are
    used there instead. Facet counts are checked against the published
    combinatorics either way.
    """
    facets = set()
    for nrm in normals:
        dots = [sum(nrm[t] * p[t] for t in range(len(nrm))) for p in vertices]
        best = max(dots)
        facet = tuple(sorted(idx for idx, d in enumerate(dots)
                             if abs(d - best) < _EPS))
        if len(facet) >= len(nrm):
            facets.add(facet)
    return sorted(facets)


def _edges_of(vertices) -> List[Tuple[int, ...]]:
    """
    Edges: the vertex pairs at minimum separation.

    Exact for regular polytopes, where every edge has the same length and no
    non-adjacent pair is that close.
    """
    n = len(vertices)
    best = None
    for i, j in combinations(range(n), 2):
        d2 = sum((vertices[i][t] - vertices[j][t]) ** 2
                 for t in range(len(vertices[i])))
        if best is None or d2 < best - _EPS:
            best = d2
    return sorted((i, j) for i, j in combinations(range(n), 2)
                  if abs(sum((vertices[i][t] - vertices[j][t]) ** 2
                             for t in range(len(vertices[i]))) - best) < _EPS)


def _ridges_of(facets, min_size: int) -> List[Tuple[int, ...]]:
    """
    Ridges: the (d-2)-faces, found as intersections of adjacent facets.

    Two octahedral cells of the 24-cell meet in a triangle; two tetrahedral
    cells of the 16-cell likewise. Intersections smaller than min_size are
    lower-rank features (edges, single vertices) and are discarded.
    """
    ridges = set()
    for a, b in combinations(facets, 2):
        shared = tuple(sorted(set(a) & set(b)))
        if len(shared) >= min_size:
            ridges.add(shared)
    return sorted(ridges)


# Rank names. For a d-polytope the facets have rank d-1: 'faces' for the
# 3D solids, 'cells' for the 4D ones. Rank 1 is always edges, which as
# sub-scales are single intervals — a way of asking that every edge of the
# polytope be consonant.
RANK_NAMES = {'cells': 3, 'faces': 2, 'edges': 1}
_RANK_LABEL = {3: 'cells', 2: 'faces', 1: 'edges'}

# dim, vertex source, facet normals (4D only), and the published counts for
# each rank as (how many, how many vertices each). The counts are asserted at
# build time: a geometry bug that produced plausible-looking garbage would
# otherwise give a structure that runs and means nothing.
_POLYTOPES = {
    'tetrahedron': {
        'dim': 3, 'vertices': _vertices_tetrahedron,
        'desc': 'regular tetrahedron',
        'counts': {2: (4, 3), 1: (6, 2)},
    },
    'octahedron': {
        'dim': 3, 'vertices': _vertices_octahedron,
        'desc': 'regular octahedron',
        'counts': {2: (8, 3), 1: (12, 2)},
    },
    'cube': {
        'dim': 3, 'vertices': _vertices_cube,
        'desc': 'cube',
        'counts': {2: (6, 4), 1: (12, 2)},
    },
    'icosahedron': {
        'dim': 3, 'vertices': _vertices_icosahedron,
        'desc': 'regular icosahedron',
        'counts': {2: (20, 3), 1: (30, 2)},
    },
    'dodecahedron': {
        'dim': 3, 'vertices': _vertices_dodecahedron,
        'desc': 'regular dodecahedron',
        'counts': {2: (12, 5), 1: (30, 2)},
    },
    '16cell': {
        'dim': 4, 'vertices': _vertices_16cell, 'normals': _normals_16cell,
        'desc': '16-cell (4D cross-polytope)',
        'counts': {3: (16, 4), 2: (32, 3), 1: (24, 2)},
    },
    '24cell': {
        'dim': 4, 'vertices': _vertices_24cell, 'normals': _normals_24cell,
        'desc': '24-cell (4D, self-dual)',
        'counts': {3: (24, 6), 2: (96, 3), 1: (96, 2)},
    },
}


def _faces_of_rank(name: str, rank: int):
    """Return (vertices, groups) for one rank of a polytope."""
    data = _POLYTOPES[name]
    dim = data['dim']
    verts = data['vertices']()

    if rank not in data['counts']:
        valid = ', '.join(f"{_RANK_LABEL[r]} (rank {r})"
                          for r in sorted(data['counts'], reverse=True))
        raise ValueError(
            f"{name} has no rank-{rank} faces usable as sub-scales. "
            f"Available: {valid}")

    if rank == dim - 1:
        groups = (_facets_3d(verts) if dim == 3
                  else _facets_by_normals(verts, data['normals']()))
    elif rank == 1:
        groups = _edges_of(verts)
    elif rank == dim - 2:
        facets = _facets_by_normals(verts, data['normals']())
        groups = _ridges_of(facets, data['counts'][rank][1])
    else:
        raise ValueError(f"{name}: rank {rank} not supported")

    exp_count, exp_size = data['counts'][rank]
    if len(groups) != exp_count:
        raise ValueError(f"{name} rank {rank}: got {len(groups)} faces, "
                         f"expected {exp_count}")
    sizes = {len(g) for g in groups}
    if sizes != {exp_size}:
        raise ValueError(f"{name} rank {rank}: face sizes {sorted(sizes)}, "
                         f"expected all {exp_size}")
    return verts, groups


def build_polytope(name: str, ranks=None) -> GroupStructure:
    """
    Build a polytope structure using one or more ranks of its face lattice.

    With several ranks the sub-scales are of mixed size — on the 16-cell,
    'cells+faces' gives 16 four-note sub-scales and 32 three-note ones over the
    same 8 slots, so the same tuning supports two levels of grouping at once.
    Every sub-scale is scored on its own terms and they are combined equally,
    so the smaller ones are not penalized for being smaller.
    """
    data = _POLYTOPES[name]
    if not ranks:
        ranks = [data['dim'] - 1]
    ranks = sorted(set(ranks), reverse=True)

    verts = None
    all_groups = []
    for rank in ranks:
        verts, groups = _faces_of_rank(name, rank)
        all_groups.extend(groups)

    # Deduplicate while keeping order (ranks can't collide, but be safe)
    seen = set()
    unique = []
    for g in all_groups:
        if g not in seen:
            seen.add(g)
            unique.append(g)

    labels = '+'.join(_RANK_LABEL[r] for r in ranks)
    full_name = name if ranks == [data['dim'] - 1] else f"{name}:{labels}"
    desc = f"{data['desc']}, grouped by {labels}"
    return GroupStructure(full_name, len(verts), unique, desc).validate()


BUILTIN_STRUCTURES = {name: (lambda n=name: build_polytope(n))
                      for name in _POLYTOPES}

# Friendly aliases
_ALIASES = {
    'tet': 'tetrahedron',
    'oct': 'octahedron',
    'hexahedron': 'cube',
    'ico': 'icosahedron',
    'icosa': 'icosahedron',
    'dodeca': 'dodecahedron',
    'dodec': 'dodecahedron',
    '16-cell': '16cell',
    'hexadecachoron': '16cell',
    'orthoplex': '16cell',
    '24-cell': '24cell',
    'icositetrachoron': '24cell',
}


# =============================================================================
# NON-POLYTOPE STRUCTURES
# =============================================================================

def _make_cyclic(spec: str) -> GroupStructure:
    """
    cyclic:N,k[,step] — N slots, sub-scales of k consecutive slots (mod N),
    with sub-scale starts every `step` slots (default 1).

    cyclic:12,7 gives the 12 rotations of a 7-slot window: each sub-scale
    shares 6 slots with the next.
    """
    body = spec.split(':', 1)[1] if ':' in spec else ''
    parts = [p.strip() for p in body.split(',') if p.strip()]
    if len(parts) < 2:
        raise ValueError("cyclic structure needs at least N,k — e.g. 'cyclic:12,7'")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"cyclic structure parameters must be integers, got '{body}'")

    num_slots, size = nums[0], nums[1]
    step = nums[2] if len(nums) > 2 else 1

    if num_slots < 2:
        raise ValueError(f"cyclic: need at least 2 slots, got {num_slots}")
    if not (1 <= size <= num_slots):
        raise ValueError(f"cyclic: sub-scale size {size} must be between 1 and {num_slots}")
    if step < 1:
        raise ValueError(f"cyclic: step must be at least 1, got {step}")

    groups = []
    for start in range(0, num_slots, step):
        groups.append(tuple(sorted((start + off) % num_slots for off in range(size))))
    # Distinct groups only (large steps can repeat)
    seen = set()
    unique = []
    for g in groups:
        if g not in seen:
            seen.add(g)
            unique.append(g)

    desc = f"cyclic: {num_slots} slots, {size}-slot windows every {step}"
    return GroupStructure(spec, num_slots, unique, desc).validate()


def _load_group_file(path: str) -> GroupStructure:
    """
    Load sub-scales from a file: one sub-scale per line as slot indices,
    separated by spaces or commas. Lines starting with '#' are comments.
    Slot count is inferred as (highest index + 1).
    """
    groups = []
    with open(path, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            tokens = [t for t in line.replace(',', ' ').split() if t]
            try:
                indices = [int(t) for t in tokens]
            except ValueError:
                raise ValueError(
                    f"{path}:{lineno}: expected slot indices (integers), got '{line}'")
            if len(indices) < 2:
                raise ValueError(
                    f"{path}:{lineno}: sub-scale needs at least 2 slots, got {len(indices)}")
            groups.append(tuple(sorted(indices)))

    if not groups:
        raise ValueError(f"{path}: no sub-scales found")

    num_slots = max(max(g) for g in groups) + 1
    name = os.path.basename(path)
    desc = f"from {name}: {num_slots} slots, {len(groups)} sub-scales"
    return GroupStructure(name, num_slots, groups, desc).validate()


# =============================================================================
# PARSING
# =============================================================================

def parse_group_spec(spec: str) -> GroupStructure:
    """
    Parse a group structure specification.

    Accepts a built-in name, 'cyclic:N,k[,step]', or a path to a group file.
    Raises ValueError with a listing of valid options if the spec isn't known.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty group specification")

    key = spec.lower()

    if key.startswith('cyclic'):
        return _make_cyclic(key)

    # A polytope may carry a rank suffix: 'name:faces', 'name:cells+faces'.
    # Split before alias lookup so '24-cell:faces' resolves too.
    base, _, rank_part = key.partition(':')
    base = _ALIASES.get(base, base)

    if base in _POLYTOPES:
        if not rank_part:
            return build_polytope(base)
        ranks = []
        for token in rank_part.split('+'):
            token = token.strip()
            if not token:
                continue
            if token in RANK_NAMES:
                ranks.append(RANK_NAMES[token])
            elif token.isdigit():
                ranks.append(int(token))
            else:
                avail = ', '.join(_RANK_LABEL[r]
                                  for r in sorted(_POLYTOPES[base]['counts'],
                                                  reverse=True))
                raise ValueError(
                    f"Unknown rank '{token}' for {base}. Available: {avail}. "
                    f"Combine with '+', e.g. '{base}:cells+faces'.")
        if not ranks:
            raise ValueError(f"'{spec}': no ranks given after ':'")
        return build_polytope(base, ranks)

    if os.path.exists(spec):
        return _load_group_file(spec)

    raise ValueError(
        f"Unknown group structure '{spec}'.\n"
        f"Built-in polytopes: {', '.join(sorted(_POLYTOPES))}\n"
        f"Add a rank suffix to group by lower-dimensional faces, "
        f"e.g. '24cell:faces' or '16cell:cells+faces'.\n"
        f"Or use 'cyclic:N,k[,step]', or the path to a group file.\n"
        f"Run with --list-groups to see details.")


def describe_structures() -> str:
    """Human-readable listing of the built-in structures."""
    lines = ["Built-in group structures:", ""]
    lines.append(f"  {'name':<14s} {'notes':>5s}  {'sub-scales':>10s}  {'size':>4s}  adjacency")
    lines.append(f"  {'-' * 14} {'-' * 5}  {'-' * 10}  {'-' * 4}  {'-' * 30}")

    for name in ['tetrahedron', 'octahedron', 'cube', 'icosahedron',
                 'dodecahedron', '16cell', '24cell']:
        s = build_polytope(name)
        adj = s.adjacency()
        if adj:
            shares = {}
            for _, _, shared in adj:
                shares[shared] = shares.get(shared, 0) + 1
            adj_desc = ', '.join(f"{count} pairs share {shared}"
                                 for shared, count in sorted(shares.items(), reverse=True))
        else:
            adj_desc = "disjoint"
        if len(adj_desc) > 30:
            adj_desc = adj_desc[:27] + '...'
        size = s.group_size
        size_s = str(size) if size is not None else 'mixed'
        lines.append(f"  {name:<14s} {s.num_slots:>5d}  {len(s.groups):>10d}  "
                     f"{size_s:>4s}  {adj_desc}")

    lines.extend([
        "",
        "Face-lattice ranks. The default groups by facets; a suffix selects a",
        "different rank, and '+' combines them so one tuning carries sub-scales",
        "of more than one size at once.",
        "",
        f"  {'spec':<24s} {'notes':>5s}  {'sub-scales':>10s}  sizes",
        f"  {'-' * 24} {'-' * 5}  {'-' * 10}  {'-' * 20}",
    ])

    for spec in ['16cell', '16cell:faces', '16cell:cells+faces', '16cell:edges',
                 '24cell', '24cell:faces', '24cell:cells+faces',
                 'dodecahedron:edges', 'cube:faces+edges']:
        try:
            st = parse_group_spec(spec)
        except ValueError:
            continue
        sizes = sorted({len(g) for g in st.groups})
        size_desc = ' + '.join(str(x) for x in sizes)
        lines.append(f"  {spec:<24s} {st.num_slots:>5d}  {len(st.groups):>10d}  {size_desc}")

    lines.extend([
        "",
        "  Rank names: cells (4D only), faces, edges.",
        "  Edges as sub-scales are single intervals — a way of asking that every",
        "  edge of the polytope be consonant.",
        "",
        "  cyclic:N,k[,step]   N slots, k-slot windows every `step` slots (wrapping)",
        "                      e.g. cyclic:12,7 — 12 heptatonic modes of a 12-note scale",
        "",
        "  <file>              one sub-scale per line as slot indices; '#' comments",
        "",
        "Slot 0 always holds the tonic 1/1.",
    ])
    return '\n'.join(lines)
