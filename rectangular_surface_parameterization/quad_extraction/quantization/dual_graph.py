"""
Dual edge graph for integer optimization (§4.2).

The dual graph represents edges in the mesh, where:
- Each dual node corresponds to a mesh half-edge
- Dual edges connect half-edges that share a vertex (form a loop)
- The graph is used for Dijkstra-based integer optimization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional


@dataclass
class HalfEdge:
    """Half-edge in the mesh."""
    vertex_from: int
    vertex_to: int
    face_idx: int
    local_idx: int  # 0, 1, or 2 within the face
    opposite: int = -1  # Index of opposite half-edge
    next_he: int = -1   # Next half-edge in face
    prev_he: int = -1   # Previous half-edge in face


@dataclass
class DualGraph:
    """Dual graph for integer optimization."""
    # Half-edges
    half_edges: List[HalfEdge]
    num_edges: int  # Number of unique edges (half of half-edges for interior)

    # Half-edge lookup: (v_from, v_to) -> half-edge index
    he_lookup: Dict[Tuple[int, int], int]

    # UV coordinates at each half-edge origin
    # uv_at_he[he_idx] = UV coordinate of vertex_from in the adjacent face
    uv_at_he: np.ndarray

    # Edge geometry (to be optimized)
    # omega[he_idx] = UV displacement along this half-edge
    omega_initial: np.ndarray


def build_half_edges(triangles: np.ndarray) -> Tuple[List[HalfEdge], Dict[Tuple[int, int], int]]:
    """
    Build half-edge structure from triangles.

    Returns:
        (half_edges, lookup) where lookup maps (v_from, v_to) to half-edge index
    """
    half_edges = []
    lookup = {}

    for fi, tri in enumerate(triangles):
        # Create three half-edges for this face
        base_idx = len(half_edges)

        for i in range(3):
            v_from = tri[i]
            v_to = tri[(i + 1) % 3]

            he = HalfEdge(
                vertex_from=v_from,
                vertex_to=v_to,
                face_idx=fi,
                local_idx=i,
                next_he=base_idx + (i + 1) % 3,
                prev_he=base_idx + (i + 2) % 3,
            )
            half_edges.append(he)
            lookup[(v_from, v_to)] = base_idx + i

    # Link opposites
    for he_idx, he in enumerate(half_edges):
        opposite_key = (he.vertex_to, he.vertex_from)
        if opposite_key in lookup:
            he.opposite = lookup[opposite_key]

    return half_edges, lookup


def compute_edge_omega(uv_from: np.ndarray, uv_to: np.ndarray) -> np.ndarray:
    """
    Compute edge geometry (displacement in UV space).

    omega = uv_to - uv_from
    """
    return uv_to - uv_from


def build_dual_graph(triangles: np.ndarray,
                     uv_per_triangle: np.ndarray) -> DualGraph:
    """
    Build dual graph from mesh with UV coordinates.

    Args:
        triangles: (M, 3) triangle indices
        uv_per_triangle: (M, 3, 2) UV coordinates per triangle corner

    Returns:
        DualGraph structure for integer optimization
    """
    half_edges, lookup = build_half_edges(triangles)

    # Compute UV at each half-edge origin and edge geometry
    uv_at_he = np.zeros((len(half_edges), 2))
    omega_initial = np.zeros((len(half_edges), 2))

    for he_idx, he in enumerate(half_edges):
        fi = he.face_idx
        local_from = he.local_idx
        local_to = (he.local_idx + 1) % 3

        uv_from = uv_per_triangle[fi, local_from]
        uv_to = uv_per_triangle[fi, local_to]

        uv_at_he[he_idx] = uv_from
        omega_initial[he_idx] = compute_edge_omega(uv_from, uv_to)

    # Count unique edges (half-edges with opposite, divided by 2, plus boundary)
    counted = set()
    num_edges = 0
    for he_idx, he in enumerate(half_edges):
        edge = (min(he.vertex_from, he.vertex_to), max(he.vertex_from, he.vertex_to))
        if edge not in counted:
            counted.add(edge)
            num_edges += 1

    return DualGraph(
        half_edges=half_edges,
        num_edges=num_edges,
        he_lookup=lookup,
        uv_at_he=uv_at_he,
        omega_initial=omega_initial,
    )


def get_vertex_one_ring(dual: DualGraph, vertex: int) -> List[int]:
    """
    Get all half-edges originating from a vertex (the one-ring).

    Returns half-edge indices in counter-clockwise order.
    """
    # Find all half-edges starting at this vertex
    outgoing = []
    for he_idx, he in enumerate(dual.half_edges):
        if he.vertex_from == vertex:
            outgoing.append(he_idx)

    if not outgoing:
        return []

    # Sort by angle in UV space for consistent ordering
    angles = []
    for he_idx in outgoing:
        omega = dual.omega_initial[he_idx]
        angle = np.arctan2(omega[1], omega[0])
        angles.append((angle, he_idx))

    angles.sort()
    return [he_idx for _, he_idx in angles]


def compute_face_closure(dual: DualGraph, face_idx: int) -> np.ndarray:
    """
    Compute the closure constraint for a face.

    For a valid parameterization, sum of omega around face = 0.

    Returns:
        Sum of edge geometries around the face
    """
    total = np.zeros(2)
    for he in dual.half_edges:
        if he.face_idx == face_idx:
            total += dual.omega_initial[he.local_idx]
    return total
