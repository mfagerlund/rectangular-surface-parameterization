"""Edge tracing for QEx quad extraction - proper ray tracing implementation.

This module traces connections between grid vertices in UV space by actually
walking through the triangle mesh, following the libQEx algorithm
(MeshExtractorT.cc find_path function).

The algorithm:
1. Build triangle adjacency (half-edge like structure)
2. For each grid vertex, create local edges pointing to adjacent integer UVs
3. For each local edge, trace a ray from start UV to target UV through triangles
4. Walk through triangles until the target is found or boundary is hit
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grid_vertex import (
    GridEdge,
    GridVertex,
    DIRECTION_PLUS_U,
    DIRECTION_PLUS_V,
    DIRECTION_MINUS_U,
    DIRECTION_MINUS_V,
    direction_to_uv_delta,
    opposite_direction,
)


@dataclass
class LocalEdgeInfo:
    """Information about a local edge from a grid vertex.

    Mirrors libQEx LocalEdgeInfo structure.

    Attributes:
        face_idx: Triangle containing the start point
        uv_from: Starting UV position (the grid vertex's UV)
        uv_to: Target UV position (adjacent integer grid point)
        direction: Which cardinal direction (0=+u, 1=+v, 2=-u, 3=-v)
        connected_to_idx: Index of target GridVertex after tracing (-1 if unconnected)
        connected_direction: Which local edge of the target connects back
    """
    face_idx: int
    uv_from: Tuple[float, float]
    uv_to: Tuple[float, float]
    direction: int
    connected_to_idx: int = -1  # -1 = unconnected
    connected_direction: int = -1


@dataclass
class HalfEdge:
    """Half-edge structure for mesh traversal.

    Attributes:
        vertex_from: Source vertex index
        vertex_to: Target vertex index
        face_idx: Face this half-edge belongs to
        next_he: Next half-edge in face (CCW)
        prev_he: Previous half-edge in face (CCW)
        opposite_he: Opposite half-edge (in adjacent triangle), -1 if boundary
        local_idx: Local index within triangle (0, 1, or 2)
    """
    vertex_from: int
    vertex_to: int
    face_idx: int
    next_he: int = -1
    prev_he: int = -1
    opposite_he: int = -1
    local_idx: int = 0


class MeshTopology:
    """Triangle mesh topology for traversal.

    Builds a half-edge-like structure from triangles for efficient
    adjacent triangle queries needed by ray tracing.
    """

    def __init__(self, triangles: np.ndarray, uvs_per_triangle: np.ndarray):
        """Build mesh topology from triangles and UVs.

        Args:
            triangles: (n_tris, 3) triangle vertex indices
            uvs_per_triangle: (n_tris, 3, 2) UV coordinates per triangle corner
        """
        self.triangles = triangles
        self.uvs_per_triangle = uvs_per_triangle
        self.n_tris = len(triangles)

        # Build half-edges: 3 per triangle
        self.half_edges: List[HalfEdge] = []

        # Map from directed edge (v0, v1) to half-edge index
        edge_to_he: Dict[Tuple[int, int], int] = {}

        for face_idx in range(self.n_tris):
            tri = triangles[face_idx]
            base_he = len(self.half_edges)

            # Create 3 half-edges for this triangle
            for local_idx in range(3):
                v_from = int(tri[local_idx])
                v_to = int(tri[(local_idx + 1) % 3])

                he = HalfEdge(
                    vertex_from=v_from,
                    vertex_to=v_to,
                    face_idx=face_idx,
                    local_idx=local_idx,
                )
                self.half_edges.append(he)

                # Link next/prev within triangle
                he_idx = base_he + local_idx
                self.half_edges[he_idx].next_he = base_he + ((local_idx + 1) % 3)
                self.half_edges[he_idx].prev_he = base_he + ((local_idx + 2) % 3)

                # Record edge for opposite linking
                edge_to_he[(v_from, v_to)] = he_idx

        # Link opposite half-edges
        for he_idx, he in enumerate(self.half_edges):
            opposite_key = (he.vertex_to, he.vertex_from)
            if opposite_key in edge_to_he:
                opp_idx = edge_to_he[opposite_key]
                he.opposite_he = opp_idx

    def get_face_half_edges(self, face_idx: int) -> Tuple[int, int, int]:
        """Get the 3 half-edge indices for a face."""
        base = face_idx * 3
        return base, base + 1, base + 2

    def get_face_uvs(self, face_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get UV coordinates for triangle corners."""
        uvs = self.uvs_per_triangle[face_idx]
        return uvs[0], uvs[1], uvs[2]

    def is_boundary_edge(self, he_idx: int) -> bool:
        """Check if half-edge is on mesh boundary."""
        return self.half_edges[he_idx].opposite_he < 0


def build_uv_index(
    grid_vertices: List[GridVertex],
) -> Dict[Tuple[int, int], List[int]]:
    """Build a spatial index mapping integer UV coordinates to vertex indices.

    Creates a dictionary where keys are (u, v) integer coordinate tuples and
    values are lists of indices into grid_vertices that have those coordinates.

    Multiple grid vertices can share the same integer UV coordinates in cases
    where the UV parameterization has seams or overlapping regions.

    Args:
        grid_vertices: List of GridVertex objects with integer UV coordinates.

    Returns:
        Dictionary mapping (u, v) tuples to lists of vertex indices.
    """
    uv_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    for idx, gv in enumerate(grid_vertices):
        uv_index[gv.uv].append(idx)

    return dict(uv_index)


def _segment_intersects_segment(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
    tolerance: float = 1e-10
) -> Tuple[bool, float]:
    """Check if two segments intersect and return intersection parameter.

    Segment 1: p1 to p2
    Segment 2: p3 to p4

    Returns:
        Tuple of (intersects, t) where t is the parameter along segment 1
        where intersection occurs (0 <= t <= 1 if intersecting)
    """
    d1 = p2 - p1
    d2 = p4 - p3
    d3 = p1 - p3

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < tolerance:
        # Parallel or collinear
        return False, 0.0

    t = (d2[0] * d3[1] - d2[1] * d3[0]) / cross
    s = (d1[0] * d3[1] - d1[1] * d3[0]) / cross

    # Check if intersection is within both segments
    if -tolerance <= t <= 1 + tolerance and -tolerance <= s <= 1 + tolerance:
        return True, t

    return False, 0.0


def _point_in_triangle(
    p: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
    tolerance: float = 1e-10
) -> bool:
    """Check if point is inside or on boundary of triangle."""
    # Compute barycentric coordinates
    d0 = v1 - v0
    d1 = v2 - v0
    d2 = p - v0

    dot00 = np.dot(d1, d1)
    dot01 = np.dot(d1, d0)
    dot02 = np.dot(d1, d2)
    dot11 = np.dot(d0, d0)
    dot12 = np.dot(d0, d2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < tolerance * tolerance:
        return False  # Degenerate triangle

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return u >= -tolerance and v >= -tolerance and (u + v) <= 1 + tolerance


def _find_grid_vertex_at_uv(
    target_uv: Tuple[int, int],
    face_idx: int,
    grid_vertices: List[GridVertex],
    face_to_gv: Dict[int, List[int]],
    uv_index: Dict[Tuple[int, int], List[int]],
) -> int:
    """Find a grid vertex at target UV, preferring one in the given face.

    Returns:
        Index of grid vertex, or -1 if not found
    """
    # First try to find a vertex in this face
    gv_indices = face_to_gv.get(face_idx, [])
    for gv_idx in gv_indices:
        if grid_vertices[gv_idx].uv == target_uv:
            return gv_idx

    # Fall back to UV index lookup
    candidates = uv_index.get(target_uv, [])
    if candidates:
        return candidates[0]

    return -1


def find_path(
    start_gv_idx: int,
    direction: int,
    grid_vertices: List[GridVertex],
    topology: MeshTopology,
    face_to_gv: Dict[int, List[int]],
    uv_index: Dict[Tuple[int, int], List[int]],
    max_iterations: int = 10000,
    tolerance: float = 1e-9,
) -> Tuple[int, int]:
    """Trace from a grid vertex in a direction to find connected grid vertex.

    This implements the libQEx find_path algorithm that walks through triangles
    following a ray from the source UV to the target UV.

    Args:
        start_gv_idx: Index of starting grid vertex
        direction: Direction to trace (0=+u, 1=+v, 2=-u, 3=-v)
        grid_vertices: List of all grid vertices
        topology: Mesh topology for traversal
        face_to_gv: Mapping from face index to grid vertex indices in that face
        uv_index: Mapping from UV to grid vertex indices
        max_iterations: Maximum triangle crossings before giving up
        tolerance: Numerical tolerance

    Returns:
        Tuple of (target_gv_idx, reverse_direction)
        target_gv_idx is -1 if no connection found (boundary or error)
    """
    start_gv = grid_vertices[start_gv_idx]
    du, dv = direction_to_uv_delta(direction)

    # Source and target UV coordinates
    uv_from = np.array([float(start_gv.uv[0]), float(start_gv.uv[1])])
    target_uv = (start_gv.uv[0] + du, start_gv.uv[1] + dv)
    uv_to = np.array([float(target_uv[0]), float(target_uv[1])])

    # Start in the grid vertex's face
    current_face = start_gv.face_idx

    # Get face UVs
    uv0, uv1, uv2 = topology.get_face_uvs(current_face)

    # Check if target is already in this triangle
    if _point_in_triangle(uv_to, uv0, uv1, uv2, tolerance):
        # Found! Look for grid vertex at target UV in this face
        target_idx = _find_grid_vertex_at_uv(
            target_uv, current_face, grid_vertices, face_to_gv, uv_index
        )
        if target_idx >= 0:
            return target_idx, opposite_direction(direction)

    # Need to trace through triangles
    he0, he1, he2 = topology.get_face_half_edges(current_face)

    # Find which edge the ray crosses first
    path_start = uv_from
    path_end = uv_to

    # Check intersection with each edge
    # Edge 0: uv0 -> uv1 (half-edge 0)
    # Edge 1: uv1 -> uv2 (half-edge 1)
    # Edge 2: uv2 -> uv0 (half-edge 2)
    edges = [
        (uv0, uv1, he0),
        (uv1, uv2, he1),
        (uv2, uv0, he2),
    ]

    # Find the edge the path crosses
    crossing_he = -1
    for edge_uv0, edge_uv1, he_idx in edges:
        intersects, t = _segment_intersects_segment(
            path_start, path_end, edge_uv0, edge_uv1, tolerance
        )
        if intersects and t > tolerance:  # Exclude starting point
            crossing_he = he_idx
            break

    if crossing_he < 0:
        # Couldn't find crossing - target might be very close or degenerate
        return -1, -1

    # Walk through triangles
    for iteration in range(max_iterations):
        he = topology.half_edges[crossing_he]

        # Check boundary
        if he.opposite_he < 0:
            # Hit boundary
            return -1, -1

        # Cross to adjacent triangle
        opp_he = topology.half_edges[he.opposite_he]
        current_face = opp_he.face_idx

        # Get new face UVs
        uv0, uv1, uv2 = topology.get_face_uvs(current_face)

        # Check if target is in this triangle
        if _point_in_triangle(uv_to, uv0, uv1, uv2, tolerance):
            # Found! Look for grid vertex at target UV
            target_idx = _find_grid_vertex_at_uv(
                target_uv, current_face, grid_vertices, face_to_gv, uv_index
            )
            if target_idx >= 0:
                return target_idx, opposite_direction(direction)
            else:
                # Target UV is in triangle but no grid vertex there
                # This can happen at boundary or degenerate cases
                return -1, -1

        # Find which edge to cross next
        # The opposite half-edge points into this triangle
        # We need to check the other two edges
        he0, he1, he2 = topology.get_face_half_edges(current_face)
        all_hes = [he0, he1, he2]

        edges = [
            (uv0, uv1, he0),
            (uv1, uv2, he1),
            (uv2, uv0, he2),
        ]

        # Don't cross back through the edge we came from
        came_from_he = he.opposite_he

        crossing_he = -1
        best_t = float('inf')

        for edge_uv0, edge_uv1, he_idx in edges:
            if he_idx == came_from_he:
                continue

            intersects, t = _segment_intersects_segment(
                path_start, path_end, edge_uv0, edge_uv1, tolerance
            )
            if intersects and t > tolerance and t < best_t:
                crossing_he = he_idx
                best_t = t

        if crossing_he < 0:
            # Couldn't find next edge - target should be in this triangle
            # but we didn't find it
            return -1, -1

    # Max iterations exceeded
    return -1, -1


def trace_edges(
    grid_vertices: List[GridVertex],
    triangles: np.ndarray = None,
    uvs_per_triangle: np.ndarray = None,
    topology: MeshTopology = None,
    verbose: bool = False,
) -> List[GridEdge]:
    """Trace connections between grid vertices using ray tracing.

    This is the main entry point for edge tracing. It uses ray tracing through
    the triangle mesh to find connections, following the libQEx algorithm.

    Args:
        grid_vertices: List of GridVertex objects.
        triangles: (n_tris, 3) triangle indices (required if topology not provided)
        uvs_per_triangle: (n_tris, 3, 2) UVs per triangle corner (required if topology not provided)
        topology: Pre-built MeshTopology (optional, built from triangles/uvs if not provided)
        verbose: If True, print progress information.

    Returns:
        List of GridEdge objects representing all traced connections.
    """
    if topology is None:
        if triangles is None or uvs_per_triangle is None:
            # Fall back to simple UV lookup when no mesh data provided
            return _trace_edges_simple(grid_vertices, verbose)
        topology = MeshTopology(triangles, uvs_per_triangle)

    # Build spatial index
    uv_index = build_uv_index(grid_vertices)

    # Build face to grid vertex mapping
    face_to_gv: Dict[int, List[int]] = defaultdict(list)
    for idx, gv in enumerate(grid_vertices):
        face_to_gv[gv.face_idx].append(idx)

    if verbose:
        print(f"  Edge tracer (ray tracing): {len(grid_vertices)} vertices, {len(uv_index)} unique UVs")

    edges: List[GridEdge] = []
    connections_made = 0
    boundary_hits = 0

    # For each grid vertex, trace in each direction
    for src_idx, src_vertex in enumerate(grid_vertices):
        for direction in range(4):
            # Skip if already connected
            if src_vertex.outgoing_edges.get(direction) is not None:
                continue

            # Try ray tracing first
            target_idx, reverse_dir = find_path(
                src_idx, direction, grid_vertices, topology,
                face_to_gv, uv_index
            )

            if target_idx >= 0:
                target_vertex = grid_vertices[target_idx]

                # Record bidirectional connection
                src_vertex.outgoing_edges[direction] = (target_idx, reverse_dir)

                if target_vertex.outgoing_edges.get(reverse_dir) is None:
                    target_vertex.outgoing_edges[reverse_dir] = (src_idx, direction)

                # Create edge
                edge = GridEdge(
                    from_vertex=src_idx,
                    to_vertex=target_idx,
                    direction=direction,
                    reverse_direction=reverse_dir,
                )
                edges.append(edge)
                connections_made += 1
            else:
                boundary_hits += 1

    if verbose:
        print(f"  Edge tracer: {connections_made} connections, {boundary_hits} boundary/unconnected")

    return edges


def _trace_edges_simple(
    grid_vertices: List[GridVertex],
    verbose: bool = False,
) -> List[GridEdge]:
    """Simple UV lookup based edge tracing (fallback).

    This is the original simplified approach that just looks up adjacent UVs.
    Used when no mesh topology is available.
    """
    uv_index = build_uv_index(grid_vertices)

    if verbose:
        print(f"  Edge tracer (simple): {len(grid_vertices)} vertices, {len(uv_index)} unique UVs")

    direction_vectors = {
        DIRECTION_PLUS_U: (1, 0),
        DIRECTION_PLUS_V: (0, 1),
        DIRECTION_MINUS_U: (-1, 0),
        DIRECTION_MINUS_V: (0, -1),
    }

    edges: List[GridEdge] = []

    for src_idx, src_vertex in enumerate(grid_vertices):
        src_u, src_v = src_vertex.uv

        for direction, (du, dv) in direction_vectors.items():
            if src_vertex.outgoing_edges.get(direction) is not None:
                continue

            target_uv = (src_u + du, src_v + dv)
            target_indices = uv_index.get(target_uv, [])

            if target_indices:
                target_idx = target_indices[0]
                target_vertex = grid_vertices[target_idx]
                reverse_dir = opposite_direction(direction)

                src_vertex.outgoing_edges[direction] = (target_idx, reverse_dir)

                if target_vertex.outgoing_edges.get(reverse_dir) is None:
                    target_vertex.outgoing_edges[reverse_dir] = (src_idx, direction)

                edge = GridEdge(
                    from_vertex=src_idx,
                    to_vertex=target_idx,
                    direction=direction,
                    reverse_direction=reverse_dir,
                )
                edges.append(edge)

    return edges


def trace_edges_with_validation(
    grid_vertices: List[GridVertex],
    verbose: bool = False,
) -> Tuple[List[GridEdge], Dict[str, int]]:
    """Trace edges with additional validation and statistics.

    Args:
        grid_vertices: List of GridVertex objects.
        verbose: If True, print detailed progress and statistics.

    Returns:
        Tuple of (edges, stats)
    """
    edges = trace_edges(grid_vertices, verbose=verbose)

    stats = {
        'total_vertices': len(grid_vertices),
        'unique_uvs': len(build_uv_index(grid_vertices)),
        'total_edges': len(edges),
        'face_connections': 0,
        'edge_connections': 0,
        'vertex_connections': 0,
        'disconnected_directions': 0,
    }

    type_map = {
        'face': 'face_connections',
        'edge': 'edge_connections',
        'vertex': 'vertex_connections',
    }

    for edge in edges:
        src_type = grid_vertices[edge.from_vertex].type
        stats[type_map[src_type]] += 1

    for gv in grid_vertices:
        for direction in range(4):
            if gv.outgoing_edges.get(direction) is None:
                stats['disconnected_directions'] += 1

    if verbose:
        print(f"  Statistics:")
        print(f"    Total vertices: {stats['total_vertices']}")
        print(f"    Unique UV positions: {stats['unique_uvs']}")
        print(f"    Total edges: {stats['total_edges']}")
        print(f"    Face vertex edges: {stats['face_connections']}")
        print(f"    Edge vertex edges: {stats['edge_connections']}")
        print(f"    Mesh vertex edges: {stats['vertex_connections']}")
        print(f"    Disconnected directions: {stats['disconnected_directions']}")

    return edges, stats


def get_connected_component(
    start_idx: int,
    grid_vertices: List[GridVertex],
) -> List[int]:
    """Find all vertices connected to a starting vertex."""
    visited = set()
    queue = [start_idx]

    while queue:
        idx = queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)

        vertex = grid_vertices[idx]
        for direction in range(4):
            connection = vertex.outgoing_edges.get(direction)
            if connection is not None:
                target_idx, _ = connection
                if target_idx not in visited:
                    queue.append(target_idx)

    return list(visited)


def find_all_components(
    grid_vertices: List[GridVertex],
) -> List[List[int]]:
    """Find all connected components in the grid vertex graph."""
    visited = set()
    components = []

    for idx in range(len(grid_vertices)):
        if idx not in visited:
            component = get_connected_component(idx, grid_vertices)
            components.append(component)
            visited.update(component)

    return components


def validate_edge_symmetry(
    grid_vertices: List[GridVertex],
    edges: List[GridEdge],
) -> List[str]:
    """Validate that all edges have symmetric reverse connections."""
    errors = []

    for edge in edges:
        src_idx = edge.from_vertex
        tgt_idx = edge.to_vertex
        direction = edge.direction
        expected_reverse = opposite_direction(direction)

        target_vertex = grid_vertices[tgt_idx]
        reverse_connection = target_vertex.outgoing_edges.get(expected_reverse)

        if reverse_connection is None:
            errors.append(
                f"Edge {src_idx}->{tgt_idx} (dir {direction}) has no reverse edge"
            )
        elif reverse_connection[0] != src_idx:
            errors.append(
                f"Edge {src_idx}->{tgt_idx} (dir {direction}) has wrong reverse target: "
                f"expected {src_idx}, got {reverse_connection[0]}"
            )

    return errors
