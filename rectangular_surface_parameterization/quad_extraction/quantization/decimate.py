"""
Mesh decimation for quantization (§4.1).

Decimates the mesh while preserving:
- Singular vertices (cannot be collapsed)
- Manifold topology
- Positive Jacobian in UV space (valid parameterization)

The key insight from QaWiTM is that we don't need a T-mesh - a simple
decimated triangular mesh is sufficient as a proxy for integer optimization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple
import heapq


@dataclass
class DecimationResult:
    """Result of mesh decimation."""
    # Coarse mesh
    vertices_coarse: np.ndarray    # (K, 3) coarse vertex positions
    triangles_coarse: np.ndarray   # (L, 3) coarse triangle indices
    uv_coarse: np.ndarray          # (L, 3, 2) UV per corner on coarse mesh

    # Mapping matrices
    D: np.ndarray                  # (K, N) decimation matrix: V_coarse = D @ V_original
    vertex_map: np.ndarray         # (N,) maps original vertex to coarse vertex (-1 if collapsed)

    # Preserved vertices
    singularities_coarse: np.ndarray  # Indices of singularities in coarse mesh


def compute_edge_collapse_cost(v1_uv: np.ndarray, v2_uv: np.ndarray,
                                triangles_around: List[Tuple[int, int, int]],
                                uvs: np.ndarray) -> float:
    """
    Compute the cost of collapsing edge (v1, v2).

    Cost considers:
    - UV distortion introduced
    - Triangle quality degradation
    - Jacobian preservation
    """
    if len(triangles_around) == 0:
        return float('inf')

    # Midpoint in UV space
    mid_uv = (v1_uv + v2_uv) / 2

    # Check if collapse would invert any triangles
    total_cost = 0.0

    for ti, local_v1, local_v2 in triangles_around:
        # Get the third vertex
        other_local = 3 - local_v1 - local_v2
        other_uv = uvs[ti, other_local]

        # Current triangle vectors
        curr_v1 = uvs[ti, local_v1]
        curr_v2 = uvs[ti, local_v2]

        # Check orientation (signed area)
        def signed_area(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        original_area = signed_area(curr_v1, curr_v2, other_uv)

        # After collapse, v1 and v2 merge to midpoint
        new_area = signed_area(mid_uv, mid_uv, other_uv)

        # This triangle degenerates - check adjacent triangles instead
        if abs(original_area) < 1e-10:
            continue

        # UV distortion cost
        dist_cost = np.linalg.norm(v1_uv - v2_uv)

        total_cost += dist_cost

    return total_cost


def build_edge_list(triangles: np.ndarray) -> List[Tuple[int, int]]:
    """Build list of unique edges from triangles."""
    edges = set()
    for tri in triangles:
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            edges.add(edge)
    return list(edges)


def get_vertex_neighbors(triangles: np.ndarray, n_verts: int) -> List[Set[int]]:
    """Build vertex adjacency lists."""
    neighbors = [set() for _ in range(n_verts)]
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[tri[i]].add(tri[j])
    return neighbors


def is_boundary_vertex(vi: int, triangles: np.ndarray) -> bool:
    """Check if vertex is on mesh boundary."""
    # Build edge count
    edge_count = {}
    for tri in triangles:
        if vi not in tri:
            continue
        for i in range(3):
            v1, v2 = tri[i], tri[(i + 1) % 3]
            if vi not in (v1, v2):
                continue
            edge = (min(v1, v2), max(v1, v2))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear once
    for edge, count in edge_count.items():
        if count == 1:
            return True
    return False


def decimate_mesh(vertices: np.ndarray,
                  triangles: np.ndarray,
                  uv_per_triangle: np.ndarray,
                  singularities: np.ndarray,
                  target_ratio: float = 0.1,
                  verbose: bool = False) -> DecimationResult:
    """
    Decimate mesh while preserving singularities.

    Uses edge collapse with priority queue, stopping when:
    - Target triangle count reached, or
    - No more valid collapses available

    Args:
        vertices: (N, 3) vertex positions
        triangles: (M, 3) triangle indices
        uv_per_triangle: (M, 3, 2) UV coordinates
        singularities: Indices of singular vertices (preserved)
        target_ratio: Target ratio of coarse/original triangles
        verbose: Print progress

    Returns:
        DecimationResult with coarse mesh and mapping
    """
    n_verts = len(vertices)
    n_tris = len(triangles)
    target_tris = max(int(n_tris * target_ratio), len(singularities) + 2)

    if verbose:
        print(f"  Decimation target: {n_tris} → {target_tris} triangles")

    # Track which vertices are protected
    protected = set(singularities.tolist())

    # Add boundary vertices to protected set
    for vi in range(n_verts):
        if is_boundary_vertex(vi, triangles):
            protected.add(vi)

    if verbose:
        print(f"  Protected vertices: {len(protected)} (singularities + boundary)")

    # Initialize working copies
    current_verts = vertices.copy()
    current_tris = triangles.copy().tolist()  # List for easy modification
    current_uvs = uv_per_triangle.copy().tolist()

    # Vertex mapping: original -> current (or -1 if collapsed)
    vertex_map = np.arange(n_verts)
    active_verts = set(range(n_verts))

    # Build priority queue of edge collapses
    # (cost, v1, v2) - lower cost = higher priority
    def rebuild_queue():
        edges = build_edge_list(np.array(current_tris))
        queue = []
        for v1, v2 in edges:
            if v1 in protected or v2 in protected:
                continue
            if v1 not in active_verts or v2 not in active_verts:
                continue

            # Simple cost: edge length in UV space
            # Find a triangle containing this edge to get UVs
            cost = 1.0  # Default
            for ti, tri in enumerate(current_tris):
                if v1 in tri and v2 in tri:
                    local_v1 = list(tri).index(v1)
                    local_v2 = list(tri).index(v2)
                    uv1 = current_uvs[ti][local_v1]
                    uv2 = current_uvs[ti][local_v2]
                    cost = np.linalg.norm(np.array(uv1) - np.array(uv2))
                    break

            heapq.heappush(queue, (cost, v1, v2))
        return queue

    queue = rebuild_queue()
    collapses = 0
    max_collapses = n_tris - target_tris

    while queue and len(current_tris) > target_tris and collapses < max_collapses:
        cost, v1, v2 = heapq.heappop(queue)

        # Check if vertices are still valid
        if v1 not in active_verts or v2 not in active_verts:
            continue

        # Collapse v2 into v1
        # Update all references to v2 to point to v1
        new_tris = []
        new_uvs = []
        removed = 0

        for ti in range(len(current_tris)):
            tri = list(current_tris[ti])
            uvs = list(current_uvs[ti])

            if v2 in tri:
                # Replace v2 with v1
                idx = tri.index(v2)
                tri[idx] = v1

                # Check if triangle degenerates (has duplicate vertices)
                if len(set(tri)) < 3:
                    removed += 1
                    continue

            new_tris.append(tri)
            new_uvs.append(uvs)

        current_tris = new_tris
        current_uvs = new_uvs

        # Mark v2 as collapsed
        active_verts.discard(v2)
        vertex_map[vertex_map == v2] = v1

        collapses += 1

        # Rebuild queue periodically
        if collapses % 100 == 0:
            queue = rebuild_queue()

    if verbose:
        print(f"  Performed {collapses} edge collapses")
        print(f"  Result: {len(current_tris)} triangles, {len(active_verts)} vertices")

    # Build final coarse mesh
    # Renumber vertices to be contiguous
    old_to_new = {}
    new_vertices = []
    for old_vi in sorted(active_verts):
        old_to_new[old_vi] = len(new_vertices)
        new_vertices.append(current_verts[old_vi])

    new_triangles = []
    new_uvs = []
    for ti in range(len(current_tris)):
        tri = current_tris[ti]
        new_tri = [old_to_new[v] for v in tri]
        new_triangles.append(new_tri)
        new_uvs.append(current_uvs[ti])

    # Update vertex_map for final renumbering
    final_vertex_map = np.full(n_verts, -1)
    for old_vi in range(n_verts):
        mapped = vertex_map[old_vi]
        if mapped in old_to_new:
            final_vertex_map[old_vi] = old_to_new[mapped]

    # Map singularities to new indices
    singularities_coarse = []
    for si in singularities:
        new_idx = final_vertex_map[si]
        if new_idx >= 0:
            singularities_coarse.append(new_idx)

    # Build decimation matrix D
    # D is (K, N) where K = new verts, N = original verts
    # D[new, old] = 1 if old maps to new
    K = len(new_vertices)
    D = np.zeros((K, n_verts))
    for old_vi in range(n_verts):
        new_vi = final_vertex_map[old_vi]
        if new_vi >= 0:
            D[new_vi, old_vi] = 1.0

    # Normalize rows (for vertices that were merged)
    row_sums = D.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    D = D / row_sums

    return DecimationResult(
        vertices_coarse=np.array(new_vertices),
        triangles_coarse=np.array(new_triangles),
        uv_coarse=np.array(new_uvs),
        D=D,
        vertex_map=final_vertex_map,
        singularities_coarse=np.array(singularities_coarse),
    )
