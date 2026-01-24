"""Find grid vertices on triangle mesh edges.

This module finds all points where integer UV iso-lines cross triangle edges.
These are grid vertices that lie ON edges (not inside faces or at mesh vertices).

Algorithm (based on libQEx MeshExtractorT.cc lines 688-800):
1. For each unique edge in the mesh:
   - Get UV coordinates of both endpoints
   - Find all integer crossings along the edge
   - Skip endpoints (those are handled by vertex_vertices)
   - For each interior integer point on the edge:
     - Compute exact UV coordinate (at least one component is integer)
     - Compute 3D position via linear interpolation along edge
     - Create a GridVertex record

Key insight: An edge from UV (u0,v0) to (u1,v1) may contain integer grid points
(points where BOTH u and v are integers). We iterate along the axis with larger
range for better numerical stability, computing the other coordinate via
interpolation and checking if it's close to an integer.
"""

from typing import Dict, List, Tuple

import numpy as np

from rectangular_surface_parameterization.quad_extraction.grid_vertex import GridVertex


def find_edge_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs_per_triangle: np.ndarray,
    tolerance: float = 1e-9,
) -> Tuple[List[GridVertex], Dict[Tuple[int, int], List[int]]]:
    """Find all grid vertices that lie on triangle edges.

    For each edge in the mesh, finds iso-line crossing points (where either
    u or v is an integer). These crossings are rounded to integer UV coords
    and deduplicated to ensure one grid vertex per integer UV location.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        3D vertex positions of the triangle mesh.
    triangles : ndarray, shape (M, 3)
        Triangle vertex indices (0-based).
    uvs_per_triangle : ndarray, shape (M, 3, 2)
        UV coordinates per triangle corner.
        uvs_per_triangle[f, i, :] is the UV for corner i of triangle f.
    tolerance : float, optional
        Numerical tolerance for point-on-segment checks. Default 1e-9.

    Returns
    -------
    edge_vertices : list of GridVertex
        List of GridVertex objects for all edge vertices found.
        Each has type='edge', integer UV coordinates, and interpolated 3D position.
        Deduplicated by UV coordinate - one vertex per unique integer UV.
    edge_to_vertices : dict
        Maps edge (as sorted tuple of vertex indices) to list of indices
        into edge_vertices for vertices on that edge.

    Notes
    -----
    When multiple iso-line crossings round to the same integer UV, only
    the first is kept. This handles cases where u and v iso-lines both
    cross near an integer grid point.
    """
    # Track seen UVs to avoid duplicates
    seen_uvs: Dict[Tuple[int, int], int] = {}  # UV tuple -> vertex index
    edge_vertices: List[GridVertex] = []
    edge_to_vertices: Dict[Tuple[int, int], List[int]] = {}

    # Process all triangle edges (not just unique mesh edges, to handle seams)
    for face_idx in range(len(triangles)):
        tri = triangles[face_idx]
        uvs = uvs_per_triangle[face_idx]  # shape (3, 2)

        for local_edge_idx in range(3):
            # Edge from corner i to corner (i+1)%3
            i0 = local_edge_idx
            i1 = (local_edge_idx + 1) % 3

            v0, v1 = int(tri[i0]), int(tri[i1])
            uv0, uv1 = uvs[i0], uvs[i1]

            # Edge key for the mapping
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key not in edge_to_vertices:
                edge_to_vertices[edge_key] = []

            # Get 3D positions of edge endpoints
            p0_3d = vertices[v0]
            p1_3d = vertices[v1]

            # Find iso-line crossings on this edge
            edge_grid_points = _find_integer_points_on_segment(uv0, uv1, tolerance)

            for uv_crossing, alpha in edge_grid_points:
                # Interpolate 3D position: p = p0 + alpha * (p1 - p0)
                pos_3d = p0_3d + alpha * (p1_3d - p0_3d)

                # Round UV to integers for grid identification
                uv_int = (int(round(uv_crossing[0])), int(round(uv_crossing[1])))

                # Check if we've already seen this UV
                if uv_int in seen_uvs:
                    # Add existing vertex index to this edge's list
                    existing_idx = seen_uvs[uv_int]
                    if existing_idx not in edge_to_vertices[edge_key]:
                        edge_to_vertices[edge_key].append(existing_idx)
                    continue

                # Create new GridVertex
                gv = GridVertex(
                    type='edge',
                    uv=uv_int,
                    pos_3d=pos_3d.copy(),
                    face_idx=face_idx,
                    edge_idx=None,
                )

                idx = len(edge_vertices)
                seen_uvs[uv_int] = idx
                edge_vertices.append(gv)
                edge_to_vertices[edge_key].append(idx)

    return edge_vertices, edge_to_vertices


def _find_integer_points_on_segment(
    uv0: np.ndarray,
    uv1: np.ndarray,
    tolerance: float = 1e-9,
) -> List[Tuple[np.ndarray, float]]:
    """Find all iso-line crossing points in the interior of a segment.

    Given a segment from uv0 to uv1, finds all points where EITHER the u
    coordinate OR the v coordinate is an integer (iso-line crossings).
    This is different from finding points where both are integers.

    QEx needs these iso-line crossings to trace the integer grid through
    the mesh. Each crossing point becomes an edge vertex that can connect
    to adjacent grid vertices.

    Parameters
    ----------
    uv0 : ndarray, shape (2,)
        UV coordinates of segment start.
    uv1 : ndarray, shape (2,)
        UV coordinates of segment end.
    tolerance : float
        Tolerance for considering a value as integer.

    Returns
    -------
    points : list of (uv, alpha)
        Each entry is a tuple of:
        - uv: ndarray (2,) with the UV coordinates (one integer, one float)
        - alpha: float in (0, 1) - interpolation parameter along segment

    Notes
    -----
    The UV coordinates returned may have non-integer components. For example,
    a u iso-line crossing at u=5 might have v=3.7. The UV is stored as (5, 3.7)
    but for grid vertex identification, we round to (5, 4) as the logical grid cell.
    """
    points: List[Tuple[np.ndarray, float]] = []
    seen_alphas: set = set()  # Avoid duplicates at same position

    u0, v0 = float(uv0[0]), float(uv0[1])
    u1, v1 = float(uv1[0]), float(uv1[1])

    # Check for degenerate edge (single point in UV)
    du = u1 - u0
    dv = v1 - v0

    if abs(du) < tolerance and abs(dv) < tolerance:
        return points

    # Find u iso-line crossings (where u = integer)
    if abs(du) >= tolerance:
        u_min, u_max = (u0, u1) if u0 < u1 else (u1, u0)
        u_int_min = int(np.ceil(u_min + tolerance))
        u_int_max = int(np.floor(u_max - tolerance))

        for u_int in range(u_int_min, u_int_max + 1):
            # Compute alpha where u = u_int
            alpha = (u_int - u0) / du

            # Skip endpoints
            if alpha <= tolerance or alpha >= 1.0 - tolerance:
                continue

            # Round alpha to avoid floating point duplicates
            alpha_key = round(alpha * 1e9) / 1e9
            if alpha_key in seen_alphas:
                continue
            seen_alphas.add(alpha_key)

            # Compute v at this crossing
            v_at_crossing = v0 + alpha * dv

            # Store as integer u, exact v (will be rounded later for grid lookup)
            uv = np.array([float(u_int), v_at_crossing], dtype=np.float64)
            points.append((uv, alpha))

    # Find v iso-line crossings (where v = integer)
    if abs(dv) >= tolerance:
        v_min, v_max = (v0, v1) if v0 < v1 else (v1, v0)
        v_int_min = int(np.ceil(v_min + tolerance))
        v_int_max = int(np.floor(v_max - tolerance))

        for v_int in range(v_int_min, v_int_max + 1):
            # Compute alpha where v = v_int
            alpha = (v_int - v0) / dv

            # Skip endpoints
            if alpha <= tolerance or alpha >= 1.0 - tolerance:
                continue

            # Round alpha to avoid floating point duplicates
            alpha_key = round(alpha * 1e9) / 1e9
            if alpha_key in seen_alphas:
                continue
            seen_alphas.add(alpha_key)

            # Compute u at this crossing
            u_at_crossing = u0 + alpha * du

            # Store as exact u, integer v
            uv = np.array([u_at_crossing, float(v_int)], dtype=np.float64)
            points.append((uv, alpha))

    # Sort by alpha for consistent ordering along the edge
    points.sort(key=lambda x: x[1])

    return points


def _point_on_segment(
    uv: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    tolerance: float = 1e-9,
) -> Tuple[bool, float]:
    """Check if a point lies on a segment and return the interpolation parameter.

    Parameters
    ----------
    uv : ndarray, shape (2,)
        Point to check.
    uv0 : ndarray, shape (2,)
        Segment start.
    uv1 : ndarray, shape (2,)
        Segment end.
    tolerance : float
        Distance tolerance.

    Returns
    -------
    on_segment : bool
        True if the point lies on the segment (strictly interior).
    alpha : float
        Interpolation parameter in (0, 1) if on_segment, else 0.
    """
    # Vector from uv0 to uv1
    d = uv1 - uv0
    length_sq = np.dot(d, d)

    if length_sq < tolerance * tolerance:
        # Degenerate segment
        return False, 0.0

    # Project point onto line
    t = uv - uv0
    alpha = np.dot(t, d) / length_sq

    # Check if strictly interior
    if alpha <= tolerance or alpha >= 1.0 - tolerance:
        return False, 0.0

    # Check distance to segment
    closest = uv0 + alpha * d
    dist_sq = np.dot(uv - closest, uv - closest)

    if dist_sq > tolerance * tolerance:
        return False, 0.0

    return True, alpha


if __name__ == "__main__":
    # Simple test case
    print("Testing find_edge_vertices...")
    print()

    # Test 1: Single triangle with edge from (0,0) to (3,0)
    # Should find grid points at (1,0) and (2,0)
    print("Test 1: Horizontal edge with integer crossings")
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.5, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    uvs = np.array([[[0.0, 0.0], [3.0, 0.0], [1.5, 3.0]]])

    edge_verts, edge_map = find_edge_vertices(vertices, triangles, uvs)

    print(f"  Found {len(edge_verts)} edge vertices")
    for i, gv in enumerate(edge_verts):
        print(f"    [{i}] UV={gv.uv}, 3D={gv.pos_3d}")

    # Edge (0,1) should have vertices at (1,0) and (2,0)
    edge_01 = (0, 1)
    print(f"  Edge {edge_01}: {len(edge_map.get(edge_01, []))} vertices")
    print()

    # Test 2: Diagonal edge from (0,0) to (2,2)
    # Should find grid point at (1,1)
    print("Test 2: Diagonal edge with integer crossing")
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    uvs = np.array([[[0.0, 0.0], [2.0, 2.0], [0.0, 2.0]]])

    edge_verts, edge_map = find_edge_vertices(vertices, triangles, uvs)

    print(f"  Found {len(edge_verts)} edge vertices")
    for i, gv in enumerate(edge_verts):
        print(f"    [{i}] UV={gv.uv}, 3D={gv.pos_3d}")
    print()

    # Test 3: Edge with no integer crossings
    print("Test 3: Edge with no integer crossings")
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    # Edge from (0.1, 0.1) to (0.9, 0.9) - no integer points
    uvs = np.array([[[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]]])

    edge_verts, edge_map = find_edge_vertices(vertices, triangles, uvs)

    print(f"  Found {len(edge_verts)} edge vertices (expected 0)")
    print()

    # Test 4: Two triangles sharing an edge
    print("Test 4: Two triangles sharing an edge")
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [3.0, 0.0, 0.0],  # 1
        [1.5, 1.0, 0.0],  # 2
        [1.5, -1.0, 0.0], # 3
    ])
    triangles = np.array([
        [0, 1, 2],
        [1, 0, 3],
    ])
    uvs = np.array([
        [[0.0, 0.0], [3.0, 0.0], [1.5, 3.0]],
        [[3.0, 0.0], [0.0, 0.0], [1.5, -3.0]],
    ])

    edge_verts, edge_map = find_edge_vertices(vertices, triangles, uvs)

    print(f"  Found {len(edge_verts)} edge vertices")
    # Edge (0,1) should still only have 2 vertices (processed once)
    edge_01 = (0, 1)
    print(f"  Edge {edge_01}: {len(edge_map.get(edge_01, []))} vertices")
    for idx in edge_map.get(edge_01, []):
        gv = edge_verts[idx]
        print(f"    UV={gv.uv}, 3D={gv.pos_3d}")
    print()

    # Test 5: Vertical edge
    print("Test 5: Vertical edge from (2,0) to (2,3)")
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [1.0, 1.5, 0.0],
    ])
    triangles = np.array([[0, 1, 2]])
    uvs = np.array([[[2.0, 0.0], [2.0, 3.0], [5.0, 1.5]]])

    edge_verts, edge_map = find_edge_vertices(vertices, triangles, uvs)

    print(f"  Found {len(edge_verts)} edge vertices")
    for i, gv in enumerate(edge_verts):
        print(f"    [{i}] UV={gv.uv}, 3D={gv.pos_3d}")

    # Should have (2,1) and (2,2) on edge (0,1)
    edge_01 = (0, 1)
    print(f"  Edge {edge_01}: {len(edge_map.get(edge_01, []))} vertices (expected 2)")
    print()

    print("All tests completed.")
