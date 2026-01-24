"""
Main quad extraction API for the Python QEx port.

This module provides the top-level `extract_quads` function that matches
the interface of libqex_wrapper.py, allowing drop-in replacement of the
C++ libQEx binary with a pure Python/NumPy implementation.

Usage:
    from rectangular_surface_parameterization.quad_extraction import extract_quads

    quad_verts, quad_faces, tri_faces = extract_quads(vertices, triangles, uvs_per_triangle)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .grid_vertex import GridVertex
from .find_face_vertices import find_face_vertices
from .find_edge_vertices import find_edge_vertices
from .find_mesh_vertices import find_mesh_vertices
from .edge_tracer import trace_edges, MeshTopology
from .quad_builder import build_quads, extract_quads_only


def extract_quads(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uv_per_triangle: np.ndarray,
    fill_holes: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract a quad mesh from a triangle mesh with UV parameterization.

    This is the main entry point for the Python QEx implementation.
    It matches the interface of libqex_wrapper.extract_quads() for
    drop-in compatibility.

    Parameters
    ----------
    vertices : ndarray, shape (n_verts, 3)
        3D vertex positions of the input triangle mesh.
    triangles : ndarray, shape (n_tris, 3)
        Triangle indices (0-based).
    uv_per_triangle : ndarray, shape (n_tris, 3, 2)
        UV coordinates for each corner of each triangle.
        uv_per_triangle[i, j, :] is the UV for the j-th corner of triangle i.
    fill_holes : bool
        If True (default), fill holes at irregular vertices with triangles.
        (Currently not implemented - placeholder for libQEx compatibility)
    verbose : bool
        Print information about extraction progress.

    Returns
    -------
    quad_vertices : ndarray, shape (n_quad_verts, 3)
        3D vertex positions of the output quad mesh.
    quad_faces : ndarray, shape (n_quads, 4)
        Quad indices (0-based).
    tri_faces : ndarray, shape (n_tris, 3) or None
        Triangle indices for hole fills (0-based), or None if fill_holes=False
        or no holes were filled.

    Algorithm
    ---------
    1. Find face vertices: integer UV points strictly inside triangles
    2. Find edge vertices: integer UV points on triangle edges
    3. Find mesh vertices: original mesh vertices with integer UVs
    4. Combine all grid vertices
    5. Trace edges: connect adjacent grid vertices
    6. Build quads: walk connected vertices to form quad faces

    Example
    -------
    >>> import numpy as np
    >>> # Simple 2-triangle mesh forming a square
    >>> verts = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=float)
    >>> tris = np.array([[0,1,2], [0,2,3]])
    >>> # UVs covering [0,2] x [0,2] - will produce 4 quads
    >>> uvs = np.array([
    ...     [[0,0], [2,0], [2,2]],
    ...     [[0,0], [2,2], [0,2]]
    ... ], dtype=float)
    >>> quad_v, quad_f, tri_f = extract_quads(verts, tris, uvs)
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    uv_per_triangle = np.asarray(uv_per_triangle, dtype=np.float64)

    n_verts = vertices.shape[0]
    n_tris = triangles.shape[0]

    if verbose:
        print(f"  Python QEx: {n_verts} vertices, {n_tris} triangles")

    # Step 1: Find face vertices (integer UV points inside triangles)
    face_vertices = find_face_vertices(vertices, triangles, uv_per_triangle)
    if verbose:
        print(f"  Found {len(face_vertices)} face vertices")

    # Step 2: Find edge vertices (integer UV points on edges)
    edge_vertices, edge_to_vertices = find_edge_vertices(
        vertices, triangles, uv_per_triangle
    )
    if verbose:
        print(f"  Found {len(edge_vertices)} edge vertices")

    # Step 3: Find mesh vertices (original vertices with integer UVs)
    mesh_vertices, vertex_to_grid = find_mesh_vertices(
        vertices, triangles, uv_per_triangle
    )
    if verbose:
        print(f"  Found {len(mesh_vertices)} mesh vertices with integer UVs")

    # Step 4: Combine all grid vertices, deduplicating by UV
    # Priority: face > edge > mesh (face vertices are most accurate for interior points)
    seen_uvs: Dict[Tuple[int, int], int] = {}
    all_grid_vertices: List[GridVertex] = []

    # Add face vertices first (highest priority for interior points)
    for gv in face_vertices:
        if gv.uv not in seen_uvs:
            seen_uvs[gv.uv] = len(all_grid_vertices)
            all_grid_vertices.append(gv)

    # Add edge vertices that don't overlap with face vertices
    edge_added = 0
    for gv in edge_vertices:
        if gv.uv not in seen_uvs:
            seen_uvs[gv.uv] = len(all_grid_vertices)
            all_grid_vertices.append(gv)
            edge_added += 1

    # Add mesh vertices that don't overlap
    mesh_added = 0
    for gv in mesh_vertices:
        if gv.uv not in seen_uvs:
            seen_uvs[gv.uv] = len(all_grid_vertices)
            all_grid_vertices.append(gv)
            mesh_added += 1

    if verbose:
        print(f"  Total grid vertices: {len(all_grid_vertices)} (dedup: {len(face_vertices) + len(edge_vertices) + len(mesh_vertices) - len(all_grid_vertices)} removed)")

    if not all_grid_vertices:
        if verbose:
            print("  No grid vertices found - returning empty mesh")
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32), None

    # Step 5: Build mesh topology and trace edges with ray tracing
    if verbose:
        print("  Building mesh topology for ray tracing...")
    topology = MeshTopology(triangles, uv_per_triangle)

    edges = trace_edges(
        all_grid_vertices,
        triangles=triangles,
        uvs_per_triangle=uv_per_triangle,
        topology=topology,
        verbose=verbose
    )
    if verbose:
        print(f"  Traced {len(edges)} edges")

    # Step 6: Build quad faces
    quad_vertices, all_faces = build_quads(
        all_grid_vertices, verbose=verbose, filter_boundary=True
    )

    # Extract only quads (4-sided faces)
    quad_vertices, quad_faces = extract_quads_only(quad_vertices, all_faces)

    if verbose:
        print(f"  Output: {len(quad_vertices)} vertices, {len(quad_faces)} quads")

    # Hole filling (placeholder - matches libqex_wrapper interface)
    tri_faces = None
    if fill_holes and len(quad_faces) > 0:
        # TODO: Implement hole filling similar to _fill_holes_with_triangles
        # For now, this is a placeholder for API compatibility
        pass

    return quad_vertices, quad_faces, tri_faces


def extract_quads_from_parameterization(
    mesh_vertices: np.ndarray,
    mesh_triangles: np.ndarray,
    uv_coords: np.ndarray,
    uv_triangles: np.ndarray,
    scale: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract quads from RSP parameterization output.

    This is a convenience wrapper that handles the common case where
    the UV coordinates are stored per-vertex with a separate UV triangle
    topology (as output by mesh_to_disk_seamless).

    Parameters
    ----------
    mesh_vertices : ndarray, shape (n_mesh_verts, 3)
        3D positions of the original mesh vertices.
    mesh_triangles : ndarray, shape (n_tris, 3)
        Original mesh triangle indices.
    uv_coords : ndarray, shape (n_uv_verts, 2)
        UV coordinates for the cut mesh vertices.
    uv_triangles : ndarray, shape (n_tris, 3)
        Triangle indices into uv_coords (may differ from mesh_triangles
        due to seam cuts).
    scale : float
        Scale factor to apply to UV coordinates before extraction.
        Higher values = more quads.
    verbose : bool
        Print progress information.

    Returns
    -------
    quad_vertices : ndarray, shape (n_quad_verts, 3)
        3D vertex positions of the output quad mesh.
    quad_faces : ndarray, shape (n_quads, 4)
        Quad indices (0-based).
    tri_faces : ndarray or None
        Triangle indices for hole fills.
    """
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    mesh_triangles = np.asarray(mesh_triangles, dtype=np.int32)
    uv_coords = np.asarray(uv_coords, dtype=np.float64)
    uv_triangles = np.asarray(uv_triangles, dtype=np.int32)

    n_tris = mesh_triangles.shape[0]

    # Scale UV coordinates
    scaled_uvs = uv_coords * scale

    # Build per-triangle UV array
    uv_per_triangle = np.zeros((n_tris, 3, 2), dtype=np.float64)

    for i in range(n_tris):
        for j in range(3):
            uv_idx = uv_triangles[i, j]
            uv_per_triangle[i, j, :] = scaled_uvs[uv_idx]

    # Use the main extract_quads function
    return extract_quads(
        mesh_vertices[mesh_triangles].reshape(-1, 3),  # Flatten to per-corner verts
        np.arange(n_tris * 3).reshape(n_tris, 3),  # Sequential triangle indices
        uv_per_triangle,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Test with a simple case
    print("Testing Python QEx extract_quads...")
    print("=" * 50)

    # Simple 2-triangle mesh forming a unit square
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    # UVs covering [0, 3] x [0, 3] - should produce 4 quads (2x2 grid)
    # Triangle 0: (0,0) -> (3,0) -> (3,3)
    # Triangle 1: (0,0) -> (3,3) -> (0,3)
    uv_per_triangle = np.array([
        [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
        [[0.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
    ], dtype=np.float64)

    print("\nTest 1: Simple square with 3x3 UV grid")
    print(f"  Input: {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"  UV range: [0,3] x [0,3]")

    quad_verts, quad_faces, tri_faces = extract_quads(
        vertices, triangles, uv_per_triangle, verbose=True
    )

    print(f"\nResult:")
    print(f"  Quad vertices: {len(quad_verts)}")
    print(f"  Quad faces: {len(quad_faces)}")
    print(f"  Triangle fills: {len(tri_faces) if tri_faces is not None else 0}")

    if len(quad_verts) > 0:
        print(f"\n  Vertex positions:")
        for i, v in enumerate(quad_verts):
            print(f"    {i}: {v}")

    if len(quad_faces) > 0:
        print(f"\n  Quad faces:")
        for i, f in enumerate(quad_faces):
            print(f"    {i}: {f}")

    # Test 2: Smaller UV range (should produce 1 quad)
    print("\n" + "=" * 50)
    print("\nTest 2: Simple square with 2x2 UV grid (1 quad expected)")

    uv_per_triangle_2 = np.array([
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
        [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
    ], dtype=np.float64)

    quad_verts, quad_faces, tri_faces = extract_quads(
        vertices, triangles, uv_per_triangle_2, verbose=True
    )

    print(f"\nResult:")
    print(f"  Quad vertices: {len(quad_verts)}")
    print(f"  Quad faces: {len(quad_faces)}")

    print("\n" + "=" * 50)
    print("All tests completed.")
