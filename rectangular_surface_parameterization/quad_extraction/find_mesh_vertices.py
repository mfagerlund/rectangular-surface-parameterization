"""Find grid vertices at mesh vertices with integer UV coordinates.

This module identifies mesh vertices where the UV coordinates are exactly
integers (within a small tolerance). These become grid vertices of type
'vertex' in the quad extraction algorithm.

Algorithm reference: libQEx MeshExtractorT.cc lines 800-900
"""

from typing import Dict, List, Tuple

import numpy as np

from rectangular_surface_parameterization.quad_extraction.grid_vertex import GridVertex


# Default tolerance for integer coordinate check
INTEGER_TOLERANCE = 1e-9


def is_integer(value: float, tolerance: float = INTEGER_TOLERANCE) -> bool:
    """Check if a value is close to an integer.

    Args:
        value: The floating-point value to check.
        tolerance: Maximum distance from nearest integer.

    Returns:
        True if value is within tolerance of an integer.
    """
    return abs(value - round(value)) < tolerance


def is_integer_uv(uv: np.ndarray, tolerance: float = INTEGER_TOLERANCE) -> bool:
    """Check if both UV coordinates are integers.

    Args:
        uv: UV coordinates as shape (2,) array.
        tolerance: Maximum distance from nearest integer.

    Returns:
        True if both u and v are within tolerance of integers.
    """
    return is_integer(uv[0], tolerance) and is_integer(uv[1], tolerance)


def find_mesh_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs_per_triangle: np.ndarray,
    tolerance: float = INTEGER_TOLERANCE,
) -> Tuple[List[GridVertex], Dict[int, int]]:
    """Find all mesh vertices with integer UV coordinates.

    For each mesh vertex, checks if its UV coordinates are integers. If so,
    creates a GridVertex record with type='vertex'.

    Note: In this simplified version, we assume seamless UVs (no seams), so
    each vertex has consistent UV coordinates across all adjacent triangles.
    We use the first occurrence of each vertex to determine its UV.

    Args:
        vertices: Mesh vertex positions, shape (N, 3).
        triangles: Triangle vertex indices, shape (M, 3).
        uvs_per_triangle: UV coordinates per triangle corner, shape (M, 3, 2).
            uvs_per_triangle[f, i, :] is the UV for the i-th corner of face f.
        tolerance: Tolerance for integer coordinate check.

    Returns:
        A tuple of:
        - grid_vertices: List of GridVertex objects for mesh vertices with
          integer UVs.
        - vertex_to_grid: Dict mapping mesh vertex index to GridVertex index
          in the returned list. Only contains vertices with integer UVs.
    """
    n_vertices = vertices.shape[0]
    n_triangles = triangles.shape[0]

    # Track which mesh vertices we've already processed
    # Maps vertex index -> (uv, face_idx) for first occurrence
    vertex_uv_map: Dict[int, Tuple[np.ndarray, int]] = {}

    # Build mapping from vertex index to its first UV occurrence
    for face_idx in range(n_triangles):
        for corner in range(3):
            vertex_idx = triangles[face_idx, corner]

            # Only record first occurrence
            if vertex_idx not in vertex_uv_map:
                uv = uvs_per_triangle[face_idx, corner, :].copy()
                vertex_uv_map[vertex_idx] = (uv, face_idx)

    # Find vertices with integer UVs
    grid_vertices: List[GridVertex] = []
    vertex_to_grid: Dict[int, int] = {}

    for vertex_idx in range(n_vertices):
        if vertex_idx not in vertex_uv_map:
            # Vertex not referenced by any triangle (unreferenced vertex)
            continue

        uv, face_idx = vertex_uv_map[vertex_idx]

        if is_integer_uv(uv, tolerance):
            # Round to exact integers for the grid vertex UV
            uv_int = (int(round(uv[0])), int(round(uv[1])))

            # Get 3D position
            pos_3d = vertices[vertex_idx].copy()

            # Create grid vertex
            grid_vertex = GridVertex(
                type='vertex',
                uv=uv_int,
                pos_3d=pos_3d,
                face_idx=face_idx,
                vertex_idx=vertex_idx,
            )

            vertex_to_grid[vertex_idx] = len(grid_vertices)
            grid_vertices.append(grid_vertex)

    return grid_vertices, vertex_to_grid


def find_mesh_vertices_with_seams(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs_per_triangle: np.ndarray,
    tolerance: float = INTEGER_TOLERANCE,
) -> Tuple[List[GridVertex], Dict[Tuple[int, int], int]]:
    """Find mesh vertices with integer UVs, handling seams.

    Unlike find_mesh_vertices(), this version handles meshes with UV seams
    where a mesh vertex can have different UV coordinates in different
    triangles. Each unique (vertex_idx, uv) pair can produce a separate
    grid vertex.

    Args:
        vertices: Mesh vertex positions, shape (N, 3).
        triangles: Triangle vertex indices, shape (M, 3).
        uvs_per_triangle: UV coordinates per triangle corner, shape (M, 3, 2).
        tolerance: Tolerance for integer coordinate check.

    Returns:
        A tuple of:
        - grid_vertices: List of GridVertex objects.
        - vertex_uv_to_grid: Dict mapping (mesh_vertex_idx, uv_tuple) to
          GridVertex index. The uv_tuple is the rounded integer UV.
    """
    n_triangles = triangles.shape[0]

    # Track unique (vertex, uv) pairs
    # Key: (vertex_idx, uv_int_tuple), Value: (uv_float, face_idx)
    seen_pairs: Dict[Tuple[int, Tuple[int, int]], Tuple[np.ndarray, int]] = {}

    for face_idx in range(n_triangles):
        for corner in range(3):
            vertex_idx = triangles[face_idx, corner]
            uv = uvs_per_triangle[face_idx, corner, :]

            if is_integer_uv(uv, tolerance):
                uv_int = (int(round(uv[0])), int(round(uv[1])))
                key = (vertex_idx, uv_int)

                # Only record first occurrence of each (vertex, uv) pair
                if key not in seen_pairs:
                    seen_pairs[key] = (uv.copy(), face_idx)

    # Build grid vertices from unique pairs
    grid_vertices: List[GridVertex] = []
    vertex_uv_to_grid: Dict[Tuple[int, int], int] = {}

    for (vertex_idx, uv_int), (uv_float, face_idx) in seen_pairs.items():
        pos_3d = vertices[vertex_idx].copy()

        grid_vertex = GridVertex(
            type='vertex',
            uv=uv_int,
            pos_3d=pos_3d,
            face_idx=face_idx,
            vertex_idx=vertex_idx,
        )

        vertex_uv_to_grid[(vertex_idx, uv_int)] = len(grid_vertices)
        grid_vertices.append(grid_vertex)

    return grid_vertices, vertex_uv_to_grid


if __name__ == "__main__":
    # Test with simple examples
    print("Testing find_mesh_vertices...")

    # Create a simple mesh: two triangles forming a quad
    # 3D positions form a unit square in the XY plane
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # v0
            [1.0, 0.0, 0.0],  # v1
            [1.0, 1.0, 0.0],  # v2
            [0.0, 1.0, 0.0],  # v3
        ],
        dtype=np.float64,
    )

    # Two triangles: (v0, v1, v2) and (v0, v2, v3)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    # Test 1: All vertices have integer UVs
    print("\n--- Test 1: All vertices at integer UVs ---")
    uvs_per_triangle_1 = np.array(
        [
            # Triangle 0: v0, v1, v2
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            # Triangle 1: v0, v2, v3
            [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    mesh_verts, v2g = find_mesh_vertices(vertices, triangles, uvs_per_triangle_1)
    print(f"Found {len(mesh_verts)} mesh vertices with integer UVs:")
    for gv in mesh_verts:
        print(f"  vertex_idx={gv.vertex_idx}, UV={gv.uv}, pos_3d={gv.pos_3d}")
    print(f"vertex_to_grid mapping: {v2g}")

    # Test 2: No vertices have integer UVs (all offset by 0.5)
    print("\n--- Test 2: No vertices at integer UVs ---")
    uvs_per_triangle_2 = np.array(
        [
            [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]],
            [[0.5, 0.5], [1.5, 1.5], [0.5, 1.5]],
        ],
        dtype=np.float64,
    )

    mesh_verts_2, v2g_2 = find_mesh_vertices(vertices, triangles, uvs_per_triangle_2)
    print(f"Found {len(mesh_verts_2)} mesh vertices with integer UVs")

    # Test 3: Some vertices have integer UVs
    print("\n--- Test 3: Some vertices at integer UVs ---")
    uvs_per_triangle_3 = np.array(
        [
            [[0.0, 0.0], [0.7, 0.0], [0.7, 0.7]],  # Only v0 is integer
            [[0.0, 0.0], [0.7, 0.7], [0.0, 1.0]],  # v0 and v3 are integer
        ],
        dtype=np.float64,
    )

    mesh_verts_3, v2g_3 = find_mesh_vertices(vertices, triangles, uvs_per_triangle_3)
    print(f"Found {len(mesh_verts_3)} mesh vertices with integer UVs:")
    for gv in mesh_verts_3:
        print(f"  vertex_idx={gv.vertex_idx}, UV={gv.uv}, pos_3d={gv.pos_3d}")

    # Test 4: Near-integer values (within tolerance)
    print("\n--- Test 4: Near-integer values ---")
    uvs_per_triangle_4 = np.array(
        [
            [[1e-10, 1e-10], [1.0 - 1e-10, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0 + 1e-10], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    mesh_verts_4, v2g_4 = find_mesh_vertices(vertices, triangles, uvs_per_triangle_4)
    print(f"Found {len(mesh_verts_4)} mesh vertices with integer UVs:")
    for gv in mesh_verts_4:
        print(f"  vertex_idx={gv.vertex_idx}, UV={gv.uv}")

    # Test 5: Seam handling
    print("\n--- Test 5: Seam handling (different UVs for same vertex) ---")
    uvs_with_seam = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],  # v0 at (0,0)
            [[2.0, 0.0], [1.0, 1.0], [2.0, 1.0]],  # v0 at (2,0) - seam!
        ],
        dtype=np.float64,
    )

    mesh_verts_seam, v2g_seam = find_mesh_vertices_with_seams(
        vertices, triangles, uvs_with_seam
    )
    print(f"Found {len(mesh_verts_seam)} mesh vertices with integer UVs (with seams):")
    for gv in mesh_verts_seam:
        print(f"  vertex_idx={gv.vertex_idx}, UV={gv.uv}")
    print(f"vertex_uv_to_grid mapping: {v2g_seam}")

    print("\nAll tests completed.")
