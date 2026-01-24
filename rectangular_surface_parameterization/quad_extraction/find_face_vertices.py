"""
Find grid vertices that lie strictly inside triangles (face vertices).

This module implements the face vertex finding algorithm from libQEx
(MeshExtractorT.cc lines 620-677). For each triangle, it finds all integer
UV points that fall strictly inside (not on the boundary) and computes
their 3D positions using barycentric interpolation.
"""

from typing import List

import numpy as np

from rectangular_surface_parameterization.quad_extraction.grid_vertex import GridVertex


def point_in_triangle_strict(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if point p is strictly inside triangle (v0, v1, v2).

    Uses barycentric coordinates. The point is strictly inside if all three
    barycentric coordinates are in the open interval (0, 1).

    Parameters
    ----------
    p : ndarray, shape (2,)
        Point to test.
    v0, v1, v2 : ndarray, shape (2,)
        Triangle vertices in UV space.

    Returns
    -------
    bool
        True if p is strictly inside the triangle.
    """
    # Compute vectors
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    # Compute dot products
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False  # Degenerate triangle

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if point is strictly inside (not on boundary)
    # Barycentric coords are (1-u-v, v, u) for vertices (v0, v1, v2)
    w = 1.0 - u - v

    eps = 1e-10
    return u > eps and v > eps and w > eps


def barycentric_coords(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates of point p with respect to triangle (v0, v1, v2).

    Parameters
    ----------
    p : ndarray, shape (2,)
        Point in UV space.
    v0, v1, v2 : ndarray, shape (2,)
        Triangle vertices in UV space.

    Returns
    -------
    ndarray, shape (3,)
        Barycentric coordinates (w, v_coord, u_coord) where:
        p = w * v0 + v_coord * v1 + u_coord * v2
    """
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = np.dot(v0v2, v0p)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = np.dot(v0v1, v0p)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0])

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    w = 1.0 - u - v

    return np.array([w, v, u])


def interpolate_3d(bary: np.ndarray, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Interpolate 3D position using barycentric coordinates.

    Parameters
    ----------
    bary : ndarray, shape (3,)
        Barycentric coordinates (w, v, u).
    p0, p1, p2 : ndarray, shape (3,)
        3D positions of triangle vertices.

    Returns
    -------
    ndarray, shape (3,)
        Interpolated 3D position.
    """
    return bary[0] * p0 + bary[1] * p1 + bary[2] * p2


def find_face_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs_per_triangle: np.ndarray,
) -> List[GridVertex]:
    """
    Find all integer UV points strictly inside triangles.

    For each triangle, computes the UV bounding box and checks every integer
    point within it. Points strictly inside (not on edges) are recorded as
    face vertices with their 3D positions computed via barycentric interpolation.

    This implements the algorithm from libQEx MeshExtractorT.cc lines 620-677.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        3D vertex positions of the input triangle mesh.
    triangles : ndarray, shape (M, 3)
        Triangle indices (0-based).
    uvs_per_triangle : ndarray, shape (M, 3, 2)
        UV coordinates for each corner of each triangle.
        uvs_per_triangle[i, j, :] is the UV for the j-th corner of triangle i.

    Returns
    -------
    List[GridVertex]
        List of GridVertex objects for all face vertices found.
        Each vertex has type='face', integer UV coordinates, 3D position,
        and the index of the containing triangle.

    Examples
    --------
    >>> import numpy as np
    >>> verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    >>> tris = np.array([[0, 1, 2], [1, 3, 2]])
    >>> uvs = np.array([
    ...     [[-0.5, -0.5], [2.5, -0.5], [-0.5, 2.5]],  # Large triangle in UV
    ...     [[2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]]
    ... ])
    >>> face_verts = find_face_vertices(verts, tris, uvs)
    >>> len(face_verts) > 0
    True
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    uvs_per_triangle = np.asarray(uvs_per_triangle, dtype=np.float64)

    n_tris = triangles.shape[0]
    grid_vertices: List[GridVertex] = []

    for face_idx in range(n_tris):
        # Get UV coordinates for this triangle's corners
        uv0 = uvs_per_triangle[face_idx, 0]
        uv1 = uvs_per_triangle[face_idx, 1]
        uv2 = uvs_per_triangle[face_idx, 2]

        # Get 3D positions of triangle vertices
        v_indices = triangles[face_idx]
        p0 = vertices[v_indices[0]]
        p1 = vertices[v_indices[1]]
        p2 = vertices[v_indices[2]]

        # Check for degenerate triangle in UV space
        edge1 = uv1 - uv0
        edge2 = uv2 - uv0
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        if abs(cross) < 1e-12:
            continue  # Degenerate triangle

        # Compute bounding box in UV space
        uv_min = np.minimum(np.minimum(uv0, uv1), uv2)
        uv_max = np.maximum(np.maximum(uv0, uv1), uv2)

        # Integer range to check (ceil for min, floor for max)
        x_min = int(np.ceil(uv_min[0]))
        x_max = int(np.floor(uv_max[0]))
        y_min = int(np.ceil(uv_min[1]))
        y_max = int(np.floor(uv_max[1]))

        # Check each integer point in the bounding box
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                p = np.array([float(x), float(y)])

                if point_in_triangle_strict(p, uv0, uv1, uv2):
                    # Compute barycentric coordinates
                    bary = barycentric_coords(p, uv0, uv1, uv2)

                    # Interpolate 3D position
                    pos_3d = interpolate_3d(bary, p0, p1, p2)

                    grid_vertices.append(
                        GridVertex(
                            type="face",
                            uv=(x, y),
                            pos_3d=pos_3d,
                            face_idx=face_idx,
                        )
                    )

    return grid_vertices


if __name__ == "__main__":
    # Test with a simple 2-triangle example
    print("Testing find_face_vertices...")

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

    # UV coordinates scaled to cover [-0.5, 2.5] x [-0.5, 2.5]
    # This means integer points (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
    # should be checked, and some will fall inside
    uvs_per_triangle = np.array(
        [
            # Triangle 0: v0, v1, v2
            [[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5]],
            # Triangle 1: v0, v2, v3
            [[-0.5, -0.5], [2.5, 2.5], [-0.5, 2.5]],
        ],
        dtype=np.float64,
    )

    face_verts = find_face_vertices(vertices, triangles, uvs_per_triangle)

    print(f"\nFound {len(face_verts)} face vertices:")
    for gv in face_verts:
        print(f"  UV={gv.uv}, pos_3d={gv.pos_3d}, face={gv.face_idx}, type={gv.type}")

    # Test with a more typical case: UVs roughly matching vertex positions
    print("\n--- Test 2: UVs matching geometry ---")
    uvs_per_triangle_2 = np.array(
        [
            # Triangle 0: v0, v1, v2 -> UVs at (0,0), (3,0), (3,3)
            [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
            # Triangle 1: v0, v2, v3 -> UVs at (0,0), (3,3), (0,3)
            [[0.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
        ],
        dtype=np.float64,
    )

    face_verts_2 = find_face_vertices(vertices, triangles, uvs_per_triangle_2)

    print(f"Found {len(face_verts_2)} face vertices:")
    for gv in face_verts_2:
        print(f"  UV={gv.uv}, pos_3d={np.round(gv.pos_3d, 4)}, face={gv.face_idx}")

    # Verify: In triangle 0 with UVs (0,0)-(3,0)-(3,3), integer points strictly inside:
    # (1,0) is on edge, (2,0) is on edge
    # (1,1), (2,1), (2,2), (3,1), (3,2) need checking
    # The triangle is the lower-right half, so points like (2,1) should be inside

    print("\n--- Test 3: Single large triangle ---")
    vertices_3 = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles_3 = np.array([[0, 1, 2]], dtype=np.int32)
    uvs_3 = np.array(
        [
            [[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]],
        ],
        dtype=np.float64,
    )

    face_verts_3 = find_face_vertices(vertices_3, triangles_3, uvs_3)
    print(f"Found {len(face_verts_3)} face vertices:")
    for gv in face_verts_3:
        print(f"  UV={gv.uv}, pos_3d={np.round(gv.pos_3d, 4)}, face={gv.face_idx}")

    print("\nAll tests completed.")
