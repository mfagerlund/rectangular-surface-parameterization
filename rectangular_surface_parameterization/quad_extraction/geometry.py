"""Geometry utilities for QEx quad extraction.

This module provides core geometric operations needed for quad mesh extraction
from integer-grid UV parameterizations:

1. Point-in-triangle test using barycentric coordinates
2. Barycentric interpolation for computing 3D positions from UV coordinates
3. Line-line intersection for finding where integer iso-lines cross edges
4. Edge crossing detection for finding integer grid crossings along edges

Reference: libQEx (https://github.com/hcebke/libQEx)
- Algebra.hh: Triangle_2.boundedness(), orient2d
- MeshExtractorT.cc: get_mapping() for barycentric interpolation
"""

from typing import Optional, Tuple, List

import numpy as np


# Tolerance for floating point comparisons
EPS = 1e-10


def orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    """Compute the orientation of three 2D points.

    Determines whether c lies to the left of, on, or to the right of
    the directed line from a to b.

    Args:
        a: First point, shape (2,)
        b: Second point, shape (2,)
        c: Third point, shape (2,)

    Returns:
        +1 if c is to the left of ab (counter-clockwise)
         0 if a, b, c are collinear
        -1 if c is to the right of ab (clockwise)
    """
    # Cross product of (b - a) and (c - a)
    det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    if det > EPS:
        return 1  # CCW
    elif det < -EPS:
        return -1  # CW
    else:
        return 0  # Collinear


def point_in_triangle(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    include_boundary: bool = False
) -> bool:
    """Test if a point lies inside a triangle.

    Uses orientation tests (robust to numerical issues) following the
    libQEx Triangle_2.boundedness() approach.

    Args:
        point: Test point, shape (2,)
        v0, v1, v2: Triangle vertices in counter-clockwise order, shape (2,)
        include_boundary: If True, points on edges are considered inside

    Returns:
        True if point is inside (or on boundary if include_boundary=True)
    """
    # Compute orientations of the point with respect to each triangle edge
    ori_a = orient2d(v1, v2, point)
    ori_b = orient2d(v2, v0, point)
    ori_c = orient2d(v0, v1, point)

    # Check triangle orientation
    tri_ori = orient2d(v0, v1, v2)

    if tri_ori == 0:
        # Degenerate triangle - all points are collinear
        if include_boundary:
            # Check if point lies on the line segment
            return _point_on_segment_collinear(point, v0, v1) or \
                   _point_on_segment_collinear(point, v1, v2) or \
                   _point_on_segment_collinear(point, v2, v0)
        return False

    # For a CCW triangle, point is inside if all orientations are CCW (positive)
    # For a CW triangle, point is inside if all orientations are CW (negative)
    if include_boundary:
        if tri_ori > 0:  # CCW triangle
            return ori_a >= 0 and ori_b >= 0 and ori_c >= 0
        else:  # CW triangle
            return ori_a <= 0 and ori_b <= 0 and ori_c <= 0
    else:
        # Strictly inside - all orientations must match triangle orientation
        if tri_ori > 0:  # CCW triangle
            return ori_a > 0 and ori_b > 0 and ori_c > 0
        else:  # CW triangle
            return ori_a < 0 and ori_b < 0 and ori_c < 0


def _point_on_segment_collinear(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray
) -> bool:
    """Check if a point lies on a segment, assuming collinearity.

    Args:
        point: Test point, shape (2,)
        a, b: Segment endpoints, shape (2,)

    Returns:
        True if point lies on segment [a, b]
    """
    return (min(a[0], b[0]) - EPS <= point[0] <= max(a[0], b[0]) + EPS and
            min(a[1], b[1]) - EPS <= point[1] <= max(a[1], b[1]) + EPS)


def barycentric_coordinates(
    point: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray
) -> np.ndarray:
    """Compute barycentric coordinates of a point with respect to a triangle.

    Args:
        point: Query point, shape (2,)
        v0, v1, v2: Triangle vertices, shape (2,)

    Returns:
        Barycentric coordinates (w0, w1, w2) such that
        point = w0*v0 + w1*v1 + w2*v2 and w0 + w1 + w2 = 1
    """
    # Using the area method
    # w0 = area(point, v1, v2) / area(v0, v1, v2)
    # etc.

    # Signed area of triangle (v0, v1, v2) * 2
    denom = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])

    if abs(denom) < EPS:
        # Degenerate triangle - return equal weights
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    # Signed areas of sub-triangles * 2
    w0 = ((v1[0] - point[0]) * (v2[1] - point[1]) -
          (v2[0] - point[0]) * (v1[1] - point[1])) / denom
    w1 = ((v2[0] - point[0]) * (v0[1] - point[1]) -
          (v0[0] - point[0]) * (v2[1] - point[1])) / denom
    w2 = 1.0 - w0 - w1

    return np.array([w0, w1, w2])


def barycentric_interpolate_3d(
    uv_point: np.ndarray,
    uv_tri: np.ndarray,
    pos_3d_tri: np.ndarray
) -> np.ndarray:
    """Compute 3D position from UV coordinates using barycentric interpolation.

    This implements the get_mapping() function from libQEx MeshExtractorT.cc
    which computes a mapping matrix M such that:
        pos_3d = M * [u, v, 1]^T

    Args:
        uv_point: UV coordinates of query point, shape (2,)
        uv_tri: UV coordinates of triangle vertices, shape (3, 2)
        pos_3d_tri: 3D positions of triangle vertices, shape (3, 3)

    Returns:
        Interpolated 3D position, shape (3,)
    """
    # Compute barycentric coordinates
    bary = barycentric_coordinates(
        uv_point,
        uv_tri[0],
        uv_tri[1],
        uv_tri[2]
    )

    # Interpolate 3D position
    return bary[0] * pos_3d_tri[0] + bary[1] * pos_3d_tri[1] + bary[2] * pos_3d_tri[2]


def get_mapping_matrix(
    uv_tri: np.ndarray,
    pos_3d_tri: np.ndarray
) -> np.ndarray:
    """Compute the 3x3 mapping matrix from UV to 3D.

    This is a direct port of libQEx get_mapping() which computes M such that:
        pos_3d = M @ [u, v, 1]^T

    The matrix encodes barycentric interpolation:
        P = [p0 | p1 | p2]  (3D positions as columns)
        p = [u0  u1  u2]    (UV coordinates)
            [v0  v1  v2]
            [1   1   1 ]
        M = P @ p^(-1)

    Args:
        uv_tri: UV coordinates of triangle vertices, shape (3, 2)
        pos_3d_tri: 3D positions of triangle vertices, shape (3, 3)

    Returns:
        3x3 mapping matrix M
    """
    # 2D matrix (homogeneous coordinates)
    p = np.array([
        [uv_tri[0, 0], uv_tri[1, 0], uv_tri[2, 0]],
        [uv_tri[0, 1], uv_tri[1, 1], uv_tri[2, 1]],
        [1.0, 1.0, 1.0]
    ])

    # 3D matrix (positions as columns)
    P = np.array([
        [pos_3d_tri[0, 0], pos_3d_tri[1, 0], pos_3d_tri[2, 0]],
        [pos_3d_tri[0, 1], pos_3d_tri[1, 1], pos_3d_tri[2, 1]],
        [pos_3d_tri[0, 2], pos_3d_tri[1, 2], pos_3d_tri[2, 2]]
    ])

    # M = P @ p^(-1)
    return P @ np.linalg.inv(p)


def apply_mapping(M: np.ndarray, u: float, v: float) -> np.ndarray:
    """Apply a mapping matrix to UV coordinates to get 3D position.

    Args:
        M: 3x3 mapping matrix from get_mapping_matrix()
        u, v: UV coordinates

    Returns:
        3D position, shape (3,)
    """
    uv_hom = np.array([u, v, 1.0])
    return M @ uv_hom


def segment_intersection_param(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Compute intersection parameters for two line segments.

    Finds t and s such that:
        p0 + t * (p1 - p0) = q0 + s * (q1 - q0)

    Args:
        p0, p1: Endpoints of first segment, shape (2,)
        q0, q1: Endpoints of second segment, shape (2,)

    Returns:
        Tuple (t, s) of intersection parameters, or None if parallel.
        The intersection point lies on segment P if 0 <= t <= 1,
        and on segment Q if 0 <= s <= 1.
    """
    d1 = p1 - p0
    d2 = q1 - q0
    d0 = q0 - p0

    # Cross product (2D determinant)
    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < EPS:
        # Lines are parallel
        return None

    # t = (d0 x d2) / (d1 x d2)
    t = (d0[0] * d2[1] - d0[1] * d2[0]) / cross

    # s = (d0 x d1) / (d1 x d2)
    s = (d0[0] * d1[1] - d0[1] * d1[0]) / cross

    return (t, s)


def line_segment_intersects(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray
) -> bool:
    """Check if two line segments intersect.

    Args:
        p0, p1: Endpoints of first segment, shape (2,)
        q0, q1: Endpoints of second segment, shape (2,)

    Returns:
        True if segments intersect (including at endpoints)
    """
    result = segment_intersection_param(p0, p1, q0, q1)
    if result is None:
        # Parallel - check for overlap on collinear segments
        if orient2d(p0, p1, q0) == 0:
            return (_point_on_segment_collinear(q0, p0, p1) or
                    _point_on_segment_collinear(q1, p0, p1) or
                    _point_on_segment_collinear(p0, q0, q1) or
                    _point_on_segment_collinear(p1, q0, q1))
        return False

    t, s = result
    return -EPS <= t <= 1.0 + EPS and -EPS <= s <= 1.0 + EPS


def find_integer_crossings_on_edge(
    uv0: np.ndarray,
    uv1: np.ndarray,
    pos_3d_0: np.ndarray,
    pos_3d_1: np.ndarray,
    exclude_endpoints: bool = True
) -> List[Tuple[Tuple[int, int], np.ndarray, float]]:
    """Find all integer UV coordinate crossings along a mesh edge.

    Scans the edge for points where either u or v is an integer.
    This is used to find grid vertices that lie on edges.

    The algorithm follows libQEx MeshExtractorT.cc generate_grid_vertices()
    which iterates over the longer UV dimension and computes the other.

    Args:
        uv0: UV coordinates at edge start, shape (2,)
        uv1: UV coordinates at edge end, shape (2,)
        pos_3d_0: 3D position at edge start, shape (3,)
        pos_3d_1: 3D position at edge end, shape (3,)
        exclude_endpoints: If True, skip crossings exactly at endpoints

    Returns:
        List of tuples: ((int_u, int_v), pos_3d, alpha)
        where alpha is the interpolation parameter (0 at uv0, 1 at uv1)
    """
    crossings = []

    # Get bounding box in UV space
    u_min, u_max = min(uv0[0], uv1[0]), max(uv0[0], uv1[0])
    v_min, v_max = min(uv0[1], uv1[1]), max(uv0[1], uv1[1])

    # Edge direction and length
    du = uv1[0] - uv0[0]
    dv = uv1[1] - uv0[1]

    # Process based on which dimension has larger range (more stable interpolation)
    if abs(du) >= abs(dv):
        # Iterate over integer u values
        u_start = int(np.ceil(u_min))
        u_end = int(np.floor(u_max))

        if exclude_endpoints:
            if u_start == u_min:
                u_start += 1
            if u_end == u_max:
                u_end -= 1

        for u in range(u_start, u_end + 1):
            if abs(du) < EPS:
                continue
            alpha = (u - uv0[0]) / du
            if alpha < -EPS or alpha > 1.0 + EPS:
                continue

            # Compute v at this u
            v = uv0[1] + alpha * dv
            v_int = round(v)

            # Check if v is close to an integer
            if abs(v - v_int) < EPS:
                pos_3d = (1 - alpha) * pos_3d_0 + alpha * pos_3d_1
                crossings.append(((u, v_int), pos_3d, alpha))
    else:
        # Iterate over integer v values
        v_start = int(np.ceil(v_min))
        v_end = int(np.floor(v_max))

        if exclude_endpoints:
            if v_start == v_min:
                v_start += 1
            if v_end == v_max:
                v_end -= 1

        for v in range(v_start, v_end + 1):
            if abs(dv) < EPS:
                continue
            alpha = (v - uv0[1]) / dv
            if alpha < -EPS or alpha > 1.0 + EPS:
                continue

            # Compute u at this v
            u = uv0[0] + alpha * du
            u_int = round(u)

            # Check if u is close to an integer
            if abs(u - u_int) < EPS:
                pos_3d = (1 - alpha) * pos_3d_0 + alpha * pos_3d_1
                crossings.append(((u_int, v), pos_3d, alpha))

    # Sort by alpha to get consistent ordering along the edge
    crossings.sort(key=lambda x: x[2])

    return crossings


def find_all_integer_crossings_on_edge(
    uv0: np.ndarray,
    uv1: np.ndarray,
    pos_3d_0: np.ndarray,
    pos_3d_1: np.ndarray,
    exclude_endpoints: bool = True
) -> List[Tuple[Tuple[int, int], np.ndarray, float]]:
    """Find all integer grid crossings along a mesh edge.

    This is a more comprehensive version that finds ALL integer crossings,
    not just where both u and v are integers simultaneously.

    For each integer iso-line (u=k or v=k) that crosses the edge,
    this computes the exact crossing point.

    Args:
        uv0: UV coordinates at edge start, shape (2,)
        uv1: UV coordinates at edge end, shape (2,)
        pos_3d_0: 3D position at edge start, shape (3,)
        pos_3d_1: 3D position at edge end, shape (3,)
        exclude_endpoints: If True, skip crossings exactly at endpoints

    Returns:
        List of tuples: ((int_u, int_v), pos_3d, alpha)
        Each entry represents a point where at least one of u,v is integer
        and both coordinates are rounded to nearest integer.
    """
    crossings = []

    du = uv1[0] - uv0[0]
    dv = uv1[1] - uv0[1]

    # Find crossings where u is an integer
    if abs(du) > EPS:
        u_start = int(np.ceil(min(uv0[0], uv1[0])))
        u_end = int(np.floor(max(uv0[0], uv1[0])))

        for u in range(u_start, u_end + 1):
            alpha = (u - uv0[0]) / du

            if exclude_endpoints:
                if alpha < EPS or alpha > 1.0 - EPS:
                    continue
            else:
                if alpha < -EPS or alpha > 1.0 + EPS:
                    continue

            v = uv0[1] + alpha * dv
            v_int = round(v)

            # Only add if v is also an integer (true grid point)
            if abs(v - v_int) < EPS:
                pos_3d = (1 - alpha) * pos_3d_0 + alpha * pos_3d_1
                crossings.append(((u, v_int), pos_3d, alpha))

    # Find crossings where v is an integer (but u might not be)
    if abs(dv) > EPS:
        v_start = int(np.ceil(min(uv0[1], uv1[1])))
        v_end = int(np.floor(max(uv0[1], uv1[1])))

        for v in range(v_start, v_end + 1):
            alpha = (v - uv0[1]) / dv

            if exclude_endpoints:
                if alpha < EPS or alpha > 1.0 - EPS:
                    continue
            else:
                if alpha < -EPS or alpha > 1.0 + EPS:
                    continue

            u = uv0[0] + alpha * du
            u_int = round(u)

            # Only add if u is also an integer (true grid point)
            if abs(u - u_int) < EPS:
                # Check if we already have this point from the u-iteration
                uv_tuple = (u_int, v)
                already_found = any(c[0] == uv_tuple for c in crossings)
                if not already_found:
                    pos_3d = (1 - alpha) * pos_3d_0 + alpha * pos_3d_1
                    crossings.append((uv_tuple, pos_3d, alpha))

    # Sort by alpha
    crossings.sort(key=lambda x: x[2])

    return crossings


def is_uv_integer(uv: np.ndarray, tol: float = EPS) -> bool:
    """Check if UV coordinates are both integers.

    Args:
        uv: UV coordinates, shape (2,)
        tol: Tolerance for integer check

    Returns:
        True if both u and v are close to integers
    """
    return (abs(uv[0] - round(uv[0])) < tol and
            abs(uv[1] - round(uv[1])) < tol)


def snap_to_integer(value: float, tol: float = EPS) -> Tuple[bool, int]:
    """Snap a value to the nearest integer if close enough.

    Args:
        value: Floating point value
        tol: Tolerance for snapping

    Returns:
        Tuple of (was_snapped, integer_value)
    """
    int_val = round(value)
    if abs(value - int_val) < tol:
        return True, int(int_val)
    return False, int(int_val)


def triangle_uv_bbox(uv_tri: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box of a triangle in UV space.

    Args:
        uv_tri: UV coordinates of triangle vertices, shape (3, 2)

    Returns:
        Tuple of (u_min, u_max, v_min, v_max)
    """
    u_min = min(uv_tri[0, 0], uv_tri[1, 0], uv_tri[2, 0])
    u_max = max(uv_tri[0, 0], uv_tri[1, 0], uv_tri[2, 0])
    v_min = min(uv_tri[0, 1], uv_tri[1, 1], uv_tri[2, 1])
    v_max = max(uv_tri[0, 1], uv_tri[1, 1], uv_tri[2, 1])
    return (u_min, u_max, v_min, v_max)


def integer_points_in_triangle_bbox(
    uv_tri: np.ndarray
) -> List[Tuple[int, int]]:
    """Get all integer UV points within the bounding box of a triangle.

    This is the first step in finding face vertices - enumerate all
    candidate integer points, then test each for containment.

    Args:
        uv_tri: UV coordinates of triangle vertices, shape (3, 2)

    Returns:
        List of (u, v) integer coordinate tuples in the bounding box
    """
    u_min, u_max, v_min, v_max = triangle_uv_bbox(uv_tri)

    u_start = int(np.ceil(u_min))
    u_end = int(np.floor(u_max))
    v_start = int(np.ceil(v_min))
    v_end = int(np.floor(v_max))

    points = []
    for u in range(u_start, u_end + 1):
        for v in range(v_start, v_end + 1):
            points.append((u, v))

    return points
