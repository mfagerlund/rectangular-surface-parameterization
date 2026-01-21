"""
Cross field computation and parallel transport.

A cross field assigns four orthogonal directions to each face (with 4-fold symmetry).
We store one representative unit vector W per face, and compute angles ξ relative
to the first edge of each face.

Phase 2 implementation:
- Simple propagation from seed face via BFS
- Parallel transport to maintain smoothness across edges
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple

from mesh import TriangleMesh
from geometry import compute_all_face_bases, compute_corner_angles


def parallel_transport(
    v: np.ndarray,
    from_normal: np.ndarray,
    to_normal: np.ndarray
) -> np.ndarray:
    """
    Parallel transport a tangent vector v from one tangent plane to another.

    Uses rotation that aligns the normals. For tangent vectors on a surface,
    this approximates Levi-Civita connection.

    Args:
        v: vector in the tangent plane of from_normal
        from_normal: source face normal
        to_normal: target face normal

    Returns:
        Transported vector in target tangent plane
    """
    # Axis of rotation is cross product of normals
    axis = np.cross(from_normal, to_normal)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(from_normal, to_normal)

    if sin_angle < 1e-10:
        # Normals are parallel, no rotation needed
        if cos_angle > 0:
            return v.copy()
        else:
            # Normals are anti-parallel (should not happen in a good mesh)
            return -v

    # Normalize axis
    axis = axis / sin_angle

    # Rodrigues' rotation formula
    v_rot = (v * cos_angle +
             np.cross(axis, v) * sin_angle +
             axis * np.dot(axis, v) * (1 - cos_angle))

    return v_rot


def propagate_cross_field(
    mesh: TriangleMesh,
    seed_face: int = 0,
    seed_direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Propagate a cross field from a seed face using BFS and parallel transport.

    Args:
        mesh: Triangle mesh
        seed_face: Starting face index
        seed_direction: Initial direction in seed face (if None, uses first edge direction)

    Returns:
        W: |F| x 3 array of one representative cross field direction per face
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    W = np.zeros((mesh.n_faces, 3), dtype=np.float64)
    visited = np.zeros(mesh.n_faces, dtype=bool)

    # Initialize seed face
    if seed_direction is None:
        # Use first edge direction (T1)
        seed_direction = T1[seed_face]
    else:
        # Project onto tangent plane and normalize
        seed_direction = seed_direction - np.dot(seed_direction, N[seed_face]) * N[seed_face]
        seed_direction = seed_direction / (np.linalg.norm(seed_direction) + 1e-30)

    W[seed_face] = seed_direction
    visited[seed_face] = True

    # BFS propagation
    queue = deque([seed_face])

    while queue:
        f = queue.popleft()

        # Visit neighbors across each edge
        for local in range(3):
            neighbor = mesh.face_adjacent(f, local)
            if neighbor == -1 or visited[neighbor]:
                continue

            # Parallel transport W[f] to neighbor
            w_transported = parallel_transport(W[f], N[f], N[neighbor])

            # Project onto neighbor's tangent plane (should already be there, but numerical safety)
            w_transported = w_transported - np.dot(w_transported, N[neighbor]) * N[neighbor]
            norm = np.linalg.norm(w_transported)
            if norm > 1e-10:
                w_transported = w_transported / norm
            else:
                # Fallback to neighbor's T1
                w_transported = T1[neighbor]

            W[neighbor] = w_transported
            visited[neighbor] = True
            queue.append(neighbor)

    return W


def cross_field_to_angles(mesh: TriangleMesh, W: np.ndarray) -> np.ndarray:
    """
    Convert cross field vectors to angles ξ relative to first edge of each face.

    From Algorithm 1 line 9:
    ξ_ijk = atan2(<T2, W>, <T1, W>)

    Args:
        W: |F| x 3 cross field directions

    Returns:
        xi: |F| array of angles in radians
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    xi = np.zeros(mesh.n_faces, dtype=np.float64)
    for f in range(mesh.n_faces):
        xi[f] = np.arctan2(np.dot(T2[f], W[f]), np.dot(T1[f], W[f]))

    return xi


def angles_to_cross_field(mesh: TriangleMesh, xi: np.ndarray) -> np.ndarray:
    """
    Convert angles back to cross field vectors.

    W = cos(ξ) * T1 + sin(ξ) * T2

    Args:
        xi: |F| array of angles in radians

    Returns:
        W: |F| x 3 cross field directions
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    W = np.zeros((mesh.n_faces, 3), dtype=np.float64)
    for f in range(mesh.n_faces):
        W[f] = np.cos(xi[f]) * T1[f] + np.sin(xi[f]) * T2[f]

    return W


def compute_xi_per_halfedge(mesh: TriangleMesh, xi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Compute cross field angle relative to each halfedge direction.

    From Algorithm 2 lines 1-4:
    For face ijk with halfedges ij, jk, ki:
    - ξ_ij = ξ_ijk (angle relative to edge ij)
    - ξ_jk = ξ_ij - (π - α_j^ki)
    - ξ_ki = ξ_jk - (π - α_k^ij)

    Args:
        xi: |F| array of cross field angles (relative to first edge)
        alpha: |C| array of corner angles

    Returns:
        xi_halfedge: |H| array of angles relative to each halfedge
    """
    xi_halfedge = np.zeros(mesh.n_halfedges, dtype=np.float64)

    for f in range(mesh.n_faces):
        # Corner indices
        c0 = 3 * f + 0  # corner at i in face ijk
        c1 = 3 * f + 1  # corner at j
        c2 = 3 * f + 2  # corner at k

        # Halfedge indices
        he_ij = 3 * f + 0
        he_jk = 3 * f + 1
        he_ki = 3 * f + 2

        # ξ_ij is given
        xi_halfedge[he_ij] = xi[f]

        # ξ_jk = ξ_ij - (π - α_j^ki)
        # α_j^ki is the angle at vertex j, which is corner index c1
        xi_halfedge[he_jk] = xi_halfedge[he_ij] - (np.pi - alpha[c1])

        # ξ_ki = ξ_jk - (π - α_k^ij)
        # α_k^ij is the angle at vertex k, which is corner index c2
        xi_halfedge[he_ki] = xi_halfedge[he_jk] - (np.pi - alpha[c2])

    return xi_halfedge


def compute_smoothness_energy(mesh: TriangleMesh, W: np.ndarray) -> float:
    """
    Compute cross field smoothness energy (sum of squared angle differences across edges).

    This measures how well the cross field is aligned across edges.

    Returns:
        Total smoothness energy (lower is better)
    """
    N, T1, T2 = compute_all_face_bases(mesh)
    energy = 0.0

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Transport W[f0] to f1's tangent plane
        w0_transported = parallel_transport(W[f0], N[f0], N[f1])

        # Compute angle difference (mod π/2 for cross field)
        angle0 = np.arctan2(np.dot(T2[f1], w0_transported), np.dot(T1[f1], w0_transported))
        angle1 = np.arctan2(np.dot(T2[f1], W[f1]), np.dot(T1[f1], W[f1]))

        diff = angle1 - angle0
        # Wrap to [-π/4, π/4] for cross field
        diff = np.arctan2(np.sin(4 * diff), np.cos(4 * diff)) / 4

        energy += diff * diff

    return energy


def initialize_smooth_cross_field(mesh: TriangleMesh, n_iters: int = 100) -> np.ndarray:
    """
    Initialize a smooth cross field by propagation then local smoothing.

    Args:
        n_iters: Number of smoothing iterations

    Returns:
        W: |F| x 3 smoothed cross field
    """
    # Initial propagation
    W = propagate_cross_field(mesh)
    xi = cross_field_to_angles(mesh, W)

    N, T1, T2 = compute_all_face_bases(mesh)

    # Simple smoothing: average neighboring angles (considering 4-fold symmetry)
    for _ in range(n_iters):
        new_xi = xi.copy()

        for f in range(mesh.n_faces):
            # Collect neighbor angles, transported to face f
            angles = [xi[f]]  # include self

            for local in range(3):
                neighbor = mesh.face_adjacent(f, local)
                if neighbor == -1:
                    continue

                # Transport W[neighbor] to face f
                w_neighbor = angles_to_cross_field(mesh, xi[neighbor:neighbor+1])[0]
                w_neighbor = np.cos(xi[neighbor]) * T1[neighbor] + np.sin(xi[neighbor]) * T2[neighbor]
                w_transported = parallel_transport(w_neighbor, N[neighbor], N[f])

                # Get angle in face f's frame
                angle_n = np.arctan2(np.dot(T2[f], w_transported), np.dot(T1[f], w_transported))

                # Find closest representative (4-fold symmetry)
                diff = angle_n - xi[f]
                diff_mod = np.arctan2(np.sin(4 * diff), np.cos(4 * diff)) / 4
                angles.append(xi[f] + diff_mod)

            # Average (simple approach, could use better weighting)
            new_xi[f] = np.mean(angles)

        xi = new_xi

    return angles_to_cross_field(mesh, xi)
