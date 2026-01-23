"""
Debug parallel transport v4 - correct holonomy computation.

The holonomy around a vertex is the sum of parallel transports across
each edge that is incident to the vertex, going around in order.

Key: d1d * para_trans = K means the sum around each vertex equals K.
The d1d operator sums with signs based on edge orientation.
"""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles, compute_all_face_bases
from mesh import TriangleMesh


def _signed_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> float:
    """Signed angle from u to v with n as rotation axis."""
    cross_uv = np.cross(u, v)
    return np.arctan2(np.dot(cross_uv, n), np.dot(u, v))


def _wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def parallel_transport_vector(v, from_normal, to_normal):
    """Parallel transport vector v from one tangent plane to another."""
    axis = np.cross(from_normal, to_normal)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(from_normal, to_normal)

    if sin_angle < 1e-10:
        if cos_angle > 0:
            return v.copy()
        else:
            return -v

    axis = axis / sin_angle
    v_rot = (v * cos_angle +
             np.cross(axis, v) * sin_angle +
             axis * np.dot(axis, v) * (1 - cos_angle))
    return v_rot


def debug():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces, {mesh.n_edges} edges")

    N, T1, T2 = compute_all_face_bases(mesh)
    alpha = compute_corner_angles(mesh)

    # Compute angle defect K
    K = np.zeros(mesh.n_vertices, dtype=np.float64)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    print(f"\nK sum: {K.sum():.6f} (expected 2πχ = {2*np.pi*2:.6f})")

    # Compute holonomy by going around each vertex
    print("\n" + "="*60)
    print("Holonomy by traversing faces around each vertex")
    print("="*60)

    holonomies = []
    for v in range(mesh.n_vertices):
        h = compute_holonomy_correct(mesh, v, N, T1, T2, alpha)
        holonomies.append(h)

    holonomies = np.array(holonomies)
    print(f"\nHolonomy statistics:")
    print(f"  Sum of holonomies: {holonomies.sum():.6f} (expected: {K.sum():.6f})")
    print(f"  Max |holonomy - K|: {np.abs(holonomies - K).max():.6f}")
    print(f"  Mean |holonomy - K|: {np.abs(holonomies - K).mean():.6f}")

    print("\nSample vertices:")
    for v in [0, 1, 10, 50, 100]:
        if v < len(holonomies):
            print(f"  v={v}: K={K[v]:.4f} ({np.degrees(K[v]):.1f}°), "
                  f"holonomy={holonomies[v]:.4f} ({np.degrees(holonomies[v]):.1f}°)")


def compute_holonomy_correct(mesh, v, N, T1, T2, alpha):
    """
    Compute holonomy by parallel transporting a reference vector around the vertex.

    1. Start with a reference direction in the first face
    2. Go around the vertex, crossing edges
    3. At each edge crossing, parallel transport the vector
    4. Also account for the rotation within each face (from one edge to the next)
    5. The total rotation when we return is the holonomy
    """
    corners = mesh.vertex_corners(v)
    if len(corners) == 0:
        return 0.0

    # Start with reference direction = T1 of first face
    first_corner = corners[0]
    first_face = first_corner // 3
    ref_dir = T1[first_face].copy()

    total_rotation = 0.0

    for i, corner in enumerate(corners):
        face = corner // 3
        local = corner % 3

        # Current direction in this face (should match ref_dir at the start)
        # The interior angle at this corner
        interior_angle = alpha[corner]

        # Rotating by the exterior angle within the face (turn to face next edge)
        # This is NOT the same as parallel transport - it's a turn in the plane
        exterior_angle = np.pi - interior_angle
        total_rotation += exterior_angle

        # Now we cross to the next face
        next_corner = corners[(i + 1) % len(corners)]
        next_face = next_corner // 3

        if next_face != face:
            # Parallel transport from face to next_face
            pt_angle = compute_pt_angle_between_faces(mesh, face, next_face, N, T1)
            total_rotation += pt_angle

    # The holonomy is the total rotation minus 2π (full circle)
    # Actually, the holonomy IS the total rotation after going around once
    # For a flat surface, total_rotation = 2π, so holonomy = 0

    holonomy = _wrap_to_pi(total_rotation - 2*np.pi)

    return holonomy


def compute_pt_angle_between_faces(mesh, f0, f1, N, T1):
    """
    Compute parallel transport angle when crossing from f0 to f1.

    This is the angle that T1 rotates by when transported across.
    """
    # Transport T1[f0] to f1's tangent plane
    T1_transported = parallel_transport_vector(T1[f0], N[f0], N[f1])

    # Project and normalize
    T1_transported = T1_transported - np.dot(T1_transported, N[f1]) * N[f1]
    norm = np.linalg.norm(T1_transported)
    if norm < 1e-10:
        return 0.0
    T1_transported = T1_transported / norm

    # Angle from transported T1[f0] to T1[f1]
    angle = _signed_angle(T1_transported, T1[f1], N[f1])

    return angle


if __name__ == "__main__":
    debug()
