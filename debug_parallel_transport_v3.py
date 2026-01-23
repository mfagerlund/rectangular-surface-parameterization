"""
Debug parallel transport v3 - compute by circulating around vertices.

The constraint d1d * para_trans = K means:
  For each vertex v, sum of signed para_trans around v = K[v]

Let's verify this by going around each vertex and tracking angle changes.
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

    # For a few vertices, manually compute the holonomy by going around
    print("\n" + "="*60)
    print("Manual holonomy computation for sample vertices")
    print("="*60)

    for v in [0, 1, 10, 50]:
        holonomy = compute_holonomy_around_vertex(mesh, v, N, T1, T2, alpha)
        print(f"\nVertex {v}:")
        print(f"  Angle defect K[v]: {K[v]:.6f} ({np.degrees(K[v]):.2f}°)")
        print(f"  Holonomy: {holonomy:.6f} ({np.degrees(holonomy):.2f}°)")
        print(f"  Difference: {abs(holonomy - K[v]):.6f}")

    # Now let's see what the parallel transport SHOULD be
    # to satisfy d1d * para_trans = K
    print("\n" + "="*60)
    print("Computing para_trans that satisfies d1d * para_trans = K")
    print("="*60)

    # For each edge, para_trans is the angle change when crossing
    # We need to define it consistently with the d1d operator

    # Let's use a simpler definition:
    # para_trans[e] = angle(T1[f1] w.r.t. T1[f0] transported across edge)

    para_trans_simple = compute_para_trans_simple(mesh, N, T1)

    # Check d1d * para_trans
    d1d_pt = np.zeros(mesh.n_vertices, dtype=np.float64)
    for e in range(mesh.n_edges):
        v1, v2 = mesh.edge_vertices[e]
        d1d_pt[v1] += para_trans_simple[e]
        d1d_pt[v2] -= para_trans_simple[e]

    print(f"\nd1d * para_trans (simple):")
    print(f"  Sum: {d1d_pt.sum():.6f}")

    diff = np.abs(np.array([_wrap_to_pi(d) for d in (d1d_pt - K)]))
    diff = np.minimum(diff, 2*np.pi - diff)
    print(f"  Max |d1d*pt - K|: {diff.max():.6f}")
    print(f"  Mean |d1d*pt - K|: {diff.mean():.6f}")


def compute_holonomy_around_vertex(mesh, v, N, T1, T2, alpha):
    """
    Compute the holonomy (total rotation) when going around vertex v.

    Go around the vertex CCW, tracking the angle of T1 relative to a fixed
    reference frame. The total rotation should equal the angle defect.
    """
    # Get corners around vertex v
    corners = mesh.vertex_corners(v)
    if len(corners) == 0:
        return 0.0

    # Track total angle traversed
    total_angle = 0.0

    for i, corner in enumerate(corners):
        face = corner // 3
        local = corner % 3

        # Interior angle at this corner
        angle_at_corner = alpha[corner]

        # Add the exterior angle (pi - interior angle)
        # This is the rotation when going from one edge to the next
        total_angle += np.pi - angle_at_corner

    # The holonomy is 2π minus the total exterior angle traversed
    # (For a flat surface, total exterior angle = 2π, so holonomy = 0)
    holonomy = 2 * np.pi - total_angle

    return holonomy


def compute_para_trans_simple(mesh, N, T1):
    """
    Simple parallel transport: angle change when crossing edge.

    For edge e between faces f0 and f1:
    para_trans[e] = angle(T1[f1] - parallel_transported(T1[f0]))
    """
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Parallel transport T1[f0] to f1's tangent plane
        # Using rotation that aligns normals
        T1_transported = parallel_transport_vector(T1[f0], N[f0], N[f1])

        # Project onto f1's tangent plane (should already be there)
        T1_transported = T1_transported - np.dot(T1_transported, N[f1]) * N[f1]
        norm = np.linalg.norm(T1_transported)
        if norm > 1e-10:
            T1_transported = T1_transported / norm
        else:
            T1_transported = T1[f1]

        # Angle from transported T1[f0] to T1[f1]
        para_trans[e] = _signed_angle(T1_transported, T1[f1], N[f1])

    return para_trans


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

    # Rodrigues' rotation formula
    v_rot = (v * cos_angle +
             np.cross(axis, v) * sin_angle +
             axis * np.dot(axis, v) * (1 - cos_angle))

    return v_rot


if __name__ == "__main__":
    debug()
