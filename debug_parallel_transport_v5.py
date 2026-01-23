"""
Debug parallel transport v5 - focus on making d1d * para_trans = K work.

The sum of holonomies is exactly negative of K sum, suggesting a sign issue.
Let's systematically try different sign conventions.
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
        return v.copy() if cos_angle > 0 else -v

    axis = axis / sin_angle
    v_rot = (v * cos_angle +
             np.cross(axis, v) * sin_angle +
             axis * np.dot(axis, v) * (1 - cos_angle))
    return v_rot


def compute_para_trans(mesh, N, T1, f0_first=True, negate_angle=False):
    """
    Compute parallel transport with configurable conventions.
    """
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        if f0_first:
            fa, fb = f0, f1
        else:
            fa, fb = f1, f0

        # Transport T1[fa] to fb's tangent plane
        T1_transported = parallel_transport_vector(T1[fa], N[fa], N[fb])
        T1_transported = T1_transported - np.dot(T1_transported, N[fb]) * N[fb]
        norm = np.linalg.norm(T1_transported)
        if norm > 1e-10:
            T1_transported = T1_transported / norm
        else:
            T1_transported = T1[fb]

        # Angle from transported T1[fa] to T1[fb]
        angle = _signed_angle(T1_transported, T1[fb], N[fb])
        para_trans[e] = -angle if negate_angle else angle

    return para_trans


def compute_d1d(mesh, x, negate=False):
    """Apply d1d operator with optional sign flip."""
    result = np.zeros(mesh.n_vertices, dtype=np.float64)

    for e in range(mesh.n_edges):
        v1, v2 = mesh.edge_vertices[e]
        if negate:
            result[v1] -= x[e]
            result[v2] += x[e]
        else:
            result[v1] += x[e]
            result[v2] -= x[e]

    return result


def check_config(mesh, N, T1, K, f0_first, negate_angle, negate_d1d, label):
    """Check a specific configuration."""
    para_trans = compute_para_trans(mesh, N, T1, f0_first, negate_angle)
    d1d_pt = compute_d1d(mesh, para_trans, negate_d1d)

    # Wrap difference
    diff_raw = d1d_pt - K
    diff_wrapped = np.array([_wrap_to_pi(d) for d in diff_raw])
    diff_wrapped = np.minimum(np.abs(diff_wrapped), 2*np.pi - np.abs(diff_wrapped))

    max_diff = diff_wrapped.max()
    mean_diff = diff_wrapped.mean()

    print(f"{label}: max={max_diff:.4f}, mean={mean_diff:.4f}, "
          f"sum_d1d_pt={d1d_pt.sum():.4f}, sum_K={K.sum():.4f}")

    return max_diff, d1d_pt, para_trans


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

    print(f"K sum: {K.sum():.6f}")

    print("\nTrying all sign combinations:")
    results = []

    for f0_first in [True, False]:
        for negate_angle in [True, False]:
            for negate_d1d in [True, False]:
                label = f"f0_first={f0_first}, neg_ang={negate_angle}, neg_d1d={negate_d1d}"
                max_diff, d1d_pt, pt = check_config(
                    mesh, N, T1, K, f0_first, negate_angle, negate_d1d, label
                )
                results.append((max_diff, label, d1d_pt, pt))

    # Find best
    results.sort(key=lambda x: x[0])
    print(f"\nBest configuration: {results[0][1]} (max_diff={results[0][0]:.4f})")

    # Show details for best
    best_d1d_pt = results[0][2]
    best_pt = results[0][3]

    print("\nSample vertex values for best config:")
    for v in [0, 1, 10, 50, 100]:
        diff_v = _wrap_to_pi(best_d1d_pt[v] - K[v])
        print(f"  v={v}: d1d_pt={best_d1d_pt[v]:.4f}, K={K[v]:.4f}, "
              f"diff_wrapped={diff_v:.4f} ({np.degrees(diff_v):.1f}°)")

    # Check if there's a pattern in the vertex valence
    print("\nGrouping vertices by valence:")
    valences = {}
    for v in range(mesh.n_vertices):
        corners = mesh.vertex_corners(v)
        val = len(corners)
        if val not in valences:
            valences[val] = []
        valences[val].append(v)

    for val, verts in sorted(valences.items()):
        diffs = [_wrap_to_pi(best_d1d_pt[v] - K[v]) for v in verts]
        print(f"  Valence {val}: {len(verts)} vertices, "
              f"mean diff={np.mean(diffs):.4f}, std={np.std(diffs):.4f}")


if __name__ == "__main__":
    debug()
