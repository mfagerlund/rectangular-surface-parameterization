"""
Debug parallel transport v6 - match E2T ordering to edge_vertices.

The key insight: d1d uses edge_vertices orientation.
- edge_vertices[e] = (v1, v2) where v1 < v2
- d1d adds para_trans[e] to v1, subtracts from v2

For consistency, when computing para_trans:
- E2T[e, 0] should be the face where edge goes from v1 to v2 (CCW direction)
- E2T[e, 1] should be the face where edge goes from v2 to v1 (CW direction)
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


def compute_para_trans_fixed(mesh, N, T1):
    """
    Compute parallel transport with proper E2T ordering.

    For edge e with vertices (v1, v2) where v1 < v2:
    - Find face f_pos where edge goes v1 -> v2 (positive direction)
    - Find face f_neg where edge goes v2 -> v1 (negative direction)
    - para_trans[e] = angle(edge, T1[f_pos]) - angle(edge, T1[f_neg])
    """
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        # Get edge vertices from the canonical ordering
        v1, v2 = mesh.edge_vertices[e]  # v1 < v2

        # Edge direction (positive = v1 -> v2)
        edge_vec = mesh.positions[v2] - mesh.positions[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Determine which halfedge goes in the positive direction
        i0, j0 = mesh.halfedge_vertices(he0)

        if i0 == v1 and j0 == v2:
            # he0 goes in positive direction
            f_pos = he0 // 3
            f_neg = he1 // 3
        else:
            # he1 goes in positive direction
            f_pos = he1 // 3
            f_neg = he0 // 3

        # Compute angle of edge in each face's basis
        angle_pos = _signed_angle(edge_vec, T1[f_pos], N[f_pos])
        angle_neg = _signed_angle(edge_vec, T1[f_neg], N[f_neg])

        para_trans[e] = _wrap_to_pi(angle_pos - angle_neg)

    return para_trans


def compute_d1d(mesh, x):
    """Apply d1d operator using edge_vertices convention."""
    result = np.zeros(mesh.n_vertices, dtype=np.float64)

    for e in range(mesh.n_edges):
        v1, v2 = mesh.edge_vertices[e]  # v1 < v2
        result[v1] += x[e]
        result[v2] -= x[e]

    return result


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

    # Compute para_trans with fixed ordering
    para_trans = compute_para_trans_fixed(mesh, N, T1)

    # Apply d1d
    d1d_pt = compute_d1d(mesh, para_trans)

    print(f"\nFixed para_trans:")
    print(f"  d1d*para_trans sum: {d1d_pt.sum():.6f}")

    # Check difference
    diff_raw = d1d_pt - K
    diff_wrapped = np.array([_wrap_to_pi(d) for d in diff_raw])
    diff_final = np.minimum(np.abs(diff_wrapped), 2*np.pi - np.abs(diff_wrapped))

    print(f"  Max |d1d*pt - K| (wrapped): {diff_final.max():.6f}")
    print(f"  Mean |d1d*pt - K| (wrapped): {diff_final.mean():.6f}")

    # Sample vertices
    print("\nSample vertices:")
    for v in [0, 1, 10, 50, 100]:
        print(f"  v={v}: d1d_pt={d1d_pt[v]:.4f}, K={K[v]:.4f}, "
              f"diff={diff_final[v]:.4f} ({np.degrees(diff_final[v]):.1f}°)")

    # Check by valence
    print("\nBy valence:")
    valences = {}
    for v in range(mesh.n_vertices):
        val = len(mesh.vertex_corners(v))
        if val not in valences:
            valences[val] = []
        valences[val].append(v)

    for val, verts in sorted(valences.items()):
        diffs = [diff_final[v] for v in verts]
        print(f"  Valence {val}: {len(verts)} vertices, "
              f"max_diff={max(diffs):.4f}, mean={np.mean(diffs):.4f}")

    return para_trans, d1d_pt, K


if __name__ == "__main__":
    debug()
