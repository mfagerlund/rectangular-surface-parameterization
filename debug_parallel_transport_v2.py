"""
Debug parallel transport - use MATLAB's edge orientation convention.

MATLAB:
  edge = Src.X(Src.E2V(:,2),:) - Src.X(Src.E2V(:,1),:);  # global edge direction
  edge_angles(e,1) = comp_angle(edge[e], e1r[f1], normal[f1])
  edge_angles(e,2) = comp_angle(edge[e], e1r[f2], normal[f2])
  para_trans = wrap(edge_angles(:,1) - edge_angles(:,2))

d1d operator:
  d1d[v, e] = +1 if v = E2V[e,1] (start)
  d1d[v, e] = -1 if v = E2V[e,2] (end)

Key insight: MATLAB's d1d uses E2V orientation, not halfedge orientation!
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


def compute_para_trans_matlab_style(mesh, N, T1):
    """
    Compute parallel transport using MATLAB's edge orientation convention.

    Uses edge_vertices (like E2V) for consistent global edge orientation.
    """
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Use MATLAB-style global edge direction: edge_vertices[e] = (v1, v2)
        # Edge goes from v1 to v2 consistently
        v1, v2 = mesh.edge_vertices[e]
        edge_vec = mesh.positions[v2] - mesh.positions[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Angle from edge to T1 in each face
        angle0 = _signed_angle(edge_vec, T1[f0], N[f0])
        angle1 = _signed_angle(edge_vec, T1[f1], N[f1])

        para_trans[e] = _wrap_to_pi(angle0 - angle1)

    return para_trans


def compute_d1d_matlab_style(mesh, x):
    """
    Apply d1d operator using MATLAB's E2V convention.

    d1d[v] = sum over edges of (+1 if v is start, -1 if v is end) * x[e]
    """
    result = np.zeros(mesh.n_vertices, dtype=np.float64)

    for e in range(mesh.n_edges):
        v1, v2 = mesh.edge_vertices[e]  # Consistent with E2V
        result[v1] += x[e]  # +1 for start
        result[v2] -= x[e]  # -1 for end

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

    # Compute parallel transport MATLAB style
    para_trans = compute_para_trans_matlab_style(mesh, N, T1)

    # Apply d1d MATLAB style
    d1d_pt = compute_d1d_matlab_style(mesh, para_trans)

    print(f"\nd1d*para_trans using MATLAB conventions:")
    print(f"  Sum: {d1d_pt.sum():.6f}")

    # Wrap and compare
    d1d_pt_wrapped = np.array([_wrap_to_pi(x) for x in d1d_pt])
    K_wrapped = np.array([_wrap_to_pi(x) for x in K])

    diff = np.abs(d1d_pt_wrapped - K_wrapped)
    diff = np.minimum(diff, 2*np.pi - diff)

    print(f"  Max |d1d*pt - K| (wrapped): {diff.max():.6f}")
    print(f"  Mean |d1d*pt - K| (wrapped): {diff.mean():.6f}")

    if diff.max() > 0.01:
        print(f"\n  Worst vertices:")
        worst_idx = np.argsort(diff)[-5:]
        for v in worst_idx:
            print(f"    v={v}: d1d_pt={d1d_pt[v]:.4f}, K={K[v]:.4f}, "
                  f"wrapped: d1d_pt={d1d_pt_wrapped[v]:.4f}, K={K_wrapped[v]:.4f}, diff={diff[v]:.4f}")

    # Now check which face ordering E2T uses
    print(f"\n\nChecking E2T (edge to face) ordering...")
    # In MATLAB: E2T[e, 1] and E2T[e, 2] are the two adjacent faces
    # The convention affects the sign of para_trans

    # Let's try: f0 is the face where edge goes CCW, f1 is where it goes CW
    # This matches MATLAB's T2E sign convention

    para_trans_v2 = compute_para_trans_oriented(mesh, N, T1)
    d1d_pt_v2 = compute_d1d_matlab_style(mesh, para_trans_v2)

    d1d_pt_v2_wrapped = np.array([_wrap_to_pi(x) for x in d1d_pt_v2])
    diff_v2 = np.abs(d1d_pt_v2_wrapped - K_wrapped)
    diff_v2 = np.minimum(diff_v2, 2*np.pi - diff_v2)

    print(f"\nOriented para_trans (match face winding):")
    print(f"  Max |d1d*pt - K| (wrapped): {diff_v2.max():.6f}")
    print(f"  Mean |d1d*pt - K| (wrapped): {diff_v2.mean():.6f}")


def compute_para_trans_oriented(mesh, N, T1):
    """
    Compute parallel transport with proper face ordering.

    Key insight: need to match E2T ordering with d1d convention.
    """
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Global edge direction
        v1, v2 = mesh.edge_vertices[e]
        edge_vec = mesh.positions[v2] - mesh.positions[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Check which face has the edge in CCW order (matching edge direction)
        # In f0, check if edge goes from v1 to v2 in CCW order
        local0 = he0 % 3
        va = mesh.faces[f0, local0]
        vb = mesh.faces[f0, (local0 + 1) % 3]

        if (va == v1 and vb == v2):
            # Edge goes CCW in f0 (same as edge_vertices direction)
            # f0 corresponds to E2T[e, 1] in MATLAB
            angle_f0 = _signed_angle(edge_vec, T1[f0], N[f0])
            angle_f1 = _signed_angle(edge_vec, T1[f1], N[f1])
        else:
            # Edge goes CW in f0 (opposite to edge_vertices direction)
            # f1 corresponds to E2T[e, 1] in MATLAB
            angle_f0 = _signed_angle(edge_vec, T1[f1], N[f1])
            angle_f1 = _signed_angle(edge_vec, T1[f0], N[f0])

        para_trans[e] = _wrap_to_pi(angle_f0 - angle_f1)

    return para_trans


if __name__ == "__main__":
    debug()
