"""
Debug parallel transport computation.

MATLAB computes:
  edge_angles(e,1) = comp_angle(edge, e1r(f1), normal(f1))  # angle of edge in face 1's basis
  edge_angles(e,2) = comp_angle(edge, e1r(f2), normal(f2))  # angle of edge in face 2's basis
  para_trans = wrap(edge_angles(:,1) - edge_angles(:,2))

Then verifies: d1d * para_trans = K (Gaussian curvature)

The key insight: parallel transport measures how much the reference frame (T1) rotates
when transported across an edge. This rotation should sum to the angle defect (K)
around each vertex.
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


def debug_parallel_transport(mesh_path: str):
    """Debug parallel transport computation."""
    print("=" * 70)
    print("Parallel Transport Debug")
    print("=" * 70)

    mesh = load_obj(mesh_path)
    print(f"\nMesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces, {mesh.n_edges} edges")

    N, T1, T2 = compute_all_face_bases(mesh)
    alpha = compute_corner_angles(mesh)

    # Compute angle defect K
    K = np.zeros(mesh.n_vertices, dtype=np.float64)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    print(f"\nAngle defect K:")
    print(f"  Sum: {K.sum():.6f} = 2πχ = {2*np.pi*2:.6f}")

    # Method 1: Current Python implementation
    print("\n--- Method 1: Current Python (edge from he0) ---")
    para_trans_v1 = compute_para_trans_v1(mesh, N, T1)
    check_d1d_K(mesh, para_trans_v1, K, "v1")

    # Method 2: Use edge direction from sorted vertices (consistent orientation)
    print("\n--- Method 2: Edge from sorted vertices ---")
    para_trans_v2 = compute_para_trans_v2(mesh, N, T1)
    check_d1d_K(mesh, para_trans_v2, K, "v2")

    # Method 3: Use MATLAB-style edge orientation (from E2V)
    print("\n--- Method 3: Consistent with face winding ---")
    para_trans_v3 = compute_para_trans_v3(mesh, N, T1, alpha)
    check_d1d_K(mesh, para_trans_v3, K, "v3")

    return para_trans_v1, para_trans_v2, para_trans_v3, K


def compute_para_trans_v1(mesh, N, T1):
    """Current Python implementation: edge from halfedge he0."""
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Edge direction from he0
        i0, j0 = mesh.halfedge_vertices(he0)
        edge_vec = mesh.positions[j0] - mesh.positions[i0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Signed angle from edge to T1 in each face
        angle0 = _signed_angle(edge_vec, T1[f0], N[f0])
        angle1 = _signed_angle(edge_vec, T1[f1], N[f1])

        para_trans[e] = _wrap_to_pi(angle0 - angle1)

    return para_trans


def compute_para_trans_v2(mesh, N, T1):
    """Edge from sorted vertex indices (consistent global orientation)."""
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Edge direction from sorted vertices
        vi, vj = mesh.edge_vertices[e]
        if vi > vj:
            vi, vj = vj, vi
        edge_vec = mesh.positions[vj] - mesh.positions[vi]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        angle0 = _signed_angle(edge_vec, T1[f0], N[f0])
        angle1 = _signed_angle(edge_vec, T1[f1], N[f1])

        para_trans[e] = _wrap_to_pi(angle0 - angle1)

    return para_trans


def compute_para_trans_v3(mesh, N, T1, alpha):
    """
    Compute parallel transport by summing interior angles.

    The parallel transport around a vertex should equal the angle defect K.
    We compute this by tracking how the angle to T1 changes as we go around edges.
    """
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Use the edge direction as seen from face f0 (going CCW around the face)
        local0 = he0 % 3
        local1 = he1 % 3

        # In face f0, halfedge he0 goes from vertex at local0 to vertex at (local0+1)%3
        v_from = mesh.faces[f0, local0]
        v_to = mesh.faces[f0, (local0 + 1) % 3]
        edge_vec = mesh.positions[v_to] - mesh.positions[v_from]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Angle from edge to T1 in f0
        angle0 = _signed_angle(edge_vec, T1[f0], N[f0])

        # In face f1, the edge goes the opposite direction
        # Twin halfedge he1 goes from vertex at local1 to (local1+1)%3 in f1
        # But this is the opposite direction of the edge as seen from f0
        # So we use -edge_vec for consistency
        angle1 = _signed_angle(-edge_vec, T1[f1], N[f1])

        para_trans[e] = _wrap_to_pi(angle0 - angle1)

    return para_trans


def check_d1d_K(mesh, para_trans, K, label):
    """Check if d1d * para_trans = K."""
    d1d_pt = np.zeros(mesh.n_vertices, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        if he0 == -1:
            continue
        i, j = mesh.halfedge_vertices(he0)
        d1d_pt[i] += para_trans[e]
        d1d_pt[j] -= para_trans[e]

    # Wrap both before comparing
    d1d_pt_wrapped = np.array([_wrap_to_pi(x) for x in d1d_pt])
    K_wrapped = np.array([_wrap_to_pi(x) for x in K])

    diff = np.abs(d1d_pt_wrapped - K_wrapped)
    # Handle wraparound at pi/-pi boundary
    diff = np.minimum(diff, 2*np.pi - diff)

    print(f"  [{label}] d1d*para_trans sum: {d1d_pt.sum():.6f}")
    print(f"  [{label}] K sum: {K.sum():.6f}")
    print(f"  [{label}] Max |d1d*pt - K| (wrapped): {diff.max():.6f}")
    print(f"  [{label}] Mean |d1d*pt - K| (wrapped): {diff.mean():.6f}")

    # Show worst vertices
    if diff.max() > 0.01:
        worst_idx = np.argsort(diff)[-5:]
        print(f"  [{label}] Worst vertices:")
        for v in worst_idx:
            print(f"    v={v}: d1d_pt={d1d_pt[v]:.4f}, K={K[v]:.4f}, diff={diff[v]:.4f}")


if __name__ == "__main__":
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    debug_parallel_transport(mesh_path)
