"""
Debug script to trace cross-field computation and compare with MATLAB formulas.

MATLAB formulas:
- omega = wrapToPi(4*(d0d*ang + para_trans))/4
- sing = (d1d*(para_trans - omega) + Kt_invisible) / (2*pi)
- Since d1d*para_trans = K and Kt_invisible ≈ 0: sing = (K - d1d*omega) / (2*pi)

Python formulas:
- omega0[e] = wrap(4*(xi[f1] - xi[f0] + para_trans[e])) / 4
- cone_index = (K + d1d*omega0) / (2*pi)  <-- potential sign issue!
"""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import (
    compute_smooth_cross_field,
    compute_parallel_transport_angles,
    compute_edge_face_map,
    compute_cross_field_singularities,
    _wrap_to_pi
)
from mesh import euler_characteristic


def debug_cross_field(mesh_path: str):
    """Run cross-field computation with detailed diagnostics."""
    print("=" * 70)
    print("Cross-Field Diagnostic")
    print("=" * 70)

    # Load mesh
    mesh = load_obj(mesh_path)
    print(f"\nMesh: {mesh_path}")
    print(f"  Vertices: {mesh.n_vertices}")
    print(f"  Faces: {mesh.n_faces}")
    print(f"  Edges: {mesh.n_edges}")
    chi = euler_characteristic(mesh)
    print(f"  Euler characteristic χ: {chi}")

    # Compute corner angles
    alpha = compute_corner_angles(mesh)

    # Compute angle defect (Gaussian curvature)
    K = np.zeros(mesh.n_vertices, dtype=np.float64)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    print(f"\n[1] Angle defect K (Gaussian curvature):")
    print(f"    Sum: {K.sum():.6f} (expected: 2πχ = {2*np.pi*chi:.6f})")
    print(f"    Error: {abs(K.sum() - 2*np.pi*chi):.2e}")

    # Compute parallel transport
    para_trans = compute_parallel_transport_angles(mesh)

    # Build d1d operator and verify d1d*para_trans = K
    d1d_para_trans = np.zeros(mesh.n_vertices, dtype=np.float64)
    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 == -1 or he1 == -1:
            continue
        i, j = mesh.halfedge_vertices(he0)
        d1d_para_trans[i] += para_trans[e]
        d1d_para_trans[j] -= para_trans[e]

    print(f"\n[2] Parallel transport verification:")
    print(f"    d1d*para_trans sum: {d1d_para_trans.sum():.6f}")
    print(f"    K sum: {K.sum():.6f}")
    error_d1d = np.abs(_wrap_to_pi(d1d_para_trans) - _wrap_to_pi(K)).max()
    print(f"    Max |d1d*para_trans - K| (wrapped): {error_d1d:.6f}")

    # Compute smooth cross field
    print(f"\n[3] Computing smooth cross field...")
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=True)

    # Compute omega (field rotation) per edge - MATLAB style
    E2T = compute_edge_face_map(mesh)
    omega_matlab = np.zeros(mesh.n_edges, dtype=np.float64)
    for e in range(mesh.n_edges):
        if E2T[e, 0] < 0 or E2T[e, 1] < 0:
            continue
        f0, f1 = E2T[e, 0], E2T[e, 1]
        # d0d*ang gives ang[f1] - ang[f0] (or opposite depending on convention)
        # MATLAB: omega = wrap(4*(d0d*ang + para_trans))/4
        # Let's try: d0d*ang[e] = xi[f1] - xi[f0]
        angle_change = xi[f1] - xi[f0] + para_trans[e]
        omega_matlab[e] = _wrap_to_pi(4 * angle_change) / 4

    # Compute singularities MATLAB style: sing = (K - d1d*omega) / (2*pi)
    d1d_omega = np.zeros(mesh.n_vertices, dtype=np.float64)
    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 == -1 or he1 == -1:
            continue
        i, j = mesh.halfedge_vertices(he0)
        d1d_omega[i] += omega_matlab[e]
        d1d_omega[j] -= omega_matlab[e]

    sing_matlab = (K - d1d_omega) / (2 * np.pi)

    print(f"\n[4] Singularities (MATLAB style: (K - d1d*omega) / 2π):")
    print(f"    Sum of indices: {sing_matlab.sum():.4f} (expected: χ = {chi})")

    # Round and count
    sing_rounded = np.round(sing_matlab * 4) / 4
    is_singular_matlab = np.abs(sing_rounded) > 0.01
    n_sing_matlab = np.sum(is_singular_matlab)
    print(f"    Singular vertices: {n_sing_matlab}")
    print(f"    Unique index values: {np.unique(sing_rounded[is_singular_matlab])}")

    # Now compute Python style: (K + d1d*omega0) / (2*pi)
    sing_python = (K + d1d_omega) / (2 * np.pi)

    print(f"\n[5] Singularities (Python style: (K + d1d*omega) / 2π):")
    print(f"    Sum of indices: {sing_python.sum():.4f} (expected: χ = {chi})")

    sing_rounded_py = np.round(sing_python * 4) / 4
    is_singular_py = np.abs(sing_rounded_py) > 0.01
    n_sing_py = np.sum(is_singular_py)
    print(f"    Singular vertices: {n_sing_py}")
    print(f"    Unique index values: {np.unique(sing_rounded_py[is_singular_py])}")

    # Compare with current implementation
    print(f"\n[6] Current implementation (compute_cross_field_singularities):")
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    n_sing_current = np.sum(is_singular)
    print(f"    Sum of indices: {cone_indices.sum():.4f}")
    print(f"    Singular vertices: {n_sing_current}")
    print(f"    Unique index values: {np.unique(cone_indices[is_singular])}")

    # Detailed comparison
    print(f"\n[7] Index comparison at singular vertices (MATLAB style):")
    if n_sing_matlab > 0:
        sing_verts = np.where(is_singular_matlab)[0][:10]
        for v in sing_verts:
            print(f"    v={v}: K={K[v]:.4f}, d1d_omega={d1d_omega[v]:.4f}, "
                  f"sing_matlab={sing_rounded[v]:.3f}, sing_python={sing_rounded_py[v]:.3f}")

    return {
        'chi': chi,
        'K_sum': K.sum(),
        'sing_matlab_sum': sing_matlab.sum(),
        'sing_python_sum': sing_python.sum(),
        'n_singular_matlab': n_sing_matlab,
        'n_singular_python': n_sing_py,
    }


if __name__ == "__main__":
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    debug_cross_field(mesh_path)
