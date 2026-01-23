"""Debug k21 computation - check sign conventions."""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_parallel_transport_angles


def _wrap_to_pi(x):
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(x), np.cos(x))


def compute_k21_variants(mesh, xi, para_trans):
    """
    Try different sign conventions for k21 computation.
    """
    n_edges = mesh.n_edges

    results = {}

    for omega_sign in [1, -1]:
        for para_sign in [1, -1]:
            for diff_sign in [1, -1]:
                label = f"omega:{'+' if omega_sign > 0 else '-'}, pt:{'+' if para_sign > 0 else '-'}, diff:{'+' if diff_sign > 0 else '-'}"

                k21 = np.ones(n_edges, dtype=np.int32)

                for e in range(n_edges):
                    he0 = mesh.edge_to_halfedge[e, 0]
                    he1 = mesh.edge_to_halfedge[e, 1]

                    if he0 == -1 or he1 == -1:
                        continue

                    f0 = he0 // 3
                    f1 = he1 // 3

                    # d0d * xi
                    d0d_xi = xi[f1] - xi[f0]

                    # omega
                    omega = _wrap_to_pi(4 * (d0d_xi + omega_sign * para_trans[e])) / 4

                    # k21 computation
                    ang1 = xi[f0]
                    ang2 = xi[f1]

                    best_k = 1
                    best_diff = float('inf')

                    for k in range(4):
                        rotated = ang2 + k * np.pi / 2 + diff_sign * (omega - para_sign * para_trans[e])
                        diff = abs(np.exp(1j * rotated) - np.exp(1j * ang1))
                        if diff < best_diff:
                            best_diff = diff
                            best_k = k + 1

                    k21[e] = best_k

                n_identity = np.sum(k21 == 1)
                results[label] = n_identity

    return results


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    para_trans = compute_parallel_transport_angles(mesh)

    print("\nTrying different sign conventions:")
    results = compute_k21_variants(mesh, xi, para_trans)

    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for label, n_identity in sorted_results:
        print(f"  {label}: {n_identity} identity edges")

    # Best result
    best_label, best_n = sorted_results[0]
    print(f"\nBest: {best_label} with {best_n} identity edges")


if __name__ == "__main__":
    main()
