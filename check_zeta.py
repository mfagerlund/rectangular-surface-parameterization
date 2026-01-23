"""Check zeta distribution and compare with MATLAB k21 computation."""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import (
    compute_smooth_cross_field,
    compute_cross_field_singularities,
    compute_xi_per_halfedge
)
from cut_graph import compute_cut_jump_data, wrap_angle


def compute_k21_matlab_style(mesh, xi, alpha):
    """
    Compute k21 like MATLAB does: directly from cross-field angles.

    MATLAB formula (reduction_from_ff2d.m line 5):
    [~,k21i] = min(abs(
        exp(1i*ang(E2T(:,2)) + (0:3)*1i*pi/2 + 1i*(omega - para_trans))
        - exp(1i*ang(E2T(:,1)))
    ), [], 2);

    This finds the rotation k ∈ {0,1,2,3} that best aligns cross-field across edge.
    """
    n_edges = mesh.n_edges
    xi_he = compute_xi_per_halfedge(mesh, xi, alpha)

    k21 = np.zeros(n_edges, dtype=np.int32)

    for e in range(n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue  # boundary edge

        # Cross-field angles at each side of edge
        xi0 = xi_he[he0]  # cross-field at face 0, relative to this halfedge
        xi1 = xi_he[he1]  # cross-field at face 1, relative to twin halfedge

        # The twin halfedge points opposite direction, so parallel transport adds π
        xi1_transported = wrap_angle(xi1 + np.pi)

        # Find k ∈ {0,1,2,3} minimizing |exp(i*xi1_transported + k*π/2) - exp(i*xi0)|
        best_k = 0
        best_diff = float('inf')

        for k in range(4):
            rotated = xi1_transported + k * np.pi / 2
            # Complex difference magnitude
            diff = abs(np.exp(1j * rotated) - np.exp(1j * xi0))
            if diff < best_diff:
                best_diff = diff
                best_k = k

        k21[e] = best_k

    return k21


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    # Compute cross-field
    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    # Current implementation: compute cut data (includes zeta)
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)

    # Convert zeta to n_star (0,1,2,3)
    n_star = np.round(2 * zeta / np.pi).astype(int) % 4

    print("\nCurrent zeta distribution (computed during BFS):")
    print(f"  n_star=0 (identity):  {np.sum(n_star == 0)} edges")
    print(f"  n_star=1 (90°):       {np.sum(n_star == 1)} edges")
    print(f"  n_star=2 (180°):      {np.sum(n_star == 2)} edges")
    print(f"  n_star=3 (270°):      {np.sum(n_star == 3)} edges")

    # MATLAB-style k21 computation
    k21 = compute_k21_matlab_style(mesh, xi, alpha)

    print("\nMATLAB-style k21 distribution (pre-computed from cross-field):")
    print(f"  k21=0 (identity):     {np.sum(k21 == 0)} edges")
    print(f"  k21=1 (90°):          {np.sum(k21 == 1)} edges")
    print(f"  k21=2 (180°):         {np.sum(k21 == 2)} edges")
    print(f"  k21=3 (270°):         {np.sum(k21 == 3)} edges")

    # Check if k21=0 edges can form a spanning tree
    n_identity = np.sum(k21 == 0)
    needed_for_spanning = mesh.n_faces - 1

    print(f"\nSpanning tree feasibility:")
    print(f"  Identity edges (k21=0): {n_identity}")
    print(f"  Needed for spanning tree: {needed_for_spanning}")
    if n_identity >= needed_for_spanning:
        print(f"  FEASIBLE: Can build spanning tree with identity edges only")
    else:
        print(f"  NOT FEASIBLE: Need {needed_for_spanning - n_identity} more identity edges")

    # Compare n_star and k21
    match = np.sum(n_star == k21)
    print(f"\nComparison:")
    print(f"  Matching: {match} / {mesh.n_edges} edges ({100*match/mesh.n_edges:.1f}%)")

    # Check cut edges
    n_cut = np.sum(Gamma)
    n_non_cut = mesh.n_edges - n_cut
    print(f"\nCurrent cut graph:")
    print(f"  Cut edges (Gamma=1): {n_cut}")
    print(f"  Non-cut edges (Gamma=0): {n_non_cut}")
    print(f"  Expected non-cut for spanning tree: {needed_for_spanning}")


if __name__ == "__main__":
    main()
