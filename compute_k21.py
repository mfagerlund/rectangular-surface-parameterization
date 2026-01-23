"""
Compute k21 (rotation jump index) MATLAB-style.

MATLAB formula (reduction_from_ff2d.m line 5):
[~,k21i] = min(abs(
    exp(1i*ang(E2T(:,2)) + (0:3)*1i*pi/2 + 1i*(omega - para_trans))
    - exp(1i*ang(E2T(:,1)))
), [], 2);

This finds k21 ∈ {1,2,3,4} that minimizes cross-field misalignment.
k21=1 means identity (no rotation needed).
"""

import numpy as np
from mesh import TriangleMesh
from geometry import compute_corner_angles
from cross_field import compute_parallel_transport_angles


def _wrap_to_pi(x):
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(x), np.cos(x))


def compute_omega(mesh, xi, para_trans):
    """
    Compute omega (connection form) per edge.

    MATLAB: omega = wrapToPi(4*(d0d*ang + para_trans))/4

    d0d*ang = ang[f1] - ang[f0] for edge with faces f0, f1

    Args:
        mesh: Triangle mesh
        xi: |F| cross-field angles per face
        para_trans: |E| parallel transport angles

    Returns:
        omega: |E| connection form per edge
    """
    n_edges = mesh.n_edges
    omega = np.zeros(n_edges, dtype=np.float64)

    for e in range(n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue  # boundary edge

        f0 = he0 // 3
        f1 = he1 // 3

        # d0d * xi = xi[f1] - xi[f0] (for edge_vertices convention)
        # Note: MATLAB uses (f1, f0) order based on E2T convention
        # We use edge_to_halfedge[e, 0] -> f0, edge_to_halfedge[e, 1] -> f1
        d0d_xi = xi[f1] - xi[f0]

        # omega = wrap_4(d0d*xi + para_trans)
        # wrap_4(x) = wrapToPi(4*x)/4
        raw = d0d_xi + para_trans[e]
        omega[e] = _wrap_to_pi(4 * raw) / 4

    return omega


def compute_k21(mesh, xi, omega, para_trans):
    """
    Compute k21 (rotation jump index) per edge, MATLAB-style.

    MATLAB: [~,k21i] = min(abs(
        exp(1i*ang(E2T(:,2)) + (0:3)*1i*pi/2 + 1i*(omega - para_trans))
        - exp(1i*ang(E2T(:,1)))
    ), [], 2);

    Returns k21 ∈ {1,2,3,4} where:
        1 = identity (no rotation)
        2 = 90° rotation
        3 = 180° rotation
        4 = 270° rotation

    Args:
        mesh: Triangle mesh
        xi: |F| cross-field angles per face
        omega: |E| connection form per edge
        para_trans: |E| parallel transport angles

    Returns:
        k21: |E| rotation index (1-indexed like MATLAB)
    """
    n_edges = mesh.n_edges
    k21 = np.ones(n_edges, dtype=np.int32)  # Default to 1 (identity)

    for e in range(n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue  # boundary edge

        f0 = he0 // 3  # face at he0
        f1 = he1 // 3  # face at he1

        # MATLAB uses E2T(e,1) and E2T(e,2) for the two faces
        # Our convention: he0 -> f0, he1 -> f1
        ang1 = xi[f0]  # E2T(:,1) in MATLAB
        ang2 = xi[f1]  # E2T(:,2) in MATLAB

        # Find k that minimizes |exp(i*(ang2 + k*π/2 + omega - para_trans)) - exp(i*ang1)|
        best_k = 1
        best_diff = float('inf')

        for k in range(4):  # k = 0,1,2,3 (will add 1 for MATLAB indexing)
            rotated_ang = ang2 + k * np.pi / 2 + omega[e] - para_trans[e]
            diff = abs(np.exp(1j * rotated_ang) - np.exp(1j * ang1))
            if diff < best_diff:
                best_diff = diff
                best_k = k + 1  # MATLAB is 1-indexed

        k21[e] = best_k

    return k21


def test_k21():
    """Test k21 computation on sphere mesh."""
    from io_obj import load_obj
    from cross_field import compute_smooth_cross_field, compute_cross_field_singularities

    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    # Compute cross-field
    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)

    # Compute parallel transport
    para_trans = compute_parallel_transport_angles(mesh)

    # Compute omega
    omega = compute_omega(mesh, xi, para_trans)

    # Compute k21
    k21 = compute_k21(mesh, xi, omega, para_trans)

    print(f"\nk21 distribution (MATLAB-style, 1-indexed):")
    print(f"  k21=1 (identity): {np.sum(k21 == 1)} edges")
    print(f"  k21=2 (90°):      {np.sum(k21 == 2)} edges")
    print(f"  k21=3 (180°):     {np.sum(k21 == 3)} edges")
    print(f"  k21=4 (270°):     {np.sum(k21 == 4)} edges")

    n_identity = np.sum(k21 == 1)
    n_non_identity = np.sum(k21 != 1)
    print(f"\nIdentity edges: {n_identity}")
    print(f"Non-identity edges: {n_non_identity} (these MUST be in cut)")

    # For MATLAB-style spanning tree, we need to use only identity edges
    # Check if identity edges can form a spanning tree
    needed_for_spanning = mesh.n_faces - 1
    print(f"\nEdges needed for spanning tree: {needed_for_spanning}")
    if n_identity >= needed_for_spanning:
        print("FEASIBLE: Enough identity edges for spanning tree")
    else:
        print(f"NOT FEASIBLE: Need {needed_for_spanning - n_identity} more identity edges")
        print("Will need to allow some non-identity edges in spanning tree")

    return k21, omega, para_trans


if __name__ == "__main__":
    test_k21()
