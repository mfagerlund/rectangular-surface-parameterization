"""
Cut graph computation following MATLAB's cut_mesh.m algorithm exactly.

Key differences from our original:
1. Pre-compute k21 for all edges BEFORE building spanning tree
2. Pre-mark non-identity edges as "visited" (can't be used for spanning tree)
3. BFS builds partial spanning tree using only identity edges
4. Cut = all edges NOT in spanning tree (includes all non-identity edges)
5. Prune leaves from cut (but non-identity edges STAY in cut)
"""

import numpy as np
from collections import deque
from typing import Tuple

from mesh import TriangleMesh
from geometry import compute_corner_angles
from cross_field import compute_parallel_transport_angles, compute_xi_per_halfedge
from compute_k21 import compute_omega, compute_k21


def wrap_angle(x: float) -> float:
    """Wrap angle to [-π, π] range."""
    return np.arctan2(np.sin(x), np.cos(x))


def compute_cut_jump_data_matlab_style(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    xi: np.ndarray,
    singularities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-style cut graph computation.

    1. Pre-compute k21 (rotation index) for all edges
    2. Build spanning tree using ONLY identity edges (k21=1)
    3. Cut = edges NOT in spanning tree
    4. Prune leaves (non-cone degree-1 vertices)

    Args:
        mesh: Triangle mesh
        alpha: |C| corner angles
        xi: |F| cross field angles
        singularities: |V| cone indices from cross-field

    Returns:
        Gamma: |E| cut edge indicator {0, 1}
        zeta: |E| quarter-rotation jump
        s: |C| sign bits {-1, +1}
        phi: |H| reference frame angle per halfedge
        omega0: |E| reference frame rotation across edge
    """
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges
    n_halfedges = mesh.n_halfedges
    n_corners = mesh.n_corners
    n_vertices = mesh.n_vertices

    # Step 1: Pre-compute k21 for all edges (MATLAB's reduction_from_ff2d.m)
    para_trans = compute_parallel_transport_angles(mesh)
    omega = compute_omega(mesh, xi, para_trans)
    k21 = compute_k21(mesh, xi, omega, para_trans)

    print(f"  k21=1 (identity): {np.sum(k21 == 1)} edges")
    print(f"  k21≠1 (non-identity): {np.sum(k21 != 1)} edges")

    # Convert k21 to edge_jump_tag (True for non-identity edges)
    edge_jump_tag = (k21 != 1)

    # Initialize outputs
    Gamma = np.ones(n_edges, dtype=np.int32)  # 1 = in cut (will be determined below)
    zeta = np.zeros(n_edges, dtype=np.float64)
    s = np.ones(n_corners, dtype=np.int32)
    phi = np.full(n_halfedges, np.inf, dtype=np.float64)
    omega0 = np.zeros(n_edges, dtype=np.float64)

    # Set zeta from k21 (k21-1 gives n_star in {0,1,2,3})
    for e in range(n_edges):
        n_star = k21[e] - 1  # Convert from 1-indexed to 0-indexed
        zeta[e] = (np.pi / 2) * n_star

    # Step 2: MATLAB's cut_mesh BFS (lines 7-31)
    # Pre-mark non-identity edges as "visited" so they can't be used
    visited_edge = edge_jump_tag.copy()  # True = can't use for spanning tree

    # Track which face each face was reached from
    tri_pred = -np.ones(n_faces, dtype=np.int32)

    # BFS queue starts from face 0
    seed_face = 0
    tri_pred[seed_face] = seed_face  # Face 0 is its own predecessor (root)

    queue = deque([seed_face])

    while queue:
        f = queue.popleft()

        # Get adjacent faces and shared edges
        for local in range(3):
            he = 3 * f + local
            he_twin = mesh.halfedge_twin[he]

            if he_twin == -1:
                continue  # boundary

            f_neighbor = he_twin // 3
            e = mesh.halfedge_to_edge[he]

            # MATLAB condition: (tri_pred(adj(i)) == -1) && ~visited_edge(adjedge(i))
            if tri_pred[f_neighbor] == -1 and not visited_edge[e]:
                tri_pred[f_neighbor] = f
                visited_edge[e] = True
                queue.append(f_neighbor)

    # Count faces reached
    n_reached = np.sum(tri_pred != -1)
    print(f"  Faces reached by BFS: {n_reached} / {n_faces}")

    # Step 3: Determine cut edges (MATLAB lines 33-35)
    # Reset edge_jump_tag edges' visited status
    visited_edge[edge_jump_tag] = False
    # Cut = edges NOT visited by BFS
    Gamma = (~visited_edge).astype(np.int32)

    print(f"  Initial cut edges: {np.sum(Gamma)}")

    # Step 4: Compute phi using a separate BFS that visits ALL faces
    # (MATLAB computes phi separately, we need it for constraints)
    xi_he = compute_xi_per_halfedge(mesh, xi, alpha)
    s_edge = np.ones(n_edges, dtype=np.int32)

    phi[3 * seed_face + 0] = wrap_angle(xi[seed_face])
    c1 = 3 * seed_face + 1
    c2 = 3 * seed_face + 2
    phi[3 * seed_face + 1] = wrap_angle(phi[3 * seed_face + 0] - (np.pi - alpha[c1]))
    phi[3 * seed_face + 2] = wrap_angle(phi[3 * seed_face + 1] - (np.pi - alpha[c2]))

    queue = deque()
    for local in range(3):
        queue.append(3 * seed_face + local)

    while queue:
        he_ij = queue.popleft()
        he_twin = mesh.halfedge_twin[he_ij]
        if he_twin == -1:
            continue

        e = mesh.halfedge_to_edge[he_ij]

        # Compute omega0 based on phi difference
        phi_ij_to_ji = wrap_angle(phi[he_ij] + np.pi)
        angle_diff = wrap_angle(phi_ij_to_ji - xi_he[he_twin])
        n_star = int(np.round(2 * angle_diff / np.pi)) % 4

        xi_star = wrap_angle((np.pi / 2) * n_star + xi_he[he_twin])
        omega0_raw = phi[he_ij] - xi_star + np.pi
        omega0[e] = wrap_angle(omega0_raw)

        he0 = mesh.edge_to_halfedge[e, 0]
        if he_ij != he0:
            omega0[e] = -omega0[e]

        # Set phi for neighbor face if not yet set
        if phi[he_twin] == np.inf:
            phi[he_twin] = xi_star

            face_to = he_twin // 3
            local_to = he_twin % 3

            he_next = 3 * face_to + (local_to + 1) % 3
            he_prev = 3 * face_to + (local_to + 2) % 3

            c_i = 3 * face_to + (local_to + 1) % 3
            c_l = 3 * face_to + (local_to + 2) % 3

            phi[he_next] = wrap_angle(phi[he_twin] - (np.pi - alpha[c_i]))
            phi[he_prev] = wrap_angle(phi[he_next] - (np.pi - alpha[c_l]))

            queue.append(he_next)
            queue.append(he_prev)

            # Compute s_edge based on n_star
            if n_star % 2 == 1:
                s_edge[e] = -1
            else:
                s_edge[e] = +1
        else:
            if n_star % 2 == 1:
                s_edge[e] = -1
            else:
                s_edge[e] = +1

    # Compute corner signs
    for v in range(n_vertices):
        corners = mesh.vertex_corners(v)
        if len(corners) == 0:
            continue
        S = 1
        for corner in corners:
            s[corner] = S
            he = corner
            e = mesh.halfedge_to_edge[he]
            S = s_edge[e] * S

    # Step 5: Prune leaves (MATLAB lines 36-45)
    # BUT: non-identity edges (edge_jump_tag) MUST stay in cut
    is_cone = np.abs(singularities) > 0.1

    pruning = True
    while pruning:
        d = np.zeros(n_vertices, dtype=np.int32)
        for e in range(n_edges):
            if Gamma[e] == 1:
                i, j = mesh.edge_vertices[e]
                d[i] += 1
                d[j] += 1

        pruning = False
        for v in range(n_vertices):
            if d[v] == 1 and not is_cone[v]:
                # Find the incident cut edge
                for e in range(n_edges):
                    if Gamma[e] == 1:
                        i, j = mesh.edge_vertices[e]
                        if i == v or j == v:
                            # MATLAB doesn't prune non-identity edges
                            # Check if this edge is non-identity
                            if not edge_jump_tag[e]:
                                # Identity edge can be pruned
                                Gamma[e] = 0
                                pruning = True
                            break

    print(f"  After pruning: {np.sum(Gamma)} cut edges")

    return Gamma, zeta, s, phi, omega0


def test_matlab_style():
    """Test MATLAB-style cut graph on sphere."""
    from io_obj import load_obj
    from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
    from optimization import solve_constraints_only
    from uv_recovery import recover_parameterization, compute_uv_quality, normalize_uvs

    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    print(f"Singularities: {np.sum(is_singular)} (sum: {cone_indices.sum():.2f})")

    print("\nComputing MATLAB-style cut graph...")
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data_matlab_style(
        mesh, alpha, xi, cone_indices
    )

    n_cut = np.sum(Gamma)
    n_non_cut = mesh.n_edges - n_cut
    print(f"\nFinal cut graph:")
    print(f"  Cut edges: {n_cut}")
    print(f"  Non-cut edges: {n_non_cut}")
    print(f"  Expected non-cut for spanning tree: {mesh.n_faces - 1}")

    # Try solving optimization
    print("\nSolving optimization...")
    try:
        u, v, theta = solve_constraints_only(mesh, alpha, Gamma, zeta, s, phi, omega0, verbose=False)

        # Recover UV
        from geometry import compute_edge_lengths
        ell = compute_edge_lengths(mesh)
        corner_uvs = recover_parameterization(mesh, Gamma, zeta, ell, alpha, phi, theta, s, u, v)

        # Compute quality
        quality = compute_uv_quality(mesh, corner_uvs)
        print(f"\nUV Quality:")
        print(f"  Flipped triangles: {quality['n_flipped']} / {mesh.n_faces}")
        print(f"  Mean angle error: {quality['mean_angle_error']:.2f}°")

    except Exception as ex:
        print(f"Error: {ex}")


if __name__ == "__main__":
    test_matlab_style()
