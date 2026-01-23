"""Analyze where flipped triangles occur in both pruned and unpruned cuts."""

import numpy as np
from collections import defaultdict

from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_jump_data
from optimization import solve_constraints_only
from uv_recovery import recover_parameterization
from geometry import compute_edge_lengths


def get_cut_adjacent_faces(mesh, Gamma):
    """Get faces that are adjacent to cut edges."""
    cut_faces = set()
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            he0 = mesh.edge_to_halfedge[e, 0]
            he1 = mesh.edge_to_halfedge[e, 1]
            if he0 != -1:
                cut_faces.add(he0 // 3)
            if he1 != -1:
                cut_faces.add(he1 // 3)
    return cut_faces


def get_singularity_adjacent_faces(mesh, singularities):
    """Get faces that contain singularity vertices."""
    sing_verts = set(np.where(np.abs(singularities) > 0.1)[0])
    sing_faces = set()
    for f in range(mesh.n_faces):
        for v in mesh.faces[f]:
            if v in sing_verts:
                sing_faces.add(f)
                break
    return sing_faces


def compute_uv_and_flips(mesh, Gamma, zeta, s, phi, omega0, alpha, xi, singularities, enable_pruning):
    """Compute UV and return flipped triangle info."""
    # Re-run with specific pruning setting
    from collections import deque
    from cut_graph import wrap_angle
    from cross_field import compute_xi_per_halfedge

    n_faces = mesh.n_faces
    n_edges = mesh.n_edges
    n_halfedges = mesh.n_halfedges
    n_corners = mesh.n_corners
    n_vertices = mesh.n_vertices

    # Recompute with pruning option
    Gamma_new = np.ones(n_edges, dtype=np.int32)
    zeta_new = np.zeros(n_edges, dtype=np.float64)
    s_new = np.ones(n_corners, dtype=np.int32)
    phi_new = np.full(n_halfedges, np.inf, dtype=np.float64)
    omega0_new = np.zeros(n_edges, dtype=np.float64)

    xi_he = compute_xi_per_halfedge(mesh, xi, alpha)

    seed_face = 0
    phi_new[3 * seed_face + 0] = wrap_angle(xi[seed_face])
    c1 = 3 * seed_face + 1
    c2 = 3 * seed_face + 2
    phi_new[3 * seed_face + 1] = wrap_angle(phi_new[3 * seed_face + 0] - (np.pi - alpha[c1]))
    phi_new[3 * seed_face + 2] = wrap_angle(phi_new[3 * seed_face + 1] - (np.pi - alpha[c2]))

    s_edge = np.ones(n_edges, dtype=np.int32)

    queue = deque()
    for local in range(3):
        queue.append(3 * seed_face + local)

    while queue:
        he_ij = queue.popleft()
        he_twin = mesh.halfedge_twin[he_ij]
        if he_twin == -1:
            continue

        e = mesh.halfedge_to_edge[he_ij]
        phi_ij_to_ji = wrap_angle(phi_new[he_ij] + np.pi)
        angle_diff = wrap_angle(phi_ij_to_ji - xi_he[he_twin])
        n_star = int(np.round(2 * angle_diff / np.pi)) % 4

        zeta_new[e] = (np.pi / 2) * n_star
        xi_star = wrap_angle(zeta_new[e] + xi_he[he_twin])
        omega0_raw = phi_new[he_ij] - xi_star + np.pi
        omega0_new[e] = wrap_angle(omega0_raw)

        he0 = mesh.edge_to_halfedge[e, 0]
        if he_ij != he0:
            omega0_new[e] = -omega0_new[e]

        if phi_new[he_twin] == np.inf:
            Gamma_new[e] = 0
            s_edge[e] = +1
            phi_new[he_twin] = xi_star

            face_to = he_twin // 3
            local_to = he_twin % 3
            he_next = 3 * face_to + (local_to + 1) % 3
            he_prev = 3 * face_to + (local_to + 2) % 3
            c_i = 3 * face_to + (local_to + 1) % 3
            c_l = 3 * face_to + (local_to + 2) % 3

            phi_new[he_next] = wrap_angle(phi_new[he_twin] - (np.pi - alpha[c_i]))
            phi_new[he_prev] = wrap_angle(phi_new[he_next] - (np.pi - alpha[c_l]))

            queue.append(he_next)
            queue.append(he_prev)
        else:
            Gamma_new[e] = 1
            if n_star % 2 == 1:
                s_edge[e] = -1
            else:
                s_edge[e] = +1

    for v in range(n_vertices):
        corners = mesh.vertex_corners(v)
        if len(corners) == 0:
            continue
        S = 1
        for corner in corners:
            s_new[corner] = S
            he = corner
            e = mesh.halfedge_to_edge[he]
            S = s_edge[e] * S

    is_cone = np.abs(singularities) > 0.1

    if enable_pruning:
        pruning = True
        while pruning:
            d = np.zeros(n_vertices, dtype=np.int32)
            for e in range(n_edges):
                if Gamma_new[e] == 1:
                    i, j = mesh.edge_vertices[e]
                    d[i] += 1
                    d[j] += 1
            pruning = False
            for v in range(n_vertices):
                if d[v] == 1 and not is_cone[v]:
                    for e in range(n_edges):
                        if Gamma_new[e] == 1:
                            i, j = mesh.edge_vertices[e]
                            if i == v or j == v:
                                Gamma_new[e] = 0
                                pruning = True
                                break
                    if pruning:
                        break

    # Solve and get UVs
    u, v, theta = solve_constraints_only(mesh, alpha, phi_new, omega0_new, s_new, verbose=False)

    ell = compute_edge_lengths(mesh)
    corner_uvs = recover_parameterization(mesh, Gamma_new, zeta_new, ell, alpha, phi_new, theta, s_new, u, v)

    # Find flipped triangles
    flipped_faces = []
    for f in range(n_faces):
        uv0 = corner_uvs[3 * f + 0]
        uv1 = corner_uvs[3 * f + 1]
        uv2 = corner_uvs[3 * f + 2]

        e1 = uv1 - uv0
        e2 = uv2 - uv0
        area = e1[0] * e2[1] - e1[1] * e2[0]

        if area < 0:
            flipped_faces.append(f)

    return Gamma_new, flipped_faces


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    sing_faces = get_singularity_adjacent_faces(mesh, cone_indices)
    print(f"\nSingularity-adjacent faces: {len(sing_faces)}")

    # Analyze WITH pruning
    print("\n=== WITH PRUNING (topologically incorrect) ===")
    Gamma_pruned, flips_pruned = compute_uv_and_flips(
        mesh, None, None, None, None, None, alpha, xi, cone_indices, enable_pruning=True
    )

    cut_faces_pruned = get_cut_adjacent_faces(mesh, Gamma_pruned)
    flips_at_cut = len(set(flips_pruned) & cut_faces_pruned)
    flips_at_sing = len(set(flips_pruned) & sing_faces)
    flips_elsewhere = len(flips_pruned) - flips_at_cut - flips_at_sing + len(set(flips_pruned) & cut_faces_pruned & sing_faces)

    print(f"  Cut edges: {np.sum(Gamma_pruned)}")
    print(f"  Flipped triangles: {len(flips_pruned)}")
    print(f"    - At cut boundary: {flips_at_cut}")
    print(f"    - At singularities: {flips_at_sing}")
    print(f"    - Elsewhere: {len(flips_pruned) - len(set(flips_pruned) & (cut_faces_pruned | sing_faces))}")

    # Analyze WITHOUT pruning
    print("\n=== WITHOUT PRUNING (topologically correct) ===")
    Gamma_unpruned, flips_unpruned = compute_uv_and_flips(
        mesh, None, None, None, None, None, alpha, xi, cone_indices, enable_pruning=False
    )

    cut_faces_unpruned = get_cut_adjacent_faces(mesh, Gamma_unpruned)
    flips_at_cut_u = len(set(flips_unpruned) & cut_faces_unpruned)
    flips_at_sing_u = len(set(flips_unpruned) & sing_faces)

    print(f"  Cut edges: {np.sum(Gamma_unpruned)}")
    print(f"  Flipped triangles: {len(flips_unpruned)}")
    print(f"    - At cut boundary: {flips_at_cut_u}")
    print(f"    - At singularities: {flips_at_sing_u}")
    print(f"    - Elsewhere: {len(flips_unpruned) - len(set(flips_unpruned) & (cut_faces_unpruned | sing_faces))}")

    # Compare overlap
    both_flipped = set(flips_pruned) & set(flips_unpruned)
    only_pruned = set(flips_pruned) - set(flips_unpruned)
    only_unpruned = set(flips_unpruned) - set(flips_pruned)

    print(f"\n=== COMPARISON ===")
    print(f"Flipped in BOTH: {len(both_flipped)}")
    print(f"Flipped ONLY in pruned: {len(only_pruned)}")
    print(f"Flipped ONLY in unpruned: {len(only_unpruned)}")


if __name__ == "__main__":
    main()
