"""Compare UV results with different cut graph configurations."""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_jump_data, count_cut_edges
from optimization import solve_constraints_only
from uv_recovery import recover_uv
from mesh import euler_characteristic


def run_pipeline(mesh, alpha, xi, cone_indices, label, enable_pruning=True):
    """Run the pipeline and return metrics."""
    # Temporarily modify the cut_graph module to enable/disable pruning
    from collections import deque
    from cut_graph import wrap_angle
    from cross_field import compute_xi_per_halfedge

    n_faces = mesh.n_faces
    n_edges = mesh.n_edges
    n_halfedges = mesh.n_halfedges
    n_corners = mesh.n_corners
    n_vertices = mesh.n_vertices

    # Initialize
    Gamma = np.ones(n_edges, dtype=np.int32)
    zeta = np.zeros(n_edges, dtype=np.float64)
    s = np.ones(n_corners, dtype=np.int32)
    phi = np.full(n_halfedges, np.inf, dtype=np.float64)
    omega0 = np.zeros(n_edges, dtype=np.float64)

    xi_he = compute_xi_per_halfedge(mesh, xi, alpha)

    # BFS setup
    seed_face = 0
    phi[3 * seed_face + 0] = wrap_angle(xi[seed_face])
    c1 = 3 * seed_face + 1
    c2 = 3 * seed_face + 2
    phi[3 * seed_face + 1] = wrap_angle(phi[3 * seed_face + 0] - (np.pi - alpha[c1]))
    phi[3 * seed_face + 2] = wrap_angle(phi[3 * seed_face + 1] - (np.pi - alpha[c2]))

    s_edge = np.ones(n_edges, dtype=np.int32)

    queue = deque()
    for local in range(3):
        queue.append(3 * seed_face + local)

    # BFS
    while queue:
        he_ij = queue.popleft()
        he_twin = mesh.halfedge_twin[he_ij]
        if he_twin == -1:
            continue

        e = mesh.halfedge_to_edge[he_ij]
        phi_ij_to_ji = wrap_angle(phi[he_ij] + np.pi)
        angle_diff = wrap_angle(phi_ij_to_ji - xi_he[he_twin])
        n_star = int(np.round(2 * angle_diff / np.pi)) % 4

        zeta[e] = (np.pi / 2) * n_star
        xi_star = wrap_angle(zeta[e] + xi_he[he_twin])
        omega0_raw = phi[he_ij] - xi_star + np.pi
        omega0[e] = wrap_angle(omega0_raw)

        he0 = mesh.edge_to_halfedge[e, 0]
        if he_ij != he0:
            omega0[e] = -omega0[e]

        if phi[he_twin] == np.inf:
            Gamma[e] = 0
            s_edge[e] = +1
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
        else:
            Gamma[e] = 1
            if n_star % 2 == 1:
                s_edge[e] = -1
            else:
                s_edge[e] = +1

    # Compute signs at corners
    for v in range(n_vertices):
        corners = mesh.vertex_corners(v)
        if len(corners) == 0:
            continue
        S = 1
        for p, corner in enumerate(corners):
            s[corner] = S
            he = corner
            e = mesh.halfedge_to_edge[he]
            S = s_edge[e] * S

    # Cone detection
    is_cone = np.abs(cone_indices) > 0.1

    # Optional pruning
    if enable_pruning:
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
                    for e in range(n_edges):
                        if Gamma[e] == 1:
                            i, j = mesh.edge_vertices[e]
                            if i == v or j == v:
                                Gamma[e] = 0
                                pruning = True
                                break
                    if pruning:
                        break

    n_cut = np.sum(Gamma)
    n_non_cut = n_edges - n_cut

    # Solve optimization
    try:
        u, v, theta = solve_constraints_only(mesh, alpha, Gamma, zeta, s, phi, omega0, verbose=False)

        # Recover UV
        uv = recover_uv(mesh, alpha, xi, Gamma, zeta, phi, omega0, u, v, theta)

        # Count flipped triangles
        n_flipped = 0
        for f in range(mesh.n_faces):
            v0, v1, v2 = mesh.faces[f]
            e1 = uv[v1] - uv[v0]
            e2 = uv[v2] - uv[v0]
            area = e1[0] * e2[1] - e1[1] * e2[0]
            if area < 0:
                n_flipped += 1

        # Compute angle error
        errors = []
        for f in range(mesh.n_faces):
            for c in range(3):
                corner = 3 * f + c
                v0 = mesh.faces[f][c]
                v1 = mesh.faces[f][(c + 1) % 3]
                v2 = mesh.faces[f][(c + 2) % 3]

                e1_3d = mesh.positions[v1] - mesh.positions[v0]
                e2_3d = mesh.positions[v2] - mesh.positions[v0]
                cos_3d = np.dot(e1_3d, e2_3d) / (np.linalg.norm(e1_3d) * np.linalg.norm(e2_3d) + 1e-10)
                angle_3d = np.arccos(np.clip(cos_3d, -1, 1))

                e1_uv = uv[v1] - uv[v0]
                e2_uv = uv[v2] - uv[v0]
                cos_uv = np.dot(e1_uv, e2_uv) / (np.linalg.norm(e1_uv) * np.linalg.norm(e2_uv) + 1e-10)
                angle_uv = np.arccos(np.clip(cos_uv, -1, 1))

                errors.append(abs(angle_3d - angle_uv))

        mean_error = np.degrees(np.mean(errors))
        success = True
    except Exception as ex:
        n_flipped = -1
        mean_error = -1
        success = False

    print(f"\n{label}:")
    print(f"  Cut edges: {n_cut}")
    print(f"  Non-cut edges: {n_non_cut}")
    print(f"  Expected non-cut for spanning tree: {n_faces - 1}")
    if success:
        print(f"  Flipped triangles: {n_flipped} / {n_faces}")
        print(f"  Mean angle error: {mean_error:.2f} deg")
    else:
        print(f"  FAILED")

    return {
        'cut_edges': n_cut,
        'non_cut': n_non_cut,
        'flipped': n_flipped,
        'angle_error': mean_error
    }


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    print(f"Singularities: {np.sum(is_singular)} (sum: {cone_indices.sum():.2f})")

    # Compare with and without pruning
    results_pruned = run_pipeline(mesh, alpha, xi, cone_indices,
                                   "WITH PRUNING (topologically wrong, but maybe better UV?)",
                                   enable_pruning=True)

    results_no_prune = run_pipeline(mesh, alpha, xi, cone_indices,
                                     "WITHOUT PRUNING (topologically correct)",
                                     enable_pruning=False)

    print("\n=== SUMMARY ===")
    print(f"Pruned:    {results_pruned['flipped']} flips, {results_pruned['angle_error']:.2f}° error, {results_pruned['cut_edges']} cuts")
    print(f"No prune:  {results_no_prune['flipped']} flips, {results_no_prune['angle_error']:.2f}° error, {results_no_prune['cut_edges']} cuts")


if __name__ == "__main__":
    main()
