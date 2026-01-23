"""Check cut graph state before and after pruning."""

import numpy as np
from collections import deque
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import (
    compute_smooth_cross_field,
    compute_cross_field_singularities,
    compute_xi_per_halfedge
)
from cut_graph import wrap_angle


def analyze_cut_graph(mesh, Gamma, is_cone, label):
    """Analyze the cut graph structure."""
    n_edges = mesh.n_edges
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces

    n_cut = np.sum(Gamma)
    n_non_cut = n_edges - n_cut

    # Cut graph vertices and edges
    cut_verts = set()
    cut_adj = {}
    for e in range(n_edges):
        if Gamma[e] == 1:
            i, j = mesh.edge_vertices[e]
            cut_verts.add(i)
            cut_verts.add(j)
            if i not in cut_adj:
                cut_adj[i] = set()
            if j not in cut_adj:
                cut_adj[j] = set()
            cut_adj[i].add(j)
            cut_adj[j].add(i)

    n_cut_verts = len(cut_verts)
    n_cut_edges = n_cut

    # Check for tree structure
    if n_cut_verts > 0:
        expected_tree_edges = n_cut_verts - 1
        if n_cut_edges == expected_tree_edges:
            structure = "TREE"
        elif n_cut_edges < expected_tree_edges:
            structure = f"FOREST (disconnected)"
        else:
            structure = f"HAS CYCLES (+{n_cut_edges - expected_tree_edges})"
    else:
        structure = "EMPTY"

    # Cones in cut
    cones_in_cut = sum(1 for v in cut_verts if is_cone[v])

    print(f"\n{label}:")
    print(f"  Cut edges: {n_cut}")
    print(f"  Non-cut edges: {n_non_cut}")
    print(f"  Expected non-cut for dual spanning tree: {n_faces - 1}")
    print(f"  Cut graph: {n_cut_verts} vertices, {n_cut_edges} edges")
    print(f"  Cut graph structure: {structure}")
    print(f"  Cones in cut: {cones_in_cut} / {np.sum(is_cone)}")


def compute_with_pruning_analysis(mesh, alpha, xi, singularities):
    """Compute cut data with analysis at each stage."""
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
        he = 3 * seed_face + local
        queue.append(he)

    # BFS main loop
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

    # Detect cones
    is_cone = np.abs(singularities) > 0.1

    print(f"\n=== BEFORE PRUNING ===")
    analyze_cut_graph(mesh, Gamma, is_cone, "After BFS")

    Gamma_before = Gamma.copy()

    # Pruning
    pruning = True
    prune_rounds = 0
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

        if pruning:
            prune_rounds += 1

    print(f"\n=== AFTER PRUNING ({prune_rounds} rounds) ===")
    analyze_cut_graph(mesh, Gamma, is_cone, "After Pruning")

    # Count edges changed
    changed = np.sum(Gamma_before != Gamma)
    print(f"\nEdges changed by pruning: {changed}")

    return Gamma, Gamma_before, is_cone


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    print(f"Singularities: {np.sum(is_singular)} (sum of indices: {cone_indices.sum():.2f})")

    Gamma, Gamma_before, is_cone = compute_with_pruning_analysis(
        mesh, alpha, xi, cone_indices
    )

    # What would a good cut look like?
    print(f"\n=== DESIRED STATE ===")
    print(f"  For disk topology:")
    print(f"    Non-cut edges (dual spanning tree): {mesh.n_faces - 1}")
    print(f"    Cut edges: {mesh.n_edges - (mesh.n_faces - 1)}")
    print(f"  Minimum Steiner tree for {np.sum(is_cone)} cones: {np.sum(is_cone) - 1} edges")


if __name__ == "__main__":
    main()
