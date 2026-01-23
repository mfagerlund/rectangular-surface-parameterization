"""
Diagnose WHY the UV is a starfish.

The cut graph should create a DISK topology (chi=1).
If it doesn't, the UV will be fragmented.
"""

import numpy as np
from collections import defaultdict, deque

from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_jump_data, count_cut_edges
from mesh import euler_characteristic


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Original mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")
    print(f"Original chi: {euler_characteristic(mesh)}")

    # Compute cut graph
    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)

    n_cuts = count_cut_edges(Gamma)
    print(f"\nCut edges: {n_cuts}")

    # Simulate cutting the mesh
    # After cutting, each cut edge becomes TWO boundary edges
    # New mesh should have:
    #   V' = V + (vertices on cut that are duplicated)
    #   E' = E + n_cuts (each cut edge becomes 2)
    #   F' = F (faces don't change)
    #   chi' = V' - E' + F' should be 1 for disk

    # Count vertices on cut edges
    cut_vertex_count = defaultdict(int)
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            v1, v2 = mesh.edge_vertices[e]
            cut_vertex_count[v1] += 1
            cut_vertex_count[v2] += 1

    # Each vertex on the cut gets duplicated based on how the cut passes through it
    # For a tree-like cut, vertices with degree 1 on cut don't need duplication
    # Vertices with degree 2+ need duplication

    n_cut_vertices = len(cut_vertex_count)
    print(f"Vertices on cut: {n_cut_vertices}")

    # Compute cut mesh Euler characteristic
    # This is an approximation - actual cutting is more complex
    # V' = V + extra vertices from cutting
    # E' = E + n_cuts (cut edges become boundary pairs)
    # F' = F

    # Better approach: compute connected components of faces after removing cut edges
    # The dual graph (faces connected through non-cut edges) should be connected

    # Build dual graph adjacency (faces connected through edges)
    dual_adj = defaultdict(set)
    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 != -1 and he1 != -1:
            f0 = he0 // 3
            f1 = he1 // 3
            if Gamma[e] == 0:  # NOT a cut edge
                dual_adj[f0].add(f1)
                dual_adj[f1].add(f0)

    # Count connected components
    visited = set()
    n_components = 0
    component_sizes = []

    for start_face in range(mesh.n_faces):
        if start_face in visited:
            continue

        # BFS from start_face
        queue = deque([start_face])
        visited.add(start_face)
        size = 0

        while queue:
            f = queue.popleft()
            size += 1
            for neighbor in dual_adj[f]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        n_components += 1
        component_sizes.append(size)

    print(f"\nDual graph (faces via non-cut edges):")
    print(f"  Connected components: {n_components}")
    if n_components == 1:
        print(f"  GOOD: Single component = disk topology possible")
    else:
        print(f"  BAD: Multiple components = mesh will be fragmented!")
        print(f"  Component sizes: {sorted(component_sizes, reverse=True)[:10]}")

    # Check if cut graph is a TREE (no cycles)
    # For a valid cut: cut graph should be a spanning tree of singularities
    cut_adj = defaultdict(set)
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            v1, v2 = mesh.edge_vertices[e]
            cut_adj[v1].add(v2)
            cut_adj[v2].add(v1)

    cut_vertices = set(cut_adj.keys())
    n_cut_edges_check = sum(len(neighbors) for neighbors in cut_adj.values()) // 2

    print(f"\nCut graph structure:")
    print(f"  Vertices: {len(cut_vertices)}")
    print(f"  Edges: {n_cut_edges_check}")

    # For a tree: E = V - 1
    if n_cut_edges_check == len(cut_vertices) - 1:
        print(f"  Structure: TREE (E = V - 1)")
    elif n_cut_edges_check < len(cut_vertices) - 1:
        print(f"  Structure: FOREST (disconnected trees)")
        print(f"  Expected edges for tree: {len(cut_vertices) - 1}")
    else:
        print(f"  Structure: HAS CYCLES (E > V - 1)")
        print(f"  Extra edges: {n_cut_edges_check - (len(cut_vertices) - 1)}")

    # Check if cut graph is connected
    if cut_vertices:
        visited_cut = set()
        start = next(iter(cut_vertices))
        queue = deque([start])
        visited_cut.add(start)

        while queue:
            v = queue.popleft()
            for neighbor in cut_adj[v]:
                if neighbor not in visited_cut:
                    visited_cut.add(neighbor)
                    queue.append(neighbor)

        if len(visited_cut) == len(cut_vertices):
            print(f"  Connectivity: CONNECTED")
        else:
            print(f"  Connectivity: DISCONNECTED!")
            print(f"  Reachable: {len(visited_cut)} / {len(cut_vertices)}")

    # Check singularities
    sing_verts = set(np.where(np.abs(cone_indices) > 0.1)[0])
    sing_on_cut = sing_verts & cut_vertices
    print(f"\nSingularities:")
    print(f"  Total: {len(sing_verts)}")
    print(f"  On cut: {len(sing_on_cut)}")
    if sing_on_cut != sing_verts:
        missing = sing_verts - sing_on_cut
        print(f"  MISSING from cut: {missing}")

    # The key question: does cutting along Gamma create a disk?
    # For sphere: original chi = 2
    # After cutting to disk: chi should be 1
    # This means: V' - E' + F' = 1

    # When we cut an edge:
    # - The edge becomes 2 boundary edges
    # - Vertices on the cut may be duplicated

    # Simpler check: the DUAL spanning tree
    # If we remove cut edges from the dual graph, we should have a spanning tree
    # A spanning tree of F faces has F-1 edges
    # So non-cut edges should be exactly F-1 = 319

    n_non_cut = mesh.n_edges - n_cuts
    expected_spanning_tree = mesh.n_faces - 1

    print(f"\nDual spanning tree check:")
    print(f"  Non-cut edges: {n_non_cut}")
    print(f"  Expected for spanning tree: {expected_spanning_tree}")
    if n_non_cut == expected_spanning_tree:
        print(f"  PERFECT: Non-cut edges form spanning tree")
    elif n_non_cut > expected_spanning_tree:
        print(f"  TOO MANY non-cut edges: dual graph has {n_non_cut - expected_spanning_tree} extra edges (cycles)")
    else:
        print(f"  TOO FEW non-cut edges: dual graph is disconnected")


if __name__ == "__main__":
    main()
