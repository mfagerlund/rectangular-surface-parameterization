"""Debug k21 computation - why doesn't BFS reach any faces?"""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_parallel_transport_angles
from compute_k21 import compute_omega, compute_k21


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")

    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)

    para_trans = compute_parallel_transport_angles(mesh)
    omega = compute_omega(mesh, xi, para_trans)
    k21 = compute_k21(mesh, xi, omega, para_trans)

    print(f"\nk21 distribution:")
    for k in range(1, 5):
        print(f"  k21={k}: {np.sum(k21 == k)} edges")

    # Check edges around face 0
    seed_face = 0
    print(f"\nFace 0 vertices: {mesh.faces[seed_face]}")

    for local in range(3):
        he = 3 * seed_face + local
        he_twin = mesh.halfedge_twin[he]
        e = mesh.halfedge_to_edge[he]

        if he_twin == -1:
            print(f"  Edge {e}: boundary")
        else:
            f_neighbor = he_twin // 3
            print(f"  Edge {e}: k21={k21[e]}, neighbor face={f_neighbor}")

    # Check if identity edges are connected
    identity_edges = np.where(k21 == 1)[0]
    print(f"\nIdentity edges: {len(identity_edges)}")

    # Build adjacency graph using identity edges
    from collections import defaultdict
    adj = defaultdict(set)
    for e in identity_edges:
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 != -1 and he1 != -1:
            f0 = he0 // 3
            f1 = he1 // 3
            adj[f0].add(f1)
            adj[f1].add(f0)

    # Find connected components
    faces_with_identity = set(adj.keys())
    print(f"Faces with identity edges: {len(faces_with_identity)}")

    # Check which faces have at least one identity edge to neighbor
    from collections import deque

    visited = set()
    n_components = 0

    for start in faces_with_identity:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        component_size = 0
        while queue:
            f = queue.popleft()
            component_size += 1
            for neighbor in adj[f]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        n_components += 1
        print(f"Component {n_components}: {component_size} faces")

    # Check what's the maximum reachable component from face 0
    if 0 in adj:
        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            f = queue.popleft()
            for neighbor in adj[f]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        print(f"\nFrom face 0, can reach {len(visited)} faces via identity edges")
    else:
        print(f"\nFace 0 has NO identity edges to neighbors!")

    # What about face 0's immediate neighbors?
    print("\nFace 0's neighbors and their connectivity:")
    for local in range(3):
        he = 3 * seed_face + local
        he_twin = mesh.halfedge_twin[he]
        if he_twin != -1:
            f_neighbor = he_twin // 3
            n_identity_at_neighbor = len(adj[f_neighbor])
            print(f"  Face {f_neighbor}: {n_identity_at_neighbor} identity edges")


if __name__ == "__main__":
    main()
