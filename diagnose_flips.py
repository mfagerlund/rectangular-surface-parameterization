"""
Diagnose remaining flipped triangles after Stage 3 fix.
"""

import numpy as np
from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_jump_data, count_cut_edges
from optimization import solve_constraints_only
from uv_recovery import recover_parameterization
from mesh import euler_characteristic


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")
    print(f"Euler characteristic: {euler_characteristic(mesh)}")

    # Stage 1: Geometry
    alpha = compute_corner_angles(mesh)
    ell = np.array([np.linalg.norm(mesh.positions[mesh.edge_vertices[e, 1]] -
                                    mesh.positions[mesh.edge_vertices[e, 0]])
                    for e in range(mesh.n_edges)])

    # Stage 2: Cross-field
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    sing_verts = np.where(is_singular)[0]
    print(f"\nCross-field singularities: {len(sing_verts)}")
    print(f"  Sum of indices: {cone_indices.sum():.4f}")
    for v in sing_verts:
        print(f"  v{v}: index={cone_indices[v]:.3f}")

    # Stage 3: Cut graph
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)

    n_cuts = count_cut_edges(Gamma)
    print(f"\nCut graph: {n_cuts} cut edges")

    # Analyze zeta values
    zeta_nonzero = np.where(zeta != 0)[0]
    print(f"Edges with non-zero zeta: {len(zeta_nonzero)}")
    zeta_values, zeta_counts = np.unique(zeta, return_counts=True)
    print(f"Zeta distribution:")
    for val, cnt in zip(zeta_values, zeta_counts):
        print(f"  zeta={val:.4f} ({val*2/np.pi:.1f}*pi/2): {cnt} edges")

    # Check cut vertices
    cut_vertices = set()
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            v1, v2 = mesh.edge_vertices[e]
            cut_vertices.add(v1)
            cut_vertices.add(v2)
    print(f"Vertices on cut: {len(cut_vertices)}")
    print(f"Singularities on cut: {len(cut_vertices & set(sing_verts))}")

    # Stage 4: Optimization
    u, v, theta = solve_constraints_only(mesh, alpha, phi, omega0, s,
                                         max_iters=500, tol=1e-6, verbose=False)

    # Stage 5: UV recovery
    corner_uvs = recover_parameterization(mesh, Gamma, zeta, ell, alpha, phi, theta, s, u, v)

    # Normalize
    uv_min = corner_uvs.min(axis=0)
    uv_max = corner_uvs.max(axis=0)
    corner_uvs = (corner_uvs - uv_min) / (uv_max - uv_min).max()

    # Find flipped triangles
    flipped = []
    for f in range(mesh.n_faces):
        c0, c1, c2 = 3*f, 3*f+1, 3*f+2
        uv0, uv1, uv2 = corner_uvs[c0], corner_uvs[c1], corner_uvs[c2]

        # Signed area
        area = 0.5 * ((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) -
                      (uv2[0] - uv0[0]) * (uv1[1] - uv0[1]))
        if area < 0:
            flipped.append(f)

    print(f"\nFlipped triangles: {len(flipped)} / {mesh.n_faces}")

    if len(flipped) > 0:
        print("\nAnalyzing flipped triangles:")
        for f in flipped[:10]:  # First 10
            v0, v1, v2 = mesh.faces[f]
            # Check if adjacent to cut
            on_cut = False
            for local in range(3):
                he = 3*f + local
                e = mesh.halfedge_to_edge[he]
                if Gamma[e] == 1:
                    on_cut = True
                    break

            # Check if adjacent to singularity
            near_sing = v0 in sing_verts or v1 in sing_verts or v2 in sing_verts

            c0, c1, c2 = 3*f, 3*f+1, 3*f+2
            area = 0.5 * ((corner_uvs[c1][0] - corner_uvs[c0][0]) * (corner_uvs[c2][1] - corner_uvs[c0][1]) -
                          (corner_uvs[c2][0] - corner_uvs[c0][0]) * (corner_uvs[c1][1] - corner_uvs[c0][1]))

            print(f"  Face {f}: verts=({v0},{v1},{v2}), on_cut={on_cut}, near_sing={near_sing}, area={area:.6f}")

        # Statistics
        on_cut_count = 0
        near_sing_count = 0
        for f in flipped:
            v0, v1, v2 = mesh.faces[f]
            for local in range(3):
                he = 3*f + local
                e = mesh.halfedge_to_edge[he]
                if Gamma[e] == 1:
                    on_cut_count += 1
                    break
            if v0 in sing_verts or v1 in sing_verts or v2 in sing_verts:
                near_sing_count += 1

        print(f"\nFlipped triangle stats:")
        print(f"  Adjacent to cut edge: {on_cut_count}/{len(flipped)}")
        print(f"  Adjacent to singularity: {near_sing_count}/{len(flipped)}")


if __name__ == "__main__":
    main()
