"""
Visualize the cut graph to verify Stage 3.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from io_obj import load_obj
from geometry import compute_corner_angles, compute_all_face_bases
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_graph, get_cone_vertices


def visualize_cutgraph(mesh_path: str, output_path: str):
    """Generate cut graph visualization."""

    # Load mesh
    mesh = load_obj(mesh_path)
    print(f"Loaded mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces, {mesh.n_edges} edges")

    # Compute corner angles
    alpha = compute_corner_angles(mesh)

    # Compute cross-field
    W, xi = compute_smooth_cross_field(mesh)
    print(f"Cross-field computed")

    # Compute singularities
    cone_index, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    sing_vertices = np.where(np.abs(cone_index) > 0.1)[0]
    print(f"Singularities: {len(sing_vertices)} (sum={cone_index.sum():.2f})")

    # Compute cut graph
    Gamma, jump = compute_cut_graph(mesh, cone_index)
    n_cut_edges = int(Gamma.sum())
    print(f"Cut graph: {n_cut_edges} cut edges")

    # Get cone vertices
    cones = get_cone_vertices(mesh, cone_index, alpha)
    print(f"Cone vertices: {len(cones)}")

    # Create figure with multiple views
    fig = plt.figure(figsize=(16, 12))

    # View 1: Cut graph + singularities
    ax1 = fig.add_subplot(221, projection='3d')
    draw_mesh_3d(ax1, mesh, alpha=0.2)
    draw_cut_edges(ax1, mesh, Gamma)
    draw_vertices(ax1, mesh, sing_vertices, 'green', 100, 'Singularities')
    ax1.set_title(f"Cut Graph ({n_cut_edges} edges) + Singularities ({len(sing_vertices)})")

    # View 2: Cut graph + cones
    ax2 = fig.add_subplot(222, projection='3d')
    draw_mesh_3d(ax2, mesh, alpha=0.2)
    draw_cut_edges(ax2, mesh, Gamma)
    draw_vertices(ax2, mesh, cones, 'red', 100, 'Cones')
    ax2.set_title(f"Cut Graph + Cones ({len(cones)})")

    # View 3: Just cut graph with vertex degrees
    ax3 = fig.add_subplot(223, projection='3d')
    draw_mesh_3d(ax3, mesh, alpha=0.2)
    draw_cut_edges(ax3, mesh, Gamma)

    # Find vertices with high cut-degree
    cut_degree = np.zeros(mesh.n_vertices, dtype=int)
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            v1, v2 = mesh.edge_vertices[e]
            cut_degree[v1] += 1
            cut_degree[v2] += 1
    high_degree_verts = np.where(cut_degree >= 3)[0]
    draw_vertices(ax3, mesh, high_degree_verts, 'orange', 80, f'Degree >= 3')
    ax3.set_title(f"Cut Graph Branching Points ({len(high_degree_verts)} vertices)")

    # View 4: Statistics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')

    # Compute cut graph statistics
    n_connected = count_connected_components_cut(mesh, Gamma)

    stats_text = f"""Cut Graph Statistics:

Cut edges: {n_cut_edges} / {mesh.n_edges} ({100*n_cut_edges/mesh.n_edges:.1f}%)
Singularities: {len(sing_vertices)}
Cones passed to cut graph: {len(cones)}
Branching vertices (degree >= 3): {len(high_degree_verts)}
Connected components of cut graph: {n_connected}

Expected for sphere (g=0):
  - Minimal cut: single tree connecting all cones
  - Expected cut edges: ~{len(sing_vertices) + mesh.n_faces // 10}
  - Actual: {n_cut_edges}

Cone indices:
"""
    for v in sing_vertices:
        stats_text += f"  v{v}: {cone_index[v]:.3f}\n"

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    return Gamma, jump, cone_index


def draw_mesh_3d(ax, mesh, alpha=0.5):
    """Draw 3D mesh."""
    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)

    collection = Poly3DCollection(triangles, facecolor='lightblue',
                                   edgecolor='gray', alpha=alpha, linewidth=0.2)
    ax.add_collection3d(collection)

    # Set axis limits
    pos = mesh.positions
    max_range = np.max(pos.max(axis=0) - pos.min(axis=0)) / 2
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def draw_cut_edges(ax, mesh, Gamma, color='red', width=2.5):
    """Draw cut edges."""
    cut_lines = []
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            i, j = mesh.edge_vertices[e]
            p1 = mesh.positions[i]
            p2 = mesh.positions[j]
            cut_lines.append([p1, p2])

    if cut_lines:
        collection = Line3DCollection(cut_lines, colors=color, linewidths=width)
        ax.add_collection3d(collection)


def draw_vertices(ax, mesh, vertices, color, size, label):
    """Draw vertex markers."""
    if len(vertices) == 0:
        return
    pos = mesh.positions[vertices]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
              c=color, s=size, marker='o', edgecolors='black', linewidths=1.5, label=label)


def count_connected_components_cut(mesh, Gamma):
    """Count connected components of the cut graph."""
    from collections import deque

    # Build adjacency list for cut edges only
    adj = {v: [] for v in range(mesh.n_vertices)}
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            v1, v2 = mesh.edge_vertices[e]
            adj[v1].append(v2)
            adj[v2].append(v1)

    # Find vertices on cut graph
    cut_vertices = set()
    for v in adj:
        if adj[v]:
            cut_vertices.add(v)

    if not cut_vertices:
        return 0

    # BFS to count components
    visited = set()
    n_components = 0

    for start in cut_vertices:
        if start in visited:
            continue

        # BFS from start
        queue = deque([start])
        visited.add(start)

        while queue:
            v = queue.popleft()
            for neighbor in adj[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        n_components += 1

    return n_components


if __name__ == "__main__":
    import os

    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    output_dir = "c:/Dev/Corman-Crane/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cutgraph_verification.png")

    visualize_cutgraph(mesh_path, output_path)
