"""
Generate cross-field visualization to verify Stage 2.
Shows cross-field vectors and singularity locations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from io_obj import load_obj
from geometry import compute_face_normals, compute_edge_lengths, compute_all_face_bases, compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities


def visualize_crossfield_with_singularities(mesh_path: str, output_path: str):
    """Generate cross-field visualization with singularities marked."""

    # Load mesh
    mesh = load_obj(mesh_path)
    print(f"Loaded mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

    # Compute corner angles
    alpha = compute_corner_angles(mesh)

    # Compute cross-field
    W, xi = compute_smooth_cross_field(mesh)
    print(f"Cross-field computed: W shape = {W.shape}, xi shape = {xi.shape}")

    # Compute singularities
    cone_index, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    # Find non-zero singularities (threshold for numerical noise)
    threshold = 0.1
    sing_vertices = np.where(np.abs(cone_index) > threshold)[0]
    sing_values = cone_index[sing_vertices]

    print(f"\nSingularities found: {len(sing_vertices)}")
    print(f"  Sum of cone indices: {cone_index.sum():.4f} (should be Euler char = 2)")
    for v, idx in zip(sing_vertices, sing_values):
        print(f"  Vertex {v}: cone index = {idx:.4f} ({idx*4:.1f}/4)")

    # Create figure with two views
    fig = plt.figure(figsize=(16, 7))

    # View 1: Cross-field vectors
    ax1 = fig.add_subplot(121, projection='3d')
    draw_mesh_3d(ax1, mesh, alpha=0.3)
    draw_cross_field(ax1, mesh, W, scale=0.25)
    draw_singularities(ax1, mesh, sing_vertices, sing_values)
    ax1.set_title(f"Cross-Field Vectors\n({len(sing_vertices)} singularities, sum={cone_index.sum():.2f})")

    # View 2: Singularities only (clearer view)
    ax2 = fig.add_subplot(122, projection='3d')
    draw_mesh_3d(ax2, mesh, alpha=0.4)
    draw_singularities(ax2, mesh, sing_vertices, sing_values, size=150)

    # Add legend
    for v, idx in zip(sing_vertices, sing_values):
        pos = mesh.positions[v]
        label = f"v{v}: {idx:.2f}"

    ax2.set_title(f"Singularities Only\n(Expected: 4 cones with sum=2 for sphere)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Also save a text summary
    summary_path = output_path.replace('.png', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Cross-Field Verification Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Mesh: {mesh_path}\n")
        f.write(f"Vertices: {mesh.n_vertices}, Faces: {mesh.n_faces}, Edges: {mesh.n_edges}\n\n")
        f.write(f"Singularities: {len(sing_vertices)}\n")
        f.write(f"Sum of cone indices: {cone_index.sum():.6f}\n")
        f.write(f"Expected sum (Euler char): 2\n\n")
        f.write(f"Individual singularities:\n")
        for v, idx in zip(sing_vertices, sing_values):
            pos = mesh.positions[v]
            f.write(f"  Vertex {v}: index={idx:.4f}, position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n")
    print(f"Saved: {summary_path}")

    return W, xi, cone_index


def draw_mesh_3d(ax, mesh, alpha=0.5):
    """Draw 3D mesh."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)

    collection = Poly3DCollection(triangles, facecolor='lightblue',
                                   edgecolor='gray', alpha=alpha, linewidth=0.3)
    ax.add_collection3d(collection)

    # Set axis limits
    pos = mesh.positions
    max_range = np.max(pos.max(axis=0) - pos.min(axis=0)) / 2
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def draw_cross_field(ax, mesh, W, scale=0.3):
    """Draw cross-field vectors on mesh faces."""
    N = compute_face_normals(mesh)

    # Compute face centroids
    centroids = np.zeros((mesh.n_faces, 3))
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        centroids[f] = (mesh.positions[v0] + mesh.positions[v1] + mesh.positions[v2]) / 3

    # Get mean edge length for scaling
    mean_len = np.mean(compute_edge_lengths(mesh))
    vec_len = scale * mean_len

    # Subsample for clarity (every 3rd face)
    step = max(1, mesh.n_faces // 100)

    # Draw cross field (all 4 directions)
    for f in range(0, mesh.n_faces, step):
        c = centroids[f]
        w = W[f]
        n = N[f]

        # Perpendicular direction in tangent plane
        w_perp = np.cross(n, w)

        # Draw 4 arms of the cross
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            direction = np.cos(angle) * w + np.sin(angle) * w_perp
            end = c + vec_len * direction
            ax.plot([c[0], end[0]], [c[1], end[1]], [c[2], end[2]],
                   color='darkred', linewidth=0.8, alpha=0.7)


def draw_singularities(ax, mesh, sing_vertices, sing_values, size=100):
    """Draw singularity markers with color based on index."""
    if len(sing_vertices) == 0:
        return

    # Color by sign: green for positive, red for negative
    colors = ['green' if idx > 0 else 'red' for idx in sing_values]

    pos = mesh.positions[sing_vertices]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
              c=colors, s=size, marker='o', edgecolors='black', linewidths=2, zorder=10)

    # Add text labels
    for i, (v, idx) in enumerate(zip(sing_vertices, sing_values)):
        p = mesh.positions[v]
        ax.text(p[0], p[1], p[2] + 0.05, f'{idx:.2f}', fontsize=8, ha='center')


if __name__ == "__main__":
    import sys
    import os

    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    output_dir = "c:/Dev/Corman-Crane/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "crossfield_verification.png")

    visualize_crossfield_with_singularities(mesh_path, output_path)
