"""
Detailed Stage 2 visualization - prove cross-field is correct.
Shows: vectors, smoothness, singularities with clear evidence.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from io_obj import load_obj
from geometry import compute_face_normals, compute_edge_lengths, compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities, compute_smoothness_energy


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")

    # Compute cross-field
    W, xi = compute_smooth_cross_field(mesh)
    alpha = compute_corner_angles(mesh)
    cone_index, is_singular = compute_cross_field_singularities(mesh, xi, alpha)

    # Compute quality metrics
    energy = compute_smoothness_energy(mesh, W)
    sing_verts = np.where(np.abs(cone_index) > 0.1)[0]
    sing_sum = cone_index.sum()

    print(f"\nStage 2 Results:")
    print(f"  Smoothness energy: {energy:.4f}")
    print(f"  Singularities: {len(sing_verts)}")
    print(f"  Singularity sum: {sing_sum:.4f} (expected: 2 for sphere)")

    # Create detailed visualization
    fig = plt.figure(figsize=(20, 10))

    # Panel 1: Cross-field vectors (zoomed, every face)
    ax1 = fig.add_subplot(231, projection='3d')
    draw_mesh_wireframe(ax1, mesh)
    draw_cross_field_all(ax1, mesh, W, scale=0.4)
    ax1.set_title("Cross-Field Vectors\n(4 arms per face)")

    # Panel 2: Singularities with labels
    ax2 = fig.add_subplot(232, projection='3d')
    draw_mesh_solid(ax2, mesh, alpha=0.4)
    draw_singularities_labeled(ax2, mesh, sing_verts, cone_index)
    ax2.set_title(f"Singularities\n{len(sing_verts)} points, sum={sing_sum:.2f}")

    # Panel 3: Smoothness visualization (color by local energy)
    ax3 = fig.add_subplot(233, projection='3d')
    draw_mesh_by_smoothness(ax3, mesh, W)
    ax3.set_title(f"Smoothness (darker = higher energy)\nTotal energy: {energy:.2f}")

    # Panel 4: Cross-field angle histogram
    ax4 = fig.add_subplot(234)
    ax4.hist(xi, bins=50, edgecolor='black')
    ax4.set_xlabel("Cross-field angle (radians)")
    ax4.set_ylabel("Frequency")
    ax4.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax4.set_title(f"Cross-field Angle Distribution\nRange: [{xi.min():.3f}, {xi.max():.3f}]")

    # Panel 5: Singularity indices
    ax5 = fig.add_subplot(235)
    ax5.bar(range(len(sing_verts)), cone_index[sing_verts], color='green', edgecolor='black')
    ax5.set_xlabel("Singularity #")
    ax5.set_ylabel("Cone Index")
    ax5.axhline(y=0.25, color='red', linestyle='--', label='Expected (1/4)')
    ax5.set_title(f"Cone Indices\nEach = {cone_index[sing_verts[0]]:.3f}")
    ax5.set_xticks(range(len(sing_verts)))
    ax5.set_xticklabels([f"v{v}" for v in sing_verts], rotation=45)

    # Panel 6: Summary text
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    summary = f"""
STAGE 2 VERIFICATION SUMMARY
============================

Cross-Field Properties:
  - Angle range: [{xi.min():.3f}, {xi.max():.3f}] rad
  - Smoothness energy: {energy:.4f}

Singularities:
  - Count: {len(sing_verts)}
  - Sum of indices: {sing_sum:.4f}
  - Expected sum (chi): 2.0000

Individual singularities:
"""
    for i, v in enumerate(sing_verts):
        pos = mesh.positions[v]
        summary += f"  {i+1}. v{v}: index={cone_index[v]:.3f}, pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})\n"

    summary += f"""
VERIFICATION STATUS:
  [{'PASS' if abs(sing_sum - 2.0) < 0.01 else 'FAIL'}] Singularity sum = chi (2.0)
  [{'PASS' if all(np.abs(cone_index[sing_verts] - 0.25) < 0.01) else 'WARN'}] All indices = 1/4
  [{'PASS' if energy < 10 else 'WARN'}] Smoothness energy reasonable
"""
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    output_path = "c:/Dev/Corman-Crane/output/stage2_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Also save just the cross-field for closer inspection
    fig2, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': '3d'})
    draw_mesh_wireframe(ax, mesh)
    draw_cross_field_all(ax, mesh, W, scale=0.5)
    draw_singularities_labeled(ax, mesh, sing_verts, cone_index)
    ax.set_title(f"Stage 2: Cross-Field + Singularities\nEnergy={energy:.2f}, Sum={sing_sum:.2f}")
    output_path2 = "c:/Dev/Corman-Crane/output/stage2_crossfield_only.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path2}")


def draw_mesh_wireframe(ax, mesh):
    """Draw mesh as wireframe."""
    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)

    collection = Poly3DCollection(triangles, facecolor='none',
                                   edgecolor='lightgray', linewidth=0.3)
    ax.add_collection3d(collection)
    set_axis_limits(ax, mesh)


def draw_mesh_solid(ax, mesh, alpha=0.5):
    """Draw mesh as solid."""
    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)

    collection = Poly3DCollection(triangles, facecolor='lightblue',
                                   edgecolor='gray', alpha=alpha, linewidth=0.2)
    ax.add_collection3d(collection)
    set_axis_limits(ax, mesh)


def draw_mesh_by_smoothness(ax, mesh, W):
    """Color mesh faces by local smoothness energy."""
    N = compute_face_normals(mesh)

    # Compute per-face energy (sum of angle differences with neighbors)
    face_energy = np.zeros(mesh.n_faces)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Angle between cross-fields
        dot = np.clip(np.dot(W[f0], W[f1]), -1, 1)
        angle_diff = np.arccos(np.abs(dot))  # 4-fold symmetry
        face_energy[f0] += angle_diff
        face_energy[f1] += angle_diff

    # Normalize
    face_energy = face_energy / face_energy.max() if face_energy.max() > 0 else face_energy

    # Draw with colors
    triangles = []
    colors = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)
        # Higher energy = darker (more red)
        r = 0.3 + 0.7 * face_energy[f]
        g = 0.3 + 0.5 * (1 - face_energy[f])
        b = 0.3 + 0.5 * (1 - face_energy[f])
        colors.append((r, g, b))

    collection = Poly3DCollection(triangles, facecolors=colors,
                                   edgecolor='gray', alpha=0.8, linewidth=0.2)
    ax.add_collection3d(collection)
    set_axis_limits(ax, mesh)


def draw_cross_field_all(ax, mesh, W, scale=0.3):
    """Draw cross-field on every face."""
    N = compute_face_normals(mesh)
    mean_len = np.mean(compute_edge_lengths(mesh))
    vec_len = scale * mean_len

    # Compute face centroids
    centroids = np.zeros((mesh.n_faces, 3))
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        centroids[f] = (mesh.positions[v0] + mesh.positions[v1] + mesh.positions[v2]) / 3

    # Draw 4 arms for each face (subsample for clarity)
    step = max(1, mesh.n_faces // 80)
    for f in range(0, mesh.n_faces, step):
        c = centroids[f]
        w = W[f]
        n = N[f]
        w_perp = np.cross(n, w)

        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            direction = np.cos(angle) * w + np.sin(angle) * w_perp
            end = c + vec_len * direction
            ax.plot([c[0], end[0]], [c[1], end[1]], [c[2], end[2]],
                   color='darkred', linewidth=1.2, alpha=0.8)


def draw_singularities_labeled(ax, mesh, sing_verts, cone_index):
    """Draw singularities with labels."""
    if len(sing_verts) == 0:
        return

    colors = ['green' if cone_index[v] > 0 else 'red' for v in sing_verts]
    pos = mesh.positions[sing_verts]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
              c=colors, s=200, marker='o', edgecolors='black', linewidths=2, zorder=10)

    # Add labels
    for v in sing_verts:
        p = mesh.positions[v]
        ax.text(p[0], p[1], p[2] + 0.08, f'{cone_index[v]:.2f}',
               fontsize=8, ha='center', fontweight='bold')


def set_axis_limits(ax, mesh):
    """Set equal axis limits."""
    pos = mesh.positions
    max_range = np.max(pos.max(axis=0) - pos.min(axis=0)) / 2
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


if __name__ == "__main__":
    main()
