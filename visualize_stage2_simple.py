"""
Simple Stage 2 visualization - prove cross-field is correct.
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

    print(f"\n" + "="*50)
    print("STAGE 2 VERIFICATION")
    print("="*50)
    print(f"Smoothness energy: {energy:.4f}")
    print(f"Singularities: {len(sing_verts)}")
    print(f"Singularity sum: {sing_sum:.4f} (expected: 2.0 for sphere)")
    print(f"Individual indices: {cone_index[sing_verts]}")

    # Create visualization
    fig = plt.figure(figsize=(16, 8))

    # Panel 1: Cross-field with singularities
    ax1 = fig.add_subplot(121, projection='3d')

    # Draw mesh
    N = compute_face_normals(mesh)
    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)
    collection = Poly3DCollection(triangles, facecolor='lightblue',
                                   edgecolor='lightgray', alpha=0.3, linewidth=0.2)
    ax1.add_collection3d(collection)

    # Draw cross-field vectors
    mean_len = np.mean(compute_edge_lengths(mesh))
    vec_len = 0.4 * mean_len

    centroids = np.zeros((mesh.n_faces, 3))
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        centroids[f] = (mesh.positions[v0] + mesh.positions[v1] + mesh.positions[v2]) / 3

    # Draw every 4th face for clarity
    for f in range(0, mesh.n_faces, 4):
        c = centroids[f]
        w = W[f]
        n = N[f]
        w_perp = np.cross(n, w)

        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            direction = np.cos(angle) * w + np.sin(angle) * w_perp
            end = c + vec_len * direction
            ax1.plot([c[0], end[0]], [c[1], end[1]], [c[2], end[2]],
                    color='darkred', linewidth=1.0, alpha=0.8)

    # Draw singularities
    if len(sing_verts) > 0:
        pos = mesh.positions[sing_verts]
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c='green', s=200, marker='o', edgecolors='black', linewidths=2, zorder=10)

        # Add labels
        for v in sing_verts:
            p = mesh.positions[v]
            ax1.text(p[0], p[1], p[2] + 0.1, f'{cone_index[v]:.2f}',
                    fontsize=9, ha='center', fontweight='bold')

    # Set axis limits
    pos = mesh.positions
    max_range = np.max(pos.max(axis=0) - pos.min(axis=0)) / 2
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax1.set_title(f"Cross-Field Vectors + Singularities\n(Energy={energy:.2f})")

    # Panel 2: Verification summary
    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    # Create verification text
    checks = []
    checks.append(("Singularity sum = chi (2.0)", abs(sing_sum - 2.0) < 0.01))
    checks.append(("All indices = 1/4", all(np.abs(cone_index[sing_verts] - 0.25) < 0.01)))
    checks.append(("Smoothness energy finite", energy < 100))

    summary = """
STAGE 2: CROSS-FIELD VERIFICATION
=================================

COMPUTED VALUES:
  Cross-field angles (xi): [{:.3f}, {:.3f}] rad
  Smoothness energy: {:.4f}
  Number of singularities: {}
  Sum of cone indices: {:.4f}

EXPECTED VALUES (sphere, chi=2):
  Singularity sum should equal chi = 2
  Each singularity index should be multiple of 1/4
  (Optimal: 4 singularities at 0.5 each, or 8 at 0.25 each)

SINGULARITY DETAILS:
""".format(xi.min(), xi.max(), energy, len(sing_verts), sing_sum)

    for i, v in enumerate(sing_verts):
        p = mesh.positions[v]
        summary += f"  #{i+1}: vertex {v}, index={cone_index[v]:.3f}\n"

    summary += "\nVERIFICATION CHECKS:\n"
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "[X]" if passed else "[ ]"
        summary += f"  {symbol} {check_name}: {status}\n"

    all_pass = all(p for _, p in checks)
    summary += f"\nOVERALL: {'STAGE 2 VERIFIED' if all_pass else 'ISSUES FOUND'}\n"

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = "c:/Dev/Corman-Crane/output/stage2_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    return sing_sum, energy, len(sing_verts)


if __name__ == "__main__":
    main()
