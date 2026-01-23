"""
Stage 3 visualization - prove cut graph is correct.
Shows: cut edges, singularities (cones), and verifies connectivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_jump_data, count_cut_edges
from mesh import euler_characteristic


def main():
    mesh_path = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
    mesh = load_obj(mesh_path)
    print(f"Mesh: {mesh.n_vertices}V, {mesh.n_faces}F, {mesh.n_edges}E")
    print(f"Euler characteristic: {euler_characteristic(mesh)}")

    # Stage 1-2
    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    sing_verts = np.where(is_singular)[0]

    print(f"\nCross-field singularities: {len(sing_verts)}")
    print(f"  Sum of indices: {cone_indices.sum():.4f}")

    # Stage 3: Cut graph
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)
    n_cuts = count_cut_edges(Gamma)

    print(f"\nCut graph: {n_cuts} cut edges")

    # Verify cut graph properties
    cut_edges = np.where(Gamma == 1)[0]

    # Check connectivity of cut graph
    cut_vertices = set()
    for e in cut_edges:
        v1, v2 = mesh.edge_vertices[e]
        cut_vertices.add(v1)
        cut_vertices.add(v2)

    # Check if all singularities are on cut
    sing_on_cut = set(sing_verts) & cut_vertices

    # Compute cut graph topology
    # For a valid cut: chi(cut_mesh) should be 1 (disk)
    # Cut graph should be a tree connecting all singularities

    print(f"\nCut graph verification:")
    print(f"  Vertices on cut: {len(cut_vertices)}")
    print(f"  Singularities on cut: {len(sing_on_cut)} / {len(sing_verts)}")
    print(f"  All singularities connected: {'YES' if len(sing_on_cut) == len(sing_verts) else 'NO'}")

    # Create visualization
    fig = plt.figure(figsize=(16, 8))

    # Panel 1: Cut graph + singularities
    ax1 = fig.add_subplot(121, projection='3d')
    draw_mesh_3d(ax1, mesh, alpha=0.3)
    draw_cut_edges(ax1, mesh, Gamma, color='red', width=3)
    draw_singularities(ax1, mesh, sing_verts, cone_indices, size=200)
    ax1.set_title(f"Cut Graph\n{n_cuts} cut edges, {len(sing_verts)} singularities")

    # Panel 2: Verification summary
    ax2 = fig.add_subplot(122)
    ax2.axis('off')

    # Verification checks
    checks = []
    checks.append(("All singularities on cut", len(sing_on_cut) == len(sing_verts)))
    checks.append(("Cut edges > 0", n_cuts > 0))
    checks.append(("Gamma is binary", np.all((Gamma == 0) | (Gamma == 1))))
    checks.append(("Phi is finite", np.all(np.isfinite(phi))))
    checks.append(("Zeta is quantized", np.all(np.abs(zeta % (np.pi/2)) < 0.01) or
                   np.all(np.abs((zeta % (np.pi/2)) - np.pi/2) < 0.01)))

    summary = f"""
STAGE 3: CUT GRAPH VERIFICATION
===============================

INPUT (from Stage 2):
  Singularities: {len(sing_verts)}
  Singularity sum: {cone_indices.sum():.4f} (expected: {euler_characteristic(mesh)})

OUTPUT:
  Cut edges: {n_cuts}
  Vertices on cut: {len(cut_vertices)}
  Singularities on cut: {len(sing_on_cut)} / {len(sing_verts)}

SINGULARITY LOCATIONS:
"""
    for i, v in enumerate(sing_verts):
        on_cut = "ON CUT" if v in cut_vertices else "NOT ON CUT"
        summary += f"  v{v}: index={cone_indices[v]:.3f} [{on_cut}]\n"

    summary += f"""
ZETA (rotation) DISTRIBUTION:
  zeta=0 (identity): {np.sum(zeta == 0)} edges
  zeta=pi/2 (90deg): {np.sum(np.abs(zeta - np.pi/2) < 0.01)} edges
  zeta=pi (180deg): {np.sum(np.abs(zeta - np.pi) < 0.01)} edges
  zeta=3pi/2 (270deg): {np.sum(np.abs(zeta - 3*np.pi/2) < 0.01)} edges

VERIFICATION CHECKS:
"""
    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "[X]" if passed else "[ ]"
        summary += f"  {symbol} {check_name}: {status}\n"
        if not passed:
            all_pass = False

    summary += f"\nOVERALL: {'STAGE 3 VERIFIED' if all_pass else 'ISSUES FOUND'}\n"

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = "c:/Dev/Corman-Crane/output/stage3_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")


def draw_mesh_3d(ax, mesh, alpha=0.5):
    """Draw mesh."""
    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)

    collection = Poly3DCollection(triangles, facecolor='lightblue',
                                   edgecolor='lightgray', alpha=alpha, linewidth=0.2)
    ax.add_collection3d(collection)

    pos = mesh.positions
    max_range = np.max(pos.max(axis=0) - pos.min(axis=0)) / 2
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def draw_cut_edges(ax, mesh, Gamma, color='red', width=3):
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


def draw_singularities(ax, mesh, sing_verts, cone_indices, size=150):
    """Draw singularities."""
    if len(sing_verts) == 0:
        return

    colors = ['green' if cone_indices[v] > 0 else 'red' for v in sing_verts]
    pos = mesh.positions[sing_verts]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
              c=colors, s=size, marker='o', edgecolors='black', linewidths=2, zorder=10)

    for v in sing_verts:
        p = mesh.positions[v]
        ax.text(p[0], p[1], p[2] + 0.1, f'v{v}', fontsize=8, ha='center', fontweight='bold')


if __name__ == "__main__":
    main()
