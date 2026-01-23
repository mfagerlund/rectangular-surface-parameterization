"""
Unified visual verification tool for the Corman-Crane RSP pipeline.

Generates per-stage visualizations saved as PNG files for manual verification.

Usage:
    python Utils/verify_pipeline.py <mesh.obj> -o output/
    python Utils/verify_pipeline.py <mesh.obj> -o output/ --stage 1
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.readOBJ import readOBJ
from Preprocess.MeshInfo import MeshInfo, mesh_info
from Preprocess.dec_tri import dec_tri
from Preprocess.preprocess_ortho_param import preprocess_ortho_param
from FrameField.compute_face_cross_field import compute_face_cross_field


# -----------------------------------------------------------------------------
# Stage 1: Geometry
# -----------------------------------------------------------------------------

def verify_geometry(Src: MeshInfo, output_dir: Path) -> dict:
    """
    Verify geometry stage with visualizations.

    Outputs:
    - stage1_mesh.png: 3D mesh wireframe from two angles
    - stage1_curvature.png: Vertex colors = discrete Gaussian curvature (angle defect)

    Returns:
        dict with metrics: euler_char, total_curvature, vertex_count, face_count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute angle defect (discrete Gaussian curvature) at each vertex
    # angle_defect[v] = 2*pi - sum of corner angles at v
    angle_defect = np.full(Src.nv, 2 * np.pi)
    for f in range(Src.nf):
        for i in range(3):
            v = Src.T[f, i]
            angle_defect[v] -= Src.corner_angle[f, i]

    # Handle boundary vertices (should have pi - sum, not 2pi - sum)
    # For now, we'll note total curvature should equal 2*pi*chi
    total_curvature = np.sum(angle_defect)
    euler_char = round(total_curvature / (2 * np.pi))

    # -------------------------------------------------------------------------
    # Plot 1: Mesh wireframe from two angles
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for ax, elev, azim, title in [(ax1, 30, 45, 'View 1 (elev=30, azim=45)'),
                                   (ax2, 30, 135, 'View 2 (elev=30, azim=135)')]:
        # Draw mesh as wireframe
        ax.plot_trisurf(Src.X[:, 0], Src.X[:, 1], Src.X[:, 2],
                        triangles=Src.T,
                        color='lightblue', edgecolor='black', linewidth=0.2, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    fig.suptitle(f'Stage 1: Geometry - Mesh Wireframe\n'
                 f'Vertices: {Src.nv}, Faces: {Src.nf}, Edges: {Src.ne}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'stage1_mesh.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'stage1_mesh.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: Curvature (angle defect) visualization
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # 3D view with curvature coloring
    # Map curvature to face colors (average of vertex curvatures)
    face_curvature = np.mean(angle_defect[Src.T], axis=1)

    # Normalize for colormap
    vmax = max(abs(face_curvature.min()), abs(face_curvature.max()))
    if vmax < 1e-10:
        vmax = 1.0

    # Create face colors
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # Build polygon collection for 3D
    verts = Src.X[Src.T]  # (nf, 3, 3)
    facecolors = cmap(norm(face_curvature))

    poly = Poly3DCollection(verts, facecolors=facecolors, edgecolor='black', linewidth=0.1, alpha=0.9)
    ax1.add_collection3d(poly)

    # Set axis limits
    ax1.set_xlim(Src.X[:, 0].min(), Src.X[:, 0].max())
    ax1.set_ylim(Src.X[:, 1].min(), Src.X[:, 1].max())
    ax1.set_zlim(Src.X[:, 2].min(), Src.X[:, 2].max())
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Discrete Gaussian Curvature\n(red=positive, blue=negative)')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, shrink=0.6, label='Angle defect (rad)')

    # Histogram of curvature values
    ax2.hist(angle_defect, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero curvature')
    ax2.set_xlabel('Angle defect (rad)')
    ax2.set_ylabel('Vertex count')
    ax2.set_title(f'Curvature Distribution\n'
                  f'Total: {total_curvature:.4f} rad = {total_curvature/(2*np.pi):.4f} × 2π\n'
                  f'Euler characteristic: {euler_char}')
    ax2.legend()

    fig.suptitle('Stage 1: Geometry - Discrete Gaussian Curvature', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'stage1_curvature.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'stage1_curvature.png'}")

    # Return metrics
    metrics = {
        'vertex_count': Src.nv,
        'face_count': Src.nf,
        'edge_count': Src.ne,
        'euler_char': euler_char,
        'total_curvature': total_curvature,
        'expected_curvature': 2 * np.pi * euler_char,
    }

    print(f"\nStage 1 Metrics:")
    print(f"  Vertices: {Src.nv}")
    print(f"  Faces: {Src.nf}")
    print(f"  Edges: {Src.ne}")
    print(f"  Euler characteristic: {euler_char} (V - E + F = {Src.nv} - {Src.ne} + {Src.nf} = {Src.nv - Src.ne + Src.nf})")
    print(f"  Total curvature: {total_curvature:.6f} rad (expected: {2*np.pi*euler_char:.6f} for χ={euler_char})")

    return metrics


# -----------------------------------------------------------------------------
# Stage 2: Cross Field
# -----------------------------------------------------------------------------

def verify_cross_field(Src: MeshInfo, param, ang: np.ndarray, sing: np.ndarray,
                       output_dir: Path) -> dict:
    """
    Verify cross field stage with visualizations.

    Outputs:
    - stage2_cross_field.png: Cross glyphs on face centroids
    - stage2_singularities.png: Mesh with singularity markers

    Returns:
        dict with metrics: singularity_count_pos, singularity_count_neg, index_sum
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count singularities
    sing_pos_mask = sing > 1/8
    sing_neg_mask = sing < -1/8
    n_sing_pos = np.sum(sing_pos_mask)
    n_sing_neg = np.sum(sing_neg_mask)
    index_sum = np.sum(sing)

    # -------------------------------------------------------------------------
    # Plot 1: Cross field streamlines (integral curves)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Compute face barycenters
    barycenters = (Src.X[Src.T[:, 0], :] + Src.X[Src.T[:, 1], :] + Src.X[Src.T[:, 2], :]) / 3

    # Compute cross directions in 3D
    # e1 = exp(1i*ang), e2 = 1i*e1
    e1_complex = np.exp(1j * ang)
    e2_complex = 1j * e1_complex

    # Map to 3D using reference frame
    E1 = np.real(e1_complex)[:, np.newaxis] * param.e1r + np.imag(e1_complex)[:, np.newaxis] * param.e2r
    E2 = np.real(e2_complex)[:, np.newaxis] * param.e1r + np.imag(e2_complex)[:, np.newaxis] * param.e2r

    # Normalize directions
    E1 = E1 / (np.linalg.norm(E1, axis=1, keepdims=True) + 1e-10)
    E2 = E2 / (np.linalg.norm(E2, axis=1, keepdims=True) + 1e-10)

    # Build a simple spatial lookup for finding nearest face
    from scipy.spatial import cKDTree
    face_tree = cKDTree(barycenters)

    def find_face(point):
        """Find the nearest face to a point."""
        _, idx = face_tree.query(point)
        return idx

    # Compute mesh center and slightly larger radius to keep lines visible
    mesh_center = np.mean(Src.X, axis=0)
    mesh_radius = np.max(np.linalg.norm(Src.X - mesh_center, axis=1)) * 1.02  # Slightly outside

    def project_to_surface(point, face_idx):
        """Project point onto mesh surface (slightly outside to be visible)."""
        direction = point - mesh_center
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            return mesh_center + direction / norm * mesh_radius
        return point

    def trace_streamline(start_face, direction_field, max_steps=200, step_size=None):
        """Trace a streamline from a starting face following the direction field."""
        if step_size is None:
            step_size = np.sqrt(np.mean(Src.SqEdgeLength)) * 0.15

        path = [barycenters[start_face].copy()]
        current_pos = path[0].copy()
        current_face = start_face
        prev_dir = None

        for _ in range(max_steps):
            # Get direction at current face
            direction = direction_field[current_face].copy()

            # Ensure consistent direction (avoid flipping back and forth)
            if prev_dir is not None:
                if np.dot(direction, prev_dir) < 0:
                    direction = -direction
            prev_dir = direction.copy()

            # Take a step
            new_pos = current_pos + direction * step_size

            # Project back to surface (for curved meshes)
            new_pos = project_to_surface(new_pos, current_face)

            # Find new face
            new_face = find_face(new_pos)

            # Check if we've gone too far from the mesh
            dist_to_face = np.linalg.norm(new_pos - barycenters[new_face])
            if dist_to_face > step_size * 3:
                break

            path.append(new_pos.copy())
            current_pos = new_pos
            current_face = new_face

        return np.array(path)

    # Generate streamlines from seed points
    n_seeds = 80
    np.random.seed(42)  # Reproducible
    seed_faces = np.random.choice(Src.nf, n_seeds, replace=False)

    streamlines_E1 = []
    streamlines_E2 = []
    skip_steps = 5  # Skip first N steps to reduce clutter at seed points

    for seed in seed_faces:
        # Trace in both directions for E1
        path_fwd = trace_streamline(seed, E1, max_steps=150)
        path_bwd = trace_streamline(seed, -E1, max_steps=150)
        if len(path_bwd) > 1:
            full_path = np.vstack([path_bwd[::-1], path_fwd[1:]])
        else:
            full_path = path_fwd
        # Skip first few steps
        if len(full_path) > skip_steps * 2:
            full_path = full_path[skip_steps:-skip_steps]
        streamlines_E1.append(full_path)

        # Trace in both directions for E2
        path_fwd = trace_streamline(seed, E2, max_steps=150)
        path_bwd = trace_streamline(seed, -E2, max_steps=150)
        if len(path_bwd) > 1:
            full_path = np.vstack([path_bwd[::-1], path_fwd[1:]])
        else:
            full_path = path_fwd
        # Skip first few steps
        if len(full_path) > skip_steps * 2:
            full_path = full_path[skip_steps:-skip_steps]
        streamlines_E2.append(full_path)

    # Draw streamlines
    for ax, elev, azim, title in [(ax1, 30, 45, 'View 1 (elev=30, azim=45)'),
                                   (ax2, 30, 135, 'View 2 (elev=30, azim=135)')]:
        # Draw mesh surface (light gray, semi-transparent)
        ax.plot_trisurf(Src.X[:, 0], Src.X[:, 1], Src.X[:, 2],
                        triangles=Src.T,
                        color='whitesmoke', edgecolor='lightgray', linewidth=0.1, alpha=0.4)

        # Draw E1 streamlines in red
        for path in streamlines_E1:
            if len(path) > 1:
                ax.plot(path[:, 0], path[:, 1], path[:, 2],
                        color='red', linewidth=1.5, alpha=0.8)

        # Draw E2 streamlines in blue
        for path in streamlines_E2:
            if len(path) > 1:
                ax.plot(path[:, 0], path[:, 1], path[:, 2],
                        color='blue', linewidth=1.5, alpha=0.8)

        # Mark singularities
        if np.any(sing_pos_mask):
            ax.scatter(Src.X[sing_pos_mask, 0], Src.X[sing_pos_mask, 1], Src.X[sing_pos_mask, 2],
                       c='orange', s=80, marker='o', edgecolors='black', linewidths=1, depthshade=False)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    fig.suptitle(f'Stage 2: Cross Field Streamlines\n'
                 f'Red = E1 direction, Blue = E2 direction (orthogonal), Orange = singularities',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'stage2_cross_field.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'stage2_cross_field.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: Singularities
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # 3D view with singularities marked
    ax1.plot_trisurf(Src.X[:, 0], Src.X[:, 1], Src.X[:, 2],
                     triangles=Src.T,
                     color='lightblue', edgecolor='gray', linewidth=0.1, alpha=0.7)

    # Mark positive singularities (red dots)
    if np.any(sing_pos_mask):
        ax1.scatter(Src.X[sing_pos_mask, 0], Src.X[sing_pos_mask, 1], Src.X[sing_pos_mask, 2],
                    c='red', s=100, marker='o', label=f'+1/4 singularities ({n_sing_pos})', depthshade=False)

    # Mark negative singularities (blue dots)
    if np.any(sing_neg_mask):
        ax1.scatter(Src.X[sing_neg_mask, 0], Src.X[sing_neg_mask, 1], Src.X[sing_neg_mask, 2],
                    c='blue', s=100, marker='s', label=f'-1/4 singularities ({n_sing_neg})', depthshade=False)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title(f'Singularities on Mesh\n'
                  f'Total: {n_sing_pos + n_sing_neg} ({n_sing_pos} pos, {n_sing_neg} neg)')

    # Histogram of singularity indices
    # Filter to show only non-zero values for clarity
    nonzero_sing = sing[np.abs(sing) > 0.01]
    if len(nonzero_sing) > 0:
        # Handle case where all values are identical
        sing_range = nonzero_sing.max() - nonzero_sing.min()
        if sing_range < 1e-10:
            # All values identical - use a simple bar
            unique_val = nonzero_sing[0]
            ax2.bar([unique_val], [len(nonzero_sing)], width=0.05, edgecolor='black', alpha=0.7)
        else:
            n_bins = min(20, len(np.unique(nonzero_sing)))
            ax2.hist(nonzero_sing, bins=max(n_bins, 5), edgecolor='black', alpha=0.7)
        ax2.axvline(x=0.25, color='red', linestyle='--', linewidth=2, label='+1/4')
        ax2.axvline(x=-0.25, color='blue', linestyle='--', linewidth=2, label='-1/4')
    else:
        ax2.text(0.5, 0.5, 'No singularities', ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xlabel('Singularity index')
    ax2.set_ylabel('Vertex count')
    ax2.set_title(f'Singularity Index Distribution\n'
                  f'Sum of indices: {index_sum:.4f} (expected: χ/4 = {index_sum:.4f})')
    ax2.legend()

    fig.suptitle('Stage 2: Cross Field - Singularities', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'stage2_singularities.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'stage2_singularities.png'}")

    # Return metrics
    metrics = {
        'singularity_count_pos': int(n_sing_pos),
        'singularity_count_neg': int(n_sing_neg),
        'singularity_count_total': int(n_sing_pos + n_sing_neg),
        'index_sum': float(index_sum),
    }

    print(f"\nStage 2 Metrics:")
    print(f"  Positive singularities (+1/4): {n_sing_pos}")
    print(f"  Negative singularities (-1/4): {n_sing_neg}")
    print(f"  Total singularities: {n_sing_pos + n_sing_neg}")
    print(f"  Sum of indices: {index_sum:.4f}")
    print(f"  Expected (χ/4 for cross field): Euler char / 4")

    return metrics


# -----------------------------------------------------------------------------
# Stage 3: Cut Graph (placeholder)
# -----------------------------------------------------------------------------

def verify_cut_graph(Src: MeshInfo, cut_edges: np.ndarray, cones: np.ndarray,
                     output_dir: Path) -> dict:
    """Placeholder for Stage 3 verification."""
    raise NotImplementedError("Stage 3 not yet implemented")


# -----------------------------------------------------------------------------
# Stage 4: Optimization (placeholder)
# -----------------------------------------------------------------------------

def verify_optimization(Src: MeshInfo, u: np.ndarray, v: np.ndarray,
                        theta: np.ndarray, output_dir: Path) -> dict:
    """Placeholder for Stage 4 verification."""
    raise NotImplementedError("Stage 4 not yet implemented")


# -----------------------------------------------------------------------------
# Stage 5: UV Recovery (placeholder)
# -----------------------------------------------------------------------------

def verify_uv_recovery(Xp: np.ndarray, T: np.ndarray, detJ: np.ndarray,
                       output_dir: Path) -> dict:
    """Placeholder for Stage 5 verification."""
    raise NotImplementedError("Stage 5 not yet implemented")


# -----------------------------------------------------------------------------
# Full pipeline verification
# -----------------------------------------------------------------------------

def verify_all(mesh_path: str, output_dir: str, stage: Optional[int] = None) -> dict:
    """
    Run full pipeline and verify all stages (or a specific stage).

    Args:
        mesh_path: Path to input OBJ mesh
        output_dir: Output directory for visualizations
        stage: If specified, only verify this stage (1-5)

    Returns:
        dict with all metrics from each stage
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading mesh: {mesh_path}")
    X, T, *_ = readOBJ(mesh_path)
    print(f"  Vertices: {X.shape[0]}, Faces: {T.shape[0]}")

    # Rescale to unit area (same as run_RSP.py)
    e1 = X[T[:, 0], :] - X[T[:, 1], :]
    e2 = X[T[:, 0], :] - X[T[:, 2], :]
    cross_prod = np.cross(e1, e2)
    area_tot = np.sum(np.sqrt(np.sum(cross_prod ** 2, axis=1))) / 2
    X = X / np.sqrt(area_tot)

    # Build MeshInfo
    print("Computing mesh connectivity...")
    Src = mesh_info(X, T)

    all_metrics = {}

    # Stage 1: Geometry
    if stage is None or stage == 1:
        print("\n" + "="*60)
        print("STAGE 1: GEOMETRY")
        print("="*60)
        all_metrics['stage1'] = verify_geometry(Src, output_dir)

    # Preprocess for cross field (needed for stages 2+)
    if stage is None or stage >= 2:
        print("Preprocessing for cross field...")
        dec = dec_tri(Src)
        param, Src, dec = preprocess_ortho_param(Src, dec, ifboundary=True, ifhardedge=True, tol_dihedral_deg=40)

    # Stage 2: Cross Field
    if stage is None or stage == 2:
        print("\n" + "="*60)
        print("STAGE 2: CROSS FIELD")
        print("="*60)
        print("Computing smooth cross field...")
        omega, ang, sing = compute_face_cross_field(Src, param, dec, smoothing_iter=10)
        all_metrics['stage2'] = verify_cross_field(Src, param, ang, sing, output_dir)

    # Stage 3: Cut Graph
    if stage is None or stage == 3:
        print("\n" + "="*60)
        print("STAGE 3: CUT GRAPH")
        print("="*60)
        print("(Not yet implemented)")

    # Stage 4: Optimization
    if stage is None or stage == 4:
        print("\n" + "="*60)
        print("STAGE 4: OPTIMIZATION")
        print("="*60)
        print("(Not yet implemented)")

    # Stage 5: UV Recovery
    if stage is None or stage == 5:
        print("\n" + "="*60)
        print("STAGE 5: UV RECOVERY")
        print("="*60)
        print("(Not yet implemented)")

    return all_metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual verification tool for Corman-Crane RSP pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Utils/verify_pipeline.py mesh.obj -o output/
  python Utils/verify_pipeline.py mesh.obj -o output/ --stage 1
  python Utils/verify_pipeline.py C:/Dev/Colonel/Data/Meshes/sphere320.obj -o Results/verify/
        """
    )

    parser.add_argument('mesh', type=str, help='Path to input OBJ mesh file')
    parser.add_argument('-o', '--output', type=str, default='output/',
                        help='Output directory for visualizations (default: output/)')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4, 5],
                        help='Only verify specific stage (1-5)')

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        metrics = verify_all(args.mesh, args.output, args.stage)
        print("\n" + "="*60)
        print("VERIFICATION COMPLETE")
        print("="*60)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
