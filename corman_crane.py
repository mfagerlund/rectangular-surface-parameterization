"""
Corman-Crane Rectangular Parameterization

Main entry point for computing rectangular surface parameterizations.

Usage:
    python corman_crane.py input.obj -o output_uv.obj

Reference: Corman & Crane, SIGGRAPH 2025
"""

import argparse
import numpy as np
from pathlib import Path

from mesh import TriangleMesh, build_connectivity, euler_characteristic, genus
from io_obj import load_obj, save_obj, mesh_info
from geometry import compute_corner_angles, compute_edge_lengths, MeshGeometry
from cross_field import (
    propagate_cross_field, cross_field_to_angles, initialize_smooth_cross_field,
    compute_smooth_cross_field, compute_cross_field_singularities
)
from cut_graph import compute_cut_jump_data, count_cut_edges
from optimization import solve_constraints_only
from uv_recovery import recover_parameterization, compute_uv_quality, normalize_uvs


def compute_rectangular_parameterization(
    mesh: TriangleMesh,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute rectangular parameterization for a triangle mesh.

    Args:
        mesh: Input triangle mesh
        verbose: Print progress information

    Returns:
        corner_uvs: |C| x 2 array of UV coordinates per corner
    """
    if verbose:
        print("=" * 60)
        print("Corman-Crane Rectangular Parameterization")
        print("=" * 60)
        print(f"\nInput mesh:")
        print(f"  Vertices: {mesh.n_vertices}")
        print(f"  Faces: {mesh.n_faces}")
        print(f"  Edges: {mesh.n_edges}")
        print(f"  Euler characteristic: {euler_characteristic(mesh)}")
        print(f"  Genus: {genus(mesh)}")

    # Phase 1-2: Geometry and cross field
    if verbose:
        print("\n[1] Computing geometry...")
    alpha = compute_corner_angles(mesh)
    ell = compute_edge_lengths(mesh)

    if verbose:
        print("[2] Generating cross field...")
    # Use connection Laplacian method for globally smooth field
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=verbose)

    # Compute singularities (always needed for cut graph)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    if verbose:
        n_sing = np.sum(is_singular)
        print(f"    Singularities: {n_sing} (sum of indices: {cone_indices.sum():.2f})")

    # Phase 3: Cut graph
    if verbose:
        print("[3] Computing cut graph...")
    # Pass singularities to cut graph (MATLAB convention)
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)
    if verbose:
        print(f"    Cut edges: {count_cut_edges(Gamma)}")

    # Phase 4-6: Optimization
    if verbose:
        print("[4-6] Solving optimization...")
    u, v, theta = solve_constraints_only(
        mesh, alpha, phi, omega0, s,
        max_iters=500, tol=1e-6, verbose=verbose
    )

    # Phase 7: UV recovery
    if verbose:
        print("[7] Recovering UV coordinates...")
    corner_uvs = recover_parameterization(
        mesh, Gamma, zeta, ell, alpha, phi, theta, s, u, v
    )

    # Normalize
    corner_uvs = normalize_uvs(corner_uvs)

    # Quality metrics
    if verbose:
        quality = compute_uv_quality(mesh, corner_uvs)
        print("\n[8] Quality metrics:")
        print(f"    Flipped triangles: {quality['flipped_count']} / {mesh.n_faces}")
        print(f"    Angle error (mean): {np.degrees(quality['angle_error_mean']):.2f} deg")

    return corner_uvs


def main():
    parser = argparse.ArgumentParser(
        description="Corman-Crane Rectangular Parameterization"
    )
    parser.add_argument("input", help="Input OBJ file")
    parser.add_argument("-o", "--output", help="Output OBJ file with UVs")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Save visualization images")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    parser.add_argument("--quads", type=int, default=0,
                        help="Extract quad mesh with approximately this many quads")
    parser.add_argument("--quad-output", help="Output path for quad mesh OBJ")

    args = parser.parse_args()

    # Load mesh
    mesh = load_obj(args.input)

    # Compute parameterization
    corner_uvs = compute_rectangular_parameterization(mesh, verbose=not args.quiet)

    # Save output
    if args.output:
        save_obj(args.output, mesh, corner_uvs)
        if not args.quiet:
            print(f"\nSaved: {args.output}")

    # Quad extraction
    if args.quads > 0:
        from quad_extract import extract_quads
        quad_output = args.quad_output or args.input.replace('.obj', '_quads.obj')
        # Use periodic mode for genus > 0 surfaces (torus, etc.)
        mesh_genus = genus(mesh)
        quad_mesh = extract_quads(
            mesh, corner_uvs,
            target_quads=args.quads,
            output_path=quad_output,
            periodic=(mesh_genus > 0),
            verbose=not args.quiet
        )
        if not args.quiet:
            print(f"Quad mesh: {quad_mesh.n_vertices} vertices, {quad_mesh.n_faces} faces")

    # Visualizations
    if args.visualize:
        import matplotlib
        matplotlib.use('Agg')
        from visualize import plot_mesh_2d, plot_uv_checkerboard
        import matplotlib.pyplot as plt

        base = Path(args.input).stem
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        # UV layout with 2 panels: plain and checkerboard
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_mesh_2d(mesh, corner_uvs, ax=axes[0], title="UV Layout")
        plot_uv_checkerboard(mesh, corner_uvs, ax=axes[1])
        plt.tight_layout()
        viz_path = output_dir / f"{base}_uv.png"
        plt.savefig(viz_path, dpi=150)
        if not args.quiet:
            print(f"Saved: {viz_path}")


if __name__ == "__main__":
    main()
