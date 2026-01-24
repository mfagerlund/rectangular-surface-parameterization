#!/usr/bin/env python3
"""
Batch migrate all modules to the new package structure.

Usage:
    python scripts/migrate_all.py --dry-run    # Preview changes
    python scripts/migrate_all.py              # Execute migration
"""

import argparse
import subprocess
import sys

# Migration mapping: (old_path, new_path)
# Order matters - migrate dependencies first
MIGRATIONS = [
    # Core utilities (no dependencies)
    ("Preprocess/signed_edge_array.py", "rectangular_surface_parameterization/core/signed_edge_array.py"),
    ("Preprocess/angles_of_triangles.py", "rectangular_surface_parameterization/preprocessing/angles_of_triangles.py"),
    ("Preprocess/gaussian_curvature.py", "rectangular_surface_parameterization/preprocessing/gaussian_curvature.py"),
    ("Utils/vec.py", "rectangular_surface_parameterization/utils/vec.py"),
    ("Utils/sparse_solve.py", "rectangular_surface_parameterization/utils/sparse_solve.py"),

    # Connectivity (depends on signed_edge_array)
    ("Preprocess/connectivity.py", "rectangular_surface_parameterization/preprocessing/connectivity.py"),

    # MeshInfo (depends on connectivity, angles_of_triangles)
    ("Preprocess/MeshInfo.py", "rectangular_surface_parameterization/core/mesh_info.py"),

    # DEC operators (depends on MeshInfo)
    ("Preprocess/dec_tri.py", "rectangular_surface_parameterization/preprocessing/dec.py"),

    # Graph algorithms
    ("Preprocess/find_graph_generator.py", "rectangular_surface_parameterization/preprocessing/find_graph_generator.py"),
    ("Preprocess/sort_triangles.py", "rectangular_surface_parameterization/preprocessing/sort_triangles.py"),
    ("Preprocess/sort_triangles_comp.py", "rectangular_surface_parameterization/preprocessing/sort_triangles_comp.py"),

    # Preprocessing orchestration
    ("Preprocess/preprocess_ortho_param.py", "rectangular_surface_parameterization/preprocessing/preprocess.py"),

    # Cross field computation
    ("FrameField/trivial_connection.py", "rectangular_surface_parameterization/cross_field/trivial_connection.py"),
    ("FrameField/compute_face_cross_field.py", "rectangular_surface_parameterization/cross_field/face_field.py"),
    ("FrameField/compute_curvature_cross_field.py", "rectangular_surface_parameterization/cross_field/curvature_field.py"),
    ("FrameField/brush_frame_field.py", "rectangular_surface_parameterization/cross_field/brush_field.py"),
    ("FrameField/plot_frame_field.py", "rectangular_surface_parameterization/cross_field/plot.py"),

    # Optimization
    ("Orthotropic/optimization_params.py", "rectangular_surface_parameterization/optimization/params.py"),
    ("Orthotropic/objective_ortho_param.py", "rectangular_surface_parameterization/optimization/objective.py"),
    ("Orthotropic/omega_from_scale.py", "rectangular_surface_parameterization/optimization/omega_from_scale.py"),
    ("Orthotropic/oracle_integrability_condition.py", "rectangular_surface_parameterization/optimization/integrability.py"),
    ("Orthotropic/reduction_from_ff2d.py", "rectangular_surface_parameterization/optimization/reduction.py"),
    ("Orthotropic/reduce_corner_var_2d.py", "rectangular_surface_parameterization/optimization/reduce_corner_var.py"),
    ("Orthotropic/reduce_corner_var_2d_cut.py", "rectangular_surface_parameterization/optimization/reduce_corner_var_cut.py"),
    ("Orthotropic/optimize_RSP.py", "rectangular_surface_parameterization/optimization/solver.py"),

    # Parameterization
    ("ComputeParam/cut_mesh.py", "rectangular_surface_parameterization/parameterization/cut_mesh.py"),
    ("ComputeParam/mesh_to_disk_seamless.py", "rectangular_surface_parameterization/parameterization/seamless.py"),
    ("ComputeParam/matrix_vector_multiplication.py", "rectangular_surface_parameterization/parameterization/matrix_ops.py"),
    ("ComputeParam/parametrization_from_scales.py", "rectangular_surface_parameterization/parameterization/integrate.py"),

    # I/O and utilities
    ("Utils/readOBJ.py", "rectangular_surface_parameterization/io/read_obj.py"),
    ("Utils/writeObj.py", "rectangular_surface_parameterization/io/write_obj.py"),
    ("Utils/save_param.py", "rectangular_surface_parameterization/io/save_param.py"),
    ("Utils/visualize_uv.py", "rectangular_surface_parameterization/io/visualize.py"),
    ("Utils/verify_pipeline.py", "rectangular_surface_parameterization/utils/verify_pipeline.py"),
    ("Utils/preprocess_mesh.py", "rectangular_surface_parameterization/utils/preprocess_mesh.py"),
    ("Utils/libqex_wrapper.py", "rectangular_surface_parameterization/utils/libqex_wrapper.py"),
    ("Utils/render_quads.py", "rectangular_surface_parameterization/utils/render_quads.py"),
    ("Utils/extract_scale_from_param.py", "rectangular_surface_parameterization/utils/extract_scale.py"),

    # Main entry point
    ("run_RSP.py", "rectangular_surface_parameterization/cli.py"),
]


def run_migration(source: str, dest: str, dry_run: bool) -> bool:
    """Run a single migration. Returns True if successful."""
    cmd = [sys.executable, "scripts/migrate_module.py", source, dest]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'='*60}")
    print(f"Migrating: {source} -> {dest}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Batch migrate all modules')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Show what would be done without making changes')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from migration index (0-based)')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of migrations to run')

    args = parser.parse_args()

    migrations = MIGRATIONS[args.start_from:]
    if args.count:
        migrations = migrations[:args.count]

    print(f"Total migrations: {len(MIGRATIONS)}")
    print(f"Running: {len(migrations)} (starting from index {args.start_from})")
    print(f"Dry run: {args.dry_run}")

    success_count = 0
    fail_count = 0

    for i, (source, dest) in enumerate(migrations, start=args.start_from):
        if run_migration(source, dest, args.dry_run):
            success_count += 1
        else:
            fail_count += 1
            print(f"FAILED: {source} -> {dest}")

    print(f"\n{'='*60}")
    print(f"Migration complete: {success_count} succeeded, {fail_count} failed")
    print('='*60)


if __name__ == '__main__':
    main()
