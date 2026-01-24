"""
Extract quad mesh from triangle mesh using RSP parameterization + libQEx.

This script runs the full pipeline:
1. Run RSP parameterization (via run_RSP.py)
2. Load the parameterized mesh with UVs
3. Extract quad mesh using libQEx

Usage:
    python extract_quads.py mesh.obj -o output/
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import numpy as np

from rectangular_surface_parameterization.io.read_obj import readOBJ
from rectangular_surface_parameterization.utils.libqex_wrapper import extract_quads, save_quad_obj


def preprocess_mesh_if_needed(mesh_path, output_dir, verbose=False):
    """
    Preprocess mesh using PyMeshLab to clean it for the pipeline.

    Returns the path to the cleaned mesh.
    """
    try:
        from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh
    except ImportError:
        print("Warning: PyMeshLab not installed. Skipping preprocessing.")
        print("Install with: pip install pymeshlab")
        return mesh_path

    mesh_name = Path(mesh_path).stem
    clean_path = output_dir / f"{mesh_name}_clean.obj"

    if verbose:
        print(f"Preprocessing mesh: {mesh_path}")

    try:
        preprocess_mesh(str(mesh_path), str(clean_path), verbose=verbose)
        return clean_path
    except Exception as e:
        print(f"Warning: Preprocessing failed: {e}")
        print("Continuing with original mesh...")
        return mesh_path


def run_rsp(mesh_path, output_dir, verbose=False):
    """
    Run the RSP parameterization pipeline.

    Returns the path to the output _param.obj file.
    """
    cmd = [
        sys.executable, "run_RSP.py",
        mesh_path,
        "-o", str(output_dir),
        "--no-quantization",  # Skip quantization (requires external program)
    ]
    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("RSP pipeline failed:")
        print(result.stderr)
        raise RuntimeError("RSP pipeline failed")

    if verbose:
        print(result.stdout)

    # Find output file
    mesh_name = Path(mesh_path).stem
    param_path = output_dir / f"{mesh_name}_param.obj"

    if not param_path.exists():
        raise FileNotFoundError(f"RSP output not found: {param_path}")

    return param_path


def load_mesh_with_uvs(obj_path):
    """
    Load OBJ file and return vertices, triangles, and per-triangle UVs.

    Returns
    -------
    X : ndarray, shape (n_verts, 3)
        Vertex positions.
    T : ndarray, shape (n_tris, 3)
        Triangle indices.
    uv_per_tri : ndarray, shape (n_tris, 3, 2)
        UV coordinates per triangle corner.
    """
    V, F, UV, TF, *_ = readOBJ(str(obj_path))

    n_tris = F.shape[0]

    # Check if we have UV data
    if UV.shape[0] == 0 or TF.shape[0] == 0:
        raise ValueError("OBJ file has no UV coordinates")

    # Convert to per-triangle UVs
    uv_per_tri = np.zeros((n_tris, 3, 2), dtype=np.float64)
    for i in range(n_tris):
        for j in range(3):
            uv_idx = TF[i, j]
            if uv_idx < 0 or uv_idx >= UV.shape[0]:
                raise ValueError(f"Invalid UV index {uv_idx} in triangle {i}")
            uv_per_tri[i, j, :] = UV[uv_idx, :2]  # Take only first 2 components

    return V, F, uv_per_tri


def main():
    parser = argparse.ArgumentParser(
        description='Extract quad mesh from triangle mesh using RSP + libQEx'
    )
    parser.add_argument('mesh', type=str, help='Path to input OBJ mesh file')
    parser.add_argument('-o', '--output', type=str, default='Results/',
                        help='Output directory (default: Results/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for UVs (controls quad density)')
    parser.add_argument('--skip-rsp', action='store_true',
                        help='Skip RSP step, use existing _param.obj file')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess mesh with PyMeshLab before running RSP')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mesh = args.mesh
    mesh_name = Path(input_mesh).stem

    # Preprocess mesh if requested
    if args.preprocess and not args.skip_rsp:
        input_mesh = preprocess_mesh_if_needed(input_mesh, output_dir, verbose=args.verbose)
        mesh_name = Path(input_mesh).stem  # Update name if preprocessing created new file

    param_path = output_dir / f"{mesh_name}_param.obj"

    # Run RSP pipeline (unless skipped)
    if not args.skip_rsp:
        print("Running RSP parameterization pipeline...")
        param_path = run_rsp(str(input_mesh), output_dir, verbose=args.verbose)
        print(f"RSP output: {param_path}")
    else:
        if not param_path.exists():
            print(f"Error: --skip-rsp specified but {param_path} not found")
            return 1
        print(f"Using existing RSP output: {param_path}")

    # Load the parameterized mesh
    print("Loading parameterized mesh...")
    X, T, uv_per_tri = load_mesh_with_uvs(param_path)

    if args.verbose:
        print(f"  Vertices: {X.shape[0]}, Triangles: {T.shape[0]}")
        uv_flat = uv_per_tri.reshape(-1, 2)
        print(f"  UV range: [{uv_flat.min():.3f}, {uv_flat.max():.3f}]")

    # Scale UVs if requested
    if args.scale != 1.0:
        uv_per_tri = uv_per_tri * args.scale
        print(f"Scaled UVs by {args.scale}")

    # Extract quads
    print("Extracting quad mesh with libQEx...")
    try:
        quad_verts, quad_faces, tri_faces = extract_quads(X, T, uv_per_tri)
        n_tris = len(tri_faces) if tri_faces is not None else 0
        print(f"Result: {len(quad_verts)} vertices, {len(quad_faces)} quads" +
              (f", {n_tris} hole-fill triangles" if n_tris > 0 else ""))

        if len(quad_faces) == 0:
            print("Warning: No quads extracted. UV range may be too small.")
            print("Try using --scale to increase UV coordinates.")
            return 1

        # Save quad mesh
        output_path = output_dir / f"{mesh_name}_quads.obj"
        save_quad_obj(output_path, quad_verts, quad_faces, tri_faces)
        print(f"Saved quad mesh to: {output_path}")

    except Exception as e:
        print(f"Error extracting quads: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
