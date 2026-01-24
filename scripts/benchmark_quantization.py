#!/usr/bin/env python3
"""
Benchmark quantization vs scaling.

Compares quad mesh quality between:
1. Current approach: --scale N (just multiply UVs)
2. Quantization: snap singularities to integers, then extract

This should show improved quality on meshes with many singularities (like pig).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tempfile


def load_param_mesh(obj_path: str):
    """Load a parameterized mesh with UVs."""
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vt':
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == 'f':
                face = []
                face_uv = []
                for p in parts[1:]:
                    if '/' in p:
                        v_idx, vt_idx = p.split('/')[:2]
                        face.append(int(v_idx) - 1)
                        if vt_idx:
                            face_uv.append(int(vt_idx) - 1)
                    else:
                        face.append(int(p) - 1)

                if len(face) == 3:
                    faces.append(face)
                    if len(face_uv) == 3:
                        face_uvs.append(face_uv)

    vertices = np.array(vertices)
    uvs = np.array(uvs) if uvs else None
    faces = np.array(faces)

    if uvs is not None and face_uvs:
        face_uvs = np.array(face_uvs)
        uv_per_triangle = uvs[face_uvs]
    else:
        uv_per_triangle = None

    return vertices, faces, uv_per_triangle


def save_param_mesh(filepath: str, vertices: np.ndarray, triangles: np.ndarray,
                    uv_per_triangle: np.ndarray):
    """Save mesh with per-triangle UVs in OBJ format."""
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write UVs (flattened)
        uv_idx = 1
        for ti in range(len(triangles)):
            for corner in range(3):
                uv = uv_per_triangle[ti, corner]
                f.write(f"vt {uv[0]} {uv[1]}\n")

        # Write faces with UV indices
        uv_idx = 1
        for ti, tri in enumerate(triangles):
            v1, v2, v3 = tri + 1  # OBJ is 1-indexed
            t1 = uv_idx
            t2 = uv_idx + 1
            t3 = uv_idx + 2
            f.write(f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}\n")
            uv_idx += 3


def extract_quads_from_uv(vertices: np.ndarray, triangles: np.ndarray,
                          uv_per_triangle: np.ndarray, verbose: bool = False):
    """Extract quads using Python QEx."""
    from rectangular_surface_parameterization.quad_extraction import extract_quads

    quad_verts, quad_faces, tri_faces = extract_quads(
        vertices, triangles, uv_per_triangle,
        fill_holes=True, verbose=verbose
    )

    return quad_verts, quad_faces


def benchmark_mesh(mesh_path: str, scale: float, verbose: bool = False):
    """
    Benchmark both approaches on a single mesh.

    Returns dict with quality metrics for both methods.
    """
    from rectangular_surface_parameterization.quad_extraction.quad_quality import (
        measure_mesh_quality, MeshQualityReport
    )
    from rectangular_surface_parameterization.quad_extraction.quantization import (
        quantize_uv
    )

    print(f"\n{'='*60}")
    print(f"Benchmarking: {Path(mesh_path).stem} at scale {scale}")
    print(f"{'='*60}")

    # Load parameterized mesh
    vertices, triangles, uv_per_triangle = load_param_mesh(mesh_path)

    if uv_per_triangle is None:
        print("  Error: No UVs in mesh")
        return None

    print(f"  Loaded: {len(vertices)} verts, {len(triangles)} tris")

    results = {}

    # Method 1: Scaling (current approach)
    print("\n  [Method 1: Scaling]")
    uv_scaled = uv_per_triangle * scale

    try:
        quad_verts, quad_faces = extract_quads_from_uv(
            vertices, triangles, uv_scaled, verbose=False
        )
        if len(quad_faces) > 0:
            report = measure_mesh_quality(quad_verts, quad_faces)
            results['scaling'] = report
            print(f"    Quads: {report.num_quads}")
            print(f"    Min angle: {report.min_angle:.1f}°")
            print(f"    Max aspect: {report.max_aspect_ratio:.2f}")
            print(f"    Min Jacobian: {report.min_scaled_jacobian:.3f}")
        else:
            print("    No quads extracted")
            results['scaling'] = None
    except Exception as e:
        print(f"    Error: {e}")
        results['scaling'] = None

    # Method 2: Quantization (new approach)
    print("\n  [Method 2: Quantization]")

    try:
        quant_result = quantize_uv(
            vertices, triangles, uv_per_triangle,
            target_scale=scale, verbose=verbose
        )

        print(f"    Singularities: {quant_result.num_singularities}")
        print(f"    On integers: {quant_result.singularities_on_integer}")

        quad_verts, quad_faces = extract_quads_from_uv(
            vertices, triangles, quant_result.uv_grid, verbose=False
        )

        if len(quad_faces) > 0:
            report = measure_mesh_quality(quad_verts, quad_faces)
            results['quantization'] = report
            print(f"    Quads: {report.num_quads}")
            print(f"    Min angle: {report.min_angle:.1f}°")
            print(f"    Max aspect: {report.max_aspect_ratio:.2f}")
            print(f"    Min Jacobian: {report.min_scaled_jacobian:.3f}")
        else:
            print("    No quads extracted")
            results['quantization'] = None

    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        results['quantization'] = None

    return results


def print_comparison(all_results: dict):
    """Print comparison table."""
    print("\n")
    print("=" * 100)
    print("QUANTIZATION vs SCALING COMPARISON")
    print("=" * 100)

    print(f"\n{'Mesh':<20} {'Method':<12} {'Quads':>7} {'MinAng':>7} {'MaxAng':>7} "
          f"{'AspMax':>7} {'JacMin':>7}")
    print("-" * 100)

    for mesh_name, results in all_results.items():
        for method in ['scaling', 'quantization']:
            report = results.get(method)
            if report is None:
                print(f"{mesh_name:<20} {method:<12} {'FAILED':>7}")
            else:
                print(f"{mesh_name:<20} {method:<12} {report.num_quads:>7} "
                      f"{report.min_angle:>7.1f} {report.max_angle:>7.1f} "
                      f"{report.max_aspect_ratio:>7.2f} {report.min_scaled_jacobian:>7.3f}")

    print("=" * 100)

    # Summary
    print("\nSummary (Quantization vs Scaling improvements):")
    for mesh_name, results in all_results.items():
        scaling = results.get('scaling')
        quant = results.get('quantization')
        if scaling and quant:
            angle_diff = quant.min_angle - scaling.min_angle
            aspect_diff = scaling.max_aspect_ratio - quant.max_aspect_ratio
            jac_diff = quant.min_scaled_jacobian - scaling.min_scaled_jacobian

            angle_emoji = "+" if angle_diff > 0 else ""
            aspect_emoji = "+" if aspect_diff > 0 else ""
            jac_emoji = "+" if jac_diff > 0 else ""

            print(f"  {mesh_name}: MinAng {angle_emoji}{angle_diff:.1f}°, "
                  f"AspMax {aspect_emoji}{aspect_diff:.2f}, "
                  f"JacMin {jac_emoji}{jac_diff:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare quantization vs scaling for quad extraction"
    )
    parser.add_argument("meshes", nargs="+", help="Parameterized meshes (*_param.obj)")
    parser.add_argument("--scale", type=float, default=30.0, help="Scale factor")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    all_results = {}
    for mesh_path in args.meshes:
        mesh_name = Path(mesh_path).stem.replace('_param', '')
        results = benchmark_mesh(mesh_path, args.scale, args.verbose)
        if results:
            all_results[mesh_name] = results

    if all_results:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
