#!/usr/bin/env python3
"""
Benchmark quad mesh quality across different scale values.

This script runs the full pipeline (RSP + QEx) at multiple scales
and reports quality metrics to establish baselines for quantization.

Usage:
    python scripts/benchmark_quad_quality.py Mesh/sphere320.obj --scales 5,10,20,30
    python scripts/benchmark_quad_quality.py Mesh/*.obj --scales 10,20
"""

import argparse
import sys
import os
import tempfile
import glob
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rectangular_surface_parameterization.quad_extraction.quad_quality import (
    load_obj_quads,
    measure_mesh_quality,
    MeshQualityReport,
)


def run_pipeline(mesh_path: str, output_dir: str, scale: float,
                 verbose: bool = False) -> str:
    """
    Run RSP + QEx pipeline and return path to output quad mesh.
    """
    import subprocess

    mesh_name = Path(mesh_path).stem

    # Normalize output dir for Windows, create scale-specific subdir
    scale_dir = Path(output_dir).resolve() / f"scale_{int(scale)}"
    scale_dir.mkdir(parents=True, exist_ok=True)
    output_dir_norm = str(scale_dir)

    # Run extract_quads.py
    cmd = [
        sys.executable, "extract_quads.py",
        mesh_path,
        "-o", output_dir_norm,
        "--scale", str(scale),
    ]

    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if verbose and result.stdout:
        print(f"  stdout: {result.stdout[:200]}")
    if verbose and result.stderr:
        print(f"  stderr: {result.stderr[:200]}")

    # Check for output file regardless of return code
    # (matplotlib warnings can cause non-zero exit)
    quad_path = Path(output_dir_norm) / f"{mesh_name}_quads.obj"
    if quad_path.exists():
        return str(quad_path)

    if result.returncode != 0:
        if verbose:
            print(f"  Pipeline failed with code {result.returncode}")
        return None

    return None


def benchmark_single_mesh(mesh_path: str, scales: list, output_dir: str,
                          verbose: bool = False) -> dict:
    """
    Benchmark a single mesh at multiple scales.

    Returns dict mapping scale -> MeshQualityReport
    """
    results = {}
    mesh_name = Path(mesh_path).stem

    print(f"\n{'='*60}")
    print(f"Benchmarking: {mesh_name}")
    print(f"{'='*60}")

    for scale in scales:
        print(f"\n  Scale {scale}...")

        quad_path = run_pipeline(mesh_path, output_dir, scale, verbose)

        if quad_path and os.path.exists(quad_path):
            try:
                vertices, quads = load_obj_quads(quad_path)
                report = measure_mesh_quality(vertices, quads)
                results[scale] = report
                print(f"    Quads: {report.num_quads}, "
                      f"Min angle: {report.min_angle:.1f}°, "
                      f"Irregular: {report.num_irregular}")
            except Exception as e:
                print(f"    Error: {e}")
                results[scale] = None
        else:
            print(f"    Failed to generate quad mesh")
            results[scale] = None

    return results


def print_comparison_table(all_results: dict):
    """
    Print a comparison table of all benchmarks.
    """
    print("\n")
    print("=" * 100)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 100)

    # Header
    print(f"{'Mesh':<20} {'Scale':>6} {'Quads':>7} {'MinAng':>7} {'MaxAng':>7} "
          f"{'AngRMS':>7} {'AspMax':>7} {'JacMin':>7} {'Irreg':>6}")
    print("-" * 100)

    for mesh_name, scale_results in all_results.items():
        for scale, report in sorted(scale_results.items()):
            if report is None:
                print(f"{mesh_name:<20} {scale:>6} {'FAILED':>7}")
                continue

            print(f"{mesh_name:<20} {scale:>6} {report.num_quads:>7} "
                  f"{report.min_angle:>7.1f} {report.max_angle:>7.1f} "
                  f"{report.angle_deviation_rms:>7.2f} {report.max_aspect_ratio:>7.2f} "
                  f"{report.min_scaled_jacobian:>7.3f} {report.num_irregular:>6}")

    print("=" * 100)
    print("\nColumn legend:")
    print("  MinAng  = Worst (smallest) corner angle in degrees (ideal: 90°)")
    print("  MaxAng  = Worst (largest) corner angle in degrees (ideal: 90°)")
    print("  AngRMS  = RMS deviation from 90° (lower is better)")
    print("  AspMax  = Worst aspect ratio (ideal: 1.0)")
    print("  JacMin  = Minimum scaled Jacobian (ideal: 1.0, negative = inverted)")
    print("  Irreg   = Irregular interior vertices (valence ≠ 4)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark quad mesh quality at different scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_quad_quality.py Mesh/sphere320.obj --scales 10,20,30
  python scripts/benchmark_quad_quality.py Mesh/sphere320.obj Mesh/torus.obj --scales 10
  python scripts/benchmark_quad_quality.py "Mesh/*.obj" --scales 5,10,20
        """
    )

    parser.add_argument("meshes", nargs="+", help="Input mesh files (supports glob)")
    parser.add_argument("--scales", default="10,20,30",
                        help="Comma-separated scale values (default: 10,20,30)")
    parser.add_argument("-o", "--output", default="Results/benchmark",
                        help="Output directory (default: Results/benchmark)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Parse scales
    scales = [float(s) for s in args.scales.split(",")]

    # Expand glob patterns
    mesh_files = []
    for pattern in args.meshes:
        expanded = glob.glob(pattern)
        if expanded:
            mesh_files.extend(expanded)
        elif os.path.exists(pattern):
            mesh_files.append(pattern)
        else:
            print(f"Warning: No files match '{pattern}'")

    if not mesh_files:
        print("Error: No mesh files found")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"Benchmarking {len(mesh_files)} mesh(es) at scales: {scales}")
    print(f"Output directory: {args.output}")

    # Run benchmarks
    all_results = {}
    for mesh_path in mesh_files:
        mesh_name = Path(mesh_path).stem
        results = benchmark_single_mesh(
            mesh_path, scales, args.output, args.verbose
        )
        all_results[mesh_name] = results

    # Print comparison table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
