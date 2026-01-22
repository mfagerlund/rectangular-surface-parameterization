"""
Automated testing for Corman-Crane rectangular parameterization.

Tests the full pipeline on various mesh types.
"""

import os
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mesh import TriangleMesh, euler_characteristic, genus
from io_obj import load_obj, save_obj
from corman_crane import compute_rectangular_parameterization
from uv_recovery import compute_uv_quality
from quad_extract import extract_quads, QuadMesh


@dataclass
class TestResult:
    """Result of a mesh test."""
    mesh_name: str
    vertices: int
    faces: int
    mesh_genus: int
    flipped_count: int
    angle_error_mean: float
    converged: bool
    uv_fill_ratio: float = 0.0  # Ratio of UV area to bounding box area
    quad_vertices: int = 0
    quad_faces: int = 0
    error: Optional[str] = None

    def passed(self, max_flip_rate: float = 0.0) -> bool:
        """Test passes if few flips and reasonable angle error."""
        if self.error is not None:
            return False
        if self.angle_error_mean >= 30.0:  # degrees
            return False
        # Allow max_flip_rate fraction of faces to be flipped
        # (genus > 0 surfaces may have some flips due to periodic BC approximation)
        flip_rate = self.flipped_count / max(self.faces, 1)
        return flip_rate <= max_flip_rate


def test_mesh(
    mesh_path: str,
    output_dir: str = "output",
    target_quads: int = 50,
    verbose: bool = True
) -> TestResult:
    """
    Run full pipeline test on a mesh.

    Args:
        mesh_path: Path to input OBJ file
        output_dir: Directory for output files
        target_quads: Number of quads to extract
        verbose: Print progress

    Returns:
        TestResult object
    """
    mesh_name = Path(mesh_path).stem

    try:
        # Load mesh
        mesh = load_obj(mesh_path)

        result = TestResult(
            mesh_name=mesh_name,
            vertices=mesh.n_vertices,
            faces=mesh.n_faces,
            mesh_genus=genus(mesh),
            flipped_count=-1,
            angle_error_mean=-1,
            converged=False
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing: {mesh_name}")
            print(f"  Vertices: {mesh.n_vertices}")
            print(f"  Faces: {mesh.n_faces}")
            print(f"  Genus: {result.mesh_genus}")

        # Compute parameterization
        corner_uvs = compute_rectangular_parameterization(mesh, verbose=verbose)

        # Quality metrics
        quality = compute_uv_quality(mesh, corner_uvs)
        result.flipped_count = quality['flipped_count']
        result.angle_error_mean = np.degrees(quality['angle_error_mean'])
        result.converged = True

        # Compute UV fill ratio (area covered vs bounding box)
        uv_min = corner_uvs.min(axis=0)
        uv_max = corner_uvs.max(axis=0)
        bbox_area = (uv_max[0] - uv_min[0]) * (uv_max[1] - uv_min[1])
        total_uv_area = 0.0
        for f in range(mesh.n_faces):
            uv0, uv1, uv2 = corner_uvs[3*f], corner_uvs[3*f+1], corner_uvs[3*f+2]
            e1 = uv1 - uv0
            e2 = uv2 - uv0
            total_uv_area += 0.5 * abs(e1[0]*e2[1] - e1[1]*e2[0])
        result.uv_fill_ratio = total_uv_area / bbox_area if bbox_area > 0 else 0

        if verbose:
            print(f"  Flipped: {result.flipped_count}")
            print(f"  Angle error: {result.angle_error_mean:.2f} deg")
            print(f"  UV fill: {result.uv_fill_ratio*100:.1f}%")

        # Save UV output
        os.makedirs(output_dir, exist_ok=True)
        uv_path = os.path.join(output_dir, f"{mesh_name}_uv.obj")
        save_obj(uv_path, mesh, corner_uvs)

        # Extract quads
        if target_quads > 0:
            quad_path = os.path.join(output_dir, f"{mesh_name}_quads.obj")
            # Use periodic mode for genus > 0 surfaces
            quad_mesh = extract_quads(
                mesh, corner_uvs,
                target_quads=target_quads,
                output_path=quad_path,
                periodic=(result.mesh_genus > 0),
                verbose=verbose
            )
            result.quad_vertices = quad_mesh.n_vertices
            result.quad_faces = quad_mesh.n_faces

            if verbose:
                print(f"  Quads: {result.quad_faces}")

    except Exception as e:
        result = TestResult(
            mesh_name=mesh_name,
            vertices=0,
            faces=0,
            mesh_genus=0,
            flipped_count=-1,
            angle_error_mean=-1,
            converged=False,
            error=str(e)
        )
        if verbose:
            print(f"  ERROR: {e}")

    return result


def run_test_suite(
    mesh_dir: str = "C:/Dev/Colonel/Data/Meshes",
    output_dir: str = "output",
    verbose: bool = True
) -> list:
    """
    Run test suite on available meshes.

    Args:
        mesh_dir: Directory containing test meshes
        output_dir: Directory for output files
        verbose: Print progress

    Returns:
        List of TestResult objects
    """
    # Test meshes and their expected properties
    test_cases = [
        ("sphere320.obj", 0, 50),    # genus 0, 50 target quads
        ("torus.obj", 1, 100),        # genus 1, 100 target quads
    ]

    results = []

    for mesh_file, expected_genus, target_quads in test_cases:
        mesh_path = os.path.join(mesh_dir, mesh_file)

        if not os.path.exists(mesh_path):
            if verbose:
                print(f"Skipping {mesh_file}: not found")
            continue

        result = test_mesh(
            mesh_path,
            output_dir=output_dir,
            target_quads=target_quads,
            verbose=verbose
        )

        # Validate genus
        if result.mesh_genus != expected_genus:
            if verbose:
                print(f"  WARNING: Expected genus {expected_genus}, got {result.mesh_genus}")

        results.append(result)

    return results


def print_summary(results: list):
    """Print summary of test results."""
    print("\n" + "=" * 78)
    print("TEST SUMMARY")
    print("=" * 78)

    header = f"{'Mesh':<16} {'V':>5} {'F':>5} {'G':>2} {'Flips':>5} {'Err':>6} {'Fill':>5} {'Quads':>5} {'Status':>6}"
    print(header)
    print("-" * 78)

    passed = 0
    failed = 0

    for r in results:
        # Allow 1% flips for genus > 0 surfaces (periodic BC approximation)
        max_flip_rate = 0.01 if r.mesh_genus > 0 else 0.0

        if r.error:
            status = "ERROR"
            err_str = "N/A"
            fill_str = "N/A"
        elif r.passed(max_flip_rate):
            status = "PASS"
            passed += 1
            err_str = f"{r.angle_error_mean:.1f}"
            fill_str = f"{r.uv_fill_ratio*100:.0f}%"
        else:
            status = "FAIL"
            failed += 1
            err_str = f"{r.angle_error_mean:.1f}"
            fill_str = f"{r.uv_fill_ratio*100:.0f}%"

        row = f"{r.mesh_name:<16} {r.vertices:>5} {r.faces:>5} {r.mesh_genus:>2} {r.flipped_count:>5} {err_str:>6} {fill_str:>5} {r.quad_faces:>5} {status:>6}"
        print(row)

    print("-" * 78)
    print(f"Passed: {passed}/{len(results)}, Failed: {failed}/{len(results)}")

    # Warn about low fill ratios
    low_fill = [r for r in results if r.uv_fill_ratio < 0.8 and not r.error]
    if low_fill:
        print(f"\nWARNING: {len(low_fill)} mesh(es) have UV fill < 80% (fragmented UV domain)")

    return passed == len(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Corman-Crane parameterization")
    parser.add_argument("--mesh-dir", default="C:/Dev/Colonel/Data/Meshes",
                        help="Directory containing test meshes")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Quiet mode")
    parser.add_argument("--mesh", help="Test single mesh file")

    args = parser.parse_args()

    if args.mesh:
        # Test single mesh
        result = test_mesh(args.mesh, args.output_dir, verbose=not args.quiet)
        # Allow 1% flips for genus > 0 surfaces
        max_flip_rate = 0.01 if result.mesh_genus > 0 else 0.0
        if result.passed(max_flip_rate):
            print("\nPASS")
            sys.exit(0)
        else:
            print(f"\nFAIL: {result.error or 'flips or high error'}")
            sys.exit(1)
    else:
        # Run full test suite
        results = run_test_suite(
            mesh_dir=args.mesh_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )

        all_passed = print_summary(results)
        sys.exit(0 if all_passed else 1)
