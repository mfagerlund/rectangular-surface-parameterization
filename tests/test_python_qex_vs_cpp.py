"""
Test script to compare Python QEx implementation against C++ libQEx.

This verifies the ray tracing implementation produces matching results.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rectangular_surface_parameterization.quad_extraction import extract_quads as python_extract
from rectangular_surface_parameterization.utils.libqex_wrapper import extract_quads as cpp_extract


def run_comparison(name, vertices, triangles, uvs_per_triangle, verbose=True):
    """Run both extractors and compare results."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Input: {len(vertices)} vertices, {len(triangles)} triangles")

    # Run Python implementation
    print("\n[Python QEx]")
    py_verts, py_quads, py_tris = python_extract(
        vertices, triangles, uvs_per_triangle, verbose=verbose
    )
    print(f"  Result: {len(py_verts)} vertices, {len(py_quads)} quads")

    # Run C++ implementation
    print("\n[C++ libQEx]")
    try:
        cpp_verts, cpp_quads, cpp_tris = cpp_extract(
            vertices, triangles, uvs_per_triangle, verbose=verbose
        )
        print(f"  Result: {len(cpp_verts)} vertices, {len(cpp_quads)} quads")

        # Compare
        print("\n[Comparison]")
        quad_diff = abs(len(py_quads) - len(cpp_quads))
        if quad_diff == 0:
            print(f"  Quads: MATCH ({len(py_quads)} quads)")
        else:
            match_pct = 100 * min(len(py_quads), len(cpp_quads)) / max(len(py_quads), len(cpp_quads))
            print(f"  Quads: Python={len(py_quads)}, C++={len(cpp_quads)} ({match_pct:.1f}% match)")

        return len(py_quads), len(cpp_quads)

    except FileNotFoundError as e:
        print(f"  C++ binary not found: {e}")
        print("  Skipping C++ comparison")
        return len(py_quads), None


def test_simple_square():
    """2 triangles forming a unit square with 3x3 UV grid.

    UV range [0, 3] has 4 integer points (0,1,2,3) along each axis.
    This gives a 4x4 grid = 16 vertices and 3x3 = 9 quads.
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    uvs = np.array([
        [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0]],
        [[0.0, 0.0], [3.0, 3.0], [0.0, 3.0]],
    ], dtype=np.float64)

    return run_comparison("Simple Square (3x3 UV, expect 9 quads)", vertices, triangles, uvs)


def test_larger_grid():
    """2 triangles with larger UV grid."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    uvs = np.array([
        [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]],
        [[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
    ], dtype=np.float64)

    return run_comparison("Simple Square (5x5 UV)", vertices, triangles, uvs)


def test_four_triangles():
    """4 triangles meeting at center point."""
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0: bottom-left
        [1.0, 0.0, 0.0],  # 1: bottom-right
        [1.0, 1.0, 0.0],  # 2: top-right
        [0.0, 1.0, 0.0],  # 3: top-left
        [0.5, 0.5, 0.0],  # 4: center
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 4],  # bottom
        [1, 2, 4],  # right
        [2, 3, 4],  # top
        [3, 0, 4],  # left
    ], dtype=np.int32)

    uvs = np.array([
        [[0.0, 0.0], [4.0, 0.0], [2.0, 2.0]],
        [[4.0, 0.0], [4.0, 4.0], [2.0, 2.0]],
        [[4.0, 4.0], [0.0, 4.0], [2.0, 2.0]],
        [[0.0, 4.0], [0.0, 0.0], [2.0, 2.0]],
    ], dtype=np.float64)

    return run_comparison("Four Triangles (4x4 UV)", vertices, triangles, uvs)


def test_mesh_file_if_exists():
    """Test with a real mesh file if available."""
    mesh_path = Path(__file__).parent.parent / "Mesh" / "sphere320.obj"
    if not mesh_path.exists():
        print(f"\nSkipping mesh test: {mesh_path} not found")
        return None, None

    # Load mesh
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path), process=False)
        vertices = np.array(mesh.vertices, dtype=np.float64)
        triangles = np.array(mesh.faces, dtype=np.int32)
    except ImportError:
        print("\nSkipping mesh test: trimesh not available")
        return None, None
    except Exception as e:
        print(f"\nSkipping mesh test: {e}")
        return None, None

    # Create simple planar projection UVs
    scale = 20.0
    uv_coords = vertices[:, :2] * scale

    # Build per-triangle UVs
    uvs_per_triangle = np.zeros((len(triangles), 3, 2), dtype=np.float64)
    for i, tri in enumerate(triangles):
        for j in range(3):
            uvs_per_triangle[i, j] = uv_coords[tri[j]]

    return run_comparison(f"sphere320 (scale={scale})", vertices, triangles, uvs_per_triangle)


def main():
    print("="*60)
    print("Python QEx vs C++ libQEx Comparison Test")
    print("="*60)

    results = []

    # Run tests
    results.append(("Simple Square 3x3", test_simple_square()))
    results.append(("Simple Square 5x5", test_larger_grid()))
    results.append(("Four Triangles", test_four_triangles()))
    results.append(("sphere320", test_mesh_file_if_exists()))

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for name, (py, cpp) in results:
        if py is None:
            continue
        if cpp is None:
            print(f"  {name}: Python={py} quads (C++ not available)")
        elif py == cpp:
            print(f"  {name}: MATCH ({py} quads)")
        else:
            match = 100 * min(py, cpp) / max(py, cpp)
            print(f"  {name}: Python={py}, C++={cpp} ({match:.1f}% match)")


if __name__ == "__main__":
    main()
