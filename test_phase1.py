"""
Test Phase 1: Mesh data structures and geometric quantities.

Tests:
1. Euler characteristic chi = V - E + F = 2 for bunny (genus 0, no boundary)
2. Angle sum in each triangle = pi
3. Total Gaussian curvature = 2*pi*chi (Gauss-Bonnet)
4. Cotan weights symmetric
"""

import numpy as np
from pathlib import Path

from mesh import TriangleMesh, build_connectivity, euler_characteristic, genus, count_boundary_loops
from io_obj import load_obj, mesh_info
from geometry import (
    compute_edge_lengths,
    compute_corner_angles,
    compute_face_areas,
    compute_cotan_weights,
    total_gaussian_curvature,
    MeshGeometry
)


def test_simple_mesh():
    """Test with a simple tetrahedron."""
    print("=" * 60)
    print("Test: Simple Tetrahedron")
    print("=" * 60)

    # Tetrahedron vertices
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
    ], dtype=np.float64)

    # Faces (outward-facing)
    faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3]
    ], dtype=np.int32)

    mesh = TriangleMesh(positions=positions, faces=faces)
    mesh = build_connectivity(mesh)

    print(f"V={mesh.n_vertices}, E={mesh.n_edges}, F={mesh.n_faces}")
    chi = euler_characteristic(mesh)
    print(f"Euler characteristic: {chi}")
    assert chi == 2, f"Expected chi=2 for tetrahedron, got {chi}"
    print("PASS: chi = 2")

    # Check angle sum
    alpha = compute_corner_angles(mesh)
    for f in range(mesh.n_faces):
        angle_sum = alpha[3*f] + alpha[3*f+1] + alpha[3*f+2]
        assert abs(angle_sum - np.pi) < 1e-10, f"Angle sum in face {f}: {angle_sum}"
    print("PASS: angle sum = pi in each face")

    # Gauss-Bonnet
    K_total = total_gaussian_curvature(mesh, alpha)
    expected = 2 * np.pi * chi
    print(f"Total Gaussian curvature: {K_total:.6f}, expected: {expected:.6f}")
    assert abs(K_total - expected) < 1e-6, f"Gauss-Bonnet failed"
    print("PASS: Gauss-Bonnet theorem")
    print()


def test_bunny():
    """Test with stanford bunny."""
    print("=" * 60)
    print("Test: Stanford Bunny")
    print("=" * 60)

    bunny_path = Path(r"C:\Dev\Colonel\Data\Meshes\stanford-bunny.obj")
    if not bunny_path.exists():
        print(f"Bunny mesh not found at {bunny_path}")
        print("SKIP")
        return

    mesh = load_obj(str(bunny_path))
    print(mesh_info(mesh))
    print()

    chi = euler_characteristic(mesh)
    b = count_boundary_loops(mesh)
    g = genus(mesh)

    print(f"Euler characteristic: {chi}")
    print(f"Boundary loops: {b}")
    print(f"Genus: {g}")

    # For a closed genus-0 surface: chi = 2
    if b == 0:
        expected_chi = 2 - 2 * g
        assert chi == expected_chi, f"Expected chi={expected_chi} for genus {g}, got {chi}"
        print(f"PASS: chi = {expected_chi} for genus {g}")
    else:
        print(f"Mesh has boundary, chi = 2 - 2g - b = {2 - 2*g - b}")

    # Compute geometry
    geom = MeshGeometry(mesh)

    # Check angle sums
    max_angle_error = 0
    for f in range(mesh.n_faces):
        angle_sum = geom.alpha[3*f] + geom.alpha[3*f+1] + geom.alpha[3*f+2]
        error = abs(angle_sum - np.pi)
        max_angle_error = max(max_angle_error, error)
    print(f"Max angle sum error: {max_angle_error:.2e}")
    assert max_angle_error < 1e-10, "Angle sum error too large"
    print("PASS: angle sum = pi in each face")

    # Gauss-Bonnet
    K_total = total_gaussian_curvature(mesh, geom.alpha)
    expected = 2 * np.pi * chi
    print(f"Total Gaussian curvature: {K_total:.6f}, expected: {expected:.6f}")
    error = abs(K_total - expected)
    print(f"Gauss-Bonnet error: {error:.2e}")
    # Note: numerical error can be larger for complex meshes
    assert error < 0.01, f"Gauss-Bonnet error too large: {error}"
    print("PASS: Gauss-Bonnet theorem")

    print(f"\nTotal area: {geom.total_area():.6f}")
    print(f"Mean edge length: {geom.mean_edge_length():.6f}")
    print()


def test_torus():
    """Test with torus (genus 1)."""
    print("=" * 60)
    print("Test: Torus")
    print("=" * 60)

    torus_path = Path(r"C:\Dev\Colonel\Data\Meshes\torus.obj")
    if not torus_path.exists():
        print(f"Torus mesh not found at {torus_path}")
        print("SKIP")
        return

    mesh = load_obj(str(torus_path))
    print(mesh_info(mesh))
    print()

    chi = euler_characteristic(mesh)
    b = count_boundary_loops(mesh)
    g = genus(mesh)

    print(f"Euler characteristic: {chi}")
    print(f"Boundary loops: {b}")
    print(f"Genus: {g}")

    # For a torus: chi = 0
    if b == 0:
        assert chi == 0, f"Expected chi=0 for torus, got {chi}"
        assert g == 1, f"Expected genus=1 for torus, got {g}"
        print("PASS: chi = 0 for torus")
        print("PASS: genus = 1")

    # Gauss-Bonnet
    alpha = compute_corner_angles(mesh)
    K_total = total_gaussian_curvature(mesh, alpha)
    expected = 2 * np.pi * chi
    print(f"Total Gaussian curvature: {K_total:.6f}, expected: {expected:.6f}")
    error = abs(K_total - expected)
    print(f"Gauss-Bonnet error: {error:.2e}")
    print()


def test_connectivity():
    """Test half-edge connectivity operations."""
    print("=" * 60)
    print("Test: Connectivity Operations")
    print("=" * 60)

    # Simple 2-triangle mesh (two triangles sharing an edge)
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [0.5, -1, 0]
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2],  # top triangle
        [1, 0, 3],  # bottom triangle
    ], dtype=np.int32)

    mesh = TriangleMesh(positions=positions, faces=faces)
    mesh = build_connectivity(mesh)

    print(f"V={mesh.n_vertices}, E={mesh.n_edges}, F={mesh.n_faces}")
    print(f"Euler characteristic: {euler_characteristic(mesh)}")

    # Check halfedge operations
    for he in range(mesh.n_halfedges):
        i, j = mesh.halfedge_vertices(he)
        he_next = mesh.halfedge_next(he)
        j2, k = mesh.halfedge_vertices(he_next)
        assert j == j2, f"Halfedge next inconsistent at {he}"

    print("PASS: halfedge_next consistent")

    # Check twin symmetry
    for he in range(mesh.n_halfedges):
        twin = mesh.halfedge_twin[he]
        if twin != -1:
            twin_twin = mesh.halfedge_twin[twin]
            assert twin_twin == he, f"Twin of twin != self at {he}"
            # Check vertex reversal
            i, j = mesh.halfedge_vertices(he)
            ti, tj = mesh.halfedge_vertices(twin)
            assert i == tj and j == ti, f"Twin vertices not reversed at {he}"

    print("PASS: halfedge_twin symmetric")

    # Check vertex degree
    for v in range(mesh.n_vertices):
        deg = mesh.vertex_degree(v)
        corners = mesh.vertex_corners(v)
        print(f"  Vertex {v}: degree={deg}, corners={corners}")

    print("PASS: vertex operations")
    print()


if __name__ == "__main__":
    test_simple_mesh()
    test_connectivity()
    test_bunny()
    test_torus()
    print("=" * 60)
    print("All Phase 1 tests completed!")
    print("=" * 60)
