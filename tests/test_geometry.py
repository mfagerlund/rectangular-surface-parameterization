"""
Pytest tests for geometry.py - Stage 1 verification.

Tests geometric computations against known expected values.
Run with: pytest tests/test_geometry.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh import TriangleMesh, build_connectivity, euler_characteristic, genus, count_boundary_loops
from geometry import (
    compute_edge_lengths,
    compute_corner_angles,
    compute_face_areas,
    compute_face_normals,
    compute_cotan_weights,
    compute_halfedge_cotan_weights,
    angle_defect,
    total_gaussian_curvature,
    MeshGeometry,
)


# =============================================================================
# Test Fixtures - Known Shapes
# =============================================================================

@pytest.fixture
def equilateral_triangle():
    """Single equilateral triangle with side length 1."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


@pytest.fixture
def right_triangle_345():
    """3-4-5 right triangle (angle at origin is 90 degrees)."""
    positions = np.array([
        [0.0, 0.0, 0.0],  # right angle here
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


@pytest.fixture
def isosceles_triangle():
    """Isosceles triangle with two equal sides of length sqrt(2), base 2."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],  # apex
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


@pytest.fixture
def unit_square_two_triangles():
    """Unit square split into two triangles."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


@pytest.fixture
def regular_tetrahedron():
    """Regular tetrahedron (closed, genus 0)."""
    # Regular tetrahedron with vertices at unit distance from origin
    positions = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    # Faces with outward normals
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


@pytest.fixture
def cube_triangulated():
    """Cube triangulated into 12 triangles (closed, genus 0)."""
    positions = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ], dtype=np.float64)
    # 6 faces, each split into 2 triangles (12 total)
    faces = np.array([
        # Bottom (z=0)
        [0, 2, 1],
        [0, 3, 2],
        # Top (z=1)
        [4, 5, 6],
        [4, 6, 7],
        # Front (y=0)
        [0, 1, 5],
        [0, 5, 4],
        # Back (y=1)
        [3, 6, 2],
        [3, 7, 6],
        # Left (x=0)
        [0, 4, 7],
        [0, 7, 3],
        # Right (x=1)
        [1, 2, 6],
        [1, 6, 5],
    ], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


@pytest.fixture
def octahedron():
    """Regular octahedron (closed, genus 0)."""
    positions = np.array([
        [1, 0, 0],   # 0: +X
        [-1, 0, 0],  # 1: -X
        [0, 1, 0],   # 2: +Y
        [0, -1, 0],  # 3: -Y
        [0, 0, 1],   # 4: +Z
        [0, 0, -1],  # 5: -Z
    ], dtype=np.float64)
    # 8 triangular faces
    faces = np.array([
        [0, 2, 4],  # +X +Y +Z
        [2, 1, 4],  # -X +Y +Z
        [1, 3, 4],  # -X -Y +Z
        [3, 0, 4],  # +X -Y +Z
        [2, 0, 5],  # +X +Y -Z
        [1, 2, 5],  # -X +Y -Z
        [3, 1, 5],  # -X -Y -Z
        [0, 3, 5],  # +X -Y -Z
    ], dtype=np.int32)
    mesh = TriangleMesh(positions=positions, faces=faces)
    return build_connectivity(mesh)


# =============================================================================
# Triangle Angle Tests
# =============================================================================

class TestTriangleAngles:
    """Test corner angle computation for single triangles."""

    def test_equilateral_angles(self, equilateral_triangle):
        """Equilateral triangle: all angles = 60 degrees = pi/3."""
        mesh = equilateral_triangle
        alpha = compute_corner_angles(mesh)

        expected_angle = np.pi / 3  # 60 degrees
        for i in range(3):
            assert abs(alpha[i] - expected_angle) < 1e-10, \
                f"Corner {i}: expected {np.degrees(expected_angle):.2f} deg, got {np.degrees(alpha[i]):.2f} deg"

    def test_right_triangle_angles(self, right_triangle_345):
        """3-4-5 right triangle: angles are 90, ~53.13, ~36.87 degrees."""
        mesh = right_triangle_345
        alpha = compute_corner_angles(mesh)

        # Corner 0 is at origin, should be 90 degrees
        assert abs(alpha[0] - np.pi/2) < 1e-10, \
            f"Right angle: expected 90 deg, got {np.degrees(alpha[0]):.2f} deg"

        # Other angles: atan(4/3) and atan(3/4)
        expected_1 = np.arctan(4/3)  # ~53.13 deg
        expected_2 = np.arctan(3/4)  # ~36.87 deg

        # Angles at corners 1 and 2 (order depends on face winding)
        angles_sorted = sorted([alpha[1], alpha[2]])
        expected_sorted = sorted([expected_1, expected_2])

        assert abs(angles_sorted[0] - expected_sorted[0]) < 1e-10
        assert abs(angles_sorted[1] - expected_sorted[1]) < 1e-10

    def test_isosceles_angles(self, isosceles_triangle):
        """Isosceles triangle: two base angles are equal."""
        mesh = isosceles_triangle
        alpha = compute_corner_angles(mesh)

        # Corners 0 and 1 are base angles (equal)
        # Corner 2 is apex
        assert abs(alpha[0] - alpha[1]) < 1e-10, \
            f"Base angles should be equal: {np.degrees(alpha[0]):.2f} vs {np.degrees(alpha[1]):.2f}"

    def test_angle_sum_equals_pi(self, equilateral_triangle, right_triangle_345, isosceles_triangle):
        """Sum of angles in any triangle = pi."""
        for mesh in [equilateral_triangle, right_triangle_345, isosceles_triangle]:
            alpha = compute_corner_angles(mesh)
            angle_sum = np.sum(alpha)
            assert abs(angle_sum - np.pi) < 1e-10, \
                f"Angle sum should be pi, got {angle_sum}"


class TestTriangleAngleSumAllFaces:
    """Test that angle sum = pi for EVERY face in multi-face meshes."""

    def test_two_triangles_angle_sum(self, unit_square_two_triangles):
        """Each triangle in the unit square has angle sum = pi."""
        mesh = unit_square_two_triangles
        alpha = compute_corner_angles(mesh)

        for f in range(mesh.n_faces):
            face_angles = alpha[3*f:3*f+3]
            angle_sum = np.sum(face_angles)
            assert abs(angle_sum - np.pi) < 1e-10, \
                f"Face {f}: angle sum = {angle_sum}, expected pi"

    def test_tetrahedron_angle_sum(self, regular_tetrahedron):
        """Each face of tetrahedron has angle sum = pi."""
        mesh = regular_tetrahedron
        alpha = compute_corner_angles(mesh)

        for f in range(mesh.n_faces):
            face_angles = alpha[3*f:3*f+3]
            angle_sum = np.sum(face_angles)
            assert abs(angle_sum - np.pi) < 1e-10, \
                f"Face {f}: angle sum = {angle_sum}, expected pi"

    def test_cube_angle_sum(self, cube_triangulated):
        """Each face of triangulated cube has angle sum = pi."""
        mesh = cube_triangulated
        alpha = compute_corner_angles(mesh)

        for f in range(mesh.n_faces):
            face_angles = alpha[3*f:3*f+3]
            angle_sum = np.sum(face_angles)
            assert abs(angle_sum - np.pi) < 1e-10, \
                f"Face {f}: angle sum = {angle_sum}, expected pi"

    def test_octahedron_angle_sum(self, octahedron):
        """Each face of octahedron has angle sum = pi."""
        mesh = octahedron
        alpha = compute_corner_angles(mesh)

        for f in range(mesh.n_faces):
            face_angles = alpha[3*f:3*f+3]
            angle_sum = np.sum(face_angles)
            assert abs(angle_sum - np.pi) < 1e-10, \
                f"Face {f}: angle sum = {angle_sum}, expected pi"


# =============================================================================
# Triangle Area Tests
# =============================================================================

class TestTriangleAreas:
    """Test face area computation."""

    def test_equilateral_area(self, equilateral_triangle):
        """Equilateral triangle with side 1: area = sqrt(3)/4."""
        mesh = equilateral_triangle
        areas = compute_face_areas(mesh)

        expected = np.sqrt(3) / 4
        assert abs(areas[0] - expected) < 1e-10, \
            f"Expected area {expected}, got {areas[0]}"

    def test_right_triangle_area(self, right_triangle_345):
        """3-4-5 right triangle: area = (1/2) * 3 * 4 = 6."""
        mesh = right_triangle_345
        areas = compute_face_areas(mesh)

        expected = 6.0
        assert abs(areas[0] - expected) < 1e-10, \
            f"Expected area {expected}, got {areas[0]}"

    def test_isosceles_area(self, isosceles_triangle):
        """Isosceles with base 2, height 1: area = 1."""
        mesh = isosceles_triangle
        areas = compute_face_areas(mesh)

        expected = 1.0  # (1/2) * base * height = (1/2) * 2 * 1
        assert abs(areas[0] - expected) < 1e-10, \
            f"Expected area {expected}, got {areas[0]}"

    def test_unit_square_area(self, unit_square_two_triangles):
        """Unit square split into triangles: total area = 1."""
        mesh = unit_square_two_triangles
        areas = compute_face_areas(mesh)

        total_area = np.sum(areas)
        assert abs(total_area - 1.0) < 1e-10, \
            f"Expected total area 1, got {total_area}"

        # Each triangle should have area 0.5
        for f in range(mesh.n_faces):
            assert abs(areas[f] - 0.5) < 1e-10, \
                f"Face {f}: expected area 0.5, got {areas[f]}"

    def test_all_areas_positive(self, regular_tetrahedron, cube_triangulated, octahedron):
        """All faces should have positive area."""
        for mesh in [regular_tetrahedron, cube_triangulated, octahedron]:
            areas = compute_face_areas(mesh)
            assert np.all(areas > 0), "All areas should be positive"


# =============================================================================
# Edge Length Tests
# =============================================================================

class TestEdgeLengths:
    """Test edge length computation."""

    def test_equilateral_edge_lengths(self, equilateral_triangle):
        """Equilateral triangle: all edges = 1."""
        mesh = equilateral_triangle
        ell = compute_edge_lengths(mesh)

        for e in range(mesh.n_edges):
            assert abs(ell[e] - 1.0) < 1e-10, \
                f"Edge {e}: expected length 1, got {ell[e]}"

    def test_right_triangle_edge_lengths(self, right_triangle_345):
        """3-4-5 right triangle: edges are 3, 4, 5."""
        mesh = right_triangle_345
        ell = compute_edge_lengths(mesh)

        lengths_sorted = sorted(ell)
        expected_sorted = [3.0, 4.0, 5.0]

        for i, (got, expected) in enumerate(zip(lengths_sorted, expected_sorted)):
            assert abs(got - expected) < 1e-10, \
                f"Edge {i}: expected {expected}, got {got}"

    def test_cube_edge_lengths(self, cube_triangulated):
        """Cube: edges are 1 (sides) or sqrt(2) (face diagonals)."""
        mesh = cube_triangulated
        ell = compute_edge_lengths(mesh)

        # All edges should be either 1 or sqrt(2)
        for e in range(mesh.n_edges):
            is_side = abs(ell[e] - 1.0) < 1e-10
            is_diagonal = abs(ell[e] - np.sqrt(2)) < 1e-10
            assert is_side or is_diagonal, \
                f"Edge {e}: length {ell[e]} is neither 1 nor sqrt(2)"

    def test_all_lengths_positive(self, regular_tetrahedron, cube_triangulated, octahedron):
        """All edges should have positive length."""
        for mesh in [regular_tetrahedron, cube_triangulated, octahedron]:
            ell = compute_edge_lengths(mesh)
            assert np.all(ell > 0), "All edge lengths should be positive"


# =============================================================================
# Euler Characteristic Tests
# =============================================================================

class TestEulerCharacteristic:
    """Test Euler characteristic: chi = V - E + F."""

    def test_single_triangle(self, equilateral_triangle):
        """Single triangle (open): chi = 3 - 3 + 1 = 1."""
        mesh = equilateral_triangle
        chi = euler_characteristic(mesh)
        assert chi == 1, f"Expected chi=1 for single triangle, got {chi}"

    def test_two_triangles(self, unit_square_two_triangles):
        """Two triangles sharing edge: chi = 4 - 5 + 2 = 1."""
        mesh = unit_square_two_triangles
        chi = euler_characteristic(mesh)
        # V=4, E=5, F=2 -> chi = 1
        assert chi == 1, f"Expected chi=1 for two triangles, got {chi}"

    def test_tetrahedron(self, regular_tetrahedron):
        """Tetrahedron (closed, genus 0): chi = 4 - 6 + 4 = 2."""
        mesh = regular_tetrahedron
        chi = euler_characteristic(mesh)
        assert chi == 2, f"Expected chi=2 for tetrahedron, got {chi}"

        # Also check V, E, F individually
        assert mesh.n_vertices == 4, f"Expected 4 vertices, got {mesh.n_vertices}"
        assert mesh.n_edges == 6, f"Expected 6 edges, got {mesh.n_edges}"
        assert mesh.n_faces == 4, f"Expected 4 faces, got {mesh.n_faces}"

    def test_cube(self, cube_triangulated):
        """Cube (closed, genus 0): chi = 8 - 18 + 12 = 2."""
        mesh = cube_triangulated
        chi = euler_characteristic(mesh)
        assert chi == 2, f"Expected chi=2 for cube, got {chi}"

        # Check V, E, F
        assert mesh.n_vertices == 8, f"Expected 8 vertices, got {mesh.n_vertices}"
        assert mesh.n_edges == 18, f"Expected 18 edges, got {mesh.n_edges}"
        assert mesh.n_faces == 12, f"Expected 12 faces, got {mesh.n_faces}"

    def test_octahedron(self, octahedron):
        """Octahedron (closed, genus 0): chi = 6 - 12 + 8 = 2."""
        mesh = octahedron
        chi = euler_characteristic(mesh)
        assert chi == 2, f"Expected chi=2 for octahedron, got {chi}"

        # Check V, E, F
        assert mesh.n_vertices == 6, f"Expected 6 vertices, got {mesh.n_vertices}"
        assert mesh.n_edges == 12, f"Expected 12 edges, got {mesh.n_edges}"
        assert mesh.n_faces == 8, f"Expected 8 faces, got {mesh.n_faces}"


# =============================================================================
# Gauss-Bonnet Tests
# =============================================================================

class TestGaussBonnet:
    """Test Gauss-Bonnet theorem: total angle defect = 2*pi*chi."""

    def test_tetrahedron_gauss_bonnet(self, regular_tetrahedron):
        """Tetrahedron: total angle defect = 2*pi*2 = 4*pi."""
        mesh = regular_tetrahedron
        alpha = compute_corner_angles(mesh)
        K_total = total_gaussian_curvature(mesh, alpha)

        chi = euler_characteristic(mesh)
        expected = 2 * np.pi * chi

        assert abs(K_total - expected) < 1e-6, \
            f"Gauss-Bonnet: expected {expected:.4f}, got {K_total:.4f}"

    def test_cube_gauss_bonnet(self, cube_triangulated):
        """Cube: total angle defect = 2*pi*2 = 4*pi."""
        mesh = cube_triangulated
        alpha = compute_corner_angles(mesh)
        K_total = total_gaussian_curvature(mesh, alpha)

        chi = euler_characteristic(mesh)
        expected = 2 * np.pi * chi

        assert abs(K_total - expected) < 1e-6, \
            f"Gauss-Bonnet: expected {expected:.4f}, got {K_total:.4f}"

    def test_octahedron_gauss_bonnet(self, octahedron):
        """Octahedron: total angle defect = 2*pi*2 = 4*pi."""
        mesh = octahedron
        alpha = compute_corner_angles(mesh)
        K_total = total_gaussian_curvature(mesh, alpha)

        chi = euler_characteristic(mesh)
        expected = 2 * np.pi * chi

        assert abs(K_total - expected) < 1e-6, \
            f"Gauss-Bonnet: expected {expected:.4f}, got {K_total:.4f}"

    def test_cube_vertex_defects(self, cube_triangulated):
        """Cube vertices: each has angle defect = pi/2 (8 vertices * pi/2 = 4*pi)."""
        mesh = cube_triangulated
        alpha = compute_corner_angles(mesh)
        K = angle_defect(mesh, alpha)

        # Each vertex of a cube has 3 right angles meeting -> total = 3*pi/2
        # Defect = 2*pi - 3*pi/2 = pi/2
        expected_per_vertex = np.pi / 2

        for v in range(mesh.n_vertices):
            assert abs(K[v] - expected_per_vertex) < 1e-6, \
                f"Vertex {v}: expected defect {expected_per_vertex:.4f}, got {K[v]:.4f}"


# =============================================================================
# Face Normal Tests
# =============================================================================

class TestFaceNormals:
    """Test face normal computation."""

    def test_xy_plane_normal(self, equilateral_triangle):
        """Triangle in XY plane should have normal along Z."""
        mesh = equilateral_triangle
        normals = compute_face_normals(mesh)

        # Normal should be [0, 0, 1] or [0, 0, -1]
        n = normals[0]
        assert abs(n[0]) < 1e-10 and abs(n[1]) < 1e-10, \
            f"Normal should be along Z axis, got {n}"
        assert abs(abs(n[2]) - 1.0) < 1e-10, \
            f"Normal Z component should be +/-1, got {n[2]}"

    def test_normals_are_unit(self, regular_tetrahedron, cube_triangulated, octahedron):
        """All face normals should be unit length."""
        for mesh in [regular_tetrahedron, cube_triangulated, octahedron]:
            normals = compute_face_normals(mesh)
            norms = np.linalg.norm(normals, axis=1)

            for f in range(mesh.n_faces):
                assert abs(norms[f] - 1.0) < 1e-10, \
                    f"Face {f}: normal length = {norms[f]}, expected 1"


# =============================================================================
# Cotangent Weights Tests
# =============================================================================

class TestCotangentWeights:
    """Test cotangent weight computation."""

    def test_equilateral_cotan_weights(self, equilateral_triangle):
        """Equilateral triangle: all cotan weights equal."""
        mesh = equilateral_triangle
        alpha = compute_corner_angles(mesh)
        weights = compute_cotan_weights(mesh, alpha)

        # All weights should be equal (by symmetry)
        # For a single triangle, weights = 0.5 * cot(60 deg) = 0.5 * 1/sqrt(3)
        expected = 0.5 / np.tan(np.pi/3)

        for e in range(mesh.n_edges):
            assert abs(weights[e] - expected) < 1e-10, \
                f"Edge {e}: expected weight {expected:.4f}, got {weights[e]:.4f}"

    def test_right_triangle_cotan_weights(self, right_triangle_345):
        """Right triangle: weight opposite to right angle = 0."""
        mesh = right_triangle_345
        alpha = compute_corner_angles(mesh)
        weights = compute_cotan_weights(mesh, alpha)

        # The edge opposite to the right angle (the hypotenuse)
        # has cotan weight = 0.5 * cot(90 deg) = 0
        # Find which edge is the hypotenuse (length 5)
        ell = compute_edge_lengths(mesh)
        for e in range(mesh.n_edges):
            if abs(ell[e] - 5.0) < 1e-10:
                # This is the hypotenuse, opposite angle is 90 deg
                assert abs(weights[e]) < 1e-10, \
                    f"Hypotenuse weight should be 0, got {weights[e]}"

    def test_cotan_weights_positive_for_acute(self, equilateral_triangle):
        """For acute triangles, all cotan weights should be positive."""
        mesh = equilateral_triangle
        alpha = compute_corner_angles(mesh)
        weights = compute_cotan_weights(mesh, alpha)

        assert np.all(weights > 0), "All cotan weights should be positive for acute triangles"

    def test_halfedge_cotan_vs_edge_cotan(self, unit_square_two_triangles):
        """Halfedge cotan weights should sum to edge cotan weights for interior edges."""
        mesh = unit_square_two_triangles
        alpha = compute_corner_angles(mesh)

        edge_weights = compute_cotan_weights(mesh, alpha)
        he_weights = compute_halfedge_cotan_weights(mesh, alpha)

        for e in range(mesh.n_edges):
            he0 = mesh.edge_to_halfedge[e, 0]
            he1 = mesh.edge_to_halfedge[e, 1]

            if he1 != -1:  # Interior edge
                he_sum = he_weights[he0] + he_weights[he1]
                assert abs(he_sum - edge_weights[e]) < 1e-10, \
                    f"Edge {e}: halfedge sum {he_sum} != edge weight {edge_weights[e]}"


# =============================================================================
# Connectivity Tests
# =============================================================================

class TestConnectivity:
    """Test mesh connectivity operations."""

    def test_halfedge_twin_symmetry(self, unit_square_two_triangles):
        """Twin of twin should be self."""
        mesh = unit_square_two_triangles

        for he in range(mesh.n_halfedges):
            twin = mesh.halfedge_twin[he]
            if twin != -1:
                twin_twin = mesh.halfedge_twin[twin]
                assert twin_twin == he, \
                    f"Halfedge {he}: twin of twin = {twin_twin}, expected {he}"

    def test_halfedge_twin_reverses_vertices(self, unit_square_two_triangles):
        """Twin halfedge should have reversed vertex order."""
        mesh = unit_square_two_triangles

        for he in range(mesh.n_halfedges):
            twin = mesh.halfedge_twin[he]
            if twin != -1:
                i, j = mesh.halfedge_vertices(he)
                ti, tj = mesh.halfedge_vertices(twin)
                assert i == tj and j == ti, \
                    f"Halfedge {he}: ({i},{j}), twin: ({ti},{tj})"

    def test_halfedge_next_consistency(self, regular_tetrahedron):
        """Next halfedge should share a vertex."""
        mesh = regular_tetrahedron

        for he in range(mesh.n_halfedges):
            i, j = mesh.halfedge_vertices(he)
            he_next = mesh.halfedge_next(he)
            j2, k = mesh.halfedge_vertices(he_next)
            assert j == j2, \
                f"Halfedge {he} -> {he_next}: end vertex {j} != start vertex {j2}"

    def test_boundary_detection_closed(self, regular_tetrahedron, cube_triangulated, octahedron):
        """Closed meshes should have no boundary halfedges."""
        for mesh in [regular_tetrahedron, cube_triangulated, octahedron]:
            n_boundary = sum(1 for he in range(mesh.n_halfedges) if mesh.is_boundary_halfedge(he))
            assert n_boundary == 0, f"Closed mesh has {n_boundary} boundary halfedges"

    def test_boundary_detection_open(self, equilateral_triangle):
        """Open mesh (single triangle) has boundary."""
        mesh = equilateral_triangle
        n_boundary = sum(1 for he in range(mesh.n_halfedges) if mesh.is_boundary_halfedge(he))
        assert n_boundary == 3, f"Single triangle should have 3 boundary halfedges, got {n_boundary}"


# =============================================================================
# Genus Tests
# =============================================================================

class TestGenus:
    """Test genus computation."""

    def test_tetrahedron_genus(self, regular_tetrahedron):
        """Tetrahedron: genus 0."""
        mesh = regular_tetrahedron
        g = genus(mesh)
        assert g == 0, f"Expected genus 0, got {g}"

    def test_cube_genus(self, cube_triangulated):
        """Cube: genus 0."""
        mesh = cube_triangulated
        g = genus(mesh)
        assert g == 0, f"Expected genus 0, got {g}"

    def test_octahedron_genus(self, octahedron):
        """Octahedron: genus 0."""
        mesh = octahedron
        g = genus(mesh)
        assert g == 0, f"Expected genus 0, got {g}"


# =============================================================================
# MeshGeometry Container Tests
# =============================================================================

class TestMeshGeometry:
    """Test the MeshGeometry container class."""

    def test_all_quantities_computed(self, regular_tetrahedron):
        """MeshGeometry should compute all quantities."""
        mesh = regular_tetrahedron
        geom = MeshGeometry(mesh)

        assert geom.edge_lengths is not None
        assert geom.halfedge_lengths is not None
        assert geom.alpha is not None
        assert geom.areas is not None
        assert geom.face_normals is not None
        assert geom.cotan_weights is not None
        assert geom.halfedge_cotan is not None
        assert geom.N is not None
        assert geom.T1 is not None
        assert geom.T2 is not None

    def test_total_area(self, cube_triangulated):
        """Cube total area = 6 (6 faces of area 1 each)."""
        mesh = cube_triangulated
        geom = MeshGeometry(mesh)

        total = geom.total_area()
        assert abs(total - 6.0) < 1e-10, \
            f"Expected total area 6, got {total}"


# =============================================================================
# Real Mesh Tests (External Files)
# =============================================================================

class TestRealMeshes:
    """Test with actual mesh files."""

    @pytest.fixture
    def sphere_mesh(self):
        """Load sphere mesh if available."""
        from io_obj import load_obj
        path = Path(r"C:\Dev\Colonel\Data\Meshes\sphere320.obj")
        if not path.exists():
            pytest.skip(f"Sphere mesh not found: {path}")
        return load_obj(str(path))

    @pytest.fixture
    def torus_mesh(self):
        """Load torus mesh if available."""
        from io_obj import load_obj
        path = Path(r"C:\Dev\Colonel\Data\Meshes\torus.obj")
        if not path.exists():
            pytest.skip(f"Torus mesh not found: {path}")
        return load_obj(str(path))

    def test_sphere_euler_characteristic(self, sphere_mesh):
        """Sphere: chi = 2."""
        chi = euler_characteristic(sphere_mesh)
        assert chi == 2, f"Sphere: expected chi=2, got {chi}"

    def test_sphere_genus(self, sphere_mesh):
        """Sphere: genus 0."""
        g = genus(sphere_mesh)
        assert g == 0, f"Sphere: expected genus 0, got {g}"

    def test_sphere_gauss_bonnet(self, sphere_mesh):
        """Sphere: total angle defect = 4*pi."""
        alpha = compute_corner_angles(sphere_mesh)
        K_total = total_gaussian_curvature(sphere_mesh, alpha)

        expected = 4 * np.pi
        # Allow slightly larger tolerance for real meshes
        assert abs(K_total - expected) < 0.01, \
            f"Sphere Gauss-Bonnet: expected {expected:.4f}, got {K_total:.4f}"

    def test_sphere_angle_sums(self, sphere_mesh):
        """All triangles in sphere should have angle sum = pi."""
        alpha = compute_corner_angles(sphere_mesh)

        max_error = 0
        for f in range(sphere_mesh.n_faces):
            angle_sum = alpha[3*f] + alpha[3*f+1] + alpha[3*f+2]
            error = abs(angle_sum - np.pi)
            max_error = max(max_error, error)

        assert max_error < 1e-10, \
            f"Max angle sum error: {max_error}"

    def test_sphere_positive_areas(self, sphere_mesh):
        """All triangles in sphere should have positive area."""
        areas = compute_face_areas(sphere_mesh)
        assert np.all(areas > 0), "All areas should be positive"

    def test_sphere_positive_edge_lengths(self, sphere_mesh):
        """All edges in sphere should have positive length."""
        ell = compute_edge_lengths(sphere_mesh)
        assert np.all(ell > 0), "All edge lengths should be positive"

    def test_torus_euler_characteristic(self, torus_mesh):
        """Torus: chi = 0."""
        chi = euler_characteristic(torus_mesh)
        assert chi == 0, f"Torus: expected chi=0, got {chi}"

    def test_torus_genus(self, torus_mesh):
        """Torus: genus 1."""
        g = genus(torus_mesh)
        assert g == 1, f"Torus: expected genus 1, got {g}"

    def test_torus_gauss_bonnet(self, torus_mesh):
        """Torus: total angle defect = 0."""
        alpha = compute_corner_angles(torus_mesh)
        K_total = total_gaussian_curvature(torus_mesh, alpha)

        expected = 0.0
        # Allow slightly larger tolerance for real meshes
        assert abs(K_total - expected) < 0.01, \
            f"Torus Gauss-Bonnet: expected {expected:.4f}, got {K_total:.4f}"

    def test_torus_angle_sums(self, torus_mesh):
        """All triangles in torus should have angle sum = pi."""
        alpha = compute_corner_angles(torus_mesh)

        max_error = 0
        for f in range(torus_mesh.n_faces):
            angle_sum = alpha[3*f] + alpha[3*f+1] + alpha[3*f+2]
            error = abs(angle_sum - np.pi)
            max_error = max(max_error, error)

        assert max_error < 1e-10, \
            f"Max angle sum error: {max_error}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_very_thin_triangle(self):
        """Very thin triangle should still compute correctly."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-6, 0.0],  # Very thin
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = TriangleMesh(positions=positions, faces=faces)
        mesh = build_connectivity(mesh)

        alpha = compute_corner_angles(mesh)
        angle_sum = np.sum(alpha)

        # Should still sum to pi
        assert abs(angle_sum - np.pi) < 1e-6, \
            f"Thin triangle angle sum: {angle_sum}"

        # Area should be positive (but small)
        areas = compute_face_areas(mesh)
        assert areas[0] > 0, "Area should be positive"

    def test_large_coordinates(self):
        """Large coordinate values should work."""
        scale = 1e6
        positions = np.array([
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.5 * scale, np.sqrt(3)/2 * scale, 0.0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = TriangleMesh(positions=positions, faces=faces)
        mesh = build_connectivity(mesh)

        alpha = compute_corner_angles(mesh)

        # Angles should still be 60 degrees each
        for i in range(3):
            assert abs(alpha[i] - np.pi/3) < 1e-8, \
                f"Corner {i}: expected pi/3, got {alpha[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
