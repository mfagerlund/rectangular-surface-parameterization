"""
Pytest tests for Preprocess/MeshInfo.py

Tests the MeshInfo dataclass and mesh_info() function.
Run with: pytest tests/test_MeshInfo.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory and Preprocess to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Preprocess"))

from Preprocess.MeshInfo import MeshInfo, mesh_info


# =============================================================================
# Test Fixtures - Known Shapes
# =============================================================================

@pytest.fixture
def single_triangle():
    """Single equilateral triangle with side length 1."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return X, T


@pytest.fixture
def right_triangle():
    """3-4-5 right triangle (right angle at vertex 0)."""
    X = np.array([
        [0.0, 0.0, 0.0],  # right angle here
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return X, T


@pytest.fixture
def tetrahedron():
    """
    Regular tetrahedron surface (4 triangles, 4 vertices, 6 edges).
    This is a closed manifold surface with genus 0.
    """
    # Regular tetrahedron vertices
    X = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    # 4 triangular faces with consistent outward orientation
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=np.int32)
    return X, T


@pytest.fixture
def two_triangles_shared_edge():
    """Two triangles sharing one edge (butterfly shape)."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],  # top triangle
        [0, 3, 1],  # bottom triangle (note: different winding for consistent normal)
    ], dtype=np.int32)
    return X, T


# =============================================================================
# Test: All Fields Populated
# =============================================================================

class TestFieldsPopulated:
    """Verify all MeshInfo fields are populated correctly."""

    def test_single_triangle_fields_exist(self, single_triangle):
        """Single triangle: all fields should be populated."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        # Check all required attributes exist
        assert hasattr(mesh, 'vertices')
        assert hasattr(mesh, 'triangles')
        assert hasattr(mesh, 'num_faces')
        assert hasattr(mesh, 'num_vertices')
        assert hasattr(mesh, 'num_edges')
        assert hasattr(mesh, 'edge_to_vertex')
        assert hasattr(mesh, 'T2E')
        assert hasattr(mesh, 'edge_to_triangle')
        assert hasattr(mesh, 'triangle_to_triangle')
        assert hasattr(mesh, 'normal')
        assert hasattr(mesh, 'area')
        assert hasattr(mesh, 'vertex_normals')
        assert hasattr(mesh, 'sq_edge_length')
        assert hasattr(mesh, 'corner_angle')
        assert hasattr(mesh, 'cot_corner_angle')

    def test_single_triangle_fields_not_none(self, single_triangle):
        """Single triangle: no fields should be None."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        assert mesh.vertices is not None
        assert mesh.triangles is not None
        assert mesh.num_faces is not None
        assert mesh.num_vertices is not None
        assert mesh.num_edges is not None
        assert mesh.edge_to_vertex is not None
        assert mesh.T2E is not None
        assert mesh.edge_to_triangle is not None
        assert mesh.triangle_to_triangle is not None
        assert mesh.normal is not None
        assert mesh.area is not None
        assert mesh.vertex_normals is not None
        assert mesh.sq_edge_length is not None
        assert mesh.corner_angle is not None
        assert mesh.cot_corner_angle is not None


# =============================================================================
# Test: Tetrahedron Counts (nv=4, nf=4, ne=6)
# =============================================================================

class TestTetrahedronCounts:
    """Verify correct counts for tetrahedron surface."""

    def test_tetrahedron_vertex_count(self, tetrahedron):
        """Tetrahedron should have 4 vertices."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)
        assert mesh.num_vertices == 4, f"Expected 4 vertices, got {mesh.num_vertices}"

    def test_tetrahedron_face_count(self, tetrahedron):
        """Tetrahedron should have 4 faces."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)
        assert mesh.num_faces == 4, f"Expected 4 faces, got {mesh.num_faces}"

    def test_tetrahedron_edge_count(self, tetrahedron):
        """Tetrahedron should have 6 edges."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)
        assert mesh.num_edges == 6, f"Expected 6 edges, got {mesh.num_edges}"

    def test_euler_formula(self, tetrahedron):
        """Verify Euler's formula: V - E + F = 2 for closed genus-0 surface."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)
        euler = mesh.num_vertices - mesh.num_edges + mesh.num_faces
        assert euler == 2, f"Euler characteristic should be 2, got {euler}"


# =============================================================================
# Test: Normals are Unit Length
# =============================================================================

class TestNormals:
    """Verify normal vectors are unit length."""

    def test_face_normals_unit_length(self, single_triangle):
        """Face normals should have unit length."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        norms = np.linalg.norm(mesh.normal, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10,
            err_msg="Face normals should have unit length")

    def test_face_normals_unit_length_tetrahedron(self, tetrahedron):
        """All tetrahedron face normals should have unit length."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)

        norms = np.linalg.norm(mesh.normal, axis=1)
        np.testing.assert_allclose(norms, np.ones(mesh.num_faces), atol=1e-10,
            err_msg="All face normals should have unit length")

    def test_vertex_normals_unit_length(self, single_triangle):
        """Vertex normals should have unit length."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        norms = np.linalg.norm(mesh.vertex_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10,
            err_msg="Vertex normals should have unit length")

    def test_vertex_normals_unit_length_tetrahedron(self, tetrahedron):
        """All tetrahedron vertex normals should have unit length."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)

        norms = np.linalg.norm(mesh.vertex_normals, axis=1)
        np.testing.assert_allclose(norms, np.ones(mesh.num_vertices), atol=1e-10,
            err_msg="All vertex normals should have unit length")

    def test_face_normal_direction_z_positive(self, single_triangle):
        """Face normal of XY-plane triangle should point in +Z or -Z."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        # For a triangle in XY plane, normal should be (0, 0, +/-1)
        assert abs(abs(mesh.normal[0, 2]) - 1.0) < 1e-10, \
            "Normal Z component should be +/-1"
        assert abs(mesh.normal[0, 0]) < 1e-10, \
            "Normal X component should be 0"
        assert abs(mesh.normal[0, 1]) < 1e-10, \
            "Normal Y component should be 0"


# =============================================================================
# Test: Areas are Positive
# =============================================================================

class TestAreas:
    """Verify face areas are computed correctly and are positive."""

    def test_areas_positive(self, single_triangle):
        """Face areas should be positive."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        assert np.all(mesh.area > 0), "All face areas should be positive"

    def test_areas_positive_tetrahedron(self, tetrahedron):
        """All tetrahedron face areas should be positive."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)

        assert np.all(mesh.area > 0), "All face areas should be positive"

    def test_equilateral_triangle_area(self, single_triangle):
        """Equilateral triangle with side 1 has area sqrt(3)/4."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        expected_area = np.sqrt(3) / 4
        np.testing.assert_allclose(mesh.area[0], expected_area, atol=1e-10,
            err_msg=f"Expected area {expected_area}, got {mesh.area[0]}")

    def test_right_triangle_area(self, right_triangle):
        """3-4-5 right triangle has area = (3*4)/2 = 6."""
        X, T = right_triangle
        mesh = mesh_info(X, T)

        expected_area = 6.0
        np.testing.assert_allclose(mesh.area[0], expected_area, atol=1e-10,
            err_msg=f"Expected area {expected_area}, got {mesh.area[0]}")

    def test_tetrahedron_equal_areas(self, tetrahedron):
        """Regular tetrahedron has 4 faces of equal area."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)

        # All faces should have the same area for a regular tetrahedron
        np.testing.assert_allclose(mesh.area, mesh.area[0] * np.ones(4), atol=1e-10,
            err_msg="All tetrahedron faces should have equal area")


# =============================================================================
# Test: Corner Angles Sum to Pi per Triangle
# =============================================================================

class TestCornerAngles:
    """Verify corner angles are computed correctly."""

    def test_angles_sum_to_pi(self, single_triangle):
        """Corner angles in each triangle should sum to pi."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        angle_sum = np.sum(mesh.corner_angle, axis=1)
        np.testing.assert_allclose(angle_sum, np.pi, atol=1e-10,
            err_msg="Corner angles should sum to pi")

    def test_angles_sum_to_pi_tetrahedron(self, tetrahedron):
        """All tetrahedron triangles: angles should sum to pi."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)

        angle_sums = np.sum(mesh.corner_angle, axis=1)
        np.testing.assert_allclose(angle_sums, np.pi * np.ones(mesh.num_faces), atol=1e-10,
            err_msg="Each triangle's angles should sum to pi")

    def test_equilateral_angles_equal(self, single_triangle):
        """Equilateral triangle has all angles = pi/3."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        expected_angle = np.pi / 3
        np.testing.assert_allclose(mesh.corner_angle[0], expected_angle, atol=1e-10,
            err_msg=f"All angles should be {np.degrees(expected_angle):.2f} degrees")

    def test_right_triangle_angles(self, right_triangle):
        """3-4-5 right triangle has a 90-degree angle at vertex 0."""
        X, T = right_triangle
        mesh = mesh_info(X, T)

        # Vertex 0 is at origin, should be 90 degrees
        np.testing.assert_allclose(mesh.corner_angle[0, 0], np.pi / 2, atol=1e-10,
            err_msg="Right angle should be pi/2")

    def test_cotangent_angles_finite(self, single_triangle):
        """Cotangent of angles should be finite (no division by zero)."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        assert np.all(np.isfinite(mesh.cot_corner_angle)), \
            "All cotangent values should be finite"

    def test_cotangent_equilateral(self, single_triangle):
        """Cotangent of 60 degrees = 1/sqrt(3)."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        expected_cot = 1.0 / np.sqrt(3)
        np.testing.assert_allclose(mesh.cot_corner_angle[0], expected_cot, atol=1e-10,
            err_msg=f"cot(60 deg) should be {expected_cot}")


# =============================================================================
# Test: Connectivity Arrays Have Correct Shapes
# =============================================================================

class TestConnectivityShapes:
    """Verify connectivity arrays have correct shapes."""

    def test_single_triangle_shapes(self, single_triangle):
        """Single triangle connectivity array shapes."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        # nv=3, nf=1, ne=3 for a single triangle
        assert mesh.vertices.shape == (3, 3), f"X shape should be (3, 3), got {mesh.vertices.shape}"
        assert mesh.triangles.shape == (1, 3), f"T shape should be (1, 3), got {mesh.triangles.shape}"
        assert mesh.edge_to_vertex.shape == (3, 2), f"E2V shape should be (3, 2), got {mesh.edge_to_vertex.shape}"
        assert mesh.T2E.shape == (1, 3), f"T2E shape should be (1, 3), got {mesh.T2E.shape}"
        assert mesh.edge_to_triangle.shape == (3, 4), f"E2T shape should be (3, 4), got {mesh.edge_to_triangle.shape}"
        assert mesh.triangle_to_triangle.shape == (1, 3), f"T2T shape should be (1, 3), got {mesh.triangle_to_triangle.shape}"
        assert mesh.normal.shape == (1, 3), f"normal shape should be (1, 3), got {mesh.normal.shape}"
        assert mesh.area.shape == (1,), f"area shape should be (1,), got {mesh.area.shape}"
        assert mesh.vertex_normals.shape == (3, 3), f"Nv shape should be (3, 3), got {mesh.vertex_normals.shape}"
        assert mesh.sq_edge_length.shape == (3,), f"SqEdgeLength shape should be (3,), got {mesh.sq_edge_length.shape}"
        assert mesh.corner_angle.shape == (1, 3), f"corner_angle shape should be (1, 3), got {mesh.corner_angle.shape}"
        assert mesh.cot_corner_angle.shape == (1, 3), f"cot_corner_angle shape should be (1, 3), got {mesh.cot_corner_angle.shape}"

    def test_tetrahedron_shapes(self, tetrahedron):
        """Tetrahedron connectivity array shapes."""
        X, T = tetrahedron
        mesh = mesh_info(X, T)

        # nv=4, nf=4, ne=6 for a tetrahedron
        assert mesh.vertices.shape == (4, 3), f"X shape should be (4, 3), got {mesh.vertices.shape}"
        assert mesh.triangles.shape == (4, 3), f"T shape should be (4, 3), got {mesh.triangles.shape}"
        assert mesh.edge_to_vertex.shape == (6, 2), f"E2V shape should be (6, 2), got {mesh.edge_to_vertex.shape}"
        assert mesh.T2E.shape == (4, 3), f"T2E shape should be (4, 3), got {mesh.T2E.shape}"
        assert mesh.edge_to_triangle.shape == (6, 4), f"E2T shape should be (6, 4), got {mesh.edge_to_triangle.shape}"
        assert mesh.triangle_to_triangle.shape == (4, 3), f"T2T shape should be (4, 3), got {mesh.triangle_to_triangle.shape}"
        assert mesh.normal.shape == (4, 3), f"normal shape should be (4, 3), got {mesh.normal.shape}"
        assert mesh.area.shape == (4,), f"area shape should be (4,), got {mesh.area.shape}"
        assert mesh.vertex_normals.shape == (4, 3), f"Nv shape should be (4, 3), got {mesh.vertex_normals.shape}"
        assert mesh.sq_edge_length.shape == (6,), f"SqEdgeLength shape should be (6,), got {mesh.sq_edge_length.shape}"
        assert mesh.corner_angle.shape == (4, 3), f"corner_angle shape should be (4, 3), got {mesh.corner_angle.shape}"
        assert mesh.cot_corner_angle.shape == (4, 3), f"cot_corner_angle shape should be (4, 3), got {mesh.cot_corner_angle.shape}"

    def test_two_triangles_shapes(self, two_triangles_shared_edge):
        """Two triangles sharing an edge: nv=4, nf=2, ne=5."""
        X, T = two_triangles_shared_edge
        mesh = mesh_info(X, T)

        # nv=4, nf=2, ne=5 for two triangles sharing one edge
        assert mesh.num_vertices == 4, f"Expected 4 vertices, got {mesh.num_vertices}"
        assert mesh.num_faces == 2, f"Expected 2 faces, got {mesh.num_faces}"
        assert mesh.num_edges == 5, f"Expected 5 edges (6 half-edges, 1 shared), got {mesh.num_edges}"

        assert mesh.edge_to_vertex.shape == (5, 2), f"E2V shape should be (5, 2), got {mesh.edge_to_vertex.shape}"
        assert mesh.T2E.shape == (2, 3), f"T2E shape should be (2, 3), got {mesh.T2E.shape}"
        assert mesh.corner_angle.shape == (2, 3), f"corner_angle shape should be (2, 3), got {mesh.corner_angle.shape}"


# =============================================================================
# Test: Edge Squared Lengths
# =============================================================================

class TestEdgeLengths:
    """Verify squared edge lengths are computed correctly."""

    def test_squared_edge_lengths_positive(self, single_triangle):
        """Squared edge lengths should be positive."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        assert np.all(mesh.sq_edge_length > 0), "All squared edge lengths should be positive"

    def test_equilateral_triangle_edge_lengths(self, single_triangle):
        """Equilateral triangle with side 1 has all edges with squared length 1."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        # All edges should have squared length 1
        np.testing.assert_allclose(mesh.sq_edge_length, np.ones(3), atol=1e-10,
            err_msg="Equilateral triangle with side 1 should have all SqEdgeLength = 1")

    def test_right_triangle_edge_lengths(self, right_triangle):
        """3-4-5 right triangle: edges have squared lengths 9, 16, 25."""
        X, T = right_triangle
        mesh = mesh_info(X, T)

        expected_sq_lengths = sorted([9.0, 16.0, 25.0])  # 3^2, 4^2, 5^2
        actual_sq_lengths = sorted(mesh.sq_edge_length)
        np.testing.assert_allclose(actual_sq_lengths, expected_sq_lengths, atol=1e-10,
            err_msg="3-4-5 triangle should have squared edge lengths 9, 16, 25")


# =============================================================================
# Test: Input Validation
# =============================================================================

class TestInputValidation:
    """Test input validation."""

    def test_non_triangulation_raises(self):
        """Input with wrong number of columns should raise assertion."""
        X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
        T = np.array([[0, 1, 2, 3]], dtype=np.int32)  # quad, not triangle

        with pytest.raises(AssertionError, match="Not a triangulation"):
            mesh_info(X, T)


# =============================================================================
# Test: MeshInfo is a Dataclass
# =============================================================================

class TestDataclass:
    """Verify MeshInfo behaves as expected dataclass."""

    def test_is_dataclass(self, single_triangle):
        """MeshInfo should be a dataclass instance."""
        from dataclasses import is_dataclass
        X, T = single_triangle
        mesh = mesh_info(X, T)

        assert is_dataclass(mesh), "MeshInfo should be a dataclass"

    def test_fields_accessible(self, single_triangle):
        """All fields should be accessible by name."""
        X, T = single_triangle
        mesh = mesh_info(X, T)

        # Access all fields
        _ = mesh.vertices
        _ = mesh.triangles
        _ = mesh.num_faces
        _ = mesh.num_vertices
        _ = mesh.num_edges
        _ = mesh.edge_to_vertex
        _ = mesh.T2E
        _ = mesh.edge_to_triangle
        _ = mesh.triangle_to_triangle
        _ = mesh.normal
        _ = mesh.area
        _ = mesh.vertex_normals
        _ = mesh.sq_edge_length
        _ = mesh.corner_angle
        _ = mesh.cot_corner_angle
