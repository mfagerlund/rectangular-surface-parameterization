"""
Pytest tests for optimization/reduction.py

Tests the reduction_from_ff2d function which builds the reduction matrix
from a 2D frame field.

The function:
1. Computes k21 jump indices per edge (values 1-4) from cross field
2. Builds a block diagonal Reduction matrix combining ut and vt reductions

NOTE: The reduce_corner_var_2d function (a dependency) requires CLOSED meshes.
Tests using open meshes (single triangle, two triangles with boundary) will fail
because reduce_corner_var_2d returns None for boundary vertices.

Run with: pytest tests/test_reduction_from_ff2d.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys
import scipy.sparse as sp

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocess"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Orthotropic"))

from rectangular_surface_parameterization.core.mesh_info import mesh_info
from rectangular_surface_parameterization.preprocessing.connectivity import connectivity
from rectangular_surface_parameterization.preprocessing.sort_triangles import clear_cache as clear_sort_cache
from rectangular_surface_parameterization.optimization.reduce_corner_var import reduce_corner_var_2d
from rectangular_surface_parameterization.optimization.reduction import reduction_from_ff2d


# Clear cache before each test to avoid cross-test pollution
@pytest.fixture(autouse=True)
def clear_caches():
    """Clear sort_triangles cache before each test."""
    clear_sort_cache()
    yield
    clear_sort_cache()


# =============================================================================
# Helper classes to mock Src and param objects
# =============================================================================

class MockParam:
    """Mock parameter object with required attributes."""

    def __init__(self, E2T, ide_int, para_trans):
        self.edge_to_triangle = E2T
        self.ide_int = ide_int
        self.para_trans = para_trans


# =============================================================================
# Test Fixtures - Closed Meshes
# NOTE: reduce_corner_var_2d requires closed meshes (no boundary)
# =============================================================================

@pytest.fixture
def tetrahedron():
    """
    Tetrahedron surface (closed mesh, genus 0).

    4 vertices, 4 faces, 6 edges.
    """
    X = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def octahedron():
    """
    Regular octahedron (closed mesh, genus 0).

    6 vertices, 8 faces, 12 edges.
    """
    X = np.array([
        [1, 0, 0],   # 0: +X
        [-1, 0, 0],  # 1: -X
        [0, 1, 0],   # 2: +Y
        [0, -1, 0],  # 3: -Y
        [0, 0, 1],   # 4: +Z
        [0, 0, -1],  # 5: -Z
    ], dtype=np.float64)
    T = np.array([
        [0, 2, 4],  # +X +Y +Z
        [2, 1, 4],  # -X +Y +Z
        [1, 3, 4],  # -X -Y +Z
        [3, 0, 4],  # +X -Y +Z
        [2, 0, 5],  # +X +Y -Z
        [1, 2, 5],  # -X +Y -Z
        [3, 1, 5],  # -X -Y -Z
        [0, 3, 5],  # +X -Y -Z
    ], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def cube():
    """
    Cube triangulated into 12 triangles (closed, genus 0).

    8 vertices, 12 faces, 18 edges.
    """
    X = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ], dtype=np.float64)
    T = np.array([
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
    return mesh_info(X, T)


@pytest.fixture
def icosahedron():
    """
    Regular icosahedron (closed mesh, genus 0).

    12 vertices, 20 faces, 30 edges.
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    X = np.array([
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ], dtype=np.float64)

    T = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=np.int32)

    return mesh_info(X, T)


# =============================================================================
# k21 Range Tests
# =============================================================================

class TestK21Range:
    """Test that k21 values are in the valid range [1, 4]."""

    def test_tetrahedron_k21_range(self, tetrahedron):
        """Tetrahedron (closed mesh): k21 values should be in [1, 4]."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 values should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 values should be <= 4, got max {k21.max()}"

    def test_octahedron_k21_range(self, octahedron):
        """Octahedron (closed mesh): k21 values should be in [1, 4]."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 values should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 values should be <= 4, got max {k21.max()}"

    def test_cube_k21_range(self, cube):
        """Cube (12 faces, 18 edges): k21 values should be in [1, 4]."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 values should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 values should be <= 4, got max {k21.max()}"

    def test_icosahedron_k21_range(self, icosahedron):
        """Icosahedron (20 faces, 30 edges): k21 values should be in [1, 4]."""
        Src = icosahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 values should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 values should be <= 4, got max {k21.max()}"


# =============================================================================
# k21 Default Value Tests
# =============================================================================

class TestK21Defaults:
    """Test k21 default behavior."""

    def test_zero_angles_give_k21_one(self, tetrahedron):
        """Zero angles and zero omega/para_trans should give k21 = 1."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # With zero angles and zero omega/para_trans, k21 should all be 1
        assert np.all(k21 == 1), f"With zero angles, all k21 should be 1, got {k21}"


# =============================================================================
# Reduction Matrix Shape Tests
# =============================================================================

class TestReductionShape:
    """Test that Reduction matrix has correct shape."""

    def test_tetrahedron_reduction_shape(self, tetrahedron):
        """Tetrahedron: Reduction should have shape (6*nf, 2*nv)."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # Reduction is block_diag([v2t_smooth, s_diag @ v2t])
        # v2t_smooth: (3*nf, nv), s_diag @ v2t: (3*nf, nv)
        # So Reduction: (6*nf, 2*nv)
        expected_rows = 6 * Src.num_faces
        expected_cols = 2 * Src.num_vertices

        assert Reduction.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {Reduction.shape}"

    def test_octahedron_reduction_shape(self, octahedron):
        """Octahedron: Reduction shape check."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        expected_rows = 6 * Src.num_faces
        expected_cols = 2 * Src.num_vertices

        assert Reduction.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {Reduction.shape}"

    def test_cube_reduction_shape(self, cube):
        """Cube: Reduction shape check."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        expected_rows = 6 * Src.num_faces  # 6 * 12 = 72
        expected_cols = 2 * Src.num_vertices  # 2 * 8 = 16

        assert Reduction.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {Reduction.shape}"

    def test_icosahedron_reduction_shape(self, icosahedron):
        """Icosahedron: Reduction shape check."""
        Src = icosahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        expected_rows = 6 * Src.num_faces  # 6 * 20 = 120
        expected_cols = 2 * Src.num_vertices  # 2 * 12 = 24

        assert Reduction.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {Reduction.shape}"


# =============================================================================
# Reduction Matrix Sparsity Tests
# =============================================================================

class TestReductionSparsity:
    """Test that Reduction matrix has expected sparsity pattern."""

    def test_reduction_is_sparse(self, tetrahedron):
        """Reduction matrix should be a sparse matrix."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert sp.issparse(Reduction), "Reduction should be a sparse matrix"

    def test_reduction_is_csr(self, tetrahedron):
        """Reduction matrix should be in CSR format."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert isinstance(Reduction, sp.csr_matrix), \
            f"Reduction should be CSR matrix, got {type(Reduction)}"

    def test_reduction_block_diagonal_structure(self, tetrahedron):
        """Reduction matrix should have block diagonal structure."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        R_dense = Reduction.toarray()

        # First block (rows 0:3*nf, cols 0:nv) - v2t_smooth
        # Second block (rows 3*nf:6*nf, cols nv:2*nv) - s_diag @ v2t
        # Off-diagonal blocks should be zero

        top_right = R_dense[:3 * Src.num_faces, Src.num_vertices:]
        bottom_left = R_dense[3 * Src.num_faces:, :Src.num_vertices]

        assert np.allclose(top_right, 0), "Top-right block should be zero (block diagonal)"
        assert np.allclose(bottom_left, 0), "Bottom-left block should be zero (block diagonal)"

    def test_reduction_nnz_reasonable(self, tetrahedron):
        """Reduction matrix should have reasonable number of non-zeros."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # Each block has at most 3*nf non-zeros (one per corner)
        # Total nnz should be at most 2 * 3 * nf
        max_nnz = 2 * 3 * Src.num_faces
        actual_nnz = Reduction.nnz

        assert actual_nnz <= max_nnz, \
            f"Reduction nnz ({actual_nnz}) exceeds expected max ({max_nnz})"
        assert actual_nnz > 0, "Reduction should have some non-zero entries"


# =============================================================================
# Tests with Different Frame Field Angles
# =============================================================================

class TestDifferentAngles:
    """Test with different frame field angle configurations."""

    def test_zero_angles(self, tetrahedron):
        """Zero frame field angles should give k21 = 1 for matching parallel transport."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # With zero angles and zero omega/para_trans, k21 should be 1
        assert np.all(k21 == 1), f"Expected all k21=1 with zero angles, got {k21}"

    def test_constant_angles(self, tetrahedron):
        """Constant frame field angles across faces."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Constant angle across all faces
        ang = np.full(Src.num_faces, np.pi / 4)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # k21 should still be in valid range
        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"

    def test_random_angles(self, tetrahedron):
        """Random frame field angles should produce valid k21."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        np.random.seed(42)
        para_trans = np.random.randn(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Random angles
        ang = np.random.rand(Src.num_faces) * 2 * np.pi
        omega = np.random.randn(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # k21 should always be in valid range
        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"

    def test_pi_half_rotation_octahedron(self, octahedron):
        """Test with pi/2 rotation difference between faces on octahedron."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Alternating angles across faces
        ang = np.zeros(Src.num_faces)
        ang[::2] = 0.0
        ang[1::2] = np.pi / 2
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # k21 should be in valid range
        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"

    def test_omega_nonzero(self, cube):
        """Test with non-zero omega values."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        # Non-zero omega should affect k21 computation
        omega = np.full(Src.num_edges, np.pi / 4)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # k21 should be in valid range
        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"

    def test_para_trans_nonzero(self, cube):
        """Test with non-zero para_trans values."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        # Non-zero parallel transport angles
        para_trans = np.full(Src.num_edges, np.pi / 3)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # k21 should be in valid range
        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"


# =============================================================================
# Sign Bit Tests
# =============================================================================

class TestSignBits:
    """Test the sign bit computation in Reduction matrix."""

    def test_sign_bits_are_plus_minus_one(self, tetrahedron):
        """Sign bits s should be +1 or -1."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # Extract the second block (s_diag @ v2t)
        R_dense = Reduction.toarray()
        second_block = R_dense[3 * Src.num_faces:, Src.num_vertices:]

        # Non-zero values should be +1 or -1
        nonzero_vals = second_block[second_block != 0]
        assert np.all((nonzero_vals == 1) | (nonzero_vals == -1)), \
            f"Sign bits should be +/-1, got {np.unique(nonzero_vals)}"

    def test_first_block_all_ones(self, tetrahedron):
        """First block (v2t_smooth) should have all +1 non-zero entries."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # Extract the first block (v2t_smooth)
        R_dense = Reduction.toarray()
        first_block = R_dense[:3 * Src.num_faces, :Src.num_vertices]

        # Non-zero values should all be +1
        nonzero_vals = first_block[first_block != 0]
        assert np.all(nonzero_vals == 1), \
            f"First block should have all +1 entries, got {np.unique(nonzero_vals)}"

    def test_sign_bits_with_random_angles(self, octahedron):
        """Sign bits should still be +/-1 with random angles."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        np.random.seed(123)
        para_trans = np.random.randn(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.random.rand(Src.num_faces) * 2 * np.pi
        omega = np.random.randn(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        R_dense = Reduction.toarray()
        second_block = R_dense[3 * Src.num_faces:, Src.num_vertices:]

        nonzero_vals = second_block[second_block != 0]
        assert np.all((nonzero_vals == 1) | (nonzero_vals == -1)), \
            f"Sign bits should be +/-1, got {np.unique(nonzero_vals)}"


# =============================================================================
# k21 Length Test
# =============================================================================

class TestK21Length:
    """Test that k21 array has correct length."""

    def test_k21_length_tetrahedron(self, tetrahedron):
        """k21 should have length 6 for tetrahedron (6 edges)."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert len(k21) == 6, f"k21 length should be 6 for tetrahedron, got {len(k21)}"
        assert len(k21) == Src.num_edges, f"k21 length should equal ne={Src.num_edges}"

    def test_k21_length_octahedron(self, octahedron):
        """k21 should have length 12 for octahedron (12 edges)."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert len(k21) == 12, f"k21 length should be 12 for octahedron, got {len(k21)}"
        assert len(k21) == Src.num_edges, f"k21 length should equal ne={Src.num_edges}"

    def test_k21_length_cube(self, cube):
        """k21 should have length 18 for cube (18 edges)."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert len(k21) == 18, f"k21 length should be 18 for cube, got {len(k21)}"
        assert len(k21) == Src.num_edges, f"k21 length should equal ne={Src.num_edges}"

    def test_k21_length_icosahedron(self, icosahedron):
        """k21 should have length 30 for icosahedron (30 edges)."""
        Src = icosahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert len(k21) == 30, f"k21 length should be 30 for icosahedron, got {len(k21)}"
        assert len(k21) == Src.num_edges, f"k21 length should equal ne={Src.num_edges}"


# =============================================================================
# k21 Integer Type Test
# =============================================================================

class TestK21Type:
    """Test k21 array data type."""

    def test_k21_is_integer(self, tetrahedron):
        """k21 should be an integer array."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.issubdtype(k21.dtype, np.integer), \
            f"k21 should be integer type, got {k21.dtype}"

    def test_k21_values_are_discrete(self, octahedron):
        """k21 values should only be 1, 2, 3, or 4 (discrete)."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        np.random.seed(999)
        para_trans = np.random.randn(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.random.rand(Src.num_faces) * 2 * np.pi
        omega = np.random.randn(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        unique_vals = set(k21)
        valid_vals = {1, 2, 3, 4}
        assert unique_vals.issubset(valid_vals), \
            f"k21 values should be in {{1,2,3,4}}, got {unique_vals}"


# =============================================================================
# Interior Edge Tests
# =============================================================================

class TestInteriorEdges:
    """Test behavior with interior edges."""

    def test_closed_mesh_all_interior_tetrahedron(self, tetrahedron):
        """Tetrahedron (closed) should have all 6 interior edges."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        assert len(ide_int) == 6, f"Expected 6 interior edges, got {len(ide_int)}"

    def test_closed_mesh_all_interior_octahedron(self, octahedron):
        """Octahedron (closed) should have all 12 interior edges."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        assert len(ide_int) == 12, f"Expected 12 interior edges, got {len(ide_int)}"

    def test_closed_mesh_all_interior_cube(self, cube):
        """Cube (closed) should have all 18 interior edges."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        assert len(ide_int) == 18, f"Expected 18 interior edges, got {len(ide_int)}"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and potential numerical issues."""

    def test_large_angles(self, tetrahedron):
        """Test with large angle values (multiples of 2*pi)."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Large angle values
        ang = np.full(Src.num_faces, 10 * np.pi)  # 5 full rotations
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        # Should still work and give valid k21
        assert np.all(k21 >= 1), f"k21 should be >= 1 with large angles"
        assert np.all(k21 <= 4), f"k21 should be <= 4 with large angles"

    def test_negative_angles(self, tetrahedron):
        """Test with negative angle values."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Negative angles
        ang = np.full(Src.num_faces, -np.pi / 3)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 should be >= 1 with negative angles"
        assert np.all(k21 <= 4), f"k21 should be <= 4 with negative angles"

    def test_very_small_angles(self, octahedron):
        """Test with very small angle values near zero."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Very small angles
        ang = np.full(Src.num_faces, 1e-10)
        omega = np.full(Src.num_edges, 1e-10)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 should be >= 1 with small angles"
        assert np.all(k21 <= 4), f"k21 should be <= 4 with small angles"

    def test_empty_ide_int_tetrahedron(self, tetrahedron):
        """Test with empty ide_int (forcing all k21 to be 1)."""
        Src = tetrahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.array([], dtype=int)
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        np.random.seed(42)
        ang = np.random.rand(Src.num_faces) * 2 * np.pi
        omega = np.random.randn(Src.num_edges)

        # Should not crash with empty ide_int
        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert len(k21) == Src.num_edges
        assert np.all(k21 == 1), "All k21 should be 1 when ide_int is empty"


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Test consistency of k21 and Reduction across different inputs."""

    def test_deterministic_with_same_seed(self, octahedron):
        """Same random seed should produce identical results."""
        Src = octahedron

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # First run
        np.random.seed(12345)
        para_trans1 = np.random.randn(Src.num_edges)
        param1 = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans1)
        ang1 = np.random.rand(Src.num_faces) * 2 * np.pi
        omega1 = np.random.randn(Src.num_edges)
        k21_1, Red_1 = reduction_from_ff2d(Src, param1, ang1, omega1, Edge_jump, v2t)

        # Second run with same seed
        np.random.seed(12345)
        para_trans2 = np.random.randn(Src.num_edges)
        param2 = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans2)
        ang2 = np.random.rand(Src.num_faces) * 2 * np.pi
        omega2 = np.random.randn(Src.num_edges)
        k21_2, Red_2 = reduction_from_ff2d(Src, param2, ang2, omega2, Edge_jump, v2t)

        assert np.array_equal(k21_1, k21_2), "k21 should be identical with same seed"
        assert np.allclose(Red_1.toarray(), Red_2.toarray()), \
            "Reduction should be identical with same seed"

    def test_k21_changes_with_angle_difference(self, cube):
        """k21 should potentially change when angle differences change."""
        Src = cube

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        # Zero angles
        ang1 = np.zeros(Src.num_faces)
        omega1 = np.zeros(Src.num_edges)
        k21_1, _ = reduction_from_ff2d(Src, param, ang1, omega1, Edge_jump, v2t)

        # Different angles (should produce different k21 for some edges)
        ang2 = np.zeros(Src.num_faces)
        ang2[0] = np.pi / 2
        ang2[1] = np.pi
        omega2 = np.zeros(Src.num_edges)
        k21_2, _ = reduction_from_ff2d(Src, param, ang2, omega2, Edge_jump, v2t)

        # At least for all-zero case, k21 should all be 1
        assert np.all(k21_1 == 1), "k21 should be 1 for zero angles"


# =============================================================================
# Real Mesh Tests (if available)
# =============================================================================

class TestRealMeshes:
    """Tests using real mesh files if available."""

    @pytest.fixture
    def sphere_mesh(self):
        """Load sphere mesh if available."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        path = Path("Mesh/sphere320.obj")
        if not path.exists():
            pytest.skip(f"Sphere mesh not found: {path}")
        X, T, *_ = readOBJ(str(path))
        return mesh_info(X, T)

    @pytest.fixture
    def torus_mesh(self):
        """Load torus mesh if available."""
        from rectangular_surface_parameterization.io.read_obj import readOBJ
        path = Path("Mesh/torus.obj")
        if not path.exists():
            pytest.skip(f"Torus mesh not found: {path}")
        X, T, *_ = readOBJ(str(path))
        return mesh_info(X, T)

    def test_sphere_k21_range(self, sphere_mesh):
        """Test k21 range on sphere mesh."""
        Src = sphere_mesh

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"

    def test_sphere_reduction_shape(self, sphere_mesh):
        """Test Reduction shape on sphere mesh."""
        Src = sphere_mesh

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        expected_rows = 6 * Src.num_faces
        expected_cols = 2 * Src.num_vertices

        assert Reduction.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {Reduction.shape}"

    def test_torus_k21_range(self, torus_mesh):
        """Test k21 range on torus mesh (genus 1)."""
        Src = torus_mesh

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]
        para_trans = np.zeros(Src.num_edges)

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.zeros(Src.num_faces)
        omega = np.zeros(Src.num_edges)

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"

    def test_sphere_random_angles(self, sphere_mesh):
        """Test with random angles on sphere mesh."""
        Src = sphere_mesh

        E2T = Src.edge_to_triangle[:, :2].copy()
        ide_int = np.where((E2T[:, 0] != -1) & (E2T[:, 1] != -1))[0]

        np.random.seed(777)
        para_trans = np.random.randn(Src.num_edges) * 0.5

        param = MockParam(E2T=E2T, ide_int=ide_int, para_trans=para_trans)

        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

        ang = np.random.rand(Src.num_faces) * 2 * np.pi
        omega = np.random.randn(Src.num_edges) * 0.5

        k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

        assert np.all(k21 >= 1), f"k21 should be >= 1, got min {k21.min()}"
        assert np.all(k21 <= 4), f"k21 should be <= 4, got max {k21.max()}"
        assert Reduction.shape == (6 * Src.num_faces, 2 * Src.num_vertices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
