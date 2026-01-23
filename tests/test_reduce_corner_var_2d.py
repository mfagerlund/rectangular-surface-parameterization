"""
Pytest tests for Orthotropic/reduce_corner_var_2d.py

Tests the reduce_corner_var_2d function which computes cross field jumps
and variable reduction for corner variables.

Run with: pytest tests/test_reduce_corner_var_2d.py -v
"""

import numpy as np
import pytest
import scipy.sparse as sp
from pathlib import Path
import sys

# Add parent directory and Orthotropic to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Orthotropic"))
sys.path.insert(0, str(project_root / "Preprocess"))

from Orthotropic.reduce_corner_var_2d import reduce_corner_var_2d
from Preprocess.MeshInfo import MeshInfo, mesh_info
from Preprocess.sort_triangles import clear_cache


# =============================================================================
# Test Fixtures - Simple Test Meshes
# =============================================================================

@pytest.fixture(autouse=True)
def clear_sort_triangles_cache():
    """Clear sort_triangles cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def single_triangle():
    """
    Single triangle mesh (3 vertices, 1 face, 3 edges).

    All vertices are boundary vertices (each has 2 boundary edges).

    NOTE: reduce_corner_var_2d does NOT support this mesh because
    sort_triangles returns None for edge_ord when there's only 1 triangle.
    """
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def two_triangles_shared_edge():
    """
    Two triangles sharing one edge (butterfly shape).

    Vertices 0, 1 are on the shared edge (interior to mesh boundary loop).
    Vertices 2, 3 are corner vertices (each on only one triangle).
    """
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],  # top triangle
        [0, 3, 1],  # bottom triangle (consistent winding for shared edge)
    ], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def tetrahedron():
    """
    Regular tetrahedron surface (4 vertices, 4 faces, 6 edges).

    This is a closed manifold surface with genus 0.
    Each vertex is surrounded by exactly 3 triangles (interior vertex).
    Euler: V - E + F = 4 - 6 + 4 = 2
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
def three_triangles_fan():
    """
    Three triangles arranged in a closed fan around vertex 0.

    Vertices 1, 2, 3 are on the outer ring.
    Vertex 0 is interior (surrounded by all 3 triangles).
    """
    X = np.array([
        [0.0, 0.0, 0.0],   # center vertex
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
    ], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def cube_triangulated():
    """
    Triangulated cube (8 vertices, 12 faces, 18 edges).

    Each face of the cube is split into 2 triangles.
    """
    X = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom vertices
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top vertices
    ], dtype=np.float64)
    T = np.array([
        # Bottom (z=0)
        [0, 2, 1], [0, 3, 2],
        # Top (z=1)
        [4, 5, 6], [4, 6, 7],
        # Front (y=0)
        [0, 1, 5], [0, 5, 4],
        # Back (y=1)
        [3, 6, 2], [3, 7, 6],
        # Left (x=0)
        [0, 4, 7], [0, 7, 3],
        # Right (x=1)
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def octahedron():
    """
    Octahedron surface (6 vertices, 8 faces, 12 edges).

    Regular octahedron - all edges equal length.
    Each vertex is surrounded by 4 triangles.
    """
    X = np.array([
        [1, 0, 0],   # +X
        [-1, 0, 0],  # -X
        [0, 1, 0],   # +Y
        [0, -1, 0],  # -Y
        [0, 0, 1],   # +Z
        [0, 0, -1],  # -Z
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


# =============================================================================
# Test: Return Types and Shapes
# =============================================================================

class TestReturnTypes:
    """Test that reduce_corner_var_2d returns correct types and shapes."""

    def test_returns_three_items(self, tetrahedron):
        """Function should return exactly 3 items."""
        result = reduce_corner_var_2d(tetrahedron)
        assert len(result) == 3, f"Expected 3 return values, got {len(result)}"

    def test_edge_jump_is_sparse_matrix(self, tetrahedron):
        """Edge_jump should be a sparse matrix."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        assert sp.issparse(Edge_jump), f"Edge_jump should be sparse, got {type(Edge_jump)}"

    def test_v2t_is_sparse_matrix(self, tetrahedron):
        """v2t should be a sparse matrix."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        assert sp.issparse(v2t), f"v2t should be sparse, got {type(v2t)}"

    def test_base_tri_is_ndarray(self, tetrahedron):
        """base_tri should be a numpy array."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        assert isinstance(base_tri, np.ndarray), f"base_tri should be ndarray, got {type(base_tri)}"


class TestReturnShapes:
    """Test that returned arrays have correct shapes."""

    def test_edge_jump_shape_tetrahedron(self, tetrahedron):
        """Edge_jump shape should be (3*nf, ne)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        expected_shape = (3 * tetrahedron.nf, tetrahedron.ne)
        assert Edge_jump.shape == expected_shape, \
            f"Edge_jump shape should be {expected_shape}, got {Edge_jump.shape}"

    def test_v2t_shape_tetrahedron(self, tetrahedron):
        """v2t shape should be (3*nf, nv)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        expected_shape = (3 * tetrahedron.nf, tetrahedron.nv)
        assert v2t.shape == expected_shape, \
            f"v2t shape should be {expected_shape}, got {v2t.shape}"

    def test_base_tri_shape_tetrahedron(self, tetrahedron):
        """base_tri shape should be (nv,)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        expected_shape = (tetrahedron.nv,)
        assert base_tri.shape == expected_shape, \
            f"base_tri shape should be {expected_shape}, got {base_tri.shape}"

    @pytest.mark.skip(reason="Single triangle mesh not supported - sort_triangles returns None for edge_ord")
    def test_shapes_single_triangle(self, single_triangle):
        """Test shapes for single triangle mesh.

        NOTE: This test is skipped because reduce_corner_var_2d does not support
        meshes where vertices have only 1 incident triangle. The underlying
        sort_triangles function returns None for edge_ord in such cases.
        """
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(single_triangle)

        expected_edge_jump_shape = (3 * single_triangle.nf, single_triangle.ne)
        expected_v2t_shape = (3 * single_triangle.nf, single_triangle.nv)
        expected_base_tri_shape = (single_triangle.nv,)

        assert Edge_jump.shape == expected_edge_jump_shape, \
            f"Edge_jump shape: expected {expected_edge_jump_shape}, got {Edge_jump.shape}"
        assert v2t.shape == expected_v2t_shape, \
            f"v2t shape: expected {expected_v2t_shape}, got {v2t.shape}"
        assert base_tri.shape == expected_base_tri_shape, \
            f"base_tri shape: expected {expected_base_tri_shape}, got {base_tri.shape}"

    def test_shapes_cube(self, cube_triangulated):
        """Test shapes for cube mesh."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(cube_triangulated)

        # Cube: nv=8, nf=12, ne=18
        expected_edge_jump_shape = (3 * 12, 18)
        expected_v2t_shape = (3 * 12, 8)
        expected_base_tri_shape = (8,)

        assert Edge_jump.shape == expected_edge_jump_shape
        assert v2t.shape == expected_v2t_shape
        assert base_tri.shape == expected_base_tri_shape


# =============================================================================
# Test: Edge_jump Values
# =============================================================================

class TestEdgeJumpValues:
    """Test Edge_jump matrix values are valid."""

    def test_edge_jump_values_in_valid_range(self, tetrahedron):
        """Edge_jump values should be -1, 0, or +1."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        data = Edge_jump.toarray()
        unique_vals = np.unique(data)
        valid_vals = {-1, 0, 1}

        for val in unique_vals:
            assert val in valid_vals, \
                f"Edge_jump contains invalid value {val}, expected only {valid_vals}"

    def test_edge_jump_values_cube(self, cube_triangulated):
        """Edge_jump values should be -1, 0, or +1 for cube."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(cube_triangulated)

        data = Edge_jump.toarray()
        unique_vals = np.unique(data)
        valid_vals = {-1, 0, 1}

        for val in unique_vals:
            assert val in valid_vals, \
                f"Edge_jump contains invalid value {val}"

    def test_edge_jump_values_octahedron(self, octahedron):
        """Edge_jump values should be -1, 0, or +1 for octahedron."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(octahedron)

        data = Edge_jump.toarray()
        unique_vals = np.unique(data)
        valid_vals = {-1, 0, 1}

        for val in unique_vals:
            assert val in valid_vals, \
                f"Edge_jump contains invalid value {val}"

    def test_edge_jump_nonzero_count(self, tetrahedron):
        """Edge_jump should have some nonzero entries for non-trivial mesh."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        nnz = Edge_jump.nnz
        assert nnz > 0, "Edge_jump should have nonzero entries for tetrahedron"

    def test_edge_jump_row_indices_valid(self, tetrahedron):
        """Edge_jump row indices should be valid corner indices [0, 3*nf)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        # Convert to COO format to access row indices
        coo = Edge_jump.tocoo()
        max_row = 3 * tetrahedron.nf - 1

        assert np.all(coo.row >= 0), "Row indices should be non-negative"
        assert np.all(coo.row <= max_row), \
            f"Row indices should be <= {max_row}, got max {np.max(coo.row)}"

    def test_edge_jump_col_indices_valid(self, tetrahedron):
        """Edge_jump column indices should be valid edge indices [0, ne)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        coo = Edge_jump.tocoo()
        max_col = tetrahedron.ne - 1

        assert np.all(coo.col >= 0), "Column indices should be non-negative"
        assert np.all(coo.col <= max_col), \
            f"Column indices should be <= {max_col}, got max {np.max(coo.col)}"


# =============================================================================
# Test: v2t Values
# =============================================================================

class TestV2tValues:
    """Test v2t (vertex to corner) matrix values."""

    def test_v2t_values_are_ones(self, tetrahedron):
        """v2t values should be 1 (mapping indicator)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        data = v2t.data
        np.testing.assert_array_equal(data, np.ones_like(data),
            err_msg="v2t values should all be 1")

    def test_v2t_row_indices_valid(self, tetrahedron):
        """v2t row indices should be valid corner indices."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        coo = v2t.tocoo()
        max_row = 3 * tetrahedron.nf - 1

        assert np.all(coo.row >= 0), "Row indices should be non-negative"
        assert np.all(coo.row <= max_row), \
            f"Row indices should be <= {max_row}"

    def test_v2t_col_indices_valid(self, tetrahedron):
        """v2t column indices should be valid vertex indices."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        coo = v2t.tocoo()
        max_col = tetrahedron.nv - 1

        assert np.all(coo.col >= 0), "Column indices should be non-negative"
        assert np.all(coo.col <= max_col), \
            f"Column indices should be <= {max_col}"

    def test_v2t_covers_all_corners(self, tetrahedron):
        """v2t should map every corner to exactly one vertex."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        # Each row (corner) should have exactly one entry
        row_nnz = np.diff(v2t.indptr)
        np.testing.assert_array_equal(row_nnz, np.ones(3 * tetrahedron.nf),
            err_msg="Each corner should map to exactly one vertex")

    def test_v2t_column_sums_equal_valence(self, tetrahedron):
        """v2t column sums should equal vertex valence (number of corners at each vertex)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        # For tetrahedron, each vertex is in 3 triangles, so valence = 3
        col_sums = np.asarray(v2t.sum(axis=0)).flatten()
        expected_valence = 3  # Each vertex of tetrahedron has 3 incident triangles

        np.testing.assert_array_equal(col_sums, np.full(tetrahedron.nv, expected_valence),
            err_msg=f"Each vertex should have valence {expected_valence}")


# =============================================================================
# Test: base_tri Values
# =============================================================================

class TestBaseTri:
    """Test base_tri (base triangle for each vertex) values."""

    def test_base_tri_values_valid(self, tetrahedron):
        """base_tri values should be valid triangle indices."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        assert np.all(base_tri >= 0), "base_tri values should be non-negative"
        assert np.all(base_tri < tetrahedron.nf), \
            f"base_tri values should be < {tetrahedron.nf}"

    def test_base_tri_contains_vertex(self, tetrahedron):
        """base_tri[v] should be a triangle that contains vertex v."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        for v in range(tetrahedron.nv):
            tri = base_tri[v]
            verts_in_tri = tetrahedron.T[tri, :]
            assert v in verts_in_tri, \
                f"base_tri[{v}]={tri} does not contain vertex {v}, triangle vertices: {verts_in_tri}"

    def test_base_tri_all_vertices_covered(self, cube_triangulated):
        """Every vertex should have a valid base triangle."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(cube_triangulated)

        for v in range(cube_triangulated.nv):
            tri = base_tri[v]
            assert 0 <= tri < cube_triangulated.nf, \
                f"Vertex {v} has invalid base_tri {tri}"
            assert v in cube_triangulated.T[tri, :], \
                f"Vertex {v} not in its base triangle {tri}"


# =============================================================================
# Test: Connectivity Assumptions
# =============================================================================

class TestConnectivity:
    """Test that the function handles different connectivity correctly."""

    def test_closed_mesh_tetrahedron(self, tetrahedron):
        """Tetrahedron (closed manifold) should work without errors."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        # Should complete without exception
        assert Edge_jump is not None
        assert v2t is not None
        assert base_tri is not None

    def test_closed_mesh_octahedron(self, octahedron):
        """Octahedron (closed manifold) should work without errors."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(octahedron)

        assert Edge_jump is not None
        assert v2t is not None
        assert base_tri is not None

    def test_closed_mesh_cube(self, cube_triangulated):
        """Triangulated cube (closed manifold) should work without errors."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(cube_triangulated)

        assert Edge_jump is not None
        assert v2t is not None
        assert base_tri is not None

    @pytest.mark.skip(reason="Single triangle mesh not supported - sort_triangles returns None for edge_ord")
    def test_boundary_mesh_single_triangle(self, single_triangle):
        """Single triangle (boundary mesh) should work.

        NOTE: This test is skipped because reduce_corner_var_2d does not support
        meshes where vertices have only 1 incident triangle.
        """
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(single_triangle)

        assert Edge_jump is not None
        assert v2t is not None
        assert base_tri is not None

    @pytest.mark.skip(reason="Two triangles mesh not supported - boundary vertices with 1 triangle cause issues")
    def test_boundary_mesh_two_triangles(self, two_triangles_shared_edge):
        """Two triangles with boundary should work.

        NOTE: This test is skipped because reduce_corner_var_2d does not support
        meshes where some vertices have only 1 incident triangle (like corner vertices).
        """
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(two_triangles_shared_edge)

        assert Edge_jump is not None
        assert v2t is not None
        assert base_tri is not None


# =============================================================================
# Test: Specific Mesh Properties
# =============================================================================

class TestSpecificMeshProperties:
    """Test properties specific to certain mesh configurations."""

    def test_tetrahedron_corner_count(self, tetrahedron):
        """Tetrahedron: 4 faces * 3 corners = 12 corners."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        n_corners = 3 * tetrahedron.nf
        assert n_corners == 12, f"Tetrahedron should have 12 corners, got {n_corners}"
        assert v2t.shape[0] == 12

    def test_tetrahedron_vertex_count(self, tetrahedron):
        """Tetrahedron: 4 vertices."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        assert len(base_tri) == 4, f"Tetrahedron should have 4 vertices, got {len(base_tri)}"

    def test_tetrahedron_edge_count(self, tetrahedron):
        """Tetrahedron: 6 edges."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        assert Edge_jump.shape[1] == 6, f"Tetrahedron should have 6 edges, got {Edge_jump.shape[1]}"

    def test_octahedron_symmetry(self, octahedron):
        """Octahedron: Each vertex has same valence (4 incident triangles)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(octahedron)

        col_sums = np.asarray(v2t.sum(axis=0)).flatten()
        # Each vertex of octahedron is in 4 triangles
        expected_valence = 4
        np.testing.assert_array_equal(col_sums, np.full(octahedron.nv, expected_valence),
            err_msg=f"Each octahedron vertex should have valence {expected_valence}")


# =============================================================================
# Test: Matrix Properties
# =============================================================================

class TestMatrixProperties:
    """Test mathematical properties of the output matrices."""

    def test_edge_jump_sparsity_reasonable(self, tetrahedron):
        """Edge_jump should be relatively sparse."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        total_elements = Edge_jump.shape[0] * Edge_jump.shape[1]
        sparsity = 1.0 - (Edge_jump.nnz / total_elements)

        # Edge_jump should be fairly sparse (most entries are 0)
        assert sparsity > 0.5, f"Edge_jump sparsity {sparsity:.2%} seems too low"

    def test_v2t_density(self, tetrahedron):
        """v2t should have exactly 3*nf nonzeros (one per corner)."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)

        expected_nnz = 3 * tetrahedron.nf
        assert v2t.nnz == expected_nnz, \
            f"v2t should have {expected_nnz} nonzeros, got {v2t.nnz}"


# =============================================================================
# Test: Corner Index Computation
# =============================================================================

class TestCornerIndices:
    """Test that corner indices are computed correctly."""

    def test_tc_indexing_tetrahedron(self, tetrahedron):
        """Verify Tc corner indexing for tetrahedron."""
        # Tc[f, c] gives corner index for face f, corner c
        # Should be arranged in column-major (Fortran) order like MATLAB
        nf = tetrahedron.nf
        Tc = np.arange(3 * nf).reshape((nf, 3), order='F')

        # First column should be [0, 1, 2, 3]
        np.testing.assert_array_equal(Tc[:, 0], np.arange(nf))
        # Second column should be [4, 5, 6, 7]
        np.testing.assert_array_equal(Tc[:, 1], np.arange(nf, 2*nf))
        # Third column should be [8, 9, 10, 11]
        np.testing.assert_array_equal(Tc[:, 2], np.arange(2*nf, 3*nf))


# =============================================================================
# Test: Consistency with MeshInfo
# =============================================================================

class TestMeshInfoConsistency:
    """Test that output is consistent with MeshInfo structure."""

    def test_edge_count_matches(self, tetrahedron):
        """Edge_jump column count should match mesh.ne."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        assert Edge_jump.shape[1] == tetrahedron.ne

    def test_vertex_count_matches(self, tetrahedron):
        """v2t column count should match mesh.nv."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        assert v2t.shape[1] == tetrahedron.nv

    def test_face_count_matches(self, tetrahedron):
        """Row counts should match 3*mesh.nf."""
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(tetrahedron)
        assert Edge_jump.shape[0] == 3 * tetrahedron.nf
        assert v2t.shape[0] == 3 * tetrahedron.nf


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skip(reason="Single triangle mesh not supported - sort_triangles returns None for edge_ord")
    def test_single_triangle_three_vertices(self, single_triangle):
        """Single triangle has 3 corners and 3 vertices.

        NOTE: Skipped - single triangle mesh not supported.
        """
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(single_triangle)

        assert v2t.shape == (3, 3), f"v2t shape should be (3, 3), got {v2t.shape}"
        assert len(base_tri) == 3, f"base_tri length should be 3, got {len(base_tri)}"

    @pytest.mark.skip(reason="Single triangle mesh not supported - sort_triangles returns None for edge_ord")
    def test_single_triangle_all_base_tri_same(self, single_triangle):
        """Single triangle: all vertices have the same (only) base triangle.

        NOTE: Skipped - single triangle mesh not supported.
        """
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(single_triangle)

        # There's only triangle 0
        np.testing.assert_array_equal(base_tri, np.zeros(3, dtype=int),
            err_msg="All base_tri should be 0 for single triangle")


# =============================================================================
# Test: Known Limitations (Boundary Meshes)
# =============================================================================

class TestKnownLimitations:
    """Test that known limitations are properly documented.

    reduce_corner_var_2d does NOT support meshes with boundaries (open meshes).
    This happens with:
    - Single triangle meshes (all edges are boundary)
    - Meshes with "corner" vertices (like 2-triangle butterfly)

    The function now explicitly rejects such meshes with ValueError.
    """

    def test_single_triangle_raises_or_fails(self, single_triangle):
        """Single triangle mesh should fail with ValueError (boundary edges)."""
        with pytest.raises(ValueError, match="closed meshes only"):
            reduce_corner_var_2d(single_triangle)

    def test_two_triangles_corner_vertex_raises(self, two_triangles_shared_edge):
        """Two triangles with corner vertices should fail (boundary edges)."""
        # Vertices 2 and 3 are corner vertices with only 1 incident triangle
        with pytest.raises(ValueError, match="closed meshes only"):
            reduce_corner_var_2d(two_triangles_shared_edge)


# =============================================================================
# Test: Larger Meshes
# =============================================================================

class TestLargerMeshes:
    """Test with larger meshes to verify scalability."""

    def test_icosahedron(self):
        """Icosahedron: 12 vertices, 20 faces, 30 edges."""
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2

        X = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ], dtype=np.float64)

        T = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ], dtype=np.int32)

        mesh = mesh_info(X, T)
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(mesh)

        # Verify shapes
        assert Edge_jump.shape == (60, 30), f"Edge_jump shape: {Edge_jump.shape}"
        assert v2t.shape == (60, 12), f"v2t shape: {v2t.shape}"
        assert base_tri.shape == (12,), f"base_tri shape: {base_tri.shape}"

        # Verify base_tri validity
        for v in range(12):
            tri = base_tri[v]
            assert v in T[tri, :], f"Vertex {v} not in base_tri {tri}"


# =============================================================================
# Test: Reproducibility
# =============================================================================

class TestReproducibility:
    """Test that results are reproducible."""

    def test_same_mesh_same_result(self, tetrahedron):
        """Calling function twice on same mesh should give same result."""
        Edge_jump1, v2t1, base_tri1 = reduce_corner_var_2d(tetrahedron)

        clear_cache()  # Clear cache to force recomputation

        Edge_jump2, v2t2, base_tri2 = reduce_corner_var_2d(tetrahedron)

        np.testing.assert_array_equal(Edge_jump1.toarray(), Edge_jump2.toarray(),
            err_msg="Edge_jump should be reproducible")
        np.testing.assert_array_equal(v2t1.toarray(), v2t2.toarray(),
            err_msg="v2t should be reproducible")
        np.testing.assert_array_equal(base_tri1, base_tri2,
            err_msg="base_tri should be reproducible")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
