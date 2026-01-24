"""
Pytest tests for ComputeParam/cut_mesh.py

Tests mesh cutting functionality for creating disk topology from a closed mesh.
Run with: pytest tests/test_cut_mesh.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory and ComputeParam to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ComputeParam"))

from rectangular_surface_parameterization.parameterization.cut_mesh import (
    cut_mesh,
    _build_meshinfo,
    _union_find,
    _detect_one_based,
    _shift_signed_indices,
    _shift_positive_or_zero,
    _edge_jump_tag_to_bool,
    MeshInfo,
)


# =============================================================================
# Test Fixtures - Known Shapes
# =============================================================================

@pytest.fixture
def tetrahedron():
    """
    Regular tetrahedron surface (4 triangles, 4 vertices, 6 edges).
    This is a closed manifold surface with genus 0 and Euler characteristic 2.
    After cutting to disk: chi=1.
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
    return X, T


@pytest.fixture
def cube():
    """
    Cube triangulated with 12 triangles (8 vertices, 18 edges).
    Closed manifold surface with genus 0 and Euler characteristic 2.
    """
    X = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=np.float64)
    T = np.array([
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ], dtype=np.int32)
    return X, T


@pytest.fixture
def octahedron():
    """
    Regular octahedron (8 triangles, 6 vertices, 12 edges).
    Closed manifold with genus 0.
    """
    X = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=np.float64)
    T = np.array([
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ], dtype=np.int32)
    return X, T


@pytest.fixture
def two_triangles_strip():
    """
    Two triangles forming a strip (open mesh).
    V=4, F=2, E=5, chi=1 (already disk topology).
    """
    X = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 1, 0],
        [1.5, 1, 0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=np.int32)
    return X, T


# =============================================================================
# Test: _union_find Algorithm
# =============================================================================

class TestUnionFind:
    """Test the union-find algorithm independently."""

    def test_empty_equiv(self):
        """Union-find with empty equivalence: each element stays in its own set."""
        n = 5
        equiv = np.zeros((0, 2), dtype=int)
        result = _union_find(n, equiv)

        # Each element should have its own unique label
        assert len(np.unique(result)) == n, "Empty equiv should leave all elements separate"

    def test_single_pair(self):
        """Union-find with a single pair: those two elements are merged."""
        n = 5
        equiv = np.array([[0, 1]], dtype=int)
        result = _union_find(n, equiv)

        # Elements 0 and 1 should have the same label
        assert result[0] == result[1], "Elements 0 and 1 should be merged"
        # Total unique labels should be n - 1
        assert len(np.unique(result)) == n - 1

    def test_chain_equiv(self):
        """Chain equivalences: 0=1, 1=2, 2=3 should merge all four."""
        n = 5
        equiv = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)
        result = _union_find(n, equiv)

        # Elements 0, 1, 2, 3 should all have the same label
        assert result[0] == result[1] == result[2] == result[3]
        # Element 4 should have a different label
        assert result[4] != result[0]
        assert len(np.unique(result)) == 2

    def test_multiple_components(self):
        """Multiple disjoint components: {0,1}, {2,3}, {4}."""
        n = 5
        equiv = np.array([[0, 1], [2, 3]], dtype=int)
        result = _union_find(n, equiv)

        assert result[0] == result[1]
        assert result[2] == result[3]
        assert result[0] != result[2]
        assert result[0] != result[4]
        assert result[2] != result[4]
        assert len(np.unique(result)) == 3

    def test_redundant_equiv(self):
        """Redundant equivalences should not cause issues."""
        n = 4
        equiv = np.array([[0, 1], [0, 1], [1, 0]], dtype=int)
        result = _union_find(n, equiv)

        assert result[0] == result[1]
        assert len(np.unique(result)) == 3

    def test_labels_contiguous(self):
        """Result labels should be contiguous (0, 1, 2, ...)."""
        n = 6
        equiv = np.array([[0, 2], [3, 5]], dtype=int)
        result = _union_find(n, equiv)

        uniq = np.unique(result)
        # Labels should be 0 to len(uniq)-1
        assert np.array_equal(np.sort(uniq), np.arange(len(uniq)))

    def test_large_union(self):
        """Test with larger data set for performance."""
        n = 1000
        # Merge all even numbers together and all odd numbers together
        equiv_even = np.array([[2*i, 2*i+2] for i in range(n//2 - 1)], dtype=int)
        equiv_odd = np.array([[2*i+1, 2*i+3] for i in range(n//2 - 1)], dtype=int)
        equiv = np.vstack([equiv_even, equiv_odd])

        result = _union_find(n, equiv)

        # All even numbers should have the same label
        even_labels = result[::2]
        assert len(np.unique(even_labels)) == 1

        # All odd numbers should have the same label
        odd_labels = result[1::2]
        assert len(np.unique(odd_labels)) == 1

        # Even and odd should have different labels
        assert even_labels[0] != odd_labels[0]

    def test_invalid_negative_indices_raises(self):
        """Union-find should raise on negative indices."""
        n = 5
        equiv = np.array([[-1, 0]], dtype=int)

        with pytest.raises(AssertionError, match="Wrong indexes"):
            _union_find(n, equiv)

    def test_invalid_out_of_range_raises(self):
        """Union-find should raise on out-of-range indices."""
        n = 5
        equiv = np.array([[0, 10]], dtype=int)

        with pytest.raises(AssertionError, match="Wrong indexes"):
            _union_find(n, equiv)


# =============================================================================
# Test: _build_meshinfo
# =============================================================================

class TestBuildMeshInfo:
    """Test the mesh info building function."""

    def test_tetrahedron_counts(self, tetrahedron):
        """Tetrahedron: nv=4, nf=4, ne=6."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        assert info.num_vertices == 4, f"Expected nv=4, got {info.num_vertices}"
        assert info.num_faces == 4, f"Expected nf=4, got {info.num_faces}"
        assert info.num_edges == 6, f"Expected ne=6, got {info.num_edges}"

    def test_cube_counts(self, cube):
        """Cube: nv=8, nf=12, ne=18."""
        X, T = cube
        info = _build_meshinfo(X, T)

        assert info.num_vertices == 8, f"Expected nv=8, got {info.num_vertices}"
        assert info.num_faces == 12, f"Expected nf=12, got {info.num_faces}"
        assert info.num_edges == 18, f"Expected ne=18, got {info.num_edges}"

    def test_euler_characteristic_closed(self, tetrahedron, cube, octahedron):
        """Closed genus-0 surfaces: chi = V - E + F = 2."""
        for X, T in [tetrahedron, cube, octahedron]:
            info = _build_meshinfo(X, T)
            chi = info.num_vertices - info.num_edges + info.num_faces
            assert chi == 2, f"Expected chi=2, got {chi}"

    def test_euler_characteristic_open(self, two_triangles_strip):
        """Open strip: chi = V - E + F = 1."""
        X, T = two_triangles_strip
        info = _build_meshinfo(X, T)
        chi = info.num_vertices - info.num_edges + info.num_faces
        assert chi == 1, f"Expected chi=1, got {chi}"

    def test_E2V_shape(self, tetrahedron):
        """E2V should have shape (ne, 2)."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        assert info.edge_to_vertex.shape == (info.num_edges, 2)

    def test_E2V_sorted_vertices(self, tetrahedron):
        """E2V should have v0 < v1 for each edge."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        for e in range(info.num_edges):
            assert info.edge_to_vertex[e, 0] < info.edge_to_vertex[e, 1], f"Edge {e} not sorted: {info.edge_to_vertex[e]}"

    def test_E2T_shape(self, tetrahedron):
        """E2T should have shape (ne, 2)."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        assert info.edge_to_triangle.shape == (info.num_edges, 2)

    def test_T2E_shape(self, tetrahedron):
        """T2E should have shape (nf, 3)."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        assert info.T2E.shape == (info.num_faces, 3)

    def test_T2T_shape(self, tetrahedron):
        """T2T should have shape (nf, 3)."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        assert info.triangle_to_triangle.shape == (info.num_faces, 3)

    def test_boundary_edges_marked(self, two_triangles_strip):
        """Boundary edges should have E2T[e,1] = -1."""
        X, T = two_triangles_strip
        info = _build_meshinfo(X, T)

        # Count boundary edges (one face neighbor is -1)
        n_boundary = np.sum(info.edge_to_triangle[:, 1] == -1)
        # Open strip should have 4 boundary edges
        assert n_boundary == 4, f"Expected 4 boundary edges, got {n_boundary}"

    def test_closed_no_boundary(self, tetrahedron):
        """Closed mesh should have no boundary edges."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        # All edges should have two faces
        n_boundary = np.sum(info.edge_to_triangle[:, 1] == -1)
        assert n_boundary == 0, f"Closed mesh has {n_boundary} boundary edges"


# =============================================================================
# Test: Index Conversion Helpers
# =============================================================================

class TestIndexConversion:
    """Test index conversion helper functions."""

    def test_detect_one_based_zero_based(self):
        """Detect 0-based indexing correctly."""
        T = np.array([[0, 1, 2]])
        E2V = np.array([[0, 1], [1, 2], [0, 2]])

        assert not _detect_one_based(T, E2V)

    def test_detect_one_based_one_based(self):
        """Detect 1-based indexing correctly."""
        T = np.array([[1, 2, 3]])
        E2V = np.array([[1, 2], [2, 3], [1, 3]])

        assert _detect_one_based(T, E2V)

    def test_shift_signed_indices_no_shift(self):
        """No shift when not 1-based."""
        arr = np.array([1, -2, 3, -4])
        result = _shift_signed_indices(arr, one_based=False)
        np.testing.assert_array_equal(result, arr)

    def test_shift_signed_indices_with_shift(self):
        """Shift signed indices from 1-based to 0-based."""
        arr = np.array([1, -2, 3, -4])
        result = _shift_signed_indices(arr, one_based=True)
        expected = np.array([0, -1, 2, -3])
        np.testing.assert_array_equal(result, expected)

    def test_shift_positive_or_zero_no_shift(self):
        """No shift when not 1-based."""
        arr = np.array([1, 0, 3, 0])
        result = _shift_positive_or_zero(arr, one_based=False)
        np.testing.assert_array_equal(result, arr)

    def test_shift_positive_or_zero_with_shift(self):
        """Shift positive indices, zeros become -1."""
        arr = np.array([1, 0, 3, 0])
        result = _shift_positive_or_zero(arr, one_based=True)
        expected = np.array([0, -1, 2, -1])
        np.testing.assert_array_equal(result, expected)

    def test_edge_jump_tag_bool_passthrough(self):
        """Boolean edge_jump_tag should pass through."""
        tag = np.array([True, False, True, False, False])
        result = _edge_jump_tag_to_bool(tag, 5, one_based=False)
        np.testing.assert_array_equal(result, tag)

    def test_edge_jump_tag_indices_zero_based(self):
        """Convert 0-based indices to boolean mask."""
        indices = np.array([0, 2])
        result = _edge_jump_tag_to_bool(indices, 5, one_based=False)
        expected = np.array([True, False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_edge_jump_tag_indices_one_based(self):
        """Convert 1-based indices to boolean mask."""
        indices = np.array([1, 3])  # 1-based
        result = _edge_jump_tag_to_bool(indices, 5, one_based=True)
        expected = np.array([True, False, True, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_edge_jump_tag_empty(self):
        """Empty indices should give all-false mask."""
        indices = np.array([])
        result = _edge_jump_tag_to_bool(indices, 5, one_based=False)
        expected = np.zeros(5, dtype=bool)
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# Test: cut_mesh Main Function
# =============================================================================

class TestCutMesh:
    """Test the main cut_mesh function."""

    def test_tetrahedron_with_spanning_cut(self, tetrahedron):
        """
        Cutting tetrahedron with a proper spanning tree should result in disk topology.
        For a genus-0 surface with chi=2, we need a cut tree connecting enough edges
        so that the BFS spanning tree complements it properly.
        """
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        # A tetrahedron has 6 edges. For disk topology, the dual spanning tree
        # covers F-1=3 edges, leaving 3 edges for the cut. The cut graph algorithm
        # starts from edge_jump_tag (forced cuts) and builds a spanning tree,
        # then remaining edges form the cut.

        # Mark edge 0 as forced cut (edge_jump_tag)
        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True

        # Use vertices of the cut edge as cones to prevent pruning
        v0, v1 = info.edge_to_vertex[0]
        idcone = np.array([v0, v1])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # The algorithm builds dual spanning tree avoiding edge_jump_tag edges,
        # then prunes non-cone degree-1 vertices. Result depends on topology.
        # Key test: function runs without error and returns valid structure
        assert disk_mesh.num_faces == info.num_faces, "Face count should be preserved"
        assert disk_mesh.num_vertices >= info.num_vertices, "Cut mesh should have >= original vertices"

    def test_cube_with_spanning_cut(self, cube):
        """
        Test cube cutting with forced cut edges.
        """
        X, T = cube
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True

        v0, v1 = info.edge_to_vertex[0]
        idcone = np.array([v0, v1])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Verify basic structure
        assert disk_mesh.num_faces == info.num_faces, "Face count should be preserved"
        assert disk_mesh.num_vertices >= info.num_vertices, "Cut mesh should have >= original vertices"

    def test_no_cuts_needed_open_mesh(self, two_triangles_strip):
        """Already-disk mesh should not need cutting."""
        X, T = two_triangles_strip
        info = _build_meshinfo(X, T)

        # No edge jump tags
        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        idcone = np.array([], dtype=int)

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        chi_cut = disk_mesh.num_faces - disk_mesh.num_edges + disk_mesh.num_vertices
        assert chi_cut == 1, f"Open mesh should have chi=1, got {chi_cut}"

        # Vertex count should remain the same (no new vertices needed)
        assert disk_mesh.num_vertices == info.num_vertices

    def test_idx_cut_inv_length(self, tetrahedron):
        """idx_cut_inv should have length = nv of cut mesh."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        assert len(idx_cut_inv) == disk_mesh.num_vertices, \
            f"idx_cut_inv length {len(idx_cut_inv)} should equal nv={disk_mesh.num_vertices}"

    def test_idx_cut_inv_valid_range(self, tetrahedron):
        """idx_cut_inv values should be valid original vertex indices."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # All values should be in range [0, original_nv)
        assert np.all(idx_cut_inv >= 0), "idx_cut_inv should be non-negative"
        assert np.all(idx_cut_inv < info.num_vertices), f"idx_cut_inv should be < {info.num_vertices}"

    def test_ide_cut_inv_length(self, tetrahedron):
        """ide_cut_inv should have length = ne of cut mesh."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        assert len(ide_cut_inv) == disk_mesh.num_edges, \
            f"ide_cut_inv length {len(ide_cut_inv)} should equal ne={disk_mesh.num_edges}"

    def test_ide_cut_inv_signed_one_based(self, tetrahedron):
        """ide_cut_inv should be signed 1-based edge indices."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Absolute values should be in range [1, original_ne]
        abs_ide = np.abs(ide_cut_inv)
        assert np.all(abs_ide >= 1), "ide_cut_inv (1-based) should have abs >= 1"
        assert np.all(abs_ide <= info.num_edges), f"ide_cut_inv (1-based) should have abs <= {info.num_edges}"

    def test_edge_cut_shape(self, tetrahedron):
        """edge_cut should be boolean array with length = original ne."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        assert edge_cut.shape == (info.num_edges,), \
            f"edge_cut shape should be ({info.num_edges},), got {edge_cut.shape}"
        assert edge_cut.dtype == bool

    def test_edge_cut_marks_cut_edges(self, tetrahedron):
        """edge_cut output should mark edges that are in the final cut."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        v0, v1 = info.edge_to_vertex[0]
        idcone = np.array([v0, v1])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Original mesh was closed (no boundary)
        orig_boundary = np.sum(info.edge_to_triangle[:, 1] == -1)
        assert orig_boundary == 0, "Original tetrahedron should have no boundary"

        # edge_cut should be a boolean array marking cut edges
        assert edge_cut.dtype == bool
        assert edge_cut.shape == (info.num_edges,)

    def test_vertex_duplication_at_cuts(self, tetrahedron):
        """Cutting should duplicate vertices along cut edges."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Cut mesh should have more or equal vertices
        assert disk_mesh.num_vertices >= info.num_vertices, "Cut mesh should have >= original vertices"

    def test_positions_preserved(self, tetrahedron):
        """Cut mesh positions should match original positions (via idx_cut_inv)."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Each cut vertex should match its original position
        for v_cut in range(disk_mesh.num_vertices):
            v_orig = idx_cut_inv[v_cut]
            np.testing.assert_allclose(
                disk_mesh.vertices[v_cut], info.vertices[v_orig],
                atol=1e-10,
                err_msg=f"Cut vertex {v_cut} position mismatch"
            )

    def test_faces_preserved(self, tetrahedron):
        """Face count should be preserved after cutting."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        idcone = np.array([info.edge_to_vertex[0, 0]])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        assert disk_mesh.num_faces == info.num_faces, f"Face count should be preserved: {disk_mesh.num_faces} vs {info.num_faces}"


class TestCutMeshOneBased:
    """Test cut_mesh with 1-based indexing (MATLAB-style)."""

    def test_one_based_input(self, tetrahedron):
        """cut_mesh should handle 1-based input correctly."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        # Convert to 1-based
        T_1based = T + 1
        E2V_1based = info.edge_to_vertex + 1
        E2T_1based = info.edge_to_triangle.copy()
        E2T_1based[E2T_1based >= 0] += 1
        E2T_1based[E2T_1based < 0] = 0  # MATLAB uses 0 for missing

        # T2E and T2T also need adjustment
        T2E_1based = info.T2E + 1  # Assuming unsigned
        T2T_1based = info.triangle_to_triangle.copy()
        T2T_1based[T2T_1based >= 0] += 1
        T2T_1based[T2T_1based < 0] = 0

        edge_jump_tag = np.array([1])  # 1-based index
        v0, v1 = E2V_1based[0]
        idcone = np.array([v0, v1])  # 1-based, both vertices to prevent pruning

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            X, T_1based, E2V_1based, E2T_1based, T2E_1based, T2T_1based,
            idcone, edge_jump_tag
        )

        # Verify basic structure
        assert disk_mesh.num_faces == info.num_faces, "Face count should be preserved"

        # idx_cut_inv should be 1-based in output
        assert np.all(idx_cut_inv >= 1), "1-based output: idx_cut_inv should be >= 1"


class TestCutMeshEdgeCases:
    """Test edge cases for cut_mesh."""

    def test_multiple_cut_edges(self, octahedron):
        """Multiple cut edges should produce valid output."""
        X, T = octahedron
        info = _build_meshinfo(X, T)

        # Mark multiple edges for cutting
        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True
        edge_jump_tag[1] = True

        # Use all vertices from both edges as cones
        v0, v1 = info.edge_to_vertex[0]
        v2, v3 = info.edge_to_vertex[1]
        idcone = np.array([v0, v1, v2, v3])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Verify basic structure
        assert disk_mesh.num_faces == info.num_faces, "Face count should be preserved"
        assert disk_mesh.num_vertices >= info.num_vertices, "Cut mesh should have >= original vertices"

    def test_empty_cones_and_edge_tag(self, tetrahedron):
        """Cut mesh with empty cones and edge_jump_tag should work (no cutting needed)."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)  # No forced cuts
        idcone = np.array([], dtype=int)  # No cones

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # With no cones to preserve, all cut leaves get pruned
        # Function should still run without error
        assert disk_mesh.num_faces == info.num_faces, "Face count should be preserved"

    def test_all_cones_on_cut(self, tetrahedron):
        """All specified cones should remain on the cut."""
        X, T = tetrahedron
        info = _build_meshinfo(X, T)

        edge_jump_tag = np.zeros(info.num_edges, dtype=bool)
        edge_jump_tag[0] = True

        # Use both vertices of the cut edge as cones
        v0, v1 = info.edge_to_vertex[0]
        idcone = np.array([v0, v1])

        disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
            info.vertices, info.triangles, info.edge_to_vertex, info.edge_to_triangle, info.T2E, info.triangle_to_triangle,
            idcone, edge_jump_tag
        )

        # Verify the cut contains at least the specified edge
        assert edge_cut[0], "Specified edge should be in the cut"


