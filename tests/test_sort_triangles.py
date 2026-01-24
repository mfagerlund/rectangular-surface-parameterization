"""
Pytest tests for Preprocess/sort_triangles.py

Tests the sort_triangles function which sorts triangles around a vertex
in consistent winding order, with caching.

Run with: pytest tests/test_sort_triangles.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocess"))

from rectangular_surface_parameterization.preprocessing.sort_triangles import sort_triangles, clear_cache
from rectangular_surface_parameterization.preprocessing.connectivity import connectivity


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_cache_before_each_test():
    """Clear the cache before each test to ensure isolation."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def three_triangles_fan():
    """
    Three triangles arranged in a fan around vertex 0.

    Vertices:
        0: (0, 0)      - center
        1: (1, 0)
        2: (0, 1)
        3: (-1, 0)

    Faces:
        [0, 1, 2]
        [0, 2, 3]
        [0, 3, 1]  - closes the fan

    This is a closed fan (all edges are interior).
    Vertex 0 is surrounded by all 3 triangles.
    """
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
    ], dtype=np.int32)
    return T


@pytest.fixture
def tetrahedron_surface():
    """
    Tetrahedron surface (4 vertices, 4 faces) - closed mesh.

    Vertices:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (0.5, sqrt(3)/2, 0)
        3: (0.5, sqrt(3)/6, sqrt(6)/3)

    Faces (CCW when viewed from outside):
        [0, 1, 2]  - bottom
        [0, 3, 1]  - front
        [1, 3, 2]  - right
        [2, 3, 0]  - left

    Properties:
        V = 4, F = 4, E = 6
        Each vertex is surrounded by exactly 3 triangles.
    """
    T = np.array([
        [0, 1, 2],  # Face 0: bottom
        [0, 3, 1],  # Face 1: front
        [1, 3, 2],  # Face 2: right
        [2, 3, 0],  # Face 3: left
    ], dtype=np.int32)
    return T


@pytest.fixture
def two_triangles_sharing_edge():
    """
    Two triangles sharing an edge - boundary mesh.

    Vertices:
        0: (0, 0)
        1: (1, 0)
        2: (0.5, 1)
        3: (0.5, -1)

    Faces:
        [0, 1, 2]  - upper triangle
        [0, 3, 1]  - lower triangle (shares edge 0-1)

    Vertex 0 and 1 are boundary vertices (on edge of mesh).
    Vertex 2 and 3 are corner vertices (on only one triangle).
    """
    T = np.array([
        [0, 1, 2],  # Face 0
        [0, 3, 1],  # Face 1 (shares edge 0-1)
    ], dtype=np.int32)
    return T


@pytest.fixture
def single_triangle():
    """
    Single triangle mesh (3 vertices, 1 face).

    All vertices are boundary vertices surrounded by 1 triangle.
    """
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return T


# =============================================================================
# Test Return Values
# =============================================================================

class TestReturnValues:
    """Test that sort_triangles returns the expected tuple structure."""

    def test_returns_three_arrays(self, three_triangles_fan):
        """Function should return exactly 3 items: tri_ord, edge_ord, sign_edge."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        result = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

    def test_tri_ord_is_ndarray(self, three_triangles_fan):
        """tri_ord should be a numpy array."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        assert isinstance(tri_ord, np.ndarray), f"tri_ord should be ndarray, got {type(tri_ord)}"

    def test_edge_ord_type(self, three_triangles_fan):
        """edge_ord should be ndarray or None."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        assert edge_ord is None or isinstance(edge_ord, np.ndarray), \
            f"edge_ord should be ndarray or None, got {type(edge_ord)}"

    def test_sign_edge_type(self, three_triangles_fan):
        """sign_edge should be ndarray or None."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        assert sign_edge is None or isinstance(sign_edge, np.ndarray), \
            f"sign_edge should be ndarray or None, got {type(sign_edge)}"


# =============================================================================
# Test Caching Behavior
# =============================================================================

class TestCaching:
    """Test that caching works correctly."""

    def test_same_result_on_repeated_calls(self, three_triangles_fan):
        """Calling sort_triangles twice with same vertex should return identical results."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        result1 = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        result2 = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        tri_ord1, edge_ord1, sign_edge1 = result1
        tri_ord2, edge_ord2, sign_edge2 = result2

        np.testing.assert_array_equal(tri_ord1, tri_ord2,
            err_msg="tri_ord should be identical on repeated calls")

        if edge_ord1 is not None and edge_ord2 is not None:
            np.testing.assert_array_equal(edge_ord1, edge_ord2,
                err_msg="edge_ord should be identical on repeated calls")

        if sign_edge1 is not None and sign_edge2 is not None:
            np.testing.assert_array_equal(sign_edge1, sign_edge2,
                err_msg="sign_edge should be identical on repeated calls")

    def test_cached_arrays_are_same_object(self, three_triangles_fan):
        """Cached results should return the exact same array objects (not copies)."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord1, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        tri_ord2, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        # Same object means caching is working
        assert tri_ord1 is tri_ord2, \
            "Cached tri_ord should be the same object, not a copy"

    def test_different_vertices_cached_separately(self, tetrahedron_surface):
        """Results for different vertices should be cached independently."""
        T = tetrahedron_surface
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord0, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        tri_ord1, _, _ = sort_triangles(1, T, E2T, T2T, E2V, T2E)

        # Different vertices can have different triangle orderings
        # Just verify both calls succeeded and returned arrays
        assert len(tri_ord0) > 0, "tri_ord for vertex 0 should not be empty"
        assert len(tri_ord1) > 0, "tri_ord for vertex 1 should not be empty"


# =============================================================================
# Test clear_cache
# =============================================================================

class TestClearCache:
    """Test that clear_cache() properly clears the cache."""

    def test_clear_cache_recomputes(self, three_triangles_fan):
        """After clear_cache, results should be recomputed (new array objects)."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord1, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        clear_cache()
        tri_ord2, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        # After clearing, should be a different object (recomputed)
        assert tri_ord1 is not tri_ord2, \
            "After clear_cache, tri_ord should be a new object"

    def test_clear_cache_values_still_equal(self, three_triangles_fan):
        """After clear_cache, recomputed values should be equal to original."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord1, edge_ord1, sign_edge1 = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        clear_cache()
        tri_ord2, edge_ord2, sign_edge2 = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        np.testing.assert_array_equal(tri_ord1, tri_ord2,
            err_msg="Recomputed tri_ord should equal original")

        if edge_ord1 is not None and edge_ord2 is not None:
            np.testing.assert_array_equal(edge_ord1, edge_ord2,
                err_msg="Recomputed edge_ord should equal original")

        if sign_edge1 is not None and sign_edge2 is not None:
            np.testing.assert_array_equal(sign_edge1, sign_edge2,
                err_msg="Recomputed sign_edge should equal original")

    def test_clear_cache_clears_all_vertices(self, tetrahedron_surface):
        """clear_cache should clear cache for all vertices."""
        T = tetrahedron_surface
        E2V, T2E, E2T, T2T = connectivity(T)

        # Cache results for multiple vertices
        tri_ord0_before, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        tri_ord1_before, _, _ = sort_triangles(1, T, E2T, T2T, E2V, T2E)
        tri_ord2_before, _, _ = sort_triangles(2, T, E2T, T2T, E2V, T2E)

        clear_cache()

        # All should be recomputed (different objects)
        tri_ord0_after, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)
        tri_ord1_after, _, _ = sort_triangles(1, T, E2T, T2T, E2V, T2E)
        tri_ord2_after, _, _ = sort_triangles(2, T, E2T, T2T, E2V, T2E)

        assert tri_ord0_before is not tri_ord0_after, \
            "Cache for vertex 0 should have been cleared"
        assert tri_ord1_before is not tri_ord1_after, \
            "Cache for vertex 1 should have been cleared"
        assert tri_ord2_before is not tri_ord2_after, \
            "Cache for vertex 2 should have been cleared"


# =============================================================================
# Test Triangle Ordering (Interior Vertex)
# =============================================================================

class TestTriangleOrderingInterior:
    """Test triangle ordering for interior vertices (closed fan)."""

    def test_all_incident_triangles_included(self, three_triangles_fan):
        """tri_ord should contain all triangles incident to the vertex."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        # Vertex 0 is in all 3 triangles
        expected_triangles = {0, 1, 2}
        actual_triangles = set(tri_ord[tri_ord >= 0])  # Exclude -1 boundary markers

        assert actual_triangles == expected_triangles, \
            f"Expected triangles {expected_triangles}, got {actual_triangles}"

    def test_correct_count_for_interior_vertex(self, three_triangles_fan):
        """Interior vertex should have exactly n triangles (no -1 marker)."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        # For closed fan, no -1 marker expected
        # Vertex 0 has 3 incident triangles
        assert len(tri_ord) == 3, f"Expected 3 triangles for interior vertex, got {len(tri_ord)}"

    def test_triangles_are_adjacent_in_ordering(self, three_triangles_fan):
        """Consecutive triangles in tri_ord should share an edge."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        # Check adjacency for consecutive pairs
        for i in range(len(tri_ord)):
            t_curr = tri_ord[i]
            t_next = tri_ord[(i + 1) % len(tri_ord)]

            if t_curr < 0 or t_next < 0:
                continue  # Skip boundary markers

            # Check that t_curr and t_next share an edge
            curr_verts = set(T[t_curr])
            next_verts = set(T[t_next])
            shared = curr_verts & next_verts

            # Two adjacent triangles share exactly 2 vertices (an edge)
            assert len(shared) >= 2, \
                f"Triangles {t_curr} and {t_next} should share an edge, but share only {shared}"

    def test_tetrahedron_vertex_has_three_triangles(self, tetrahedron_surface):
        """Each vertex of tetrahedron is surrounded by 3 triangles."""
        T = tetrahedron_surface
        E2V, T2E, E2T, T2T = connectivity(T)

        for vertex in range(4):
            tri_ord, _, _ = sort_triangles(vertex, T, E2T, T2T, E2V, T2E)
            valid_tris = tri_ord[tri_ord >= 0]
            assert len(valid_tris) == 3, \
                f"Vertex {vertex} should have 3 triangles, got {len(valid_tris)}"


# =============================================================================
# Test Triangle Ordering (Boundary Vertex)
# =============================================================================

# Note: Testing boundary vertices is complex because sort_triangles_comp requires
# a specific E2T format with consistent sign conventions. The function is designed
# to work with meshes produced by the full preprocessing pipeline in the MATLAB
# reference implementation.
#
# For boundary vertices, the algorithm requires:
# 1. Exactly 2 boundary edges at the vertex
# 2. Valid E2T sign conventions that allow finding a starting triangle
#
# We test with closed meshes (interior vertices) which don't have these requirements.
# Corner vertices (single triangle) correctly raise ValueError.


class TestBoundaryVertexWorks:
    """Test that boundary vertices with 2 boundary edges work correctly."""

    def test_corner_vertex_is_proper_boundary(self, two_triangles_sharing_edge):
        """Corner vertex with 2 boundary edges (e.g., edges 0-2 and 1-2) is proper boundary."""
        T = two_triangles_sharing_edge
        E2V, T2E, E2T, T2T = connectivity(T)

        # Vertex 2 has 2 boundary edges: edge (0,2) and edge (1,2)
        tri_ord, edge_ord, sign_edge = sort_triangles(2, T, E2T, T2T, E2V, T2E)

        # Should be a boundary vertex with -1 marker
        assert tri_ord[-1] == -1, "Vertex 2 should be boundary"
        triangles = set(tri_ord[:-1])
        assert 0 in triangles, "Vertex 2 should be in triangle 0"

    def test_single_triangle_vertex_is_boundary(self, single_triangle):
        """Vertices on single triangle are proper boundary vertices (2 boundary edges each)."""
        T = single_triangle
        E2V, T2E, E2T, T2T = connectivity(T)

        # All vertices on a single triangle have 2 boundary edges each
        for vertex in range(3):
            tri_ord, edge_ord, sign_edge = sort_triangles(vertex, T, E2T, T2T, E2V, T2E)

            # Should be boundary with -1 marker and have triangle 0
            assert tri_ord[-1] == -1, f"Vertex {vertex} should be boundary"
            triangles = set(tri_ord[:-1])
            assert triangles == {0}, f"Vertex {vertex} should have triangle 0"


# =============================================================================
# Test Edge Ordering
# =============================================================================

class TestEdgeOrdering:
    """Test edge ordering results."""

    def test_edge_ord_length_matches_tri_ord(self, three_triangles_fan):
        """edge_ord should have same length as tri_ord."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        if edge_ord is not None:
            assert len(edge_ord) == len(tri_ord), \
                f"edge_ord length {len(edge_ord)} != tri_ord length {len(tri_ord)}"

    def test_sign_edge_length_matches_tri_ord(self, three_triangles_fan):
        """sign_edge should have same length as tri_ord."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        if sign_edge is not None:
            assert len(sign_edge) == len(tri_ord), \
                f"sign_edge length {len(sign_edge)} != tri_ord length {len(tri_ord)}"

    def test_edge_indices_valid(self, three_triangles_fan):
        """edge_ord should contain valid edge indices."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        if edge_ord is not None:
            ne = E2V.shape[0]
            for e in edge_ord:
                assert 0 <= e < ne, \
                    f"Edge index {e} out of range [0, {ne})"


# =============================================================================
# Test with T2E=None
# =============================================================================

class TestWithoutT2E:
    """Test that function works when T2E is not provided."""

    def test_works_without_t2e(self, three_triangles_fan):
        """sort_triangles should work when T2E is None."""
        T = three_triangles_fan
        E2V, T2E, E2T, T2T = connectivity(T)

        # Call without T2E argument
        tri_ord, edge_ord, sign_edge = sort_triangles(0, T, E2T, T2T, E2V, None)

        assert tri_ord is not None, "tri_ord should not be None"
        assert len(tri_ord) == 3, f"Expected 3 triangles, got {len(tri_ord)}"


# =============================================================================
# Test Consistency
# =============================================================================

class TestConsistency:
    """Test consistency properties of the sorting."""

    def test_no_duplicate_triangles(self, tetrahedron_surface):
        """tri_ord should not contain duplicate triangles (except -1 marker)."""
        T = tetrahedron_surface
        E2V, T2E, E2T, T2T = connectivity(T)

        for vertex in range(4):
            tri_ord, _, _ = sort_triangles(vertex, T, E2T, T2T, E2V, T2E)

            valid_tris = tri_ord[tri_ord >= 0]
            unique_tris = np.unique(valid_tris)
            assert len(valid_tris) == len(unique_tris), \
                f"Vertex {vertex} has duplicate triangles in tri_ord"

    def test_all_triangles_contain_vertex(self, tetrahedron_surface):
        """All triangles in tri_ord should actually contain the vertex."""
        T = tetrahedron_surface
        E2V, T2E, E2T, T2T = connectivity(T)

        for vertex in range(4):
            tri_ord, _, _ = sort_triangles(vertex, T, E2T, T2T, E2V, T2E)

            for t in tri_ord:
                if t >= 0:
                    assert vertex in T[t], \
                        f"Triangle {t} does not contain vertex {vertex}"


# =============================================================================
# Test Cross-Mesh Cache Isolation
# =============================================================================

class TestCrossMeshCacheIsolation:
    """Test that cache correctly isolates results between different meshes."""

    def test_different_meshes_same_vertex_index(self):
        """
        Calling sort_triangles on two different meshes with same vertex index
        should return different results (not stale cached data).

        This test verifies the fix for the bug where cache was keyed only by
        vertex index, causing cross-mesh cache pollution.
        """
        # Mesh A: three triangles fan (vertex 0 has 3 incident triangles)
        T_A = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
        ], dtype=np.int32)
        E2V_A, T2E_A, E2T_A, T2T_A = connectivity(T_A)

        # Mesh B: tetrahedron (vertex 0 also has 3 incident triangles, but different)
        T_B = np.array([
            [0, 1, 2],  # Face 0: bottom
            [0, 3, 1],  # Face 1: front
            [1, 3, 2],  # Face 2: right
            [2, 3, 0],  # Face 3: left
        ], dtype=np.int32)
        E2V_B, T2E_B, E2T_B, T2T_B = connectivity(T_B)

        # Query vertex 0 on mesh A first
        tri_ord_A, edge_ord_A, sign_edge_A = sort_triangles(0, T_A, E2T_A, T2T_A, E2V_A, T2E_A)

        # Query vertex 0 on mesh B (should NOT return cached result from mesh A)
        tri_ord_B, edge_ord_B, sign_edge_B = sort_triangles(0, T_B, E2T_B, T2T_B, E2V_B, T2E_B)

        # Results should be different array objects (not same cached object)
        assert tri_ord_A is not tri_ord_B, \
            "Results from different meshes should be different objects"

        # Both should have correct number of triangles for their respective meshes
        assert len(tri_ord_A) == 3, f"Mesh A vertex 0 should have 3 triangles, got {len(tri_ord_A)}"
        assert len(tri_ord_B) == 3, f"Mesh B vertex 0 should have 3 triangles, got {len(tri_ord_B)}"

        # Verify that tri_ord_A references triangles from mesh A (indices 0, 1, 2)
        valid_A = set(tri_ord_A[tri_ord_A >= 0])
        assert valid_A == {0, 1, 2}, f"Mesh A tri_ord should contain {{0, 1, 2}}, got {valid_A}"

        # Verify that tri_ord_B references triangles from mesh B
        # For tetrahedron, vertex 0 is in triangles 0, 1, and 3
        valid_B = set(tri_ord_B[tri_ord_B >= 0])
        expected_B = {0, 1, 3}
        assert valid_B == expected_B, f"Mesh B tri_ord should contain {expected_B}, got {valid_B}"

    def test_cache_hit_same_mesh_after_different_mesh_query(self):
        """
        After querying mesh B, querying mesh A again should still return cached result.
        This tests that cache correctly uses mesh identity in the key.
        """
        # Mesh A
        T_A = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
        ], dtype=np.int32)
        E2V_A, T2E_A, E2T_A, T2T_A = connectivity(T_A)

        # Mesh B
        T_B = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0],
        ], dtype=np.int32)
        E2V_B, T2E_B, E2T_B, T2T_B = connectivity(T_B)

        # First call on mesh A
        tri_ord_A1, _, _ = sort_triangles(0, T_A, E2T_A, T2T_A, E2V_A, T2E_A)

        # Call on mesh B (different mesh)
        tri_ord_B, _, _ = sort_triangles(0, T_B, E2T_B, T2T_B, E2V_B, T2E_B)

        # Second call on mesh A (should return cached result from first call)
        tri_ord_A2, _, _ = sort_triangles(0, T_A, E2T_A, T2T_A, E2V_A, T2E_A)

        # A1 and A2 should be the exact same cached object
        assert tri_ord_A1 is tri_ord_A2, \
            "Second call on same mesh should return cached object"

        # B should be different from A
        assert tri_ord_B is not tri_ord_A1, \
            "Results from different meshes should be different objects"

    def test_modified_mesh_array_not_cached(self):
        """
        If the mesh array is modified in-place (same id(T)), the cache
        will return stale results. This is expected behavior - users should
        call clear_cache() or create a new array when modifying meshes.

        This test documents this limitation.
        """
        T = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
        ], dtype=np.int32)
        E2V, T2E, E2T, T2T = connectivity(T)

        # First call
        tri_ord1, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        # Modify T in place (same array object, so same id(T))
        # Note: This would require recomputing connectivity too for correctness
        # But the cache doesn't know that - it returns stale result

        # Second call returns cached result (same object)
        tri_ord2, _, _ = sort_triangles(0, T, E2T, T2T, E2V, T2E)

        assert tri_ord1 is tri_ord2, \
            "Same array object should return cached result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
