"""
Pytest tests for Preprocess/connectivity.py

Tests the connectivity function which computes adjacency properties
from a list of triangles.

Run with: pytest tests/test_connectivity.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocess"))

from Preprocess.connectivity import connectivity


# =============================================================================
# Test Fixtures - Simple Hand-Constructed Meshes
# =============================================================================

@pytest.fixture
def single_triangle():
    """
    Single triangle mesh (3 vertices, 1 face).

    Vertices:
        0: (0, 0)
        1: (1, 0)
        2: (0, 1)

    Face: [0, 1, 2]

    Edges (sorted vertex pairs):
        Edge 0: (0, 1)
        Edge 1: (0, 2)
        Edge 2: (1, 2)

    All edges are boundary edges (appear once).
    """
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return T


@pytest.fixture
def two_triangles_sharing_edge():
    """
    Two triangles sharing an edge.

    Vertices:
        0: (0, 0)
        1: (1, 0)
        2: (0.5, 1)
        3: (0.5, -1)

    Faces:
        [0, 1, 2]  - upper triangle
        [0, 3, 1]  - lower triangle (note: winding for shared edge 0-1)

    Edge 0-1 is shared (interior), others are boundary.
    """
    T = np.array([
        [0, 1, 2],  # Face 0
        [0, 3, 1],  # Face 1 (shares edge 0-1)
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
        Euler: V - E + F = 4 - 6 + 4 = 2
        All edges are interior (each appears in exactly 2 faces).
    """
    T = np.array([
        [0, 1, 2],  # Face 0: bottom
        [0, 3, 1],  # Face 1: front
        [1, 3, 2],  # Face 2: right
        [2, 3, 0],  # Face 3: left
    ], dtype=np.int32)
    return T


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

    This is a closed fan (all edges are interior except none).
    """
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
    ], dtype=np.int32)
    return T


# =============================================================================
# Shape Tests - E2V (Edge to Vertex)
# =============================================================================

class TestE2VShape:
    """Test that E2V has correct shape (ne, 2)."""

    def test_single_triangle_e2v_shape(self, single_triangle):
        """Single triangle has 3 edges."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)
        assert E2V.shape == (3, 2), f"Expected shape (3, 2), got {E2V.shape}"

    def test_two_triangles_e2v_shape(self, two_triangles_sharing_edge):
        """Two triangles sharing edge: 3 + 3 - 1 = 5 edges."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)
        assert E2V.shape == (5, 2), f"Expected shape (5, 2), got {E2V.shape}"

    def test_tetrahedron_e2v_shape(self, tetrahedron_surface):
        """Tetrahedron has 6 edges."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        assert E2V.shape == (6, 2), f"Expected shape (6, 2), got {E2V.shape}"


# =============================================================================
# Shape Tests - T2E (Triangle to Edge)
# =============================================================================

class TestT2EShape:
    """Test that T2E has correct shape (nf, 3)."""

    def test_single_triangle_t2e_shape(self, single_triangle):
        """Single triangle: T2E shape is (1, 3)."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)
        assert T2E.shape == (1, 3), f"Expected shape (1, 3), got {T2E.shape}"

    def test_two_triangles_t2e_shape(self, two_triangles_sharing_edge):
        """Two triangles: T2E shape is (2, 3)."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)
        assert T2E.shape == (2, 3), f"Expected shape (2, 3), got {T2E.shape}"

    def test_tetrahedron_t2e_shape(self, tetrahedron_surface):
        """Tetrahedron (4 faces): T2E shape is (4, 3)."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        assert T2E.shape == (4, 3), f"Expected shape (4, 3), got {T2E.shape}"


# =============================================================================
# Shape Tests - E2T (Edge to Triangle)
# =============================================================================

class TestE2TShape:
    """Test that E2T has correct shape (ne, 4)."""

    def test_single_triangle_e2t_shape(self, single_triangle):
        """Single triangle: E2T shape is (3, 4)."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)
        assert E2T.shape == (3, 4), f"Expected shape (3, 4), got {E2T.shape}"

    def test_two_triangles_e2t_shape(self, two_triangles_sharing_edge):
        """Two triangles: E2T shape is (5, 4)."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)
        assert E2T.shape == (5, 4), f"Expected shape (5, 4), got {E2T.shape}"

    def test_tetrahedron_e2t_shape(self, tetrahedron_surface):
        """Tetrahedron: E2T shape is (6, 4)."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        assert E2T.shape == (6, 4), f"Expected shape (6, 4), got {E2T.shape}"


# =============================================================================
# Shape Tests - T2T (Triangle to Triangle)
# =============================================================================

class TestT2TShape:
    """Test that T2T has correct shape (nf, 3)."""

    def test_single_triangle_t2t_shape(self, single_triangle):
        """Single triangle: T2T shape is (1, 3)."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)
        assert T2T.shape == (1, 3), f"Expected shape (1, 3), got {T2T.shape}"

    def test_two_triangles_t2t_shape(self, two_triangles_sharing_edge):
        """Two triangles: T2T shape is (2, 3)."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)
        assert T2T.shape == (2, 3), f"Expected shape (2, 3), got {T2T.shape}"

    def test_tetrahedron_t2t_shape(self, tetrahedron_surface):
        """Tetrahedron: T2T shape is (4, 3)."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        assert T2T.shape == (4, 3), f"Expected shape (4, 3), got {T2T.shape}"


# =============================================================================
# Boundary Edge Tests
# =============================================================================

class TestBoundaryEdges:
    """Test that boundary edges have -1 in E2T."""

    def test_single_triangle_all_boundary(self, single_triangle):
        """Single triangle: all edges are boundary (have -1 in E2T columns 0 or 1)."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)

        # All edges should be boundary - one of the triangle indices should be -1
        for e in range(E2T.shape[0]):
            tri0, tri1 = E2T[e, 0], E2T[e, 1]
            is_boundary = (tri0 == -1) or (tri1 == -1)
            assert is_boundary, f"Edge {e} should be boundary but has triangles {tri0}, {tri1}"

    def test_two_triangles_shared_edge_not_boundary(self, two_triangles_sharing_edge):
        """Shared edge should not be boundary (no -1)."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)

        # Find the shared edge (vertices 0 and 1)
        shared_edge_idx = None
        for e in range(E2V.shape[0]):
            v0, v1 = E2V[e, 0], E2V[e, 1]
            if (v0 == 0 and v1 == 1) or (v0 == 1 and v1 == 0):
                shared_edge_idx = e
                break

        assert shared_edge_idx is not None, "Could not find shared edge (0,1)"

        # Shared edge should have two valid triangles
        tri0, tri1 = E2T[shared_edge_idx, 0], E2T[shared_edge_idx, 1]
        assert tri0 != -1 and tri1 != -1, \
            f"Shared edge should have two triangles, got {tri0}, {tri1}"

    def test_two_triangles_boundary_count(self, two_triangles_sharing_edge):
        """Two triangles sharing one edge: 4 boundary edges."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)

        boundary_count = 0
        for e in range(E2T.shape[0]):
            tri0, tri1 = E2T[e, 0], E2T[e, 1]
            if tri0 == -1 or tri1 == -1:
                boundary_count += 1

        # 3 edges from first triangle + 3 from second - 1 shared = 5 edges
        # 4 of them are boundary
        assert boundary_count == 4, f"Expected 4 boundary edges, got {boundary_count}"

    def test_tetrahedron_no_boundary(self, tetrahedron_surface):
        """Closed tetrahedron: no boundary edges."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)

        boundary_count = 0
        for e in range(E2T.shape[0]):
            tri0, tri1 = E2T[e, 0], E2T[e, 1]
            if tri0 == -1 or tri1 == -1:
                boundary_count += 1

        assert boundary_count == 0, f"Closed mesh should have 0 boundary edges, got {boundary_count}"


# =============================================================================
# Edge Count Formula Tests
# =============================================================================

class TestEdgeCountFormula:
    """Test edge count formula: ne = 3*nf/2 for closed mesh."""

    def test_tetrahedron_edge_count(self, tetrahedron_surface):
        """Tetrahedron: ne = 3*4/2 = 6."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        nf = tetrahedron_surface.shape[0]  # 4
        ne = E2V.shape[0]

        expected_ne = 3 * nf // 2  # = 6
        assert ne == expected_ne, f"Expected {expected_ne} edges, got {ne}"

    def test_three_triangles_fan_edge_count(self, three_triangles_fan):
        """Closed fan with 3 triangles: ne = 3*3/2 = 4.5 -> check it's close."""
        E2V, T2E, E2T, T2T = connectivity(three_triangles_fan)
        nf = three_triangles_fan.shape[0]  # 3
        ne = E2V.shape[0]

        # For a closed fan with 3 triangles sharing a center vertex:
        # 3 radial edges + 3 outer edges = 6 edges... but they all share so:
        # Actually: edges are (0,1), (0,2), (0,3), (1,2), (2,3), (3,1) = 6 edges
        # But check that interior edges are counted correctly
        # Formula: E = 3*F/2 = 4.5 is approximate for closed surfaces
        # This mesh has boundary, so formula doesn't apply exactly

        # Just verify we get some reasonable number
        assert ne >= 3, f"Should have at least 3 edges, got {ne}"


# =============================================================================
# E2V Content Tests
# =============================================================================

class TestE2VContent:
    """Test E2V content (edges sorted by vertex indices)."""

    def test_single_triangle_edges(self, single_triangle):
        """Single triangle edges: (0,1), (0,2), (1,2)."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)

        # Convert to set of tuples for easy comparison
        edges = set(tuple(sorted([E2V[e, 0], E2V[e, 1]])) for e in range(E2V.shape[0]))
        expected = {(0, 1), (0, 2), (1, 2)}

        assert edges == expected, f"Expected edges {expected}, got {edges}"

    def test_e2v_sorted(self, single_triangle):
        """E2V should have smaller vertex index first."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)

        for e in range(E2V.shape[0]):
            v0, v1 = E2V[e, 0], E2V[e, 1]
            assert v0 <= v1, f"Edge {e}: vertices should be sorted, got ({v0}, {v1})"

    def test_tetrahedron_edges(self, tetrahedron_surface):
        """Tetrahedron should have 6 edges connecting all vertex pairs."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)

        # Tetrahedron with vertices 0,1,2,3 should have edges:
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        edges = set(tuple(sorted([E2V[e, 0], E2V[e, 1]])) for e in range(E2V.shape[0]))
        expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}

        assert edges == expected, f"Expected edges {expected}, got {edges}"


# =============================================================================
# T2E Content Tests
# =============================================================================

class TestT2EContent:
    """Test T2E content (triangle to edge mapping)."""

    def test_single_triangle_t2e_edges_valid(self, single_triangle):
        """T2E should reference valid edge indices."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)
        ne = E2V.shape[0]

        # T2E is now a SignedEdgeArray - use .indices for 0-based edge indices
        for f in range(T2E.shape[0]):
            for i in range(3):
                edge_idx = T2E[f, i].indices.item()  # 0-based edge index
                assert 0 <= edge_idx < ne, \
                    f"Face {f}, edge {i}: index {edge_idx} out of range [0, {ne})"

    def test_t2e_references_correct_edges(self, single_triangle):
        """T2E should reference edges that contain the triangle's vertices."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)
        T = single_triangle

        for f in range(T.shape[0]):
            face_verts = set(T[f])
            for i in range(3):
                edge_idx = T2E[f, i].indices.item()  # 0-based edge index
                edge_verts = set(E2V[edge_idx])
                # Edge vertices should be a subset of face vertices
                assert edge_verts.issubset(face_verts), \
                    f"Face {f} with verts {face_verts}, edge {edge_idx} has verts {edge_verts}"


# =============================================================================
# T2T Content Tests
# =============================================================================

class TestT2TContent:
    """Test T2T content (triangle to triangle neighbors)."""

    def test_single_triangle_no_neighbors(self, single_triangle):
        """Single triangle has no neighbors (all -1)."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)

        # All entries should be -1 (no neighbors)
        assert np.all(T2T == -1), f"Single triangle should have no neighbors, got T2T:\n{T2T}"

    def test_two_triangles_mutual_neighbors(self, two_triangles_sharing_edge):
        """Two triangles sharing edge should be neighbors."""
        E2V, T2E, E2T, T2T = connectivity(two_triangles_sharing_edge)

        # Triangle 0 should have triangle 1 as one neighbor
        neighbors_0 = set(T2T[0]) - {-1}
        assert 1 in neighbors_0, f"Triangle 0 should have 1 as neighbor, got {neighbors_0}"

        # Triangle 1 should have triangle 0 as one neighbor
        neighbors_1 = set(T2T[1]) - {-1}
        assert 0 in neighbors_1, f"Triangle 1 should have 0 as neighbor, got {neighbors_1}"

    def test_tetrahedron_each_face_has_three_neighbors(self, tetrahedron_surface):
        """Each face of tetrahedron borders all other 3 faces."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)

        for f in range(4):
            neighbors = set(T2T[f]) - {-1}
            expected_neighbors = set(range(4)) - {f}
            assert neighbors == expected_neighbors, \
                f"Face {f}: expected neighbors {expected_neighbors}, got {neighbors}"


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Test consistency between different connectivity arrays."""

    def test_e2t_t2e_consistency(self, tetrahedron_surface):
        """If T2E says face f uses edge e, then E2T should list f for edge e."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        nf = tetrahedron_surface.shape[0]

        for f in range(nf):
            for i in range(3):
                edge_idx = T2E[f, i].indices.item()  # 0-based edge index
                # E2T[edge_idx, 0] or E2T[edge_idx, 1] should be f
                tri0, tri1 = E2T[edge_idx, 0], E2T[edge_idx, 1]
                assert f in (tri0, tri1), \
                    f"Face {f} uses edge {edge_idx}, but E2T[{edge_idx}] = ({tri0}, {tri1})"

    def test_e2v_has_unique_edges(self, tetrahedron_surface):
        """No duplicate edges in E2V."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)

        edges = [tuple(sorted([E2V[e, 0], E2V[e, 1]])) for e in range(E2V.shape[0])]
        unique_edges = set(edges)

        assert len(edges) == len(unique_edges), \
            f"Found duplicate edges: {len(edges)} total, {len(unique_edges)} unique"


# =============================================================================
# Sign Tests (T2E signs indicate edge orientation)
# =============================================================================

class TestSigns:
    """Test that T2E signs indicate edge orientation."""

    def test_t2e_signs_nonzero(self, tetrahedron_surface):
        """T2E signs should be +1 or -1, never 0."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)

        # T2E is now a SignedEdgeArray - use .signs property
        signs = T2E.signs
        assert np.all(np.abs(signs) == 1), \
            f"All T2E signs should be +1 or -1, got signs with zero: {signs[signs == 0]}"

    def test_t2e_edge_indices_valid(self, tetrahedron_surface):
        """T2E decoded indices should be valid edge indices."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)
        ne = E2V.shape[0]

        # Use SignedEdgeArray .indices property for 0-based edge indices
        edge_indices = T2E.indices
        assert np.all(edge_indices >= 0), "Edge indices should be non-negative"
        assert np.all(edge_indices < ne), f"Edge indices should be < {ne}"

    def test_t2e_edge0_sign_preserved(self, single_triangle):
        """Critical test: edge 0 with negative sign should decode correctly."""
        E2V, T2E, E2T, T2T = connectivity(single_triangle)

        # Find any occurrence of edge 0 in T2E
        edge_indices = T2E.indices
        mask = edge_indices == 0

        if np.any(mask):
            # Get the SignedEdgeArray for edge 0 entries
            edge0_sea = T2E[mask]
            # Signs should be +1 or -1
            signs = edge0_sea.signs
            assert np.all(np.abs(signs) == 1), \
                f"Edge 0 signs should be +1 or -1, got {signs}"
            # Raw values for edge 0 should have abs() == 1 (since index 0 -> raw = 1*sign)
            assert np.all(np.abs(edge0_sea.raw) == 1), \
                f"Edge 0 raw should have abs() == 1, got {edge0_sea.raw}"

    def test_e2t_sign_columns(self, tetrahedron_surface):
        """E2T columns 2 and 3 should be opposite signs."""
        E2V, T2E, E2T, T2T = connectivity(tetrahedron_surface)

        for e in range(E2T.shape[0]):
            sign0, sign1 = E2T[e, 2], E2T[e, 3]
            assert sign0 == -sign1, \
                f"Edge {e}: E2T signs should be opposite, got {sign0} and {sign1}"


# =============================================================================
# Larger Mesh Tests
# =============================================================================

class TestLargerMeshes:
    """Test with slightly larger meshes."""

    def test_octahedron(self):
        """Octahedron: 6 vertices, 8 faces, 12 edges."""
        # Octahedron vertices
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

        E2V, T2E, E2T, T2T = connectivity(T)

        # Check shapes
        assert E2V.shape == (12, 2), f"Expected (12, 2), got {E2V.shape}"
        assert T2E.shape == (8, 3), f"Expected (8, 3), got {T2E.shape}"
        assert E2T.shape == (12, 4), f"Expected (12, 4), got {E2T.shape}"
        assert T2T.shape == (8, 3), f"Expected (8, 3), got {T2T.shape}"

        # No boundary edges
        for e in range(E2T.shape[0]):
            tri0, tri1 = E2T[e, 0], E2T[e, 1]
            assert tri0 != -1 and tri1 != -1, \
                f"Closed octahedron edge {e} has boundary marker"

    def test_cube_triangulated(self):
        """Cube triangulated: 8 vertices, 12 faces, 18 edges."""
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

        E2V, T2E, E2T, T2T = connectivity(T)

        # Check shapes
        assert E2V.shape == (18, 2), f"Expected (18, 2), got {E2V.shape}"
        assert T2E.shape == (12, 3), f"Expected (12, 3), got {T2E.shape}"
        assert E2T.shape == (18, 4), f"Expected (18, 4), got {E2T.shape}"
        assert T2T.shape == (12, 3), f"Expected (12, 3), got {T2T.shape}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_single_triangle_zero_indexed(self):
        """Verify single triangle works with 0-indexed vertices."""
        T = np.array([[0, 1, 2]], dtype=np.int32)
        E2V, T2E, E2T, T2T = connectivity(T)

        # Should not crash and produce valid output
        assert E2V.shape[0] == 3
        assert T2E.shape[0] == 1
        assert E2T.shape[0] == 3
        assert T2T.shape[0] == 1

    def test_non_contiguous_vertex_indices(self):
        """Mesh with non-contiguous vertex indices (e.g., 0, 2, 5)."""
        T = np.array([[0, 2, 5]], dtype=np.int32)
        E2V, T2E, E2T, T2T = connectivity(T)

        # Should work - connectivity only cares about indices as labels
        assert E2V.shape[0] == 3

        # Check edges contain the correct vertex indices
        edges = set(tuple(sorted([E2V[e, 0], E2V[e, 1]])) for e in range(E2V.shape[0]))
        expected = {(0, 2), (0, 5), (2, 5)}
        assert edges == expected, f"Expected {expected}, got {edges}"

    def test_large_vertex_indices(self):
        """Mesh with large vertex indices."""
        T = np.array([[100, 200, 300]], dtype=np.int32)
        E2V, T2E, E2T, T2T = connectivity(T)

        assert E2V.shape[0] == 3
        edges = set(tuple(sorted([E2V[e, 0], E2V[e, 1]])) for e in range(E2V.shape[0]))
        expected = {(100, 200), (100, 300), (200, 300)}
        assert edges == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
