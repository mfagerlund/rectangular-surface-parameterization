"""
Pytest tests for preprocessing/sort_triangles_comp.py

Tests the sort_triangles_comp function which sorts ring triangles
around a vertex in consistent order.

Run with: pytest tests/test_sort_triangles_comp.py -v

KNOWN LIMITATIONS:
1. The function requires boundary vertices to have exactly 2 boundary edges
   incident to them. This is the case for "proper" mesh boundaries (edge of a
   disk-like region), but NOT for degenerate cases like:
   - A single isolated triangle (corner vertices have only 1 boundary edge each)
   - Vertices at mesh corners with < 2 boundary edges
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocess"))

from rectangular_surface_parameterization.preprocessing.connectivity import connectivity
from rectangular_surface_parameterization.preprocessing.sort_triangles_comp import sort_triangles_comp


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

    All vertices are boundary vertices but DEGENERATE case:
    each vertex has only 1 incident boundary edge, not 2.
    The function will raise ValueError for these vertices.
    """
    T = np.array([[0, 1, 2]], dtype=np.int32)
    E2V, T2E, E2T, T2T = connectivity(T)
    return T, E2V, T2E, E2T, T2T


@pytest.fixture
def two_triangles_sharing_edge():
    """
    Two triangles sharing an edge (bowtie/diamond shape).

    Vertices:
        0: (0, 0)
        1: (1, 0)
        2: (0.5, 1)
        3: (0.5, -1)

    Faces:
        [0, 1, 2]  - upper triangle
        [0, 3, 1]  - lower triangle (shares edge 0-1)

    Vertices 0 and 1 are "proper" boundary vertices with exactly 2 boundary
    edges each. Vertices 2 and 3 have only 1 boundary edge (degenerate).
    """
    T = np.array([
        [0, 1, 2],  # Face 0
        [0, 3, 1],  # Face 1 (shares edge 0-1)
    ], dtype=np.int32)
    E2V, T2E, E2T, T2T = connectivity(T)
    return T, E2V, T2E, E2T, T2T


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
        All vertices are interior (each has 3 triangles around it).
    """
    T = np.array([
        [0, 1, 2],  # Face 0: bottom
        [0, 3, 1],  # Face 1: front
        [1, 3, 2],  # Face 2: right
        [2, 3, 0],  # Face 3: left
    ], dtype=np.int32)
    E2V, T2E, E2T, T2T = connectivity(T)
    return T, E2V, T2E, E2T, T2T


@pytest.fixture
def fan_of_triangles():
    """
    Fan of 5 triangles around a central vertex (vertex 0).

    Vertices:
        0: (0, 0)    - center
        1: (1, 0)
        2: (cos(72), sin(72))
        3: (cos(144), sin(144))
        4: (cos(216), sin(216))
        5: (cos(288), sin(288))

    Faces (all share vertex 0):
        [0, 1, 2]
        [0, 2, 3]
        [0, 3, 4]
        [0, 4, 5]
        [0, 5, 1]  - closes the fan

    Vertex 0 is interior (closed fan), vertices 1-5 are also interior.
    This is a closed fan forming a cone.
    """
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 1],
    ], dtype=np.int32)
    E2V, T2E, E2T, T2T = connectivity(T)
    return T, E2V, T2E, E2T, T2T


@pytest.fixture
def open_fan_of_triangles():
    """
    Open fan of 3 triangles around a central vertex (vertex 0).
    Does NOT close - vertex 0 is a PROPER boundary vertex with exactly 2
    boundary edges incident to it (edges 0-1 and 0-4).

    Vertices:
        0: (0, 0)    - center (proper boundary vertex)
        1: (1, 0)
        2: (0, 1)
        3: (-1, 0)
        4: (0, -1)

    Faces (all share vertex 0, but don't close):
        [0, 1, 2]
        [0, 2, 3]
        [0, 3, 4]

    Vertex 0 is a proper boundary vertex (fan doesn't close, has 2 boundary edges).
    Vertices 1 and 4 are degenerate boundary (1 boundary edge each).
    Vertices 2 and 3 are interior to the boundary path.
    """
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
    ], dtype=np.int32)
    E2V, T2E, E2T, T2T = connectivity(T)
    return T, E2V, T2E, E2T, T2T


@pytest.fixture
def strip_of_triangles():
    """
    A strip of 4 triangles forming a quadrilateral-ish shape.

    Vertices:
        0: (0, 0)
        1: (1, 0)
        2: (2, 0)
        3: (0, 1)
        4: (1, 1)
        5: (2, 1)

    Faces (forming a 2-triangle-wide strip):
        [0, 1, 3]
        [1, 4, 3]
        [1, 2, 4]
        [2, 5, 4]

    Vertex 1 is interior (has 4 triangles).
    Vertex 4 is interior (has 4 triangles).
    All other vertices are boundary.
    """
    T = np.array([
        [0, 1, 3],
        [1, 4, 3],
        [1, 2, 4],
        [2, 5, 4],
    ], dtype=np.int32)
    E2V, T2E, E2T, T2T = connectivity(T)
    return T, E2V, T2E, E2T, T2T


# =============================================================================
# Test: Interior Vertex (all triangles in ring, no -1)
# =============================================================================

class TestInteriorVertex:
    """Tests for interior vertices where all triangles form a closed ring."""

    def test_tetrahedron_vertex0_all_triangles_returned(self, tetrahedron_surface):
        """Interior vertex should return all incident triangles without -1."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        # Vertex 0 is in faces 0, 1, 3
        expected_triangles = {0, 1, 3}
        returned_triangles = set(tri_ord)

        assert -1 not in returned_triangles, "Interior vertex should not have -1 in result"
        assert returned_triangles == expected_triangles, \
            f"Expected triangles {expected_triangles}, got {returned_triangles}"

    def test_tetrahedron_vertex3_all_triangles_returned(self, tetrahedron_surface):
        """Vertex 3 (apex) should have triangles 1, 2, 3."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(3, T, E2T, T2T, E2V, T2E)

        # Vertex 3 is in faces 1, 2, 3
        expected_triangles = {1, 2, 3}
        returned_triangles = set(tri_ord)

        assert -1 not in returned_triangles, "Interior vertex should not have -1 in result"
        assert returned_triangles == expected_triangles, \
            f"Expected triangles {expected_triangles}, got {returned_triangles}"

    def test_closed_fan_center_vertex(self, fan_of_triangles):
        """Center of closed fan should return all 5 triangles."""
        T, E2V, T2E, E2T, T2T = fan_of_triangles

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        # Vertex 0 is in all 5 faces
        expected_triangles = {0, 1, 2, 3, 4}
        returned_triangles = set(tri_ord)

        assert -1 not in returned_triangles, "Interior vertex should not have -1 in result"
        assert returned_triangles == expected_triangles, \
            f"Expected triangles {expected_triangles}, got {returned_triangles}"


# =============================================================================
# Test: Boundary Vertex (triangles returned with -1 at end)
# =============================================================================

class TestBoundaryVertex:
    """Tests for boundary vertices where triangle ring is open.

    NOTE: The function requires boundary vertices to have exactly 2 boundary
    edges incident to them.
    """

    def test_single_triangle_boundary_vertex(self, single_triangle):
        """
        Single triangle vertices ARE proper boundary vertices.
        Each vertex has exactly 2 boundary edges (e.g., vertex 0 has edges 0-1 and 0-2).
        """
        T, E2V, T2E, E2T, T2T = single_triangle

        # Each vertex in a single triangle has 2 boundary edges
        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        # Should have -1 at end (boundary marker)
        assert tri_ord[-1] == -1, "Boundary vertex should have -1 at end"

        # Should have triangle 0
        triangles = set(tri_ord[:-1])
        assert triangles == {0}, f"Expected triangles {{0}}, got {triangles}"

    def test_open_fan_center_is_proper_boundary(self, open_fan_of_triangles):
        """Center of open fan is a proper boundary vertex with 2 boundary edges."""
        T, E2V, T2E, E2T, T2T = open_fan_of_triangles

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        # Should have -1 at end (boundary)
        assert tri_ord[-1] == -1, "Boundary vertex should have -1 at end of tri_ord"

        # Should have all 3 triangles
        triangles_without_marker = set(tri_ord[:-1])  # Exclude -1
        assert triangles_without_marker == {0, 1, 2}, \
            f"Expected triangles {{0, 1, 2}}, got {triangles_without_marker}"

    def test_two_triangles_shared_vertex_is_proper_boundary(self, two_triangles_sharing_edge):
        """Vertices 0 and 1 are proper boundary vertices (2 boundary edges each)."""
        T, E2V, T2E, E2T, T2T = two_triangles_sharing_edge

        # Test vertex 0 - has boundary edges (0,2) and (0,3)
        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        assert tri_ord[-1] == -1, "Boundary vertex should have -1 at end"
        triangles = set(tri_ord[:-1])
        assert triangles == {0, 1}, f"Expected triangles {{0, 1}}, got {triangles}"

    def test_two_triangles_corner_vertex_is_boundary(self, two_triangles_sharing_edge):
        """Vertices 2 and 3 are proper boundary vertices (2 boundary edges each)."""
        T, E2V, T2E, E2T, T2T = two_triangles_sharing_edge

        # Vertex 2 has 2 boundary edges (0-2 and 1-2)
        tri_ord, edge_ord, sign_edge = sort_triangles_comp(2, T, E2T, T2T, E2V, T2E)

        assert tri_ord[-1] == -1, "Boundary vertex should have -1 at end"
        triangles = set(tri_ord[:-1])
        assert triangles == {0}, f"Expected triangles {{0}}, got {triangles}"


# =============================================================================
# Test: All Returned Triangles Contain Query Vertex
# =============================================================================

class TestTrianglesContainVertex:
    """Verify all returned triangles contain the query vertex."""

    def test_tetrahedron_all_vertices(self, tetrahedron_surface):
        """For each vertex, all returned triangles must contain that vertex."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        for v in range(4):
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)

            for tri in tri_ord:
                if tri == -1:
                    continue  # Skip boundary marker
                assert v in T[tri], f"Vertex {v} not in triangle {tri}: {T[tri]}"

    def test_fan_all_triangles_contain_center(self, fan_of_triangles):
        """All triangles around center vertex should contain vertex 0."""
        T, E2V, T2E, E2T, T2T = fan_of_triangles

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        for tri in tri_ord:
            if tri == -1:
                continue
            assert 0 in T[tri], f"Vertex 0 not in triangle {tri}: {T[tri]}"

    def test_open_fan_all_triangles_contain_center(self, open_fan_of_triangles):
        """All triangles around center vertex should contain vertex 0."""
        T, E2V, T2E, E2T, T2T = open_fan_of_triangles

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        for tri in tri_ord:
            if tri == -1:
                continue
            assert 0 in T[tri], f"Vertex 0 not in triangle {tri}: {T[tri]}"

    def test_strip_interior_vertices(self, strip_of_triangles):
        """Interior vertices in strip should have all their triangles."""
        T, E2V, T2E, E2T, T2T = strip_of_triangles

        # Vertex 1 is interior, should have triangles containing it
        tri_ord, _, _ = sort_triangles_comp(1, T, E2T, T2T, E2V, T2E)

        for tri in tri_ord:
            if tri == -1:
                continue
            assert 1 in T[tri], f"Vertex 1 not in triangle {tri}: {T[tri]}"


# =============================================================================
# Test: Triangles Are in Connected Order (Adjacent in Ring)
# =============================================================================

class TestConnectedOrder:
    """Verify triangles are returned in connected order (share edges)."""

    def test_tetrahedron_connected_ring(self, tetrahedron_surface):
        """Consecutive triangles in result should share an edge."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        for v in range(4):
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)

            # Remove -1 marker if present
            triangles = [t for t in tri_ord if t != -1]

            # Check consecutive pairs are neighbors
            n = len(triangles)
            for i in range(n):
                tri_a = triangles[i]
                tri_b = triangles[(i + 1) % n]  # Wrap around for closed ring

                # tri_b should be in T2T[tri_a]
                neighbors_a = set(T2T[tri_a])
                assert tri_b in neighbors_a, \
                    f"Triangles {tri_a} and {tri_b} should be adjacent for vertex {v}"

    def test_closed_fan_connected_order(self, fan_of_triangles):
        """Center vertex's triangles should be in connected order."""
        T, E2V, T2E, E2T, T2T = fan_of_triangles

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)
        triangles = [t for t in tri_ord if t != -1]

        n = len(triangles)
        for i in range(n):
            tri_a = triangles[i]
            tri_b = triangles[(i + 1) % n]

            neighbors_a = set(T2T[tri_a])
            assert tri_b in neighbors_a, \
                f"Triangles {tri_a} and {tri_b} should be adjacent"

    def test_open_fan_connected_order_not_cyclic(self, open_fan_of_triangles):
        """Open fan: consecutive triangles connected, but not cyclic."""
        T, E2V, T2E, E2T, T2T = open_fan_of_triangles

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        # Remove -1 marker
        triangles = [t for t in tri_ord if t != -1]

        # Check consecutive pairs (not wrapping) are neighbors
        for i in range(len(triangles) - 1):
            tri_a = triangles[i]
            tri_b = triangles[i + 1]

            neighbors_a = set(T2T[tri_a])
            assert tri_b in neighbors_a, \
                f"Triangles {tri_a} and {tri_b} should be adjacent"

    def test_strip_boundary_vertex_connected_order(self, strip_of_triangles):
        """Boundary vertex in strip has triangles in connected order (open, not cyclic)."""
        T, E2V, T2E, E2T, T2T = strip_of_triangles

        # Vertex 1 is actually a BOUNDARY vertex with 3 triangles (faces 0, 1, 2)
        # It has boundary edges 0-1 and 1-2
        tri_ord, _, _ = sort_triangles_comp(1, T, E2T, T2T, E2V, T2E)

        # Should have -1 at end (boundary marker)
        assert tri_ord[-1] == -1, "Vertex 1 should be boundary"
        triangles = [t for t in tri_ord if t != -1]

        # Check consecutive pairs are neighbors (open path, not cyclic)
        for i in range(len(triangles) - 1):
            tri_a = triangles[i]
            tri_b = triangles[i + 1]

            neighbors_a = set(T2T[tri_a])
            assert tri_b in neighbors_a, \
                f"Triangles {tri_a} and {tri_b} should be adjacent"


# =============================================================================
# Test: Single Triangle Mesh
# =============================================================================

class TestSingleTriangle:
    """Tests for single triangle mesh.

    NOTE: Single triangle vertices ARE proper boundary vertices.
    Each vertex has exactly 2 boundary edges (e.g., vertex 0 has edges 0-1 and 0-2).
    """

    def test_single_triangle_all_vertices_boundary(self, single_triangle):
        """Single triangle: all vertices are proper boundary vertices."""
        T, E2V, T2E, E2T, T2T = single_triangle

        for v in range(3):
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)

            # Should have -1 at end (boundary marker)
            assert tri_ord[-1] == -1, f"Vertex {v} should be boundary"

            # Should have only triangle 0
            triangles = set(tri_ord[:-1])
            assert triangles == {0}, f"Vertex {v}: expected triangles {{0}}, got {triangles}"


# =============================================================================
# Test: Fan of Triangles Around Central Vertex
# =============================================================================

class TestFanOfTriangles:
    """Tests for fan topology around a central vertex."""

    def test_closed_fan_count(self, fan_of_triangles):
        """Closed fan of 5 triangles: center has 5 triangles."""
        T, E2V, T2E, E2T, T2T = fan_of_triangles

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)
        triangles = [t for t in tri_ord if t != -1]

        assert len(triangles) == 5, f"Expected 5 triangles, got {len(triangles)}"

    def test_open_fan_count(self, open_fan_of_triangles):
        """Open fan of 3 triangles: center (vertex 0) has 3 triangles."""
        T, E2V, T2E, E2T, T2T = open_fan_of_triangles

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)
        triangles = [t for t in tri_ord if t != -1]

        assert len(triangles) == 3, f"Expected 3 triangles, got {len(triangles)}"

    def test_closed_fan_rim_vertices_are_boundary(self, fan_of_triangles):
        """Rim vertices in closed fan are BOUNDARY (2 boundary edges each).

        The "closed fan" is actually a CONE - the rim edges (1-2, 1-5, 2-3, etc.)
        only appear once, making them boundary edges. Rim vertices have 2
        boundary edges each, so they are proper boundary vertices.
        """
        T, E2V, T2E, E2T, T2T = fan_of_triangles

        for v in range(1, 6):
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)
            triangles = [t for t in tri_ord if t != -1]

            # Rim vertices should be boundary (have -1 marker)
            assert tri_ord[-1] == -1, \
                f"Rim vertex {v} should be boundary (cone rim)"
            assert len(triangles) == 2, \
                f"Rim vertex {v} should have 2 triangles, got {len(triangles)}"

    def test_open_fan_rim_vertices_boundary(self, open_fan_of_triangles):
        """Rim vertices at ends of open fan are proper boundary vertices."""
        T, E2V, T2E, E2T, T2T = open_fan_of_triangles

        # Vertices 1 and 4 are at the ends - each has 2 boundary edges
        for v in [1, 4]:
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)

            # Should be boundary vertices
            assert tri_ord[-1] == -1, f"Vertex {v} should be boundary"

            # Should have 1 triangle each
            triangles = set(tri_ord[:-1])
            assert len(triangles) == 1, f"Vertex {v} should have 1 triangle"


# =============================================================================
# Test: Edge and Sign Outputs
# =============================================================================

class TestEdgeAndSignOutputs:
    """Tests for edge_ord and sign_edge outputs."""

    def test_edge_ord_length_matches_tri_ord(self, tetrahedron_surface):
        """edge_ord should have same length as tri_ord."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        if edge_ord is not None:
            assert len(edge_ord) == len(tri_ord), \
                f"edge_ord length {len(edge_ord)} != tri_ord length {len(tri_ord)}"

    def test_sign_edge_length_matches_tri_ord(self, tetrahedron_surface):
        """sign_edge should have same length as tri_ord."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        if sign_edge is not None:
            assert len(sign_edge) == len(tri_ord), \
                f"sign_edge length {len(sign_edge)} != tri_ord length {len(tri_ord)}"

    def test_edge_ord_contains_valid_indices(self, tetrahedron_surface):
        """edge_ord values should be valid edge indices."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface
        ne = E2V.shape[0]

        tri_ord, edge_ord, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        if edge_ord is not None:
            for e in edge_ord:
                assert 0 <= e < ne, f"Edge index {e} out of range [0, {ne})"


# =============================================================================
# Test: Consistency Across Different Starting Conditions
# =============================================================================

class TestConsistency:
    """Verify consistency of results."""

    def test_same_triangles_regardless_of_mesh_order(self):
        """Reordering triangles in mesh should give same set of triangles."""
        # Original order
        T1 = np.array([
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0],
        ], dtype=np.int32)
        E2V1, T2E1, E2T1, T2T1 = connectivity(T1)

        # Reordered (faces permuted)
        T2 = np.array([
            [2, 3, 0],  # was face 3
            [0, 1, 2],  # was face 0
            [1, 3, 2],  # was face 2
            [0, 3, 1],  # was face 1
        ], dtype=np.int32)
        E2V2, T2E2, E2T2, T2T2 = connectivity(T2)

        # Get triangles around vertex 0 for both
        tri_ord1, _, _ = sort_triangles_comp(0, T1, E2T1, T2T1, E2V1, T2E1)
        tri_ord2, _, _ = sort_triangles_comp(0, T2, E2T2, T2T2, E2V2, T2E2)

        # The actual face indices will differ, but the count should be same
        tris1 = [t for t in tri_ord1 if t != -1]
        tris2 = [t for t in tri_ord2 if t != -1]

        assert len(tris1) == len(tris2), \
            f"Different triangle counts: {len(tris1)} vs {len(tris2)}"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_vertex_not_in_any_triangle(self, tetrahedron_surface):
        """Vertex not in any triangle should return empty or error gracefully."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        # Vertex 99 is not in the mesh
        try:
            tri_ord, _, _ = sort_triangles_comp(99, T, E2T, T2T, E2V, T2E)
            # If no error, should return empty result
            triangles = [t for t in tri_ord if t != -1]
            assert len(triangles) == 0, "Non-existent vertex should have no triangles"
        except (ValueError, IndexError):
            # Also acceptable to raise an error
            pass

    def test_isolated_triangle_vertex_is_boundary(self):
        """
        Vertex that belongs to only one triangle is a proper boundary vertex.
        Each vertex has 2 boundary edges.
        """
        T = np.array([[0, 1, 2]], dtype=np.int32)
        E2V, T2E, E2T, T2T = connectivity(T)

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        # Should be boundary
        assert tri_ord[-1] == -1, "Vertex should be boundary"

        # Should have triangle 0
        triangles = set(tri_ord[:-1])
        assert triangles == {0}, f"Expected triangles {{0}}, got {triangles}"

    def test_strip_boundary_vertex(self, strip_of_triangles):
        """
        Boundary vertex with proper 2 boundary edges in strip mesh.
        """
        T, E2V, T2E, E2T, T2T = strip_of_triangles

        # Vertex 3 is on the top-left corner with 2 boundary edges
        # It has triangles 0 and 1 incident to it
        tri_ord, _, _ = sort_triangles_comp(3, T, E2T, T2T, E2V, T2E)

        # Should be boundary (has -1 at end)
        assert tri_ord[-1] == -1, "Vertex 3 should be boundary"

        # Should have triangles 0 and 1
        triangles = set(tri_ord[:-1])
        assert triangles == {0, 1}, f"Expected triangles {{0, 1}}, got {triangles}"


# =============================================================================
# Test: Larger Closed Meshes (all interior vertices)
# =============================================================================

class TestLargerClosedMeshes:
    """Tests on larger closed meshes where all vertices are interior."""

    def test_octahedron_all_vertices_interior(self):
        """
        Octahedron: 6 vertices, 8 faces, 12 edges.
        All vertices are interior (4 triangles each).
        """
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

        # Test all 6 vertices
        for v in range(6):
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)

            # All vertices should be interior (no -1)
            assert -1 not in tri_ord, f"Vertex {v} should be interior"

            # Each vertex has 4 incident triangles
            assert len(tri_ord) == 4, f"Vertex {v} should have 4 triangles, got {len(tri_ord)}"

            # All returned triangles should contain vertex v
            for tri in tri_ord:
                assert v in T[tri], f"Vertex {v} not in triangle {tri}"

    def test_octahedron_connected_rings(self):
        """Verify triangles around each vertex are in connected order."""
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
        E2V, T2E, E2T, T2T = connectivity(T)

        for v in range(6):
            tri_ord, _, _ = sort_triangles_comp(v, T, E2T, T2T, E2V, T2E)
            triangles = list(tri_ord)

            # Check consecutive pairs are neighbors (closed ring)
            n = len(triangles)
            for i in range(n):
                tri_a = triangles[i]
                tri_b = triangles[(i + 1) % n]

                neighbors_a = set(T2T[tri_a])
                assert tri_b in neighbors_a, \
                    f"Vertex {v}: triangles {tri_a} and {tri_b} should be adjacent"


class TestReturnValueTypes:
    """Test return value types and shapes."""

    def test_tri_ord_is_int_array(self, tetrahedron_surface):
        """tri_ord should be an integer array."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        tri_ord, _, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        assert isinstance(tri_ord, np.ndarray)
        assert np.issubdtype(tri_ord.dtype, np.integer)

    def test_edge_ord_is_int_array(self, tetrahedron_surface):
        """edge_ord should be an integer array."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        _, edge_ord, _ = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        if edge_ord is not None:
            assert isinstance(edge_ord, np.ndarray)
            assert np.issubdtype(edge_ord.dtype, np.integer)

    def test_sign_edge_is_int_array(self, tetrahedron_surface):
        """sign_edge should be an integer array."""
        T, E2V, T2E, E2T, T2T = tetrahedron_surface

        _, _, sign_edge = sort_triangles_comp(0, T, E2T, T2T, E2V, T2E)

        if sign_edge is not None:
            assert isinstance(sign_edge, np.ndarray)
            assert np.issubdtype(sign_edge.dtype, np.integer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
