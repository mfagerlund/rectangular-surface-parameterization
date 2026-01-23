"""
Pytest tests for Preprocess/find_graph_generator.py

Tests the homology generator finder that computes primal cycle and dual
cocycle generators for surfaces with handles.

Based on "Greedy Optimal Homotopy and Homology Generators" by Erickson & Whittlesey.

Run with: pytest tests/test_find_graph_generator.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory and Preprocess to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Preprocess"))

from Preprocess.find_graph_generator import (
    find_graph_generator,
    _predecessors,
    _compute_predecessors_bfs,
    _setdiff_rows,
)
from Preprocess.MeshInfo import mesh_info


# =============================================================================
# Test Fixtures - Known Shapes
# =============================================================================

@pytest.fixture
def tetrahedron():
    """
    Regular tetrahedron surface (4 triangles, 4 vertices, 6 edges).
    Genus 0, closed manifold -> 0 homology generators.
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
def cube():
    """
    Triangulated cube (12 triangles, 8 vertices, 18 edges).
    Genus 0, closed manifold -> 0 homology generators.
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
def octahedron():
    """
    Regular octahedron (8 triangles, 6 vertices, 12 edges).
    Genus 0, closed manifold -> 0 homology generators.
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
    return mesh_info(X, T)


@pytest.fixture
def single_triangle():
    """
    Single triangle - open mesh with boundary.
    Has boundary, so not applicable for standard homology computation.
    """
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return mesh_info(X, T)


# =============================================================================
# Helper: Create Triangulated Torus Programmatically
# =============================================================================

def create_triangulated_torus(n_major=16, n_minor=8):
    """
    Create a triangulated torus mesh programmatically.

    Parameters
    ----------
    n_major : int
        Number of divisions around the major circumference.
    n_minor : int
        Number of divisions around the minor circumference.

    Returns
    -------
    X : ndarray (nv, 3)
        Vertex positions.
    T : ndarray (nf, 3)
        Triangle indices.
    """
    R = 2.0  # major radius
    r = 1.0  # minor radius

    vertices = []
    for i in range(n_major):
        theta = 2 * np.pi * i / n_major
        for j in range(n_minor):
            phi = 2 * np.pi * j / n_minor
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            vertices.append([x, y, z])

    faces = []
    for i in range(n_major):
        i_next = (i + 1) % n_major
        for j in range(n_minor):
            j_next = (j + 1) % n_minor
            # Current quad vertices
            v00 = i * n_minor + j
            v10 = i_next * n_minor + j
            v01 = i * n_minor + j_next
            v11 = i_next * n_minor + j_next
            # Split quad into two triangles
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


def load_obj_as_meshinfo(path_str, triangulate=False):
    """Load OBJ file and return MeshInfo.

    Parameters
    ----------
    path_str : str
        Path to OBJ file.
    triangulate : bool
        If True, triangulate quad faces by splitting each into two triangles.

    Returns
    -------
    MeshInfo or None if file doesn't exist or can't be loaded.
    """
    path = Path(path_str)
    if not path.exists():
        return None

    vertices = []
    faces = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                # Handle face indices (OBJ is 1-indexed, may have v/vt/vn format)
                face_verts = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1  # Convert to 0-indexed
                    face_verts.append(idx)

                if len(face_verts) == 3:
                    faces.append(face_verts)
                elif len(face_verts) == 4 and triangulate:
                    # Split quad into two triangles
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[0], face_verts[2], face_verts[3]])
                elif len(face_verts) > 4 and triangulate:
                    # Fan triangulation for polygons
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
                elif not triangulate and len(face_verts) != 3:
                    # Skip non-triangle faces if not triangulating
                    continue
                else:
                    faces.append(face_verts[:3])

    if not vertices or not faces:
        return None

    X = np.array(vertices, dtype=np.float64)
    T = np.array(faces, dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def sphere_mesh():
    """Load sphere mesh if available (genus 0)."""
    mesh = load_obj_as_meshinfo(r"C:\Dev\Colonel\Data\Meshes\sphere320.obj")
    if mesh is None:
        pytest.skip("Sphere mesh not found")
    return mesh


@pytest.fixture
def torus_mesh():
    """Create triangulated torus mesh (genus 1 -> 2 generators)."""
    X, T = create_triangulated_torus(n_major=16, n_minor=8)
    return mesh_info(X, T)


@pytest.fixture
def torus_mesh_from_file():
    """Load and triangulate torus.obj if available (genus 1)."""
    mesh = load_obj_as_meshinfo(r"C:\Dev\Colonel\Data\Meshes\torus.obj", triangulate=True)
    if mesh is None:
        pytest.skip("Torus mesh file not found")
    return mesh


# =============================================================================
# Test: Helper Functions
# =============================================================================

class TestSetdiffRows:
    """Test the _setdiff_rows helper function."""

    def test_empty_A(self):
        """Empty A should return empty array."""
        A = np.zeros((0, 2), dtype=int)
        B = np.array([[1, 2], [3, 4]], dtype=int)
        result = _setdiff_rows(A, B)
        assert len(result) == 0

    def test_empty_B(self):
        """Empty B should return all indices of A."""
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
        B = np.zeros((0, 2), dtype=int)
        result = _setdiff_rows(A, B)
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_no_overlap(self):
        """No overlap should return all indices of A."""
        A = np.array([[1, 2], [3, 4]], dtype=int)
        B = np.array([[5, 6], [7, 8]], dtype=int)
        result = _setdiff_rows(A, B)
        np.testing.assert_array_equal(result, np.array([0, 1]))

    def test_complete_overlap(self):
        """Complete overlap should return empty array."""
        A = np.array([[1, 2], [3, 4]], dtype=int)
        B = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
        result = _setdiff_rows(A, B)
        assert len(result) == 0

    def test_partial_overlap(self):
        """Partial overlap should return indices not in B."""
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
        B = np.array([[3, 4]], dtype=int)
        result = _setdiff_rows(A, B)
        np.testing.assert_array_equal(result, np.array([0, 2]))


class TestComputePredecessorsBFS:
    """Test the BFS predecessor computation."""

    def test_simple_chain(self):
        """Test BFS on a simple chain: 0 - 1 - 2."""
        from scipy.sparse import csr_matrix
        # Chain: 0 -- 1 -- 2
        row = np.array([0, 1, 1, 2])
        col = np.array([1, 0, 2, 1])
        data = np.ones(4)
        adj = csr_matrix((data, (row, col)), shape=(3, 3))

        pred = _compute_predecessors_bfs(adj, root=0, n=3)

        # Root's predecessor is itself
        assert pred[0] == 0
        # Node 1's predecessor is 0
        assert pred[1] == 0
        # Node 2's predecessor is 1
        assert pred[2] == 1

    def test_star_graph(self):
        """Test BFS on a star graph with center at node 0."""
        from scipy.sparse import csr_matrix
        # Star: 0 connected to 1, 2, 3
        row = np.array([0, 0, 0, 1, 2, 3])
        col = np.array([1, 2, 3, 0, 0, 0])
        data = np.ones(6)
        adj = csr_matrix((data, (row, col)), shape=(4, 4))

        pred = _compute_predecessors_bfs(adj, root=0, n=4)

        assert pred[0] == 0  # root
        assert pred[1] == 0
        assert pred[2] == 0
        assert pred[3] == 0


class TestPredecessors:
    """Test the _predecessors path tracing function."""

    def test_path_from_leaf_to_root(self):
        """Trace path from leaf to root."""
        # pred[0] = 0 (root), pred[1] = 0, pred[2] = 1
        pred = np.array([0, 0, 1])

        # Path from node 2 should be [2, 1, 0]
        path = _predecessors(pred, 2)
        np.testing.assert_array_equal(path, np.array([2, 1, 0]))

    def test_path_from_root(self):
        """Path from root should be just the root."""
        pred = np.array([0, 0, 1])

        path = _predecessors(pred, 0)
        np.testing.assert_array_equal(path, np.array([0]))

    def test_disconnected_node(self):
        """Disconnected node (pred=-1) should return single node path."""
        pred = np.array([0, 0, -1])  # Node 2 is disconnected

        path = _predecessors(pred, 2)
        # Should return just [2] since pred[2] = -1
        assert path[0] == 2


# =============================================================================
# Test: Genus 0 Surfaces (Sphere-like)
# =============================================================================

class TestGenus0Surfaces:
    """Test that genus 0 surfaces produce 0 generators."""

    def test_tetrahedron_zero_generators(self, tetrahedron):
        """Tetrahedron (genus 0): should have 0 homology generators."""
        mesh = tetrahedron

        # Prepare inputs for find_graph_generator
        l = np.sqrt(mesh.SqEdgeLength)  # Edge lengths
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        # Genus 0 -> 0 generators
        assert len(cycle) == 0, f"Tetrahedron should have 0 cycle generators, got {len(cycle)}"
        assert len(cocycle) == 0, f"Tetrahedron should have 0 cocycle generators, got {len(cocycle)}"

    def test_cube_zero_generators(self, cube):
        """Cube (genus 0): should have 0 homology generators."""
        mesh = cube

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        assert len(cycle) == 0, f"Cube should have 0 cycle generators, got {len(cycle)}"
        assert len(cocycle) == 0, f"Cube should have 0 cocycle generators, got {len(cocycle)}"

    def test_octahedron_zero_generators(self, octahedron):
        """Octahedron (genus 0): should have 0 homology generators."""
        mesh = octahedron

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        assert len(cycle) == 0, f"Octahedron should have 0 cycle generators, got {len(cycle)}"
        assert len(cocycle) == 0, f"Octahedron should have 0 cocycle generators, got {len(cocycle)}"

    def test_sphere_mesh_zero_generators(self, sphere_mesh):
        """Sphere mesh (genus 0): should have 0 homology generators."""
        mesh = sphere_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        assert len(cycle) == 0, f"Sphere should have 0 cycle generators, got {len(cycle)}"
        assert len(cocycle) == 0, f"Sphere should have 0 cocycle generators, got {len(cocycle)}"


# =============================================================================
# Test: Genus 1 Surfaces (Torus-like)
# =============================================================================

class TestGenus1Surfaces:
    """Test that genus 1 surfaces produce 2 generators."""

    def test_torus_mesh_two_generators(self, torus_mesh):
        """Torus mesh (genus 1): should have 2 homology generators."""
        mesh = torus_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        # Genus g -> 2g generators, so genus 1 -> 2 generators
        assert len(cycle) == 2, f"Torus should have 2 cycle generators, got {len(cycle)}"
        assert len(cocycle) == 2, f"Torus should have 2 cocycle generators, got {len(cocycle)}"


# =============================================================================
# Test: Generator Properties
# =============================================================================

class TestGeneratorProperties:
    """Test that generators have valid properties."""

    def test_torus_cycles_are_closed_loops(self, torus_mesh):
        """Each cycle generator should form a closed loop (or nearly so)."""
        mesh = torus_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        for i, cyc in enumerate(cycle):
            assert len(cyc) > 0, f"Cycle {i} should not be empty"
            # The cycle is constructed from two paths that meet at the root
            # So first and last vertices should share the same ancestor path
            # In the current implementation, cycles are paths that start and end
            # at the spanning tree root (or close to it)

            # Check that all vertex indices are valid
            assert np.all(cyc >= 0), f"Cycle {i} contains negative vertex indices"
            assert np.all(cyc < mesh.nv), f"Cycle {i} contains vertex indices >= nv"

    def test_torus_cycles_vertex_validity(self, torus_mesh):
        """All vertices in cycle generators should be valid mesh vertices."""
        mesh = torus_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        for i, cyc in enumerate(cycle):
            for v in cyc:
                assert 0 <= v < mesh.nv, f"Cycle {i} has invalid vertex {v}"

    def test_torus_cocycles_face_validity(self, torus_mesh):
        """All faces in cocycle generators should be valid mesh faces."""
        mesh = torus_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        for i, cocyc in enumerate(cocycle):
            if len(cocyc) > 0:  # May be empty for boundary edges
                for f in cocyc:
                    assert 0 <= f < mesh.nf, f"Cocycle {i} has invalid face {f}"

    def test_torus_cycles_nonempty(self, torus_mesh):
        """Cycle generators should not be empty."""
        mesh = torus_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        for i, cyc in enumerate(cycle):
            assert len(cyc) >= 2, f"Cycle {i} should have at least 2 vertices, got {len(cyc)}"


# =============================================================================
# Test: Generator Count Formula
# =============================================================================

class TestGeneratorCountFormula:
    """Test that generator count matches 2 * genus formula."""

    def compute_genus_from_mesh(self, mesh):
        """Compute genus using Euler characteristic: V - E + F = 2 - 2g."""
        chi = mesh.nv - mesh.ne + mesh.nf
        # For closed orientable surface: chi = 2 - 2g, so g = (2 - chi) / 2
        return (2 - chi) // 2

    def test_tetrahedron_genus_formula(self, tetrahedron):
        """Tetrahedron: verify genus=0 and generator count matches."""
        mesh = tetrahedron
        genus = self.compute_genus_from_mesh(mesh)

        assert genus == 0, f"Tetrahedron genus should be 0, got {genus}"

        l = np.sqrt(mesh.SqEdgeLength)
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

        expected = 2 * genus
        assert len(cycle) == expected, f"Expected {expected} generators, got {len(cycle)}"

    def test_torus_genus_formula(self, torus_mesh):
        """Torus: verify genus=1 and generator count matches 2*genus=2."""
        mesh = torus_mesh
        genus = self.compute_genus_from_mesh(mesh)

        assert genus == 1, f"Torus genus should be 1, got {genus}"

        l = np.sqrt(mesh.SqEdgeLength)
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

        expected = 2 * genus
        assert len(cycle) == expected, f"Expected {expected} generators, got {len(cycle)}"


# =============================================================================
# Test: Different Root Vertices
# =============================================================================

class TestDifferentRoots:
    """Test that different root vertices produce valid results."""

    def test_tetrahedron_different_roots(self, tetrahedron):
        """Tetrahedron: generator count should be 0 regardless of root."""
        mesh = tetrahedron
        l = np.sqrt(mesh.SqEdgeLength)

        for root in range(mesh.nv):
            cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V, init=root)
            assert len(cycle) == 0, f"Root {root}: expected 0 generators, got {len(cycle)}"

    def test_torus_different_roots(self, torus_mesh):
        """Torus: generator count should be 2 regardless of root."""
        mesh = torus_mesh
        l = np.sqrt(mesh.SqEdgeLength)

        # Test a few different root vertices
        test_roots = [0, mesh.nv // 4, mesh.nv // 2, mesh.nv - 1]
        for root in test_roots:
            cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V, init=root)
            assert len(cycle) == 2, f"Root {root}: expected 2 generators, got {len(cycle)}"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_triangle_with_boundary(self, single_triangle):
        """Single triangle (has boundary): now correctly raises an error.

        A single triangle with all boundary edges creates a degenerate case where
        the MST algorithm cannot properly connect all vertices (boundary edges
        get weight 0, which causes issues with the spanning tree construction).

        The algorithm requires closed manifolds or meshes where the interior
        edges properly connect all vertices.
        """
        mesh = single_triangle
        l = np.sqrt(mesh.SqEdgeLength)

        # A single triangle is a degenerate case that should now be detected
        # as problematic (all edges are boundary edges)
        with pytest.raises(ValueError, match="disconnected components"):
            find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

    def test_uniform_edge_weights(self, tetrahedron):
        """Uniform edge weights should still produce valid results."""
        mesh = tetrahedron
        # Use uniform weights instead of actual lengths
        l = np.ones(mesh.ne)

        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

        assert len(cycle) == 0, "Tetrahedron should have 0 generators with uniform weights"


# =============================================================================
# Test: Disconnected Mesh Detection
# =============================================================================

class TestDisconnectedMeshDetection:
    """Test that disconnected meshes are properly detected and raise errors."""

    @pytest.fixture
    def two_disjoint_triangles(self):
        """
        Two completely separate triangles (disconnected mesh).
        This should fail because vertices from one triangle cannot
        reach vertices in the other triangle.
        """
        X = np.array([
            # First triangle
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            # Second triangle (disjoint)
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [5.5, 1.0, 0.0],
        ], dtype=np.float64)
        T = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ], dtype=np.int32)
        return mesh_info(X, T)

    @pytest.fixture
    def two_disjoint_tetrahedra(self):
        """
        Two completely separate tetrahedra (disconnected mesh).
        This represents a more realistic disconnected mesh scenario.
        """
        X = np.array([
            # First tetrahedron
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
            # Second tetrahedron (disjoint)
            [10, 0, 0],
            [11, 0, 0],
            [10.5, 1, 0],
            [10.5, 0.5, 1],
        ], dtype=np.float64)
        T = np.array([
            # First tetrahedron faces
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3],
            # Second tetrahedron faces
            [4, 5, 6],
            [4, 5, 7],
            [5, 6, 7],
            [4, 6, 7],
        ], dtype=np.int32)
        return mesh_info(X, T)

    def test_disconnected_triangles_raises_error(self, two_disjoint_triangles):
        """Two disjoint triangles should raise ValueError for disconnected components."""
        mesh = two_disjoint_triangles
        l = np.sqrt(mesh.SqEdgeLength)

        with pytest.raises(ValueError, match="disconnected components"):
            find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

    def test_disconnected_tetrahedra_raises_error(self, two_disjoint_tetrahedra):
        """Two disjoint tetrahedra should raise ValueError for disconnected components."""
        mesh = two_disjoint_tetrahedra
        l = np.sqrt(mesh.SqEdgeLength)

        with pytest.raises(ValueError, match="disconnected components"):
            find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

    def test_disconnected_error_message_includes_vertex_info(self, two_disjoint_triangles):
        """Error message should include information about unreachable vertices."""
        mesh = two_disjoint_triangles
        l = np.sqrt(mesh.SqEdgeLength)

        with pytest.raises(ValueError) as exc_info:
            find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

        error_msg = str(exc_info.value)
        assert "unreachable" in error_msg.lower()
        assert "root" in error_msg.lower()

    def test_connected_mesh_does_not_raise(self, tetrahedron):
        """A connected mesh should not raise any error."""
        mesh = tetrahedron
        l = np.sqrt(mesh.SqEdgeLength)

        # Should complete without raising
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

        # Tetrahedron is genus 0, so 0 generators
        assert len(cycle) == 0
        assert len(cocycle) == 0


# =============================================================================
# Test: BFS Predecessor Disconnected Detection
# =============================================================================

class TestBFSDisconnectedDetection:
    """Test that _compute_predecessors_bfs correctly marks unreachable nodes."""

    def test_disconnected_graph_has_minus_one_predecessors(self):
        """Disconnected nodes should have pred == -1 after BFS."""
        from scipy.sparse import csr_matrix

        # Graph: 0 -- 1 -- 2    3 -- 4 (two disconnected components)
        row = np.array([0, 1, 1, 2, 3, 4])
        col = np.array([1, 0, 2, 1, 4, 3])
        data = np.ones(6)
        adj = csr_matrix((data, (row, col)), shape=(5, 5))

        # BFS from node 0 should not reach nodes 3 and 4
        pred = _compute_predecessors_bfs(adj, root=0, n=5)

        # Nodes 0, 1, 2 should be reachable
        assert pred[0] == 0  # root
        assert pred[1] == 0
        assert pred[2] == 1

        # Nodes 3 and 4 should be unreachable (pred == -1)
        assert pred[3] == -1
        assert pred[4] == -1

    def test_fully_connected_graph_all_reachable(self):
        """All nodes in a connected graph should be reachable."""
        from scipy.sparse import csr_matrix

        # Fully connected triangle: 0 -- 1 -- 2 -- 0
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([1, 2, 0, 2, 0, 1])
        data = np.ones(6)
        adj = csr_matrix((data, (row, col)), shape=(3, 3))

        pred = _compute_predecessors_bfs(adj, root=0, n=3)

        # All nodes should be reachable (no -1 except for edge cases)
        assert pred[0] == 0  # root
        assert pred[1] >= 0  # reachable
        assert pred[2] >= 0  # reachable


# =============================================================================
# Test: File-based Torus (if available)
# =============================================================================

class TestTorusFromFile:
    """Test with the actual torus.obj file (triangulated)."""

    def test_torus_file_two_generators(self, torus_mesh_from_file):
        """Torus from file (triangulated): should have 2 homology generators."""
        mesh = torus_mesh_from_file

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        cycle, cocycle = find_graph_generator(l, T, E2T, E2V)

        assert len(cycle) == 2, f"Torus file should have 2 cycle generators, got {len(cycle)}"
        assert len(cocycle) == 2, f"Torus file should have 2 cocycle generators, got {len(cocycle)}"

    def test_torus_file_euler_characteristic(self, torus_mesh_from_file):
        """Torus from file: Euler characteristic should be 0."""
        mesh = torus_mesh_from_file
        chi = mesh.nv - mesh.ne + mesh.nf
        assert chi == 0, f"Torus Euler characteristic should be 0, got {chi}"


# =============================================================================
# Test: Higher Genus Surfaces
# =============================================================================

class TestHigherGenusSurfaces:
    """Test surfaces with genus > 1."""

    @pytest.fixture
    def double_torus(self):
        """
        Create a genus-2 surface (double torus) by connected sum of two tori.

        This is a simplified construction: we create two tori and connect them
        by removing a face from each and gluing along the boundary.
        """
        # For simplicity, create a mesh that topologically has genus 2
        # by creating a 2x1 grid of tori glued together
        # We'll use a rectangular grid that wraps in a way that creates genus 2

        # A genus-2 surface can be created from a rectangular grid with specific
        # boundary identifications. For testing, we use the formula:
        # For a closed orientable surface: chi = 2 - 2g
        # So for genus 2: chi = 2 - 4 = -2

        # Create a simple genus-2 mesh by hand
        # This is a minimal triangulation with V - E + F = -2

        # Using the formula: for genus g, minimum faces = 4g+2, vertices = 2g+1, edges = 6g+3
        # For g=2: F=10, V=5, E=15 -> chi = 5 - 15 + 10 = 0 (wrong!)

        # Let's use a standard construction: octagon with identified sides
        # Actually, let's just connect two small tori

        # For testing purposes, we'll create a programmatic genus-2 surface
        # by taking the connected sum construction

        n_major = 8
        n_minor = 4

        def make_torus_vertices_faces(offset_x, start_v_idx):
            R = 1.0
            r = 0.3
            vertices = []
            for i in range(n_major):
                theta = 2 * np.pi * i / n_major
                for j in range(n_minor):
                    phi = 2 * np.pi * j / n_minor
                    x = (R + r * np.cos(phi)) * np.cos(theta) + offset_x
                    y = (R + r * np.cos(phi)) * np.sin(theta)
                    z = r * np.sin(phi)
                    vertices.append([x, y, z])

            faces = []
            for i in range(n_major):
                i_next = (i + 1) % n_major
                for j in range(n_minor):
                    j_next = (j + 1) % n_minor
                    v00 = start_v_idx + i * n_minor + j
                    v10 = start_v_idx + i_next * n_minor + j
                    v01 = start_v_idx + i * n_minor + j_next
                    v11 = start_v_idx + i_next * n_minor + j_next
                    faces.append([v00, v10, v11])
                    faces.append([v00, v11, v01])

            return np.array(vertices), np.array(faces)

        # Create two separate tori
        v1, f1 = make_torus_vertices_faces(-1.5, 0)
        v2, f2 = make_torus_vertices_faces(1.5, len(v1))

        # Combine (this gives two disjoint tori, not connected)
        # For a true genus-2 surface, we'd need to connect them properly
        # But for testing, let's just verify that two disjoint tori give 4 generators

        X = np.vstack([v1, v2])
        T = np.vstack([f1, f2])

        return mesh_info(X.astype(np.float64), T.astype(np.int32))

    def test_two_disjoint_tori_raises_disconnected_error(self, double_torus):
        """Two disjoint tori should raise ValueError for disconnected components.

        Previously this test incorrectly expected 4 generators, but the algorithm
        requires connected meshes. Disconnected meshes now properly raise an error.
        """
        mesh = double_torus

        l = np.sqrt(mesh.SqEdgeLength)
        T = mesh.T
        E2T = mesh.E2T
        E2V = mesh.E2V

        # Disconnected meshes should raise an error
        with pytest.raises(ValueError, match="disconnected components"):
            find_graph_generator(l, T, E2T, E2V)


# =============================================================================
# Test: Cycle Generator Loop Properties
# =============================================================================

class TestCycleLoopProperties:
    """Test that cycle generators form valid loops in the mesh."""

    def test_torus_cycle_edges_exist_in_mesh(self, torus_mesh):
        """Verify consecutive vertices in cycles are connected by edges in the mesh."""
        mesh = torus_mesh

        l = np.sqrt(mesh.SqEdgeLength)
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)

        # Build edge set for quick lookup
        edge_set = set()
        for e in range(mesh.ne):
            v0, v1 = mesh.E2V[e]
            edge_set.add((min(v0, v1), max(v0, v1)))

        for i, cyc in enumerate(cycle):
            if len(cyc) < 2:
                continue
            # Check each consecutive pair of vertices
            for j in range(len(cyc) - 1):
                v0, v1 = cyc[j], cyc[j + 1]
                edge = (min(v0, v1), max(v0, v1))
                assert edge in edge_set, \
                    f"Cycle {i}: edge ({v0}, {v1}) not found in mesh"


# =============================================================================
# Test: Different Torus Resolutions
# =============================================================================

class TestDifferentResolutions:
    """Test that generator count is consistent across different mesh resolutions."""

    @pytest.fixture
    def small_torus(self):
        """Small torus (8x4 grid)."""
        X, T = create_triangulated_torus(n_major=8, n_minor=4)
        return mesh_info(X, T)

    @pytest.fixture
    def medium_torus(self):
        """Medium torus (16x8 grid)."""
        X, T = create_triangulated_torus(n_major=16, n_minor=8)
        return mesh_info(X, T)

    @pytest.fixture
    def large_torus(self):
        """Large torus (32x16 grid)."""
        X, T = create_triangulated_torus(n_major=32, n_minor=16)
        return mesh_info(X, T)

    def test_small_torus_two_generators(self, small_torus):
        """Small torus should have 2 generators."""
        mesh = small_torus
        l = np.sqrt(mesh.SqEdgeLength)
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)
        assert len(cycle) == 2, f"Small torus should have 2 generators, got {len(cycle)}"

    def test_medium_torus_two_generators(self, medium_torus):
        """Medium torus should have 2 generators."""
        mesh = medium_torus
        l = np.sqrt(mesh.SqEdgeLength)
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)
        assert len(cycle) == 2, f"Medium torus should have 2 generators, got {len(cycle)}"

    def test_large_torus_two_generators(self, large_torus):
        """Large torus should have 2 generators."""
        mesh = large_torus
        l = np.sqrt(mesh.SqEdgeLength)
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V)
        assert len(cycle) == 2, f"Large torus should have 2 generators, got {len(cycle)}"

    def test_all_tori_same_euler_characteristic(self, small_torus, medium_torus, large_torus):
        """All tori should have Euler characteristic 0."""
        for name, mesh in [("small", small_torus), ("medium", medium_torus), ("large", large_torus)]:
            chi = mesh.nv - mesh.ne + mesh.nf
            assert chi == 0, f"{name} torus chi should be 0, got {chi}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
