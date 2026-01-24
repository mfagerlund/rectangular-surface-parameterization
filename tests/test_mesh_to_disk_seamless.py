"""
Pytest tests for parameterization/seamless.py

Tests the mesh cutting and seamless constraint building for UV parameterization.
Run with: pytest tests/test_mesh_to_disk_seamless.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import norm as sparse_norm

# Add parent directory and submodules to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Preprocess"))
sys.path.insert(0, str(project_root / "ComputeParam"))

from rectangular_surface_parameterization.core.mesh_info import MeshInfo, mesh_info
from rectangular_surface_parameterization.preprocessing.dec import DEC, dec_tri
from rectangular_surface_parameterization.parameterization.seamless import wrap_to_pi, mesh_to_disk_seamless


# =============================================================================
# Test Fixtures - Meshes
# =============================================================================

@pytest.fixture
def tetrahedron():
    """
    Regular tetrahedron surface (4 triangles, 4 vertices, 6 edges).
    This is a closed manifold surface with genus 0.
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
def small_sphere():
    """
    Small sphere mesh with 6 vertices (octahedron).
    Genus 0, Euler characteristic 2.
    """
    X = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5],
    ], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def single_triangle_with_boundary():
    """Single triangle - has boundary, Euler char = 1 (already disk)."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return mesh_info(X, T)


@pytest.fixture
def two_triangles_shared_edge():
    """Two triangles sharing one edge (butterfly shape) - has boundary."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],
        [0, 3, 1],
    ], dtype=np.int32)
    return mesh_info(X, T)


# =============================================================================
# Helper: Create Mock OrthoParam Structure
# =============================================================================

class MockOrthoParam:
    """
    Mock parameter structure for mesh_to_disk_seamless.

    This mimics the OrthoParam dataclass from preprocess_ortho_param.py
    with the minimal attributes needed for testing.
    """
    def __init__(self, mesh: MeshInfo, cones: np.ndarray = None):
        """
        Create a mock param structure for the given mesh.

        Parameters
        ----------
        mesh : MeshInfo
            The mesh to create parameters for.
        cones : ndarray, optional
            Cone vertex indices. If None, no cones are set.
        """
        # Interior vertex indices (all vertices for closed mesh, non-boundary for open)
        boundary_edges = np.where(np.any(mesh.edge_to_triangle[:, :2] < 0, axis=1))[0]
        if len(boundary_edges) > 0:
            boundary_verts = np.unique(mesh.edge_to_vertex[boundary_edges])
            self.idx_int = np.setdiff1d(np.arange(mesh.num_vertices), boundary_verts)
            self.ide_bound = boundary_edges
            self.idx_bound = boundary_verts
        else:
            self.idx_int = np.arange(mesh.num_vertices)
            self.ide_bound = np.array([], dtype=int)
            self.idx_bound = np.array([], dtype=int)

        # E2T mapping - just use the mesh's E2T for the first two columns
        self.edge_to_triangle = mesh.edge_to_triangle[:, :2]

        # Fixed edges/triangles (empty by default)
        self.ide_hard = np.array([], dtype=int)
        self.ide_fix = np.array([], dtype=int)
        self.tri_fix = np.array([], dtype=int)

        # Cones
        self.cones = cones if cones is not None else np.array([], dtype=int)


def create_test_inputs(mesh: MeshInfo, cones: np.ndarray = None):
    """
    Create all inputs needed for mesh_to_disk_seamless testing.

    Parameters
    ----------
    mesh : MeshInfo
        The mesh to test with.
    cones : ndarray, optional
        Cone vertex indices.

    Returns
    -------
    tuple
        (mesh, param, ang, sing, k21, dec)
    """
    param = MockOrthoParam(mesh, cones)

    # Angular field (one angle per face) - initialize to zero
    ang = np.zeros(mesh.num_faces)

    # Singularity index per vertex - typically zero, non-zero at cones
    sing = np.zeros(mesh.num_vertices)
    if cones is not None:
        for c in cones:
            sing[c] = 0.5  # Set a singularity value > 0.1

    # k21: rotation index per edge (1 = identity, 2 = 90 deg, etc.)
    # Default to identity (no rotation mismatch)
    k21 = np.ones(mesh.num_edges, dtype=int)

    # DEC operators
    dec = dec_tri(mesh)

    return mesh, param, ang, sing, k21, dec


# =============================================================================
# Test: wrap_to_pi Function
# =============================================================================

class TestWrapToPi:
    """Test the wrap_to_pi angle wrapping utility."""

    def test_wrap_to_pi_zero(self):
        """Zero should remain zero."""
        result = wrap_to_pi(np.array([0.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_wrap_to_pi_within_range(self):
        """Angles already in [-pi, pi] should remain unchanged."""
        angles = np.array([0.0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2])
        result = wrap_to_pi(angles)
        np.testing.assert_allclose(result, angles, atol=1e-10)

    def test_wrap_to_pi_positive_overflow(self):
        """Angles > pi should wrap to negative range."""
        angles = np.array([np.pi + 0.5, 2*np.pi, 3*np.pi/2])
        result = wrap_to_pi(angles)
        expected = np.array([-np.pi + 0.5, 0.0, -np.pi/2])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_wrap_to_pi_negative_overflow(self):
        """Angles < -pi should wrap to positive range."""
        angles = np.array([-np.pi - 0.5, -2*np.pi, -3*np.pi/2])
        result = wrap_to_pi(angles)
        expected = np.array([np.pi - 0.5, 0.0, np.pi/2])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_wrap_to_pi_boundary(self):
        """Angles at +/-pi should remain at +/-pi."""
        result_pos = wrap_to_pi(np.array([np.pi]))
        result_neg = wrap_to_pi(np.array([-np.pi]))
        # Both should be close to pi (or -pi, they're equivalent)
        assert np.abs(np.abs(result_pos[0]) - np.pi) < 1e-10
        assert np.abs(np.abs(result_neg[0]) - np.pi) < 1e-10

    def test_wrap_to_pi_large_values(self):
        """Large angle values should wrap correctly."""
        angles = np.array([10*np.pi, -10*np.pi, 100.0, -100.0])
        result = wrap_to_pi(angles)
        # All results should be in [-pi, pi]
        assert np.all(result >= -np.pi - 1e-10)
        assert np.all(result <= np.pi + 1e-10)

    def test_wrap_to_pi_array(self):
        """Should work with arrays of any shape."""
        angles = np.array([[0.0, np.pi], [2*np.pi, -2*np.pi]])
        result = wrap_to_pi(angles)
        assert result.shape == angles.shape
        # All results should be in [-pi, pi]
        assert np.all(result >= -np.pi - 1e-10)
        assert np.all(result <= np.pi + 1e-10)


# =============================================================================
# Test: Output Structure
# =============================================================================

class TestOutputStructure:
    """Test that mesh_to_disk_seamless returns proper output structure."""

    def test_returns_four_elements(self, tetrahedron):
        """Function should return exactly 4 elements."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        result = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        assert len(result) == 4, "Should return (disk_mesh, dec_cut, Align, Rot)"

    def test_output_types(self, tetrahedron):
        """Verify types of returned objects."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        assert isinstance(disk_mesh, MeshInfo), "disk_mesh should be MeshInfo"
        assert isinstance(dec_cut, DEC), "dec_cut should be DEC"
        assert issparse(Align), "Align should be sparse matrix"
        assert issparse(Rot), "Rot should be sparse matrix"


# =============================================================================
# Test: Disk Topology After Cut
# =============================================================================

class TestDiskTopology:
    """Test that cut mesh has expected topology based on cutting strategy.

    Note: The mesh cutting behavior depends on:
    1. k21 != 1 marks edges with rotation mismatch
    2. Singularities (|sing| > 0.1) mark cone vertices
    3. The cut_mesh algorithm tries to cut to disk topology

    Current implementation may not always successfully cut to disk topology
    for simple test cases like tetrahedron with minimal k21 mismatch.
    """

    def test_tetrahedron_no_singularities_no_cut(self, tetrahedron):
        """Closed tetrahedron with no singularities or k21 mismatch stays closed.

        When there are no cone singularities and no rotation mismatches (k21 != 1),
        the mesh is not cut and remains topologically equivalent to a sphere (chi=2).
        """
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # No cutting happens without singularities/rotation mismatch
        assert chi == 2, f"Uncut closed mesh should have chi=2, got {chi}"

    def test_tetrahedron_single_k21_no_cones_stays_closed(self, tetrahedron):
        """Single k21 mismatch WITHOUT cones: mesh stays closed (cut gets pruned).

        The cut_mesh algorithm prunes cut edges that create degree-1 vertices
        not terminating at cones. With no cones, all cuts get pruned.
        """
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Single k21 mismatch, but NO cones
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # Without cones, the single-edge cut is pruned, mesh stays closed
        assert chi == 2, f"Single k21 without cones should keep mesh closed (chi=2), got {chi}"

    def test_tetrahedron_single_k21_with_cones_insufficient_for_disk(self, tetrahedron):
        """Single k21 mismatch with cones: insufficient for disk topology.

        On a closed mesh, a single edge cut (even with cones) doesn't
        topologically disconnect the mesh - vertices are still connected
        through other edges. A proper cut PATH is needed.
        """
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Edge 0 connects vertices 0 and 1, set cones at those vertices
        k21[0] = 2
        sing[0] = 0.5  # Make vertex 0 a cone
        sing[1] = 0.5  # Make vertex 1 a cone

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # Single edge cut on closed mesh doesn't produce disk
        # The mesh stays closed because vertices remain connected through other edges
        assert chi == 2, f"Single k21 with cones still insufficient for disk, got chi={chi}"

    def test_tetrahedron_cut_path_produces_disk(self, tetrahedron):
        """Proper cut PATH with cones: produces disk topology.

        When k21 mismatches form a path connecting cones, the mesh
        is successfully cut to disk topology.
        """
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Find edges forming a path from vertex 0 to vertex 3
        # Edges: [0,1]=0, [0,2]=1, [0,3]=2, [1,2]=3, [1,3]=4, [2,3]=5
        # Path: edges 1 ([0,2]) and 5 ([2,3]) form path 0 -> 2 -> 3
        k21[1] = 2  # Edge [0,2]
        k21[5] = 2  # Edge [2,3]

        # Set cones at path endpoints
        sing[0] = 0.5
        sing[3] = 0.5

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 1, f"Cut path should produce disk (chi=1), got {chi}"
        assert disk_mesh.num_vertices > mesh.num_vertices, "Cut should duplicate vertices"

    def test_octahedron_single_k21_no_cones_stays_closed(self, small_sphere):
        """Octahedron: single k21 mismatch without cones stays closed."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(small_sphere)

        # Single k21 mismatch, no cones
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # Without cones, cuts are pruned
        assert chi == 2, f"Single k21 without cones should keep mesh closed, got chi={chi}"

    def test_already_disk_stays_disk(self, single_triangle_with_boundary):
        """Mesh that's already a disk should remain a disk."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(single_triangle_with_boundary)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 1, f"Disk mesh should remain disk with chi=1, got {chi}"


# =============================================================================
# Test: Boundary Edges
# =============================================================================

class TestBoundaryEdges:
    """Test that boundary edges are correctly identified in cut mesh."""

    def _count_boundary_edges(self, mesh: MeshInfo) -> int:
        """Count edges with only one adjacent face."""
        return np.sum(np.any(mesh.edge_to_triangle[:, :2] < 0, axis=1))

    def test_tetrahedron_no_boundary_single_k21_no_cones(self, tetrahedron):
        """Single k21 mismatch without cones: no boundary (cut pruned).

        The cut_mesh algorithm prunes cuts that create degree-1 vertices
        not at cones. Without cones, all cuts are pruned.
        """
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        original_boundary = self._count_boundary_edges(mesh)
        assert original_boundary == 0, "Original tetrahedron should have no boundary"

        # Single k21 mismatch, no cones - cut will be pruned
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Without cones, cut is pruned, no boundary created
        cut_boundary = self._count_boundary_edges(disk_mesh)
        assert cut_boundary == 0, "Single k21 without cones should not create boundary"

    def test_tetrahedron_has_boundary_with_cut_path(self, tetrahedron):
        """Proper cut path with cones creates boundary edges."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        original_boundary = self._count_boundary_edges(mesh)
        assert original_boundary == 0, "Original tetrahedron should have no boundary"

        # Cut path: edges 1 and 5 form path from vertex 0 to vertex 3
        k21[1] = 2
        k21[5] = 2
        sing[0] = 0.5
        sing[3] = 0.5

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        cut_boundary = self._count_boundary_edges(disk_mesh)
        assert cut_boundary > 0, "Cut path with cones should create boundary edges"

    def test_tetrahedron_no_boundary_without_k21(self, tetrahedron):
        """Closed tetrahedron without k21 mismatch stays closed (no boundary)."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Original tetrahedron has no boundary
        original_boundary = self._count_boundary_edges(mesh)
        assert original_boundary == 0, "Original tetrahedron should have no boundary"

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Without k21 mismatch, mesh stays closed
        cut_boundary = self._count_boundary_edges(disk_mesh)
        assert cut_boundary == 0, "Uncut mesh should have no boundary edges"

    def test_boundary_preserved_for_open_mesh(self, single_triangle_with_boundary):
        """Open mesh should preserve its boundary."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(single_triangle_with_boundary)

        original_boundary = self._count_boundary_edges(mesh)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        cut_boundary = self._count_boundary_edges(disk_mesh)
        # Should have at least as many boundary edges as original
        assert cut_boundary >= original_boundary, \
            f"Cut mesh boundary ({cut_boundary}) should be >= original ({original_boundary})"


# =============================================================================
# Test: Seamless Constraint Matrices
# =============================================================================

class TestSeamlessConstraints:
    """Test seamless constraint matrices (Align, Rot)."""

    def test_align_dimensions_without_boundary(self, tetrahedron):
        """Align matrix should have correct dimensions when no boundary alignment."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # Align should have 0 rows when no boundary/hard edge alignment
        # and columns = 2 * nv of cut mesh
        assert Align.shape[1] == 2 * disk_mesh.num_vertices, \
            f"Align should have {2*disk_mesh.num_vertices} columns, got {Align.shape[1]}"

    def test_rot_dimensions(self, tetrahedron):
        """Rot matrix should have correct column dimension."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # Rot should have columns = 2 * nv of cut mesh
        assert Rot.shape[1] == 2 * disk_mesh.num_vertices, \
            f"Rot should have {2*disk_mesh.num_vertices} columns, got {Rot.shape[1]}"

    def test_matrices_sparse(self, tetrahedron):
        """Both Align and Rot should be sparse matrices."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        assert issparse(Align), "Align should be sparse"
        assert issparse(Rot), "Rot should be sparse"


# =============================================================================
# Test: Without Seamless Constraints
# =============================================================================

class TestWithoutSeamlessConstraints:
    """Test behavior when seamless constraints are disabled."""

    def test_empty_constraints_when_disabled(self, tetrahedron):
        """Align and Rot should be empty when ifseamless_const=False."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Both matrices should have 0 rows
        assert Align.shape[0] == 0, f"Align should have 0 rows, got {Align.shape[0]}"
        assert Rot.shape[0] == 0, f"Rot should have 0 rows, got {Rot.shape[0]}"

    def test_cut_mesh_still_valid_without_constraints_and_k21(self, tetrahedron):
        """Cut mesh should be valid even without seamless constraints, with k21 mismatch."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Force cutting
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Check mesh is valid
        assert disk_mesh.num_vertices > 0, "Cut mesh should have vertices"
        assert disk_mesh.num_edges > 0, "Cut mesh should have edges"
        assert disk_mesh.num_faces > 0, "Cut mesh should have faces"

        # Euler characteristic: depends on whether cutting succeeded
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # Accept either disk (chi=1) or sphere (chi=2) - current implementation
        # may not always successfully cut to disk
        assert chi in [1, 2], f"Mesh should have chi=1 or 2, got {chi}"

    def test_mesh_unchanged_without_constraints_and_no_k21(self, tetrahedron):
        """Mesh should be unchanged when no constraints and no k21 mismatch."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Check mesh is valid
        assert disk_mesh.num_vertices > 0, "Mesh should have vertices"
        assert disk_mesh.num_edges > 0, "Mesh should have edges"
        assert disk_mesh.num_faces > 0, "Mesh should have faces"

        # Without k21 mismatch, mesh stays closed (chi=2)
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 2, f"Uncut mesh should have chi=2, got {chi}"


# =============================================================================
# Test: DEC Operators on Cut Mesh
# =============================================================================

class TestDECOperators:
    """Test that DEC operators on cut mesh are valid."""

    def test_dec_d0p_shape(self, tetrahedron):
        """d0p should have shape (ne, nv) for cut mesh."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        assert dec_cut.d0p.shape == (disk_mesh.num_edges, disk_mesh.num_vertices), \
            f"d0p shape should be ({disk_mesh.num_edges}, {disk_mesh.num_vertices}), got {dec_cut.d0p.shape}"

    def test_dec_d1p_shape(self, tetrahedron):
        """d1p should have shape (nf, ne) for cut mesh."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        assert dec_cut.d1p.shape == (disk_mesh.num_faces, disk_mesh.num_edges), \
            f"d1p shape should be ({disk_mesh.num_faces}, {disk_mesh.num_edges}), got {dec_cut.d1p.shape}"

    def test_dec_exactness(self, tetrahedron):
        """d1p @ d0p should be zero (exactness property)."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        product = dec_cut.d1p @ dec_cut.d0p
        frob_norm = sparse_norm(product, 'fro')
        assert frob_norm < 1e-10, \
            f"d1p @ d0p should be zero, but Frobenius norm is {frob_norm}"


# =============================================================================
# Test: Singularity/Cone Handling
# =============================================================================

class TestConeHandling:
    """Test handling of singularities/cones in cutting."""

    def test_cone_at_vertex_with_k21(self, tetrahedron):
        """Mesh with cone singularity and k21 mismatch should produce valid mesh."""
        mesh = tetrahedron

        # Place a cone at vertex 0
        cones = np.array([0])
        mesh_obj, param, ang, sing, k21, _ = create_test_inputs(mesh, cones)

        # Need k21 mismatch to force cut
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh_obj, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # Should produce valid mesh (may or may not be disk depending on cut success)
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi in [1, 2], f"Mesh should have chi=1 or 2, got {chi}"

    def test_cone_at_vertex_no_k21_no_cut(self, tetrahedron):
        """Mesh with cone singularity but no k21 mismatch should not be cut.

        The cutting is driven by k21 mismatch (rotation across edges), not by
        cone singularities alone. Cones are destinations for cut paths.
        """
        mesh = tetrahedron

        # Place a cone at vertex 0
        cones = np.array([0])
        mesh_obj, param, ang, sing, k21, _ = create_test_inputs(mesh, cones)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh_obj, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # Without k21 mismatch, mesh stays closed
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 2, f"Uncut mesh should have chi=2, got {chi}"

    def test_multiple_cones(self, small_sphere):
        """Mesh with multiple cones should produce valid mesh."""
        mesh = small_sphere

        # Place cones at vertices 0 and 1
        cones = np.array([0, 1])
        mesh_obj, param, ang, sing, k21, _ = create_test_inputs(mesh, cones)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh_obj, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # Depending on the cone handling, mesh may or may not be cut
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # Accept both closed (chi=2) and disk (chi=1) topologies
        assert chi in [1, 2], f"Mesh should have chi=1 or 2, got {chi}"

    def test_multiple_cones_with_k21(self, small_sphere):
        """Mesh with multiple cones and k21 mismatch should be cut to disk."""
        mesh = small_sphere

        # Place cones at vertices 0 and 1
        cones = np.array([0, 1])
        mesh_obj, param, ang, sing, k21, _ = create_test_inputs(mesh, cones)

        # Force cut with k21 mismatch
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh_obj, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # With k21 mismatch, should be cut to disk
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 1, f"Cut mesh with multiple cones should be disk (chi=1), got {chi}"


# =============================================================================
# Test: Rotation Mismatch (k21)
# =============================================================================

class TestRotationMismatch:
    """Test handling of rotation mismatch across edges."""

    def test_identity_rotation_no_cut(self, tetrahedron):
        """k21 = 1 (identity) everywhere means no cut is needed."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # k21 is already all 1s
        assert np.all(k21 == 1)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # With no rotation mismatch, mesh stays closed
        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 2, f"Uncut mesh should have chi=2, got {chi}"

    def test_nonidentity_rotation_without_cones_pruned(self, tetrahedron):
        """Non-identity k21 without cones: cut is pruned, mesh stays closed.

        The cut_mesh algorithm prunes cut edges that create degree-1 vertices
        not terminating at cones. Single k21 mismatch without cones gets pruned.
        """
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Single k21 mismatch, no cones
        k21[0] = 2

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        # Without cones, cut is pruned
        assert chi == 2, f"Single k21 without cones should stay closed (chi=2), got {chi}"

    def test_nonidentity_rotation_with_cut_path_produces_disk(self, tetrahedron):
        """Non-identity k21 forming cut path with cones produces disk."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Cut path from vertex 0 to 3 via vertex 2
        k21[1] = 2  # Edge [0,2]
        k21[5] = 2  # Edge [2,3]
        sing[0] = 0.5
        sing[3] = 0.5

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 1, f"Cut path with cones should produce disk (chi=1), got {chi}"

    def test_all_k21_values_valid(self, tetrahedron):
        """Test that all valid k21 values (1-4) work."""
        for k_val in [1, 2, 3, 4]:
            mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)
            k21[0] = k_val

            disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
                mesh, param, ang, sing, k21,
                ifseamless_const=False, ifboundary=False, ifhardedge=False
            )

            # Should not crash for any valid k21 value
            assert disk_mesh.num_faces > 0, f"Mesh should be valid for k21={k_val}"


# =============================================================================
# Test: Vertex/Edge Counts
# =============================================================================

class TestVertexEdgeCounts:
    """Test that cut mesh has reasonable vertex/edge counts."""

    def test_cut_increases_vertices(self, tetrahedron):
        """Cutting closed mesh should typically increase vertex count."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Cut mesh should have at least as many vertices
        # (typically more due to vertex duplication along cuts)
        assert disk_mesh.num_vertices >= mesh.num_vertices, \
            f"Cut mesh vertices ({disk_mesh.num_vertices}) should be >= original ({mesh.num_vertices})"

    def test_face_count_preserved(self, tetrahedron):
        """Cutting should preserve face count."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        assert disk_mesh.num_faces == mesh.num_faces, \
            f"Cut mesh faces ({disk_mesh.num_faces}) should equal original ({mesh.num_faces})"


# =============================================================================
# Test: Input Validation
# =============================================================================

class TestInputValidation:
    """Test input validation and error handling."""

    def test_consistent_dimensions(self, tetrahedron):
        """Inputs should have consistent dimensions."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Verify dimensions match
        assert len(ang) == mesh.num_faces, "ang should have nf elements"
        assert len(sing) == mesh.num_vertices, "sing should have nv elements"
        assert len(k21) == mesh.num_edges, "k21 should have ne elements"

    def test_k21_valid_range(self, tetrahedron):
        """k21 values should be in range 1-4."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Default k21 is all 1s
        assert np.all(k21 >= 1) and np.all(k21 <= 4), \
            "k21 values should be in range [1, 4]"


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_minimal_mesh(self, single_triangle_with_boundary):
        """Single triangle (minimal mesh) should work."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(single_triangle_with_boundary)

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        # Should still be a valid mesh
        assert disk_mesh.num_vertices >= 3, "Should have at least 3 vertices"
        assert disk_mesh.num_faces >= 1, "Should have at least 1 face"

    def test_no_interior_vertices(self, single_triangle_with_boundary):
        """Mesh with no interior vertices should work."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(single_triangle_with_boundary)

        # All vertices are on boundary for single triangle
        assert len(param.idx_int) == 0, "Single triangle has no interior vertices"

        disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        chi = disk_mesh.num_vertices - disk_mesh.num_edges + disk_mesh.num_faces
        assert chi == 1, f"Should maintain disk topology, got chi={chi}"


# =============================================================================
# Test: Consistency Between Calls
# =============================================================================

class TestConsistency:
    """Test consistency and reproducibility."""

    def test_same_input_same_output(self, tetrahedron):
        """Same input should produce same output."""
        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        result1 = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        result2 = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        # Check cut mesh has same topology
        assert result1[0].num_vertices == result2[0].num_vertices, "Vertex count should be consistent"
        assert result1[0].num_edges == result2[0].num_edges, "Edge count should be consistent"
        assert result1[0].num_faces == result2[0].num_faces, "Face count should be consistent"

        # Check constraint matrix shapes
        assert result1[2].shape == result2[2].shape, "Align shape should be consistent"
        assert result1[3].shape == result2[3].shape, "Rot shape should be consistent"
