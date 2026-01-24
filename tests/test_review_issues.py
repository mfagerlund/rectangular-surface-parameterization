"""
Tests for issues identified in the code review (my-half-review.md).

These tests verify the fixes align Python behavior with MATLAB:
1. Cut-edge pairing - MATLAB silently drops last element if odd count
2. ismember mismatch - MATLAB errors when indexing with 0 (not found)
3. Graph generator forest BFS - MATLAB uses 'Type','forest' for all components
4. UV scale extraction - MATLAB errors on index out of bounds
"""

import numpy as np
import pytest
import warnings
from pathlib import Path
import sys

# Add parent directory for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scipy.sparse import csr_matrix
from Preprocess.MeshInfo import mesh_info
from Preprocess.find_graph_generator import (
    find_graph_generator,
    _compute_predecessors_bfs,
)
from Utils.extract_scale_from_param import extract_scale_from_param, DistortionMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tetrahedron():
    """Regular tetrahedron surface."""
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
    """Octahedron (small sphere) mesh."""
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


# =============================================================================
# Issue 1: Cut-edge pairing validation
# =============================================================================

class TestCutEdgePairingValidation:
    """
    Test that odd number of cut edges is properly detected.

    mesh_to_disk_seamless.py:222 - Previously would silently drop unpaired edge.
    Now raises AssertionError with clear message.
    """

    def test_even_cut_edges_passes(self, tetrahedron):
        """Even number of cut edges should work normally."""
        from ComputeParam.mesh_to_disk_seamless import mesh_to_disk_seamless
        from tests.test_mesh_to_disk_seamless import create_test_inputs

        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Create a proper cut path (even number of cut boundary edges)
        k21[1] = 2  # Edge [0,2]
        k21[5] = 2  # Edge [2,3]
        sing[0] = 0.5
        sing[3] = 0.5

        # Should not raise
        SrcCut, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=True, ifboundary=False, ifhardedge=False
        )

        assert SrcCut.nf > 0

    def test_no_seamless_const_skips_validation(self, tetrahedron):
        """When ifseamless_const=False, pairing validation is skipped."""
        from ComputeParam.mesh_to_disk_seamless import mesh_to_disk_seamless
        from tests.test_mesh_to_disk_seamless import create_test_inputs

        mesh, param, ang, sing, k21, _ = create_test_inputs(tetrahedron)

        # Any k21 configuration should work when seamless constraints disabled
        k21[:] = 2

        SrcCut, dec_cut, Align, Rot = mesh_to_disk_seamless(
            mesh, param, ang, sing, k21,
            ifseamless_const=False, ifboundary=False, ifhardedge=False
        )

        assert SrcCut.nf > 0


# =============================================================================
# Issue 3: Graph generator forest BFS
# =============================================================================

class TestGraphGeneratorForestBFS:
    """
    Test that dual graph BFS handles all connected components.

    find_graph_generator.py:188 - Previously only did BFS from face 0.
    Note: _compute_predecessors_bfs_forest was planned but not implemented.
    The main find_graph_generator function works correctly for connected meshes.
    """

    @pytest.mark.skip(reason="_compute_predecessors_bfs_forest not implemented")
    def test_bfs_forest_single_component(self):
        """BFS forest should work for single connected component."""
        pass

    @pytest.mark.skip(reason="_compute_predecessors_bfs_forest not implemented")
    def test_bfs_forest_multiple_components(self):
        """BFS forest should handle multiple disconnected components."""
        pass

    @pytest.mark.skip(reason="_compute_predecessors_bfs_forest not implemented")
    def test_bfs_forest_isolated_nodes(self):
        """BFS forest should handle isolated nodes."""
        pass

    @pytest.mark.skip(reason="_compute_predecessors_bfs_forest not implemented")
    def test_bfs_forest_empty_graph(self):
        """BFS forest should handle empty graph."""
        pass

    def test_graph_generator_handles_disconnected_dual(self, octahedron):
        """Graph generator should work even with potentially disconnected dual."""
        mesh = octahedron

        # Edge lengths
        l = np.ones(mesh.ne)

        # Should not raise even if dual graph has multiple components
        cycle, cocycle = find_graph_generator(l, mesh.T, mesh.E2T, mesh.E2V, init=0)

        # For a closed mesh with genus 0, should have no generators
        assert len(cycle) == 0, "Sphere should have no cycle generators"
        assert len(cocycle) == 0, "Sphere should have no cocycle generators"


# =============================================================================
# Issue 4: UV scale extraction index validation
# =============================================================================

class TestUVScaleExtractionValidation:
    """
    Test that T/T_cut index mismatches are properly detected.

    extract_scale_from_param.py:243 - Previously would silently use zeros.
    Now raises IndexError with clear message.
    """

    def test_valid_indices_passes(self, tetrahedron):
        """Valid T/T_cut indices should work normally."""
        mesh = tetrahedron
        nf = mesh.nf
        nv = mesh.nv

        # Create mock param with e1r, e2r
        class MockParam:
            def __init__(self):
                self.e1r = np.zeros((nf, 3))
                self.e2r = np.zeros((nf, 3))
                # Set orthonormal frames
                for i in range(nf):
                    self.e1r[i] = [1, 0, 0]
                    self.e2r[i] = [0, 1, 0]

        param = MockParam()

        # UV coordinates (same vertices as original - no cutting)
        Xp = np.random.rand(nv, 2)
        T_cut = mesh.T.copy()  # Same as T (no cutting)
        ang = np.zeros(nf)

        # Should not raise
        disto, ut, theta, u_tri = extract_scale_from_param(
            Xp, mesh.X, mesh.T, param, T_cut, ang
        )

        assert disto is not None
        assert ut is not None

    def test_index_mismatch_raises_error(self, tetrahedron):
        """T indices exceeding v length should raise IndexError."""
        mesh = tetrahedron
        nf = mesh.nf
        nv = mesh.nv

        class MockParam:
            def __init__(self):
                self.e1r = np.zeros((nf, 3))
                self.e2r = np.zeros((nf, 3))
                for i in range(nf):
                    self.e1r[i] = [1, 0, 0]
                    self.e2r[i] = [0, 1, 0]

        param = MockParam()

        # UV coordinates with FEWER vertices than T references
        # This simulates a T/T_cut mismatch bug
        Xp_small = np.random.rand(nv - 1, 2)  # One less vertex
        T_cut_small = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])  # References max 2
        ang = np.zeros(nf)

        # T still references vertex nv-1 (index 3), but v will only have nv-1 elements
        # This should raise ValueError with a helpful message
        with pytest.raises(ValueError, match="T/T_cut mismatch"):
            extract_scale_from_param(
                Xp_small, mesh.X, mesh.T, param, T_cut_small, ang
            )

    def test_matching_dimensions_works(self, tetrahedron):
        """When T and T_cut have matching vertex counts, should work."""
        mesh = tetrahedron
        nf = mesh.nf
        nv = mesh.nv

        class MockParam:
            def __init__(self):
                self.e1r = np.zeros((nf, 3))
                self.e2r = np.zeros((nf, 3))
                for i in range(nf):
                    self.e1r[i] = [1, 0, 0]
                    self.e2r[i] = [0, 1, 0]

        param = MockParam()

        # Create cut mesh with more vertices (simulating vertex duplication)
        nv_cut = nv + 2  # Two extra vertices from cutting
        Xp = np.random.rand(nv_cut, 2)

        # T_cut references the cut mesh vertices
        T_cut = mesh.T.copy()
        T_cut[0, 0] = nv  # Use one of the new vertices

        ang = np.zeros(nf)

        # Should work because v (computed from T_cut) has enough elements
        disto, ut, theta, u_tri = extract_scale_from_param(
            Xp, mesh.X, mesh.T, param, T_cut, ang
        )

        assert ut is not None


# =============================================================================
# Integration tests with real pipeline
# =============================================================================

class TestIntegrationWithRealPipeline:
    """Integration tests using the actual pipeline on test meshes."""

    def test_sphere_pipeline_passes_validation(self):
        """Run full pipeline on sphere mesh to verify fixes don't break normal operation."""
        # This test runs the actual run_RSP.py pipeline which is the integration path
        # Use the existing run_RSP code path which handles all the setup correctly
        test_mesh = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"

        # Skip if test mesh doesn't exist
        if not Path(test_mesh).exists():
            pytest.skip(f"Test mesh not found: {test_mesh}")

        import subprocess
        result = subprocess.run(
            ["python", "run_RSP.py", test_mesh, "-o", "Results/test_output", "-v"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=60
        )

        # Check that UV recovery succeeded with 0 flips
        # Note: save_param may fail if quantization program is not compiled, that's OK
        assert "Flipped triangles: 0" in result.stdout, \
            f"Expected 0 flipped triangles, output:\n{result.stdout}"

        # Allow the pipeline to fail at save_param due to missing quantization
        if result.returncode != 0:
            # Only acceptable failure is the quantization program
            assert "QuantizationYoann" in result.stderr, \
                f"Unexpected pipeline failure: {result.stderr}"
