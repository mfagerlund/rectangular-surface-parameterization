"""
Pytest tests for Orthotropic/optimize_RSP.py

Tests the main optimization loop for Rectangular Surface Parameterization
using Newton's method on the KKT conditions.

Run with: pytest tests/test_optimize_RSP.py -v

Note: Integration tests using the full optimizer are marked with @pytest.mark.slow
and can be skipped with: pytest tests/test_optimize_RSP.py -v -m "not slow"
"""

import numpy as np
import pytest
import scipy.sparse as sp
from pathlib import Path
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

# Add parent directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Orthotropic"))
sys.path.insert(0, str(project_root / "Preprocess"))

from rectangular_surface_parameterization.optimization.solver import optimize_RSP, OptimizeResult, wrap_to_pi, _zero_rows


# =============================================================================
# Helper Classes for Testing
# =============================================================================

@dataclass
class MockParam:
    """Mock parameter structure for testing."""
    tri_fix: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_fix: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_hard: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_bound: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_free: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ang_basis: Optional[np.ndarray] = None
    edge_to_triangle: Optional[np.ndarray] = None
    para_trans: Optional[np.ndarray] = None


@dataclass
class MockWeight:
    """Mock weight structure for testing."""
    w_conf_ar: float = 0.5
    om: Optional[np.ndarray] = None
    w_gradv: float = 0.0


# =============================================================================
# Test wrap_to_pi utility function
# =============================================================================

class TestWrapToPi:
    """Test the wrap_to_pi utility function."""

    def test_zero(self):
        """Zero should remain zero."""
        result = wrap_to_pi(np.array([0.0]))
        assert np.allclose(result, [0.0], atol=1e-10)

    def test_pi(self):
        """Pi should remain pi or -pi."""
        result = wrap_to_pi(np.array([np.pi]))
        assert np.allclose(np.abs(result), [np.pi], atol=1e-10)

    def test_positive_wrap(self):
        """Angles > pi should wrap to negative."""
        result = wrap_to_pi(np.array([1.5 * np.pi]))
        assert np.allclose(result, [-0.5 * np.pi], atol=1e-10)

    def test_negative_wrap(self):
        """Angles < -pi should wrap to positive."""
        result = wrap_to_pi(np.array([-1.5 * np.pi]))
        assert np.allclose(result, [0.5 * np.pi], atol=1e-10)

    def test_multiple_wraps(self):
        """Large angles should wrap correctly."""
        result = wrap_to_pi(np.array([5 * np.pi]))
        assert np.allclose(np.abs(result), [np.pi], atol=1e-10)

    def test_array_input(self):
        """Should work on arrays."""
        angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        result = wrap_to_pi(angles)
        assert result.shape == angles.shape
        # Check specific values
        assert np.allclose(result[0], 0.0, atol=1e-10)
        assert np.allclose(result[1], np.pi/2, atol=1e-10)
        # pi wraps to +/- pi
        assert np.allclose(np.abs(result[2]), np.pi, atol=1e-10)
        assert np.allclose(result[3], -np.pi/2, atol=1e-10)
        assert np.allclose(result[4], 0.0, atol=1e-10)

    def test_empty_array(self):
        """Empty array should return empty array."""
        result = wrap_to_pi(np.array([]))
        assert len(result) == 0

    def test_scalar_like(self):
        """Single element array should work."""
        result = wrap_to_pi(np.array([2.5 * np.pi]))
        assert np.allclose(result, [0.5 * np.pi], atol=1e-10)


# =============================================================================
# Test _zero_rows utility function
# =============================================================================

class TestZeroRows:
    """Test the _zero_rows utility function."""

    def test_zero_single_row(self):
        """Zeroing a single row should work."""
        mat = sp.csr_matrix(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float))
        result = _zero_rows(mat, np.array([1]))
        expected = np.array([
            [1, 2, 3],
            [0, 0, 0],
            [7, 8, 9]
        ])
        assert np.allclose(result.toarray(), expected)

    def test_zero_multiple_rows(self):
        """Zeroing multiple rows should work."""
        mat = sp.csr_matrix(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float))
        result = _zero_rows(mat, np.array([0, 2]))
        expected = np.array([
            [0, 0, 0],
            [4, 5, 6],
            [0, 0, 0]
        ])
        assert np.allclose(result.toarray(), expected)

    def test_zero_empty_indices(self):
        """Empty indices should leave matrix unchanged."""
        mat = sp.csr_matrix(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=float))
        result = _zero_rows(mat, np.array([], dtype=int))
        assert np.allclose(result.toarray(), mat.toarray())

    def test_preserves_sparse_format(self):
        """Result should be in CSR format."""
        mat = sp.csr_matrix(np.eye(3))
        result = _zero_rows(mat, np.array([0]))
        assert isinstance(result, sp.csr_matrix)


# =============================================================================
# Test OptimizeResult dataclass
# =============================================================================

class TestOptimizeResult:
    """Test the OptimizeResult dataclass structure."""

    def test_result_fields(self):
        """OptimizeResult should have all expected fields."""
        result = OptimizeResult(
            u=np.array([1.0, 2.0]),
            v=np.array([3.0, 4.0]),
            ut=np.array([[1, 2, 3]]),
            vt=np.array([[4, 5, 6]]),
            om=np.array([0.1, 0.2, 0.3]),
            angn=np.array([0.0]),
            flag=0,
            it=10
        )

        assert hasattr(result, 'u')
        assert hasattr(result, 'v')
        assert hasattr(result, 'ut')
        assert hasattr(result, 'vt')
        assert hasattr(result, 'om')
        assert hasattr(result, 'angn')
        assert hasattr(result, 'flag')
        assert hasattr(result, 'it')

    def test_flag_value_zero(self):
        """Flag 0 means convergence."""
        result = OptimizeResult(
            u=np.array([]), v=np.array([]), ut=np.array([]),
            vt=np.array([]), om=np.array([]), angn=np.array([]),
            flag=0, it=5
        )
        assert result.flag == 0

    def test_flag_value_one(self):
        """Flag 1 means reached max iterations."""
        result = OptimizeResult(
            u=np.array([]), v=np.array([]), ut=np.array([]),
            vt=np.array([]), om=np.array([]), angn=np.array([]),
            flag=1, it=300
        )
        assert result.flag == 1

    def test_flag_value_negative_one(self):
        """Flag -1 means linesearch failed."""
        result = OptimizeResult(
            u=np.array([]), v=np.array([]), ut=np.array([]),
            vt=np.array([]), om=np.array([]), angn=np.array([]),
            flag=-1, it=50
        )
        assert result.flag == -1

    def test_iteration_count_positive(self):
        """Iteration count should be positive."""
        result = OptimizeResult(
            u=np.array([1.0]), v=np.array([1.0]), ut=np.array([[1, 2, 3]]),
            vt=np.array([[1, 2, 3]]), om=np.array([0.1]), angn=np.array([0.0]),
            flag=0, it=15
        )
        assert result.it > 0

    def test_arrays_are_numpy(self):
        """All array fields should be numpy arrays."""
        result = OptimizeResult(
            u=np.array([1.0]),
            v=np.array([1.0]),
            ut=np.array([[1, 2, 3]]),
            vt=np.array([[1, 2, 3]]),
            om=np.array([0.1]),
            angn=np.array([0.0]),
            flag=0,
            it=1
        )
        assert isinstance(result.u, np.ndarray)
        assert isinstance(result.v, np.ndarray)
        assert isinstance(result.ut, np.ndarray)
        assert isinstance(result.vt, np.ndarray)
        assert isinstance(result.om, np.ndarray)
        assert isinstance(result.angn, np.ndarray)


# =============================================================================
# Test MockParam structure
# =============================================================================

class TestMockParam:
    """Test MockParam for correct default values."""

    def test_default_empty_arrays(self):
        """Default arrays should be empty."""
        param = MockParam()
        assert len(param.tri_fix) == 0
        assert len(param.ide_fix) == 0
        assert len(param.ide_hard) == 0
        assert len(param.ide_bound) == 0
        assert len(param.ide_free) == 0
        assert len(param.ide_int) == 0
        assert len(param.idx_int) == 0

    def test_default_none_fields(self):
        """Optional fields should be None by default."""
        param = MockParam()
        assert param.ang_basis is None
        assert param.edge_to_triangle is None
        assert param.para_trans is None

    def test_custom_values(self):
        """Custom values should be accepted."""
        param = MockParam(
            tri_fix=np.array([1, 2, 3]),
            ide_fix=np.array([0, 1]),
            ang_basis=np.zeros((4, 3))
        )
        assert len(param.tri_fix) == 3
        assert len(param.ide_fix) == 2
        assert param.ang_basis.shape == (4, 3)


# =============================================================================
# Test MockWeight structure
# =============================================================================

class TestMockWeight:
    """Test MockWeight for correct default values."""

    def test_default_values(self):
        """Default values should be set correctly."""
        weight = MockWeight()
        assert weight.w_conf_ar == 0.5
        assert weight.om is None
        assert weight.w_gradv == 0.0

    def test_custom_values(self):
        """Custom values should be accepted."""
        weight = MockWeight(w_conf_ar=0.8, w_gradv=0.1)
        assert weight.w_conf_ar == 0.8
        assert weight.w_gradv == 0.1


# =============================================================================
# Integration Test Fixtures
# =============================================================================

@pytest.fixture
def tetrahedron_setup():
    """
    Set up a tetrahedron mesh with all necessary structures for optimization.
    This is a closed mesh (genus 0) so there's no boundary.
    """
    from rectangular_surface_parameterization.core.mesh_info import mesh_info
    from rectangular_surface_parameterization.preprocessing.dec import dec_tri
    from rectangular_surface_parameterization.optimization.reduce_corner_var import reduce_corner_var_2d
    from rectangular_surface_parameterization.optimization.reduction import reduction_from_ff2d

    # Create mesh - regular tetrahedron
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

    Src = mesh_info(X, T)
    dec = dec_tri(Src)

    # Create param structure
    param = MockParam()
    param.ide_fix = np.array([], dtype=int)
    param.ide_hard = np.array([], dtype=int)
    param.ide_bound = np.array([], dtype=int)
    param.ide_free = np.arange(Src.num_edges)
    param.ide_int = np.arange(Src.num_edges)  # All edges are interior for closed mesh
    param.idx_int = np.arange(Src.num_vertices)  # All vertices are interior for closed mesh
    param.tri_fix = np.array([], dtype=int)

    # Build E2T (face pairs per edge)
    E2T = np.zeros((Src.num_edges, 2), dtype=int)
    for e in range(Src.num_edges):
        E2T[e, 0] = max(Src.edge_to_triangle[e, 0], 0)
        E2T[e, 1] = max(Src.edge_to_triangle[e, 1], 0) if Src.edge_to_triangle[e, 1] >= 0 else E2T[e, 0]
    param.edge_to_triangle = E2T

    # Compute local basis angles
    edge = Src.vertices[Src.edge_to_vertex[:, 1], :] - Src.vertices[Src.edge_to_vertex[:, 0], :]
    edge = edge / np.linalg.norm(edge, axis=1, keepdims=True)
    e1r = Src.vertices[Src.triangles[:, 1], :] - Src.vertices[Src.triangles[:, 0], :]
    e1r = e1r / np.linalg.norm(e1r, axis=1, keepdims=True)

    def comp_angle(u, v, n):
        cross_uv = np.cross(u, v)
        sin_angle = np.sum(cross_uv * n, axis=1)
        cos_angle = np.sum(u * v, axis=1)
        return np.arctan2(sin_angle, cos_angle)

    ang_basis = np.column_stack([
        comp_angle(Src.vertices[Src.triangles[:, 0], :] - Src.vertices[Src.triangles[:, 1], :], e1r, Src.normal),
        comp_angle(Src.vertices[Src.triangles[:, 1], :] - Src.vertices[Src.triangles[:, 2], :], e1r, Src.normal),
        comp_angle(Src.vertices[Src.triangles[:, 2], :] - Src.vertices[Src.triangles[:, 0], :], e1r, Src.normal)
    ])
    param.ang_basis = ang_basis

    # Compute parallel transport (zero for simplified test)
    param.para_trans = np.zeros(Src.num_edges)

    # Create weight structure
    weight = MockWeight()
    weight.w_conf_ar = 0.5

    # Build reduction matrix
    Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)

    # Initial variables
    u = np.zeros(Src.num_vertices)
    v = np.zeros(Src.num_vertices)
    ang = np.zeros(Src.num_faces)
    omega = np.zeros(Src.num_edges)

    k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

    return {
        'Src': Src,
        'dec': dec,
        'param': param,
        'weight': weight,
        'Reduction': Reduction,
        'u': u,
        'v': v,
        'ang': ang,
        'omega': omega,
    }


# =============================================================================
# Integration Tests (marked slow)
# =============================================================================

@pytest.mark.slow
@pytest.mark.skip(reason="Integration tests require significant computation time (>2 min each)")
class TestOptimizeRSPIntegration:
    """Integration tests for the full optimize_RSP function.

    These tests are slow because they run the full Newton optimization
    with KKT system solving. They may take several minutes each.

    To run these tests manually:
        pytest tests/test_optimize_RSP.py -m slow --no-skip
    """

    def test_result_is_optimize_result(self, tetrahedron_setup):
        """Result should be an OptimizeResult instance."""
        setup = tetrahedron_setup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert isinstance(result, OptimizeResult)

    def test_output_dimensions(self, tetrahedron_setup):
        """Output arrays should have correct dimensions."""
        setup = tetrahedron_setup
        Src = setup['Src']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=Src,
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert result.u.shape == (Src.num_vertices,), f"u shape mismatch: {result.u.shape}"
        assert result.v.shape == (Src.num_vertices,), f"v shape mismatch: {result.v.shape}"
        assert result.ut.shape == (Src.num_faces, 3), f"ut shape mismatch: {result.ut.shape}"
        assert result.vt.shape == (Src.num_faces, 3), f"vt shape mismatch: {result.vt.shape}"
        assert result.om.shape == (Src.num_edges,), f"om shape mismatch: {result.om.shape}"
        assert result.angn.shape == (Src.num_faces,), f"angn shape mismatch: {result.angn.shape}"

    def test_flag_is_valid(self, tetrahedron_setup):
        """Flag should be one of: -1, 0, or 1."""
        setup = tetrahedron_setup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert result.flag in [-1, 0, 1], f"Invalid flag value: {result.flag}"

    def test_max_iter_respected(self, tetrahedron_setup):
        """Optimizer should not exceed itmax iterations."""
        setup = tetrahedron_setup
        itmax = 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=itmax,
            )

        assert result.it <= itmax, f"Iterations {result.it} exceeded max {itmax}"

    def test_output_is_finite(self, tetrahedron_setup):
        """All outputs should be finite (no NaN or Inf)."""
        setup = tetrahedron_setup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert np.all(np.isfinite(result.u)), "u contains non-finite values"
        assert np.all(np.isfinite(result.v)), "v contains non-finite values"
        assert np.all(np.isfinite(result.ut)), "ut contains non-finite values"
        assert np.all(np.isfinite(result.vt)), "vt contains non-finite values"
        assert np.all(np.isfinite(result.om)), "om contains non-finite values"
        assert np.all(np.isfinite(result.angn)), "angn contains non-finite values"


@pytest.mark.slow
@pytest.mark.skip(reason="Integration tests require significant computation time (>2 min each)")
class TestEnergyTypes:
    """Test different energy types in optimization."""

    def test_distortion_energy(self, tetrahedron_setup):
        """Distortion energy type should run without error."""
        setup = tetrahedron_setup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert result is not None

    def test_chebyshev_energy(self, tetrahedron_setup):
        """Chebyshev energy type should run without error."""
        setup = tetrahedron_setup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='chebyshev',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert result is not None

    def test_alignment_energy(self, tetrahedron_setup):
        """Alignment energy type should run without error."""
        setup = tetrahedron_setup

        # Alignment energy requires additional weight parameters
        weight = MockWeight()
        weight.ang_dir = setup['ang'].copy()
        weight.aspect_ratio = 1.0
        weight.w_ratio = 1.0
        weight.w_ang = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='alignment',
                weight=weight,
                if_plot=False,
                itmax=2,
            )

        assert result is not None

    def test_invalid_energy_type_raises(self, tetrahedron_setup):
        """Invalid energy type should raise ValueError."""
        setup = tetrahedron_setup

        with pytest.raises(ValueError, match="does not exist"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimize_RSP(
                    omega=setup['omega'],
                    ang=setup['ang'],
                    u=setup['u'],
                    v=setup['v'],
                    Src=setup['Src'],
                    param=setup['param'],
                    dec=setup['dec'],
                    Reduction=setup['Reduction'],
                    energy_type='invalid_energy',
                    weight=setup['weight'],
                    if_plot=False,
                    itmax=2,
                )


@pytest.mark.slow
@pytest.mark.skip(reason="Integration tests require significant computation time (>2 min each)")
class TestWeightParameters:
    """Test different weight parameter configurations."""

    def test_w_conf_ar_zero(self, tetrahedron_setup):
        """w_conf_ar=0 (pure area) should work."""
        setup = tetrahedron_setup
        weight = MockWeight()
        weight.w_conf_ar = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=weight,
                if_plot=False,
                itmax=2,
            )

        assert result is not None

    def test_w_conf_ar_one(self, tetrahedron_setup):
        """w_conf_ar=1 (pure conformal) should work."""
        setup = tetrahedron_setup
        weight = MockWeight()
        weight.w_conf_ar = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize_RSP(
                omega=setup['omega'],
                ang=setup['ang'],
                u=setup['u'],
                v=setup['v'],
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=weight,
                if_plot=False,
                itmax=2,
            )

        assert result is not None


@pytest.mark.slow
@pytest.mark.skip(reason="Integration tests require significant computation time (>2 min each)")
class TestReproducibility:
    """Test that results are reproducible."""

    def test_deterministic_output(self, tetrahedron_setup):
        """Same inputs should produce same outputs."""
        setup = tetrahedron_setup

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result1 = optimize_RSP(
                omega=setup['omega'].copy(),
                ang=setup['ang'].copy(),
                u=setup['u'].copy(),
                v=setup['v'].copy(),
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

            result2 = optimize_RSP(
                omega=setup['omega'].copy(),
                ang=setup['ang'].copy(),
                u=setup['u'].copy(),
                v=setup['v'].copy(),
                Src=setup['Src'],
                param=setup['param'],
                dec=setup['dec'],
                Reduction=setup['Reduction'],
                energy_type='distortion',
                weight=setup['weight'],
                if_plot=False,
                itmax=2,
            )

        assert np.allclose(result1.u, result2.u), "u not reproducible"
        assert np.allclose(result1.v, result2.v), "v not reproducible"
        assert result1.flag == result2.flag, "flag not reproducible"
        assert result1.it == result2.it, "it not reproducible"

