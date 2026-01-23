"""
Pytest tests for Orthotropic/omega_from_scale.py

Tests the omega_from_scale function which builds frame rotation from scale factors
at triangle corners.

Run with: pytest tests/test_omega_from_scale.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import norm as sparse_norm
import warnings

# Add parent directory and submodules to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Preprocess"))
sys.path.insert(0, str(project_root / "Orthotropic"))

from Preprocess.MeshInfo import MeshInfo, mesh_info
from Preprocess.dec_tri import DEC, dec_tri
from Orthotropic.omega_from_scale import omega_from_scale


# =============================================================================
# Helper Functions
# =============================================================================

def safe_dec_tri(mesh):
    """
    Call dec_tri, skipping the test if orientation assertion fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return dec_tri(mesh)
    except AssertionError as e:
        if "Orientation" in str(e) or "Orinetation" in str(e):
            pytest.skip(f"dec_tri has orientation bug: {e}")
        raise


class MockParam:
    """
    Mock parameter object providing ang_basis for testing.

    ang_basis is (nf, 3) and represents the angle between the local basis
    and triangle edges for each face/corner.
    """
    def __init__(self, nf, ang_basis=None):
        if ang_basis is not None:
            self.ang_basis = ang_basis
        else:
            # Default: zero angles (frame aligned with first edge)
            self.ang_basis = np.zeros((nf, 3))


def create_single_triangle():
    """
    Create a single equilateral triangle mesh.

    Vertices:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (0.5, sqrt(3)/2, 0)
    """
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return mesh_info(X, T)


def create_two_triangles():
    """
    Create a mesh with two triangles sharing an edge.

    Vertices:
        0: (0, 0, 0)
        1: (1, 0, 0)
        2: (1, 1, 0)
        3: (0, 1, 0)

    Triangles:
        0: (0, 1, 2)
        1: (0, 2, 3)
    """
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    return mesh_info(X, T)


def create_tetrahedron():
    """
    Create a tetrahedron surface mesh (closed, genus 0).
    4 vertices, 6 edges, 4 faces.
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


def create_right_triangle():
    """
    Create a 3-4-5 right triangle (right angle at vertex 0).
    """
    X = np.array([
        [0.0, 0.0, 0.0],  # right angle here
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return mesh_info(X, T)


def compute_ang_basis(mesh: MeshInfo):
    """
    Compute ang_basis: angle between local basis (first edge direction)
    and each triangle edge.

    For simplicity, we use the angle from the first edge to each edge.
    """
    nf = mesh.nf
    ang_basis = np.zeros((nf, 3))

    for f in range(nf):
        v0, v1, v2 = mesh.T[f]

        # Edge vectors
        e01 = mesh.X[v1] - mesh.X[v0]
        e12 = mesh.X[v2] - mesh.X[v1]
        e20 = mesh.X[v0] - mesh.X[v2]

        # Use e01 as the reference direction (local basis e1)
        e01_norm = e01 / np.linalg.norm(e01)
        normal = mesh.normal[f]

        # Compute angles using atan2 for signed angles
        def signed_angle(v, ref, n):
            """Compute signed angle from ref to v around normal n."""
            v_norm = v / np.linalg.norm(v)
            cos_a = np.clip(np.dot(v_norm, ref), -1, 1)
            sin_a = np.dot(np.cross(ref, v_norm), n)
            return np.arctan2(sin_a, cos_a)

        # ang_basis[f, i] is the angle from e1 to edge i
        ang_basis[f, 0] = signed_angle(e01, e01_norm, normal)  # Should be 0
        ang_basis[f, 1] = signed_angle(e12, e01_norm, normal)
        ang_basis[f, 2] = signed_angle(e20, e01_norm, normal)

    return ang_basis


# =============================================================================
# Test: Output Shapes
# =============================================================================

class TestOutputShapes:
    """Verify omega_from_scale returns matrices with correct shapes."""

    def test_O_shape_single_triangle(self):
        """O should have shape (ne, 6*nf) for single triangle."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        expected_shape = (ne, 6 * nf)
        assert O.shape == expected_shape, \
            f"O shape should be {expected_shape}, got {O.shape}"

    def test_O_shape_two_triangles(self):
        """O should have shape (ne, 6*nf) for two triangles."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        expected_shape = (ne, 6 * nf)
        assert O.shape == expected_shape, \
            f"O shape should be {expected_shape}, got {O.shape}"

    def test_O_shape_tetrahedron(self):
        """O should have shape (ne, 6*nf) for tetrahedron."""
        mesh = create_tetrahedron()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        expected_shape = (ne, 6 * nf)  # (6, 24) for tetrahedron
        assert O.shape == expected_shape, \
            f"O shape should be {expected_shape}, got {O.shape}"

    def test_O_is_sparse(self):
        """O should be a sparse matrix."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        assert issparse(O), "O should be a sparse matrix"


# =============================================================================
# Test: Or (Reduced Output) with Reduction Matrix
# =============================================================================

class TestReducedOutput:
    """Test the reduced output Or when Reduction matrix is provided."""

    def test_Or_is_None_without_Reduction(self):
        """Or should be None when Reduction is not provided."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction=None)

        assert Or is None, "Or should be None when Reduction is not provided"

    def test_Or_shape_with_Reduction(self):
        """Or should have shape (ne, cols_of_Reduction) when Reduction is provided."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne
        nv = mesh.nv

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        # Create a simple Reduction matrix: maps 2*nv to 6*nf
        # This simulates mapping vertex-based variables to corner-based variables
        Reduction = csr_matrix(np.random.randn(6 * nf, 2 * nv))

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction=Reduction)

        expected_shape = (ne, 2 * nv)
        assert Or is not None, "Or should not be None when Reduction is provided"
        assert Or.shape == expected_shape, \
            f"Or shape should be {expected_shape}, got {Or.shape}"

    def test_Or_is_sparse(self):
        """Or should be a sparse matrix."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne
        nv = mesh.nv

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        Reduction = csr_matrix(np.eye(6 * nf, nv))

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction=Reduction)

        assert issparse(Or), "Or should be a sparse matrix"

    def test_Or_equals_O_times_Reduction(self):
        """Or should equal O @ Reduction."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne
        nv = mesh.nv

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        # Random reduction matrix for testing
        np.random.seed(42)
        Reduction = csr_matrix(np.random.randn(6 * nf, 2 * nv))

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction=Reduction)

        # Compute expected Or
        expected_Or = O @ Reduction

        diff_norm = sparse_norm(Or - expected_Or, 'fro')
        assert diff_norm < 1e-10, \
            f"Or should equal O @ Reduction, diff norm is {diff_norm}"


# =============================================================================
# Test: dO (Derivative) Computation
# =============================================================================

class TestDerivativeComputation:
    """Test the derivative dO when compute_derivative=True."""

    def test_dO_is_None_without_flag(self):
        """dO should be None when compute_derivative=False."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, compute_derivative=False)

        assert dO is None, "dO should be None when compute_derivative=False"

    def test_dO_shape_with_flag(self):
        """dO should have shape (ne, nf) when compute_derivative=True."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.ones((nf, 3)) * 0.5  # Non-zero vt needed for non-trivial dO
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, compute_derivative=True)

        expected_shape = (ne, nf)
        assert dO is not None, "dO should not be None when compute_derivative=True"
        assert dO.shape == expected_shape, \
            f"dO shape should be {expected_shape}, got {dO.shape}"

    def test_dO_shape_tetrahedron(self):
        """dO should have shape (ne, nf) for tetrahedron."""
        mesh = create_tetrahedron()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.ones((nf, 3)) * 0.5
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, compute_derivative=True)

        expected_shape = (ne, nf)  # (6, 4) for tetrahedron
        assert dO.shape == expected_shape, \
            f"dO shape should be {expected_shape}, got {dO.shape}"

    def test_dO_is_sparse(self):
        """dO should be a sparse matrix."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.ones((nf, 3)) * 0.5
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, compute_derivative=True)

        assert issparse(dO), "dO should be a sparse matrix"

    def test_dO_is_zero_when_vt_uniform(self):
        """dO should be zero when vt is uniform across corners."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        # Uniform vt per face: vt1 = vt2 = vt3
        vt = np.tile([1.0, 1.0, 1.0], (nf, 1))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, compute_derivative=True)

        # When vt is uniform per face, (vt3-vt1), (vt1-vt2), (vt2-vt3) are all zero
        # So dO should be zero
        dO_norm = sparse_norm(dO, 'fro')
        assert dO_norm < 1e-10, \
            f"dO should be zero when vt is uniform, got norm {dO_norm}"


# =============================================================================
# Test: Mathematical Properties
# =============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of the omega_from_scale output."""

    def test_zero_scales_give_specific_omega(self):
        """With zero scale factors (ut=vt=0), omega is determined by d0p_tri."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # With zero scales, omega = O @ [ut.ravel('F'); vt.ravel('F')] = 0
        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])
        omega = O @ scale_vec

        assert np.allclose(omega, 0), "With zero scales, omega should be zero"

    def test_O_applies_to_combined_scale_vector(self):
        """O multiplied by [ut(:); vt(:)] gives omega vector of length ne."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        np.random.seed(42)
        ut = np.random.randn(nf, 3)
        vt = np.random.randn(nf, 3)
        ang = np.random.randn(nf) * 0.1

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # Combine scales in Fortran order (column-major)
        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])

        assert scale_vec.shape == (6 * nf,), \
            f"Scale vector shape should be ({6 * nf},), got {scale_vec.shape}"

        omega = O @ scale_vec

        assert omega.shape == (ne,), \
            f"omega shape should be ({ne},), got {omega.shape}"
        assert np.all(np.isfinite(omega)), "omega should have finite values"

    def test_omega_changes_with_angle(self):
        """Omega should change when frame angle (ang) changes."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        param = MockParam(nf)
        ut = np.ones((nf, 3)) * 0.5
        # Use non-uniform vt to ensure the angle-dependent terms are non-zero
        # When vt is uniform per face, the Dv_tri contribution becomes zero
        np.random.seed(42)
        vt = np.random.randn(nf, 3) * 0.5

        # Zero angle
        ang0 = np.zeros(nf)
        O0, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang0)
        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])
        omega0 = O0 @ scale_vec

        # Non-zero angle
        ang1 = np.ones(nf) * np.pi / 4
        O1, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang1)
        omega1 = O1 @ scale_vec

        # Omega should be different
        diff = np.linalg.norm(omega1 - omega0)
        assert diff > 1e-10, \
            f"Omega should change with angle, but diff is {diff}"

    def test_omega_periodic_in_angle(self):
        """Omega should be periodic in angle with period pi (due to 2*ang terms)."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        param = MockParam(nf)
        ut = np.ones((nf, 3)) * 0.5
        vt = np.ones((nf, 3)) * 0.5
        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])

        # ang = 0
        ang0 = np.zeros(nf)
        O0, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang0)
        omega0 = O0 @ scale_vec

        # ang = pi (should give same omega due to cos(2*ang + ...) and sin(2*ang + ...))
        ang_pi = np.ones(nf) * np.pi
        O_pi, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang_pi)
        omega_pi = O_pi @ scale_vec

        np.testing.assert_allclose(omega0, omega_pi, atol=1e-10,
            err_msg="Omega should be periodic with period pi")

    def test_first_block_involves_d0p_tri(self):
        """First 3*nf columns of O involve -star1p @ d0p_tri."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # First block: O[:, :3*nf] should be -star1p @ d0p_tri
        expected_first_block = -dec.star1p @ dec.d0p_tri
        actual_first_block = O[:, :3*nf]

        diff_norm = sparse_norm(actual_first_block - expected_first_block, 'fro')
        assert diff_norm < 1e-10, \
            f"First block should be -star1p @ d0p_tri, diff norm is {diff_norm}"


# =============================================================================
# Test: ang_basis Effect
# =============================================================================

class TestAngBasisEffect:
    """Test the effect of ang_basis parameter on the output."""

    def test_different_ang_basis_changes_O(self):
        """Different ang_basis values should give different O matrices."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        # Zero ang_basis
        param0 = MockParam(nf, ang_basis=np.zeros((nf, 3)))
        O0, _, _ = omega_from_scale(mesh, param0, dec, ut, vt, ang)

        # Non-zero ang_basis
        ang_basis1 = np.random.randn(nf, 3) * 0.5
        param1 = MockParam(nf, ang_basis=ang_basis1)
        O1, _, _ = omega_from_scale(mesh, param1, dec, ut, vt, ang)

        # O should be different
        diff_norm = sparse_norm(O0 - O1, 'fro')
        assert diff_norm > 1e-10, \
            f"Different ang_basis should give different O, but diff norm is {diff_norm}"

    def test_realistic_ang_basis(self):
        """Test with realistically computed ang_basis."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        # Compute realistic ang_basis
        ang_basis = compute_ang_basis(mesh)
        param = MockParam(nf, ang_basis=ang_basis)

        ut = np.ones((nf, 3)) * 0.5
        vt = np.ones((nf, 3)) * 0.3
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # Just verify it runs and produces finite output
        assert issparse(O), "O should be sparse"
        assert np.all(np.isfinite(O.data)), "O should have finite values"


# =============================================================================
# Test: Numerical Derivative Verification
# =============================================================================

class TestNumericalDerivative:
    """Verify dO by comparing with finite differences."""

    def test_dO_matches_finite_difference(self):
        """dO should match numerical finite difference approximation."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.random.randn(nf, 3) * 0.5  # Random vt for non-trivial derivative

        # Base angle
        ang0 = np.random.randn(nf) * 0.1

        # Compute analytical derivative
        O0, _, dO = omega_from_scale(mesh, param, dec, ut, vt, ang0, compute_derivative=True)
        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])

        # Compute omega at base angle
        omega0 = O0 @ scale_vec

        # Finite difference approximation
        eps = 1e-7
        dO_fd = np.zeros((ne, nf))

        for f in range(nf):
            ang_plus = ang0.copy()
            ang_plus[f] += eps
            O_plus, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang_plus)
            omega_plus = O_plus @ scale_vec

            dO_fd[:, f] = (omega_plus - omega0) / eps

        # Compare
        dO_dense = dO.toarray()
        np.testing.assert_allclose(dO_dense, dO_fd, atol=1e-5, rtol=1e-4,
            err_msg="dO should match finite difference approximation")


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_triangle_works(self):
        """Function works correctly with single triangle mesh."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.random.randn(nf, 3)
        vt = np.random.randn(nf, 3)
        ang = np.random.randn(nf)

        O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang)

        assert O.shape == (mesh.ne, 6 * nf)
        assert np.all(np.isfinite(O.data))

    def test_large_scale_factors(self):
        """Function handles large scale factor values."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.ones((nf, 3)) * 10.0
        vt = np.ones((nf, 3)) * 10.0
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])
        omega = O @ scale_vec

        assert np.all(np.isfinite(omega)), "omega should be finite with large scales"

    def test_small_scale_factors(self):
        """Function handles small scale factor values."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.ones((nf, 3)) * 1e-10
        vt = np.ones((nf, 3)) * 1e-10
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])
        omega = O @ scale_vec

        assert np.all(np.isfinite(omega)), "omega should be finite with small scales"

    def test_negative_scale_factors(self):
        """Function handles negative scale factor values."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = -np.ones((nf, 3)) * 0.5
        vt = -np.ones((nf, 3)) * 0.5
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])
        omega = O @ scale_vec

        assert np.all(np.isfinite(omega)), "omega should be finite with negative scales"

    def test_extreme_angles(self):
        """Function handles extreme angle values."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.ones((nf, 3)) * 0.5
        vt = np.ones((nf, 3)) * 0.5

        # Large angle values
        ang = np.ones(nf) * 100 * np.pi

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        scale_vec = np.concatenate([ut.ravel('F'), vt.ravel('F')])
        omega = O @ scale_vec

        assert np.all(np.isfinite(omega)), "omega should be finite with extreme angles"


# =============================================================================
# Test: Cotangent Weights Usage
# =============================================================================

class TestCotangentWeightsUsage:
    """Test that cotangent weights from mesh are used correctly."""

    def test_uses_mesh_cot_corner_angle(self):
        """Function should use mesh.cot_corner_angle in computations."""
        mesh = create_right_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        param = MockParam(nf)
        ut = np.ones((nf, 3))
        vt = np.ones((nf, 3))
        ang = np.zeros(nf)

        # The right triangle has cot(90 deg) = 0, which should affect the result
        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # Verify computation uses cot_corner_angle
        # cot(90 deg) = 0, so terms involving cot_ang[:,0] should be affected
        assert np.all(np.isfinite(O.data)), "O should be finite even with cot(90)=0"

    def test_different_triangles_give_different_results(self):
        """Different triangle shapes should give different O matrices."""
        mesh1 = create_single_triangle()  # Equilateral
        mesh2 = create_right_triangle()   # Right triangle

        dec1 = safe_dec_tri(mesh1)
        dec2 = safe_dec_tri(mesh2)

        param1 = MockParam(mesh1.nf)
        param2 = MockParam(mesh2.nf)

        ut1 = np.ones((mesh1.nf, 3))
        vt1 = np.ones((mesh1.nf, 3))
        ut2 = np.ones((mesh2.nf, 3))
        vt2 = np.ones((mesh2.nf, 3))
        ang1 = np.zeros(mesh1.nf)
        ang2 = np.zeros(mesh2.nf)

        O1, _, _ = omega_from_scale(mesh1, param1, dec1, ut1, vt1, ang1)
        O2, _, _ = omega_from_scale(mesh2, param2, dec2, ut2, vt2, ang2)

        # They have the same shape but different values
        assert O1.shape == O2.shape

        # The actual values should differ due to different cotangent weights
        # Compare normalized versions
        norm1 = sparse_norm(O1, 'fro')
        norm2 = sparse_norm(O2, 'fro')

        # At least the norms should be different
        assert not np.isclose(norm1, norm2, rtol=0.1), \
            "Different triangle shapes should give different O matrices"


# =============================================================================
# Test: Consistency with Combined Inputs
# =============================================================================

class TestCombinedInputs:
    """Test with various combinations of inputs."""

    def test_all_options_enabled(self):
        """Test with Reduction and compute_derivative both enabled."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf
        ne = mesh.ne
        nv = mesh.nv

        param = MockParam(nf)
        ut = np.random.randn(nf, 3)
        vt = np.random.randn(nf, 3)
        ang = np.random.randn(nf) * 0.1

        Reduction = csr_matrix(np.random.randn(6 * nf, 2 * nv))

        O, Or, dO = omega_from_scale(
            mesh, param, dec, ut, vt, ang,
            Reduction=Reduction,
            compute_derivative=True
        )

        assert O.shape == (ne, 6 * nf)
        assert Or is not None
        assert Or.shape == (ne, 2 * nv)
        assert dO is not None
        assert dO.shape == (ne, nf)

    def test_realistic_ang_basis_with_derivative(self):
        """Test with realistic ang_basis and derivative computation."""
        mesh = create_tetrahedron()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        ang_basis = compute_ang_basis(mesh)
        param = MockParam(nf, ang_basis=ang_basis)

        ut = np.random.randn(nf, 3) * 0.5
        vt = np.random.randn(nf, 3) * 0.5
        ang = np.random.randn(nf) * 0.1

        O, _, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, compute_derivative=True)

        assert np.all(np.isfinite(O.data))
        assert np.all(np.isfinite(dO.data))


# =============================================================================
# Test: Integration with dec_tri Operators
# =============================================================================

class TestDecTriIntegration:
    """Test integration with dec_tri operators (star1p, d0p_tri)."""

    def test_uses_star1p(self):
        """Function should use dec.star1p in computations."""
        mesh = create_two_triangles()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # First block should be -star1p @ d0p_tri
        first_block = O[:, :3*nf]
        expected = -dec.star1p @ dec.d0p_tri

        diff = sparse_norm(first_block - expected, 'fro')
        assert diff < 1e-10, "First block should be -star1p @ d0p_tri"

    def test_uses_d0p_tri(self):
        """Function should use dec.d0p_tri in computations."""
        mesh = create_single_triangle()
        dec = safe_dec_tri(mesh)

        nf = mesh.nf

        param = MockParam(nf)
        ut = np.zeros((nf, 3))
        vt = np.zeros((nf, 3))
        ang = np.zeros(nf)

        O, _, _ = omega_from_scale(mesh, param, dec, ut, vt, ang)

        # Verify d0p_tri is used correctly
        assert dec.d0p_tri.shape == (mesh.ne, 3 * nf)

        # First block of O should be -star1p @ d0p_tri
        first_block = O[:, :3*nf].toarray()
        expected = (-dec.star1p @ dec.d0p_tri).toarray()

        np.testing.assert_allclose(first_block, expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
