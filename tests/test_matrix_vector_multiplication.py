"""
Pytest tests for parameterization/matrix_ops.py.

Tests the matrix_vector_multiplication function which builds block-diagonal
sparse matrices for element-wise matrix-vector multiplication (used for
rotation matrices in UV parameterization).

Run with: pytest tests/test_matrix_vector_multiplication.py -v
"""

import numpy as np
import pytest
from scipy.sparse import issparse, csr_matrix
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rectangular_surface_parameterization.parameterization.matrix_ops import matrix_vector_multiplication


# =============================================================================
# Helper Functions
# =============================================================================

def rotation_matrix_2x2(theta: float) -> np.ndarray:
    """Create a 2x2 rotation matrix for angle theta (radians), flattened row-major."""
    c, s = np.cos(theta), np.sin(theta)
    # Row-major flattening: [cos, -sin, sin, cos]
    return np.array([c, -s, s, c])


def apply_rotation_manually(R_flat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Manually apply a 2x2 rotation matrix (flattened) to a 2D vector."""
    R = R_flat.reshape(2, 2)
    return R @ v


# =============================================================================
# Basic Structure Tests
# =============================================================================

class TestBasicStructure:
    """Test basic matrix structure and properties."""

    def test_returns_sparse_matrix(self):
        """Output should be a sparse matrix."""
        A = np.array([[1, 0, 0, 1]])  # Single 2x2 identity
        M = matrix_vector_multiplication(A)
        assert issparse(M), "Output should be sparse"

    def test_returns_csr_matrix(self):
        """Output should specifically be a CSR matrix."""
        A = np.array([[1, 0, 0, 1]])
        M = matrix_vector_multiplication(A)
        assert isinstance(M, csr_matrix), "Output should be CSR matrix"

    def test_output_shape_single_2x2(self):
        """Single 2x2 matrix: output shape should be (2, 2)."""
        A = np.array([[1, 0, 0, 1]])  # 1 row, 4 columns
        M = matrix_vector_multiplication(A)
        assert M.shape == (2, 2), f"Expected (2, 2), got {M.shape}"

    def test_output_shape_multiple_2x2(self):
        """Multiple 2x2 matrices: output shape should be (2*nv, 2*nv)."""
        nv = 5
        A = np.tile([1, 0, 0, 1], (nv, 1))  # nv rows of identity
        M = matrix_vector_multiplication(A)
        expected_shape = (2 * nv, 2 * nv)
        assert M.shape == expected_shape, f"Expected {expected_shape}, got {M.shape}"

    def test_output_shape_3x3(self):
        """Single 3x3 matrix: output shape should be (3, 3)."""
        # 3x3 identity flattened row-major
        A = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])
        M = matrix_vector_multiplication(A)
        assert M.shape == (3, 3), f"Expected (3, 3), got {M.shape}"

    def test_assertion_on_non_square_entries(self):
        """Should raise assertion error if columns != n^2."""
        A = np.array([[1, 2, 3]])  # 3 columns is not a perfect square
        with pytest.raises(AssertionError):
            matrix_vector_multiplication(A)


# =============================================================================
# Identity Matrix Tests
# =============================================================================

class TestIdentityRotation:
    """Test with identity matrices (0 degree rotation)."""

    def test_single_identity_2x2(self):
        """Single 2x2 identity: M should be identity."""
        A = np.array([[1, 0, 0, 1]])
        M = matrix_vector_multiplication(A)

        expected = np.eye(2)
        np.testing.assert_array_almost_equal(
            M.toarray(), expected, decimal=10,
            err_msg="Single identity should produce identity matrix"
        )

    def test_multiple_identity_2x2(self):
        """Multiple 2x2 identities: M should be block diagonal identity."""
        nv = 4
        A = np.tile([1, 0, 0, 1], (nv, 1))
        M = matrix_vector_multiplication(A)

        expected = np.eye(2 * nv)
        np.testing.assert_array_almost_equal(
            M.toarray(), expected, decimal=10,
            err_msg="Multiple identities should produce block diagonal identity"
        )

    def test_identity_preserves_vector(self):
        """Identity rotation should preserve input vectors."""
        nv = 3
        A = np.tile([1, 0, 0, 1], (nv, 1))
        M = matrix_vector_multiplication(A)

        # Input vector: [x0, x1, x2, y0, y1, y2]
        vec_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        vec_out = M @ vec_in

        np.testing.assert_array_almost_equal(
            vec_out, vec_in, decimal=10,
            err_msg="Identity should preserve vectors"
        )


# =============================================================================
# Standard Rotation Tests
# =============================================================================

class TestStandardRotations:
    """Test standard rotation angles: 90, 180, 270 degrees."""

    def test_90_degree_rotation_single(self):
        """Single 90-degree rotation matrix."""
        theta = np.pi / 2
        A = rotation_matrix_2x2(theta).reshape(1, -1)
        M = matrix_vector_multiplication(A)

        # 90-degree rotation: [x, y] -> [-y, x]
        vec_in = np.array([1.0, 0.0])  # [x, y]
        vec_out = M @ vec_in

        expected = np.array([0.0, 1.0])  # [-y, x] = [0, 1]
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="90-degree rotation failed"
        )

    def test_90_degree_rotation_multiple(self):
        """Multiple 90-degree rotations."""
        theta = np.pi / 2
        nv = 3
        R_flat = rotation_matrix_2x2(theta)
        A = np.tile(R_flat, (nv, 1))
        M = matrix_vector_multiplication(A)

        # Input: 3 vectors as [x0, x1, x2, y0, y1, y2]
        vec_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        vec_out = M @ vec_in

        # 90-degree rotation: [xi, yi] -> [-yi, xi]
        # Output: [-y0, -y1, -y2, x0, x1, x2] = [-4, -5, -6, 1, 2, 3]
        expected = np.array([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="90-degree rotation on multiple vectors failed"
        )

    def test_180_degree_rotation_single(self):
        """Single 180-degree rotation matrix."""
        theta = np.pi
        A = rotation_matrix_2x2(theta).reshape(1, -1)
        M = matrix_vector_multiplication(A)

        # 180-degree rotation: [x, y] -> [-x, -y]
        vec_in = np.array([3.0, 4.0])
        vec_out = M @ vec_in

        expected = np.array([-3.0, -4.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="180-degree rotation failed"
        )

    def test_270_degree_rotation_single(self):
        """Single 270-degree rotation matrix (-90 degrees)."""
        theta = 3 * np.pi / 2
        A = rotation_matrix_2x2(theta).reshape(1, -1)
        M = matrix_vector_multiplication(A)

        # 270-degree rotation: [x, y] -> [y, -x]
        vec_in = np.array([1.0, 0.0])
        vec_out = M @ vec_in

        expected = np.array([0.0, -1.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="270-degree rotation failed"
        )

    def test_45_degree_rotation(self):
        """45-degree rotation preserves vector magnitude."""
        theta = np.pi / 4
        A = rotation_matrix_2x2(theta).reshape(1, -1)
        M = matrix_vector_multiplication(A)

        vec_in = np.array([1.0, 0.0])
        vec_out = M @ vec_in

        # Magnitude should be preserved
        assert abs(np.linalg.norm(vec_out) - np.linalg.norm(vec_in)) < 1e-10, \
            "Rotation should preserve vector magnitude"

        # Expected output: [cos(45), sin(45)] = [sqrt(2)/2, sqrt(2)/2]
        expected = np.array([np.sqrt(2)/2, np.sqrt(2)/2])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="45-degree rotation failed"
        )


# =============================================================================
# Block Diagonal Structure Tests
# =============================================================================

class TestBlockDiagonalStructure:
    """Test that the matrix has correct block diagonal structure."""

    def test_block_diagonal_sparsity_pattern(self):
        """Check that non-zeros are only in 2x2 diagonal blocks."""
        nv = 4
        A = np.random.rand(nv, 4)  # Random 2x2 matrices
        M = matrix_vector_multiplication(A)

        # Convert to dense for inspection
        M_dense = M.toarray()

        # Check that elements outside 2x2 blocks along diagonal are zero
        for i in range(2 * nv):
            for j in range(2 * nv):
                # Block index for row i: i // nv
                # Block index for col j: j // nv
                block_row = i // nv
                block_col = j // nv

                # Element (i, j) belongs to block (block_row, block_col)
                # For block diagonal, block_row == block_col
                # But within-block indices: (i % nv, j % nv) should match
                vertex_row = i % nv
                vertex_col = j % nv

                if vertex_row != vertex_col:
                    # Off-diagonal within blocks (across vertices) should be zero
                    assert abs(M_dense[i, j]) < 1e-15, \
                        f"M[{i},{j}] = {M_dense[i, j]} should be zero (different vertices)"

    def test_each_vertex_has_independent_2x2_block(self):
        """Each vertex's 2x2 matrix acts independently."""
        nv = 3

        # Give each vertex a different rotation
        angles = [0, np.pi/4, np.pi/2]
        A = np.array([rotation_matrix_2x2(theta) for theta in angles])
        M = matrix_vector_multiplication(A)

        # Apply to unit vectors for each vertex separately
        for v in range(nv):
            # Input: unit x vector for vertex v only
            vec_in = np.zeros(2 * nv)
            vec_in[v] = 1.0  # x component for vertex v

            vec_out = M @ vec_in

            # Expected: rotated vector for vertex v
            expected = np.zeros(2 * nv)
            expected[v] = np.cos(angles[v])      # x output
            expected[nv + v] = np.sin(angles[v])  # y output

            np.testing.assert_array_almost_equal(
                vec_out, expected, decimal=10,
                err_msg=f"Vertex {v} rotation failed"
            )

    def test_nnz_count(self):
        """Non-zero count should be 4*nv for dense 2x2 blocks."""
        nv = 10
        A = np.random.rand(nv, 4) + 0.1  # Ensure non-zero entries
        M = matrix_vector_multiplication(A)

        # Each vertex contributes 4 non-zeros (2x2 block)
        expected_nnz = 4 * nv
        assert M.nnz == expected_nnz, \
            f"Expected {expected_nnz} non-zeros, got {M.nnz}"


# =============================================================================
# Vector Application Tests
# =============================================================================

class TestVectorApplication:
    """Test application of block diagonal matrix to vectors."""

    def test_element_wise_rotation(self):
        """Verify element-wise rotation behavior."""
        nv = 4
        angles = np.linspace(0, np.pi, nv)
        A = np.array([rotation_matrix_2x2(theta) for theta in angles])
        M = matrix_vector_multiplication(A)

        # Input vectors: x = [1,1,1,1], y = [0,0,0,0]
        x = np.ones(nv)
        y = np.zeros(nv)
        vec_in = np.concatenate([x, y])

        vec_out = M @ vec_in

        # Verify each output separately
        for v in range(nv):
            # Input vector for vertex v: [1, 0]
            # Rotated by angles[v]: [cos(angles[v]), sin(angles[v])]
            expected_x = np.cos(angles[v])
            expected_y = np.sin(angles[v])

            assert abs(vec_out[v] - expected_x) < 1e-10, \
                f"Vertex {v}: x component {vec_out[v]} != {expected_x}"
            assert abs(vec_out[nv + v] - expected_y) < 1e-10, \
                f"Vertex {v}: y component {vec_out[nv + v]} != {expected_y}"

    def test_different_rotations_per_vertex(self):
        """Different rotation matrices for different vertices."""
        # 3 vertices with different rotations
        A = np.array([
            [1, 0, 0, 1],    # Identity (0 deg)
            [0, -1, 1, 0],   # 90 deg
            [-1, 0, 0, -1],  # 180 deg
        ])
        M = matrix_vector_multiplication(A)

        # Input: [1, 2, 3] for x, [4, 5, 6] for y
        vec_in = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        vec_out = M @ vec_in

        # Expected outputs:
        # v0 (identity): [1, 4]
        # v1 (90 deg): [2, 5] -> [-5, 2]
        # v2 (180 deg): [3, 6] -> [-3, -6]
        expected = np.array([1.0, -5.0, -3.0, 4.0, 2.0, -6.0])

        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Different rotations per vertex failed"
        )

    def test_compose_two_rotations(self):
        """Applying M twice should compose rotations."""
        nv = 2
        theta = np.pi / 4  # 45 degrees
        A = np.array([rotation_matrix_2x2(theta) for _ in range(nv)])
        M = matrix_vector_multiplication(A)

        # Apply M twice = 90 degree rotation
        M2 = M @ M

        vec_in = np.array([1.0, 0.0, 0.0, 1.0])  # [1,0] and [0,1]
        vec_out = M2 @ vec_in

        # 90-degree rotation
        expected = np.array([0.0, -1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Composed rotation failed"
        )


# =============================================================================
# Numerical Precision Tests
# =============================================================================

class TestNumericalPrecision:
    """Test numerical precision and edge cases."""

    def test_very_small_angles(self):
        """Very small rotation angles should work correctly."""
        theta = 1e-10
        A = rotation_matrix_2x2(theta).reshape(1, -1)
        M = matrix_vector_multiplication(A)

        vec_in = np.array([1.0, 0.0])
        vec_out = M @ vec_in

        # Should be approximately [1, theta]
        expected = np.array([np.cos(theta), np.sin(theta)])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Very small angle rotation failed"
        )

    def test_large_number_of_vertices(self):
        """Should handle large number of vertices efficiently."""
        nv = 1000
        A = np.random.rand(nv, 4)
        M = matrix_vector_multiplication(A)

        assert M.shape == (2 * nv, 2 * nv), f"Shape mismatch for large nv"
        assert M.nnz == 4 * nv, f"Wrong nnz for large nv"

    def test_orthogonality_preserved(self):
        """Rotation matrices should be orthogonal: R^T R = I."""
        nv = 5
        angles = np.random.rand(nv) * 2 * np.pi
        A = np.array([rotation_matrix_2x2(theta) for theta in angles])
        M = matrix_vector_multiplication(A)

        # M^T M should be identity (block-wise)
        MTM = M.T @ M
        expected = np.eye(2 * nv)

        np.testing.assert_array_almost_equal(
            MTM.toarray(), expected, decimal=10,
            err_msg="M^T M should be identity for rotation matrices"
        )

    def test_determinant_is_one(self):
        """Each 2x2 rotation block should have determinant 1."""
        nv = 4
        angles = np.random.rand(nv) * 2 * np.pi
        A = np.array([rotation_matrix_2x2(theta) for theta in angles])
        M = matrix_vector_multiplication(A)
        M_dense = M.toarray()

        # Extract each 2x2 block and check determinant
        for v in range(nv):
            # Block for vertex v: rows [v, nv+v], cols [v, nv+v]
            block = np.array([
                [M_dense[v, v], M_dense[v, nv + v]],
                [M_dense[nv + v, v], M_dense[nv + v, nv + v]]
            ])
            det = np.linalg.det(block)
            assert abs(det - 1.0) < 1e-10, \
                f"Vertex {v}: det = {det}, expected 1"


# =============================================================================
# 3x3 Matrix Tests
# =============================================================================

class Test3x3Matrices:
    """Test with 3x3 matrices (not just 2x2)."""

    def test_single_3x3_identity(self):
        """Single 3x3 identity matrix."""
        A = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]])  # Flattened 3x3 identity
        M = matrix_vector_multiplication(A)

        assert M.shape == (3, 3), f"Expected (3, 3), got {M.shape}"
        np.testing.assert_array_almost_equal(
            M.toarray(), np.eye(3), decimal=10,
            err_msg="3x3 identity failed"
        )

    def test_multiple_3x3(self):
        """Multiple 3x3 matrices."""
        nv = 3
        A = np.tile([1, 0, 0, 0, 1, 0, 0, 0, 1], (nv, 1))
        M = matrix_vector_multiplication(A)

        assert M.shape == (3 * nv, 3 * nv), f"Shape mismatch"
        np.testing.assert_array_almost_equal(
            M.toarray(), np.eye(3 * nv), decimal=10,
            err_msg="Multiple 3x3 identities failed"
        )

    def test_3x3_rotation_z_axis(self):
        """3x3 rotation around z-axis."""
        theta = np.pi / 2
        c, s = np.cos(theta), np.sin(theta)
        # Rotation around z: [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        R = np.array([c, -s, 0, s, c, 0, 0, 0, 1]).reshape(1, -1)
        M = matrix_vector_multiplication(R)

        vec_in = np.array([1.0, 0.0, 0.0])  # x-axis
        vec_out = M @ vec_in

        expected = np.array([0.0, 1.0, 0.0])  # y-axis
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="3x3 z-rotation failed"
        )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_vertex(self):
        """Single vertex (nv=1) should work."""
        A = np.array([[0, -1, 1, 0]])  # 90 degree rotation
        M = matrix_vector_multiplication(A)

        assert M.shape == (2, 2), f"Shape mismatch for single vertex"

    def test_zero_matrix(self):
        """Zero rotation matrix (degenerate case)."""
        A = np.array([[0, 0, 0, 0]])
        M = matrix_vector_multiplication(A)

        vec_in = np.array([1.0, 2.0])
        vec_out = M @ vec_in

        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Zero matrix should produce zero output"
        )

    def test_scaling_matrix(self):
        """Scaling matrix (non-rotation) should also work."""
        scale = 2.0
        A = np.array([[scale, 0, 0, scale]])  # 2x uniform scaling
        M = matrix_vector_multiplication(A)

        vec_in = np.array([1.0, 3.0])
        vec_out = M @ vec_in

        expected = np.array([2.0, 6.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Scaling matrix failed"
        )

    def test_non_uniform_scaling(self):
        """Non-uniform scaling matrix."""
        A = np.array([[2, 0, 0, 3]])  # Scale x by 2, y by 3
        M = matrix_vector_multiplication(A)

        vec_in = np.array([1.0, 1.0])
        vec_out = M @ vec_in

        expected = np.array([2.0, 3.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Non-uniform scaling failed"
        )

    def test_shear_matrix(self):
        """Shear matrix (non-orthogonal transformation)."""
        A = np.array([[1, 2, 0, 1]])  # Shear in x
        M = matrix_vector_multiplication(A)

        vec_in = np.array([1.0, 1.0])
        vec_out = M @ vec_in

        # [1 2][1] = [3]
        # [0 1][1]   [1]
        expected = np.array([3.0, 1.0])
        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Shear matrix failed"
        )


# =============================================================================
# Consistency with Manual Computation
# =============================================================================

class TestConsistencyWithManual:
    """Test that matrix multiplication matches manual computation."""

    def test_random_rotations_match_manual(self):
        """Random rotations should match manual application."""
        np.random.seed(42)
        nv = 10
        angles = np.random.rand(nv) * 2 * np.pi
        A = np.array([rotation_matrix_2x2(theta) for theta in angles])
        M = matrix_vector_multiplication(A)

        # Random input vectors
        x = np.random.rand(nv)
        y = np.random.rand(nv)
        vec_in = np.concatenate([x, y])

        vec_out = M @ vec_in

        # Manually compute expected output
        expected_x = np.zeros(nv)
        expected_y = np.zeros(nv)
        for v in range(nv):
            R = A[v].reshape(2, 2)
            result = R @ np.array([x[v], y[v]])
            expected_x[v] = result[0]
            expected_y[v] = result[1]
        expected = np.concatenate([expected_x, expected_y])

        np.testing.assert_array_almost_equal(
            vec_out, expected, decimal=10,
            err_msg="Matrix application doesn't match manual computation"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
