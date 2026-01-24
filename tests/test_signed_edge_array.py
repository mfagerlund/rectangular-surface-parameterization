"""Tests for SignedEdgeArray class."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from rectangular_surface_parameterization.core.signed_edge_array import SignedEdgeArray


class TestConstruction:
    """Test SignedEdgeArray construction methods."""

    def test_from_edges_and_signs_basic(self):
        """Basic construction from edges and signs."""
        edges = np.array([0, 1, 2])
        signs = np.array([1, -1, 1])
        sea = SignedEdgeArray.from_edges_and_signs(edges, signs)

        assert_array_equal(sea.indices, edges)
        assert_array_equal(sea.signs, signs)

    def test_from_edges_and_signs_2d(self):
        """Construction from 2D arrays (like T2E)."""
        edges = np.array([[0, 1, 2], [1, 2, 3]])
        signs = np.array([[1, -1, 1], [-1, 1, -1]])
        sea = SignedEdgeArray.from_edges_and_signs(edges, signs)

        assert sea.shape == (2, 3)
        assert_array_equal(sea.indices, edges)
        assert_array_equal(sea.signs, signs)

    def test_from_raw(self):
        """Construction from raw 1-based encoding."""
        raw = np.array([1, -2, 3, -4])  # (0+1)*1, (1+1)*-1, (2+1)*1, (3+1)*-1
        sea = SignedEdgeArray.from_raw(raw)

        assert_array_equal(sea.indices, [0, 1, 2, 3])
        assert_array_equal(sea.signs, [1, -1, 1, -1])

    def test_edge_zero_with_positive_sign(self):
        """Edge index 0 with positive sign should work."""
        sea = SignedEdgeArray.from_edges_and_signs(np.array([0]), np.array([1]))
        assert_array_equal(sea.indices, [0])
        assert_array_equal(sea.signs, [1])
        assert_array_equal(sea.raw, [1])  # (0+1)*1 = 1

    def test_edge_zero_with_negative_sign(self):
        """Edge index 0 with negative sign should work."""
        sea = SignedEdgeArray.from_edges_and_signs(np.array([0]), np.array([-1]))
        assert_array_equal(sea.indices, [0])
        assert_array_equal(sea.signs, [-1])
        assert_array_equal(sea.raw, [-1])  # (0+1)*-1 = -1


class TestIndexing:
    """Test SignedEdgeArray indexing."""

    @pytest.fixture
    def sample_array(self):
        """Create a sample 2x3 SignedEdgeArray."""
        edges = np.array([[0, 1, 2], [1, 2, 3]])
        signs = np.array([[1, -1, 1], [-1, 1, -1]])
        return SignedEdgeArray.from_edges_and_signs(edges, signs)

    def test_integer_index(self, sample_array):
        """Integer indexing returns SignedEdgeArray."""
        row = sample_array[0]
        assert isinstance(row, SignedEdgeArray)
        assert_array_equal(row.indices, [0, 1, 2])
        assert_array_equal(row.signs, [1, -1, 1])

    def test_slice_index(self, sample_array):
        """Slice indexing returns SignedEdgeArray."""
        sliced = sample_array[:, 1:]
        assert isinstance(sliced, SignedEdgeArray)
        assert sliced.shape == (2, 2)

    def test_boolean_mask(self, sample_array):
        """Boolean mask indexing."""
        mask = np.array([True, False])
        result = sample_array[mask]
        assert result.shape == (1, 3)

    def test_column_index(self, sample_array):
        """Column indexing."""
        col = sample_array[:, 0]
        assert_array_equal(col.indices, [0, 1])
        assert_array_equal(col.signs, [1, -1])


class TestOperations:
    """Test SignedEdgeArray operations."""

    @pytest.fixture
    def sample_array(self):
        edges = np.array([0, 1, 2, 1, 0])
        signs = np.array([1, -1, 1, 1, -1])
        return SignedEdgeArray.from_edges_and_signs(edges, signs)

    def test_index_into(self, sample_array):
        """Test indexing into per-edge arrays."""
        per_edge = np.array([10, 20, 30])
        result = sample_array.index_into(per_edge)
        assert_array_equal(result, [10, 20, 30, 20, 10])

    def test_signed_index_into(self, sample_array):
        """Test signed indexing."""
        per_edge = np.array([10, 20, 30])
        result = sample_array.signed_index_into(per_edge)
        assert_array_equal(result, [10, -20, 30, 20, -10])

    def test_unique_indices(self, sample_array):
        """Test unique indices extraction."""
        unique = sample_array.unique_indices()
        assert_array_equal(unique, [0, 1, 2])

    def test_bincount(self, sample_array):
        """Test bincount operation."""
        counts = sample_array.bincount(minlength=4)
        assert_array_equal(counts, [2, 2, 1, 0])  # edges 0,1 appear twice, 2 once, 3 never

    def test_bincount_with_weights(self, sample_array):
        """Test bincount with weights."""
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sample_array.bincount(weights=weights, minlength=4)
        # Edge 0: weights 1.0 + 5.0 = 6.0
        # Edge 1: weights 2.0 + 4.0 = 6.0
        # Edge 2: weights 3.0
        assert_array_almost_equal(result, [6.0, 6.0, 3.0, 0.0])


class TestShapeOperations:
    """Test shape-related operations."""

    def test_ravel(self):
        """Test ravel returns SignedEdgeArray."""
        edges = np.array([[0, 1], [2, 3]])
        signs = np.array([[1, -1], [-1, 1]])
        sea = SignedEdgeArray.from_edges_and_signs(edges, signs)

        flat = sea.ravel()
        assert isinstance(flat, SignedEdgeArray)
        assert flat.shape == (4,)
        assert_array_equal(flat.indices, [0, 1, 2, 3])

    def test_flatten(self):
        """Test flatten returns SignedEdgeArray copy."""
        edges = np.array([[0, 1], [2, 3]])
        signs = np.ones((2, 2), dtype=int)
        sea = SignedEdgeArray.from_edges_and_signs(edges, signs)

        flat = sea.flatten()
        assert isinstance(flat, SignedEdgeArray)
        assert flat.shape == (4,)

    def test_reshape(self):
        """Test reshape."""
        sea = SignedEdgeArray.from_edges_and_signs(np.arange(6), np.ones(6, dtype=int))
        reshaped = sea.reshape(2, 3)
        assert reshaped.shape == (2, 3)

    def test_copy(self):
        """Test copy creates independent array."""
        sea = SignedEdgeArray.from_edges_and_signs(np.array([0, 1]), np.array([1, -1]))
        sea_copy = sea.copy()

        # Modify original
        sea._data[0] = 99

        # Copy should be unchanged
        assert sea_copy.raw[0] != 99


class TestNumpyInterop:
    """Test numpy interoperability."""

    def test_asarray(self):
        """np.asarray returns raw data."""
        sea = SignedEdgeArray.from_edges_and_signs(np.array([0, 1]), np.array([1, -1]))
        arr = np.asarray(sea)
        assert_array_equal(arr, [1, -2])  # raw 1-based encoding

    def test_len(self):
        """len() works."""
        sea = SignedEdgeArray.from_edges_and_signs(np.array([0, 1, 2]), np.ones(3, dtype=int))
        assert len(sea) == 3


class TestSparseTriplets:
    """Test sparse matrix construction helpers."""

    def test_to_sparse_triplets(self):
        """Test triplet generation for sparse matrices."""
        # 2 faces, 3 edges each
        edges = np.array([[0, 1, 2], [0, 2, 1]])
        signs = np.array([[1, -1, 1], [-1, 1, 1]])
        sea = SignedEdgeArray.from_edges_and_signs(edges, signs)

        row_idx, col_idx, data = sea.to_sparse_triplets(n_rows=2)

        # Should have 6 entries (2 faces * 3 edges)
        assert len(row_idx) == 6
        assert len(col_idx) == 6
        assert len(data) == 6

        # Row indices should be [0,1,0,1,0,1] (column-major from 2x3)
        assert_array_equal(row_idx, [0, 1, 0, 1, 0, 1])

        # Col indices are the 0-based edge indices
        assert_array_equal(col_idx, sea.indices.ravel())

        # Data is the signs
        assert_array_equal(data, sea.signs.ravel())


class TestRealWorldPatterns:
    """Test patterns that appear in the actual codebase."""

    def test_dec_operator_pattern(self):
        """Pattern from dec_tri.py: building curl operator d1p."""
        # Simulated T2E for 4 triangles, 6 edges
        T2E_raw = np.array([
            [1, -2, 3],    # face 0: edges 0, 1, 2 with signs +, -, +
            [-1, 4, -5],   # face 1: edges 0, 3, 4 with signs -, +, -
            [2, -4, 6],    # face 2: edges 1, 3, 5 with signs +, -, +
            [-3, 5, -6],   # face 3: edges 2, 4, 5 with signs -, +, -
        ])

        sea = SignedEdgeArray.from_raw(T2E_raw)

        # This is how d1p is built
        nf = 4
        ne = 6
        flat = sea.ravel('F')  # Column-major for MATLAB compatibility

        row_idx = np.tile(np.arange(nf), 3)
        col_idx = flat.indices
        data = flat.signs.astype(float)

        # Verify we can build a sparse matrix
        from scipy.sparse import csr_matrix
        d1p = csr_matrix((data, (row_idx, col_idx)), shape=(nf, ne))

        assert d1p.shape == (nf, ne)
        # Each row should sum to 0 (curl of gradient is zero)
        # (This won't be true for arbitrary T2E, but structure is correct)

    def test_omega_indexing_pattern(self):
        """Pattern from omega_from_scale.py: indexing per-edge omega."""
        omega = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # per-edge rotation

        # T2E for 2 faces
        T2E_raw = np.array([[1, -3, 4], [2, 3, -5]])
        sea = SignedEdgeArray.from_raw(T2E_raw)

        # Get omega values at edges
        omega_at_edges = sea.index_into(omega)
        expected = np.array([[0.1, 0.3, 0.4], [0.2, 0.3, 0.5]])
        assert_array_almost_equal(omega_at_edges, expected)

        # Get signed omega
        signed_omega = sea.signed_index_into(omega)
        expected_signed = np.array([[0.1, -0.3, 0.4], [0.2, 0.3, -0.5]])
        assert_array_almost_equal(signed_omega, expected_signed)
