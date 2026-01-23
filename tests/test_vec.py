"""
Pytest tests for Utils/vec.py - vector flattening utility.

Tests that vec() correctly flattens arrays to 1D.
Run with: pytest tests/test_vec.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Utils.vec import vec


# =============================================================================
# Test Cases
# =============================================================================

class TestVec:
    """Tests for the vec() function."""

    def test_1d_array_unchanged(self):
        """1D array should be returned as-is (same values, 1D shape)."""
        x = np.array([1, 2, 3, 4, 5])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 5
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_2d_array_flattened(self):
        """2D array should be flattened to 1D in column-major (F) order like MATLAB."""
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 6
        # MATLAB x(:) uses column-major (Fortran) order: column-by-column
        np.testing.assert_array_equal(result, [1, 4, 2, 5, 3, 6])

    def test_2d_column_vector(self):
        """2D column vector (n,1) should be flattened to 1D."""
        x = np.array([[1], [2], [3]])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 3
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_2d_row_vector(self):
        """2D row vector (1,n) should be flattened to 1D."""
        x = np.array([[1, 2, 3]])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 3
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_list_input(self):
        """Python list should be converted and returned as 1D array."""
        x = [1, 2, 3, 4]
        result = vec(x)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 4
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_nested_list_input(self):
        """Nested Python list (2D) should be flattened to 1D column-major."""
        x = [[1, 2], [3, 4]]
        result = vec(x)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 4
        # Column-major: [[1,2],[3,4]] -> [1, 3, 2, 4]
        np.testing.assert_array_equal(result, [1, 3, 2, 4])

    def test_empty_array(self):
        """Empty array should return empty 1D array."""
        x = np.array([])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 0
        np.testing.assert_array_equal(result, [])

    def test_single_element(self):
        """Single element should return 1D array with one element."""
        x = np.array([42])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 1
        assert result[0] == 42

    def test_single_element_2d(self):
        """Single element in 2D array should return 1D array with one element."""
        x = np.array([[42]])
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 1
        assert result[0] == 42

    def test_scalar_input(self):
        """Scalar input should return 1D array with one element."""
        x = 42
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 1
        assert result[0] == 42

    def test_float_array(self):
        """Float array should preserve dtype and values."""
        x = np.array([1.5, 2.5, 3.5])
        result = vec(x)

        assert result.ndim == 1
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.5, 2.5, 3.5])

    def test_matches_numpy_ravel_fortran(self):
        """vec() should produce same result as np.ravel('F') (column-major)."""
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        result = vec(x)
        expected = np.asarray(x).ravel('F')

        np.testing.assert_array_equal(result, expected)

    def test_3d_array_flattened(self):
        """3D array should also be flattened to 1D column-major."""
        x = np.arange(24).reshape(2, 3, 4)
        result = vec(x)

        assert result.ndim == 1
        assert len(result) == 24
        # Column-major flattening
        np.testing.assert_array_equal(result, x.ravel('F'))
