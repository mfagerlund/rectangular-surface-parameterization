# Python implementation based on MATLAB code from commit 7d1aab4
# https://github.com/etcorman/RectangularSurfaceParameterization
"""
SignedEdgeArray: Abstraction for signed 1-based edge encoding.

The MATLAB code uses a signed 1-based encoding for triangle-to-edge connectivity:
    T2E[f, i] = (edge_idx + 1) * sign

where:
    - edge_idx is the 0-based edge index (0 to ne-1)
    - sign is +1 or -1 indicating edge orientation relative to the triangle

This encoding allows storing both the edge index and orientation in a single
integer, but requires careful decoding:
    edge_idx = abs(T2E) - 1
    sign = sign(T2E)

This class encapsulates the encoding to prevent off-by-one errors.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union, Optional


class SignedEdgeArray:
    """
    Array of signed edge references with 1-based internal encoding.

    This class wraps the signed 1-based edge encoding used in MATLAB:
        internal_value = (edge_idx + 1) * sign

    It provides a clean API that works with 0-based indices externally
    while maintaining compatibility with the original algorithm.

    Examples
    --------
    >>> # Create from 0-based indices and signs
    >>> edges = np.array([0, 1, 2])
    >>> signs = np.array([1, -1, 1])
    >>> sea = SignedEdgeArray.from_edges_and_signs(edges, signs)
    >>> sea.indices  # 0-based
    array([0, 1, 2])
    >>> sea.signs
    array([ 1, -1,  1])

    >>> # Index into per-edge arrays
    >>> omega = np.array([0.1, 0.2, 0.3])
    >>> sea.index_into(omega)
    array([0.1, 0.2, 0.3])

    >>> # Get signed values
    >>> sea.signed_index_into(omega)
    array([ 0.1, -0.2,  0.3])
    """

    __slots__ = ('_data',)

    def __init__(self, data: np.ndarray):
        """
        Create from raw 1-based signed data.

        Use from_edges_and_signs() for the common case of creating
        from 0-based indices and signs.

        Parameters
        ----------
        data : ndarray
            Raw signed 1-based encoded data. Values are (edge_idx + 1) * sign.
        """
        self._data = np.asarray(data)

    @classmethod
    def from_edges_and_signs(
        cls,
        edges: np.ndarray,
        signs: np.ndarray
    ) -> SignedEdgeArray:
        """
        Create from 0-based edge indices and orientation signs.

        Parameters
        ----------
        edges : ndarray
            0-based edge indices.
        signs : ndarray
            Orientation signs (+1 or -1).

        Returns
        -------
        SignedEdgeArray
            New instance with the encoded data.
        """
        edges = np.asarray(edges)
        signs = np.asarray(signs)
        data = (edges + 1) * signs
        return cls(data)

    @classmethod
    def from_raw(cls, data: np.ndarray) -> SignedEdgeArray:
        """
        Create from raw 1-based signed encoding.

        Alias for __init__ for clarity when working with existing encoded data.
        """
        return cls(data)

    @property
    def indices(self) -> np.ndarray:
        """
        Get 0-based edge indices.

        Returns
        -------
        ndarray
            0-based edge indices (always non-negative).
        """
        return np.abs(self._data) - 1

    @property
    def signs(self) -> np.ndarray:
        """
        Get orientation signs.

        Returns
        -------
        ndarray
            Signs (+1 or -1) for each edge reference.
        """
        return np.sign(self._data)

    @property
    def raw(self) -> np.ndarray:
        """
        Get the raw 1-based signed encoding.

        Returns
        -------
        ndarray
            Internal representation: (edge_idx + 1) * sign
        """
        return self._data

    def __getitem__(self, key) -> SignedEdgeArray:
        """
        Index into the array, returning a new SignedEdgeArray.

        Supports all numpy indexing: integers, slices, boolean masks, etc.
        """
        return SignedEdgeArray(self._data[key])

    def __setitem__(self, key, value):
        """
        Set values. Accepts SignedEdgeArray or raw encoded values.
        """
        if isinstance(value, SignedEdgeArray):
            self._data[key] = value._data
        else:
            self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def ravel(self, order: str = 'C') -> SignedEdgeArray:
        """Return a flattened SignedEdgeArray."""
        return SignedEdgeArray(self._data.ravel(order))

    def flatten(self, order: str = 'C') -> SignedEdgeArray:
        """Return a flattened copy as SignedEdgeArray."""
        return SignedEdgeArray(self._data.flatten(order))

    def reshape(self, *shape) -> SignedEdgeArray:
        """Return a reshaped SignedEdgeArray."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return SignedEdgeArray(self._data.reshape(shape))

    def copy(self) -> SignedEdgeArray:
        """Return a copy."""
        return SignedEdgeArray(self._data.copy())

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Return raw data for numpy interop.

        This allows np.asarray(sea) to work, returning the 1-based encoding.
        """
        if dtype is None:
            return self._data
        return self._data.astype(dtype)

    # Common operations

    def unique_indices(self) -> np.ndarray:
        """
        Get unique 0-based edge indices.

        Returns
        -------
        ndarray
            Sorted unique edge indices.
        """
        return np.unique(self.indices)

    def index_into(self, arr: np.ndarray) -> np.ndarray:
        """
        Index into a per-edge array using the 0-based indices.

        Parameters
        ----------
        arr : ndarray
            Array to index into. First dimension should be num_edges.

        Returns
        -------
        ndarray
            arr[self.indices]
        """
        return arr[self.indices]

    def signed_index_into(self, arr: np.ndarray) -> np.ndarray:
        """
        Index into array and multiply by signs.

        Useful for getting oriented edge values.

        Parameters
        ----------
        arr : ndarray
            Array to index into.

        Returns
        -------
        ndarray
            arr[self.indices] * self.signs
        """
        return arr[self.indices] * self.signs

    def bincount(
        self,
        weights: Optional[np.ndarray] = None,
        minlength: int = 0
    ) -> np.ndarray:
        """
        Count occurrences of each edge index.

        Parameters
        ----------
        weights : ndarray, optional
            Weights for each element.
        minlength : int, optional
            Minimum length of output array.

        Returns
        -------
        ndarray
            Counts (or weighted sums) per edge index.
        """
        flat_indices = self.indices.ravel()
        if weights is not None:
            weights = np.asarray(weights).ravel()
        return np.bincount(flat_indices, weights=weights, minlength=minlength)

    def to_sparse_triplets(
        self,
        n_rows: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triplets for sparse matrix construction.

        Returns (row_indices, col_indices, data) where:
        - row_indices: range(n_rows) repeated for each column
        - col_indices: the 0-based edge indices
        - data: the signs

        This is useful for building operators like d1p (curl).

        Parameters
        ----------
        n_rows : int
            Number of rows (typically num_faces for T2E).

        Returns
        -------
        row_idx : ndarray
            Row indices for sparse matrix.
        col_idx : ndarray
            Column indices (0-based edge indices).
        data : ndarray
            Data values (signs).
        """
        flat = self.ravel()
        n_entries = flat.size
        n_cols = n_entries // n_rows if n_rows > 0 else 0

        row_idx = np.tile(np.arange(n_rows), n_cols)
        col_idx = flat.indices
        data = flat.signs

        return row_idx, col_idx, data

    def __repr__(self) -> str:
        return f"SignedEdgeArray({self._data!r})"

    def __str__(self) -> str:
        return f"SignedEdgeArray(shape={self.shape}, indices={self.indices}, signs={self.signs})"
