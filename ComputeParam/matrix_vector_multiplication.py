# === ISSUES ===
# - None: straightforward sparse matrix construction
# === END ISSUES ===

# function M = matrix_vector_multiplication(A)
# % A = [A1 A2 A3]
# %     [A4 A5 A6]
# %     [A7 A8 A9]
# % M * B = A*B
#
# nv = size(A,1);
# n = sqrt(size(A,2));
#
# I = repmat((1:nv)', [n,n]) + repelem((0:n-1)*nv, n*nv,1);
# J = repmat((1:nv)', [n,n]) + repelem(repmat((0:n-1)'*nv, [1,n]), nv,1);
# S = reshape(A, [n*nv,n]);
# M = sparse(I, J, S, n*nv, n*nv);

import numpy as np
from scipy.sparse import csr_matrix
from typing import Union


def matrix_vector_multiplication(A: np.ndarray) -> csr_matrix:
    """
    Build a sparse matrix M such that M * vec(B) = vec(A*B) for each row of A.

    Each row of A represents a flattened n x n matrix (in row-major order).
    The output M is a block-diagonal sparse matrix where each block corresponds
    to one row of A, allowing element-wise matrix-vector multiplication.

    This is used to apply rotation matrices to 2D vectors stored as stacked
    columns [u; v].

    Parameters
    ----------
    A : ndarray (nv, n*n)
        Each row is a flattened n x n matrix.

    Returns
    -------
    M : sparse matrix (n*nv, n*nv)
        Block-diagonal sparse matrix for element-wise matrix-vector multiplication.

    Example
    -------
    For 2x2 rotation matrices with nv rows:
    >>> R = np.array([[1, 0, 0, 1], [0, -1, 1, 0]])  # Two 2x2 matrices flattened
    >>> M = matrix_vector_multiplication(R)
    >>> # M @ [x0, x1, y0, y1] = [R0 @ [x0,y0], R1 @ [x1,y1]] stacked
    """

    # nv = size(A,1);
    # n = sqrt(size(A,2));

    nv = A.shape[0]
    n2 = A.shape[1]
    n = int(np.sqrt(n2))
    assert n * n == n2, f"A must have n^2 columns, got {n2}"

    # For each row i of A, we have an n x n matrix.
    # M is constructed so that M @ vec([x; y]) applies the i-th matrix to the i-th element.
    #
    # MATLAB constructs:
    # I = repmat((1:nv)', [n,n]) + repelem((0:n-1)*nv, n*nv,1)
    # J = repmat((1:nv)', [n,n]) + repelem(repmat((0:n-1)'*nv, [1,n]), nv,1)
    # S = reshape(A, [n*nv,n])
    #
    # In Python, let's build this more explicitly.
    # For n=2 and matrix [a b; c d]:
    #   The matrix M such that M @ [x0,x1,...; y0,y1,...] = [a*x + b*y; c*x + d*y]
    #   where the operations are element-wise across the nv rows.

    # Build row indices, column indices, and values
    # For each (i, j) entry of the n x n matrix, and each vertex v:
    #   M[i*nv + v, j*nv + v] = A[v, i*n + j]

    rows = []
    cols = []
    vals = []

    for i in range(n):
        for j in range(n):
            # Row offset for block i: i*nv
            # Column offset for block j: j*nv
            # A[:, i*n + j] gives the (i,j) element of each matrix
            row_idx = i * nv + np.arange(nv)
            col_idx = j * nv + np.arange(nv)
            val = A[:, i * n + j]

            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(val)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    M = csr_matrix((vals, (rows, cols)), shape=(n * nv, n * nv))

    return M
