# Sparse linear algebra utilities
# Provides MATLAB-like robustness for singular matrices

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, lsqr
import warnings


def regularized_solve(A, b, reg=1e-10):
    """
    Solve Ax = b with regularization fallback for singular matrices.

    MATLAB vs scipy difference:
        MATLAB's backslash operator (\\) automatically handles singular matrices
        by switching to least-squares or QR factorization. Python's spsolve
        uses strict LU decomposition that fails on singular matrices.

    This function provides MATLAB-like robustness by:
    1. Trying direct solve first (fast path)
    2. Adding small diagonal regularization if singular/failed
    3. Falling back to least-squares (lsqr) as last resort

    Args:
        A: Sparse matrix (can be singular)
        b: Right-hand side vector
        reg: Regularization strength (default 1e-10)

    Returns:
        Solution vector x that minimizes ||Ax - b||
    """
    A_csr = A.tocsr() if not sp.isspmatrix_csr(A) else A

    # Try direct solve first
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Matrix is exactly singular')
            warnings.filterwarnings('ignore', 'Matrix is singular')
            x = spsolve(A_csr, b)
        if not np.any(np.isnan(x)):
            return x
    except Exception:
        pass  # Fall through to regularization

    # Add regularization and retry
    n = A.shape[0]
    A_reg = A_csr + reg * sp.eye(n, format='csr')
    try:
        x = spsolve(A_reg, b)
        if not np.any(np.isnan(x)):
            return x
    except Exception:
        pass  # Fall through to least-squares

    # If still failing, use least-squares (always works)
    result = lsqr(A_csr, b, atol=1e-10, btol=1e-10)
    return result[0]
