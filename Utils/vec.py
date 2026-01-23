# function y = vec(x)
# y = x(:);

import numpy as np

def vec(x):
    """Flatten array to 1D column-major (MATLAB x(:) semantics).

    MATLAB's x(:) flattens in column-major (Fortran) order.
    This matches that behavior for compatibility.
    """
    return np.asarray(x).ravel('F')
