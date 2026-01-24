# Python implementation based on MATLAB code from commit 7d1aab4
# https://github.com/etcorman/RectangularSurfaceParameterization
"""
Optimization parameters for the Newton solver in optimize_RSP.

These were previously hardcoded in optimize_RSP.py. Extracting them to a
dataclass makes them:
1. Documented - each parameter has a clear description
2. Tunable - users can override defaults for specific meshes
3. Testable - parameter combinations can be unit tested
"""

from dataclasses import dataclass


@dataclass
class OptimizationParams:
    """Parameters controlling the Newton optimization solver.

    These parameters affect convergence speed and robustness of the
    rectangular surface parameterization optimization.

    Attributes:
        itmax: Maximum number of Newton iterations.
        rho: Line search backtracking factor. On failure, step *= rho.
        beta_init: Initial step size parameter for line search.
        step_min: Minimum step size before line search is considered failed.
        err_tol: L2 error norm threshold for convergence.
        angle_change_tol: Maximum allowed frame field angle change (degrees)
            for convergence. Must be satisfied along with err_tol.
        kkt_residual_tol: Tolerance for KKT system solve residual check.
        lsmr_atol: Absolute tolerance for LSMR iterative solver.
        lsmr_btol: Relative tolerance for LSMR iterative solver.
        lsmr_maxiter: Maximum iterations for LSMR solver.
        lsqr_damp: Damping factor for LSQR fallback solver.
        lsqr_iter_lim: Iteration limit for LSQR fallback solver.
        reg_factor: Diagonal regularization factor for ill-conditioned systems.
    """

    # Iteration control
    itmax: int = 300

    # Line search parameters
    rho: float = 0.9
    beta_init: float = 1.0
    step_min: float = 1e-12

    # Convergence tolerances
    err_tol: float = 1e-5
    angle_change_tol: float = 1e-3

    # KKT system solve
    kkt_residual_tol: float = 1e-5

    # LSMR iterative solver parameters
    lsmr_atol: float = 1e-10
    lsmr_btol: float = 1e-10
    lsmr_maxiter: int = 10000

    # LSQR fallback solver parameters
    lsqr_damp: float = 1e-6
    lsqr_iter_lim: int = 10000

    # Regularization for ill-conditioned systems
    reg_factor: float = 1e-8


# Default instance for backwards compatibility
DEFAULT_PARAMS = OptimizationParams()
