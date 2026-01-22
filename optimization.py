"""
Optimization for Corman-Crane rectangular parameterization.

Implements:
- Dirichlet energy objective (default)
- Newton solver with line search (Algorithm 3-4)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Callable

from mesh import TriangleMesh
from geometry import compute_face_areas, compute_corner_angles
from sparse_ops import (
    build_constraint_system,
    build_jacobian,
    build_hessian,
    build_laplacian
)


def dirichlet_energy(
    mesh: TriangleMesh,
    areas: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray
) -> float:
    """
    Compute Dirichlet energy: Phi = sum_f A_f * (|grad u|^2 + |grad v|^2)

    For simplicity, we use the cotan-Laplacian formulation:
    Phi = (1/2) * u^T L u + (1/2) * v^T L v

    Args:
        mesh: Triangle mesh
        areas: |F| face areas
        u, v: |V| log scale factors
        theta: |F| frame angles (not used for Dirichlet)

    Returns:
        Energy value
    """
    L = build_laplacian(mesh)
    energy = 0.5 * u @ (L @ u) + 0.5 * v @ (L @ v)
    return energy


def grad_dirichlet_energy(
    mesh: TriangleMesh,
    areas: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray
) -> np.ndarray:
    """
    Gradient of Dirichlet energy.

    grad_u Phi = L u
    grad_v Phi = L v
    grad_theta Phi = 0

    Returns:
        g: (2|V| + |F|) gradient vector
    """
    L = build_laplacian(mesh)
    grad_u = L @ u
    grad_v = L @ v
    grad_theta = np.zeros(mesh.n_faces)

    return np.concatenate([grad_u, grad_v, grad_theta])


def hess_dirichlet_energy(
    mesh: TriangleMesh,
    areas: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray
) -> sparse.csr_matrix:
    """
    Hessian of Dirichlet energy.

    H = [[L, 0, 0],
         [0, L, 0],
         [0, 0, 0]]

    Returns:
        H: (2|V| + |F|) x (2|V| + |F|) sparse matrix
    """
    L = build_laplacian(mesh)
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces

    Z_VV = sparse.csr_matrix((n_vertices, n_vertices))
    Z_VF = sparse.csr_matrix((n_vertices, n_faces))
    Z_FF = sparse.csr_matrix((n_faces, n_faces))

    H = sparse.bmat([
        [L, Z_VV, Z_VF],
        [Z_VV, L, Z_VF],
        [Z_VF.T, Z_VF.T, Z_FF]
    ])

    return H.tocsr()


def solve_constraints_only(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    phi: np.ndarray,
    omega0: np.ndarray,
    s: np.ndarray,
    max_iters: int = 500,
    tol: float = 1e-8,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve only the constraint equations (find feasible point).
    Uses damped Gauss-Newton iteration.
    """
    from scipy.sparse.linalg import lsqr

    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces

    u = np.zeros(n_vertices)
    v = np.zeros(n_vertices)
    theta = np.zeros(n_faces)

    if verbose:
        print("Solving for feasible point...")

    for iteration in range(max_iters):
        F = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)
        norm_F = np.linalg.norm(F)

        if verbose and iteration % 50 == 0:
            print(f"  Iter {iteration:3d}: |F| = {norm_F:.2e}")

        if norm_F < tol:
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break

        J = build_jacobian(mesh, alpha, u, v, s, phi, theta)
        result = lsqr(J, -F, atol=1e-10, btol=1e-10)
        dx = result[0]

        # Adaptive step size
        tau = min(1.0, 10.0 / (np.linalg.norm(dx) + 1))

        u = u + tau * dx[:n_vertices]
        v = v + tau * dx[n_vertices:2*n_vertices]
        theta = theta + tau * dx[2*n_vertices:]

    # NOTE: Do NOT center u,v here - it breaks constraint satisfaction
    # due to sign bits s that can be +1 or -1
    # The constraint terms involve s[c]*v[i], so shifting v doesn't cancel

    return u, v, theta


def solve_optimization(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    phi: np.ndarray,
    omega0: np.ndarray,
    s: np.ndarray,
    max_iters: int = 100,
    tol: float = 1e-8,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Algorithm 3: SolveOptimizationProblem

    Solve the constrained optimization using Newton's method with line search.

    Args:
        mesh: Triangle mesh
        alpha: |C| corner angles
        phi: |H| reference frame angles
        omega0: |E| frame rotation across edges
        s: |C| sign bits
        max_iters: Maximum Newton iterations
        tol: Convergence tolerance
        verbose: Print progress

    Returns:
        u, v: |V| log scale factors
        theta: |F| frame angles
    """
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges

    areas = compute_face_areas(mesh)

    # Initialize variables
    u = np.zeros(n_vertices)
    v = np.zeros(n_vertices)
    theta = np.zeros(n_faces)
    lambda_ = np.zeros(n_edges)

    if verbose:
        print("Starting Newton solver...")
        print(f"Variables: {2*n_vertices + n_faces} primal, {n_edges} dual")

    for iteration in range(max_iters):
        # Build system components
        H = hess_dirichlet_energy(mesh, areas, u, v, theta)
        g = grad_dirichlet_energy(mesh, areas, u, v, theta)
        D = build_hessian(mesh, alpha, u, v, lambda_, s, phi, theta)
        J = build_jacobian(mesh, alpha, u, v, s, phi, theta)
        F = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)

        # KKT system (Algorithm 3 line 11-12)
        # [H + D, J^T] [y ]   [g + J^T lambda]
        # [J    , 0  ] [delta] = [F            ]

        n_primal = 2 * n_vertices + n_faces

        # Build KKT matrix
        A = sparse.bmat([
            [H + D, J.T],
            [J, sparse.csr_matrix((n_edges, n_edges))]
        ])

        # Build right-hand side
        JT_lambda = J.T @ lambda_
        b = np.concatenate([g + JT_lambda, F])

        # Add regularization to make system non-singular
        # The Laplacian has constants in null space, and theta has no strong regularization
        reg = 1e-8 * sparse.eye(A.shape[0])
        A_reg = A + reg

        # Solve KKT system (line 13)
        try:
            y_delta = spsolve(A_reg.tocsc(), b)
        except Exception as e:
            if verbose:
                print(f"Linear solve failed: {e}")
            # Try stronger regularization
            A_reg = A + 1e-4 * sparse.eye(A.shape[0])
            y_delta = spsolve(A_reg.tocsc(), b)

        y = y_delta[:n_primal]
        delta = y_delta[n_primal:]

        # Line search (line 14)
        x = np.concatenate([u, v, theta, lambda_])
        tau = line_search(mesh, alpha, s, phi, omega0, areas, J, F, g, x, y_delta)

        # Update (line 15)
        u = u - tau * y[:n_vertices]
        v = v - tau * y[n_vertices:2*n_vertices]
        theta = theta - tau * y[2*n_vertices:]
        lambda_ = lambda_ - tau * delta

        # Check convergence (line 16-17)
        g_new = grad_dirichlet_energy(mesh, areas, u, v, theta)
        J_new = build_jacobian(mesh, alpha, u, v, s, phi, theta)
        F_new = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)
        JT_lambda_new = J_new.T @ lambda_

        merit = np.linalg.norm(g_new + JT_lambda_new) + np.linalg.norm(F_new)

        if verbose and (iteration % 5 == 0 or merit < tol):
            energy = dirichlet_energy(mesh, areas, u, v, theta)
            print(f"  Iter {iteration:3d}: merit={merit:.2e}, energy={energy:.4f}, tau={tau:.4f}")

        if merit < tol:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break

    return u, v, theta


def line_search(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    s: np.ndarray,
    phi: np.ndarray,
    omega0: np.ndarray,
    areas: np.ndarray,
    J: sparse.csr_matrix,
    F: np.ndarray,
    g: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    backtrack_ratio: float = 0.9,
    max_backtrack: int = 50
) -> float:
    """
    Algorithm 4: LineSearch

    Backtracking line search for KKT system.

    Returns:
        tau: step size
    """
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges
    n_primal = 2 * n_vertices + n_faces

    u = x[:n_vertices]
    v = x[n_vertices:2*n_vertices]
    theta = x[2*n_vertices:n_primal]
    lambda_ = x[n_primal:]

    # Current merit (line 2)
    JT_lambda = J.T @ lambda_
    E = np.linalg.norm(g + JT_lambda) + np.linalg.norm(F)

    tau = 1.0

    for _ in range(max_backtrack):
        # Trial point (line 5)
        u_new = u - tau * y[:n_vertices]
        v_new = v - tau * y[n_vertices:2*n_vertices]
        theta_new = theta - tau * y[2*n_vertices:n_primal]
        lambda_new = lambda_ - tau * y[n_primal:]

        # New objective gradient (line 6)
        g_new = grad_dirichlet_energy(mesh, areas, u_new, v_new, theta_new)

        # New constraint (line 7-8)
        J_new = build_jacobian(mesh, alpha, u_new, v_new, s, phi, theta_new)
        F_new = build_constraint_system(mesh, alpha, u_new, v_new, s, phi, theta_new, omega0)

        # New merit (line 9)
        JT_lambda_new = J_new.T @ lambda_new
        E_new = np.linalg.norm(g_new + JT_lambda_new) + np.linalg.norm(F_new)

        # Sufficient decrease (line 10)
        if E_new <= (1 - tau / 2) * E:
            break

        # Backtrack (line 13)
        tau *= backtrack_ratio

    return tau


def solve_with_callback(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    phi: np.ndarray,
    omega0: np.ndarray,
    s: np.ndarray,
    callback: Optional[Callable] = None,
    max_iters: int = 100,
    tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Solve optimization with optional callback for visualization.

    Args:
        callback: Function(iteration, u, v, theta, merit) called each iteration

    Returns:
        u, v, theta: Solution
        history: List of (merit, energy) tuples
    """
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges

    areas = compute_face_areas(mesh)

    u = np.zeros(n_vertices)
    v = np.zeros(n_vertices)
    theta = np.zeros(n_faces)
    lambda_ = np.zeros(n_edges)

    history = []

    for iteration in range(max_iters):
        H = hess_dirichlet_energy(mesh, areas, u, v, theta)
        g = grad_dirichlet_energy(mesh, areas, u, v, theta)
        D = build_hessian(mesh, alpha, u, v, lambda_, s, phi, theta)
        J = build_jacobian(mesh, alpha, u, v, s, phi, theta)
        F = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)

        n_primal = 2 * n_vertices + n_faces

        A = sparse.bmat([
            [H + D, J.T],
            [J, sparse.csr_matrix((n_edges, n_edges))]
        ])

        JT_lambda = J.T @ lambda_
        b = np.concatenate([g + JT_lambda, F])

        try:
            y_delta = spsolve(A.tocsc(), b)
        except:
            A_reg = A + 1e-6 * sparse.eye(A.shape[0])
            y_delta = spsolve(A_reg.tocsc(), b)

        y = y_delta[:n_primal]
        delta = y_delta[n_primal:]

        x = np.concatenate([u, v, theta, lambda_])
        tau = line_search(mesh, alpha, s, phi, omega0, areas, J, F, g, x, y_delta)

        u = u - tau * y[:n_vertices]
        v = v - tau * y[n_vertices:2*n_vertices]
        theta = theta - tau * y[2*n_vertices:]
        lambda_ = lambda_ - tau * delta

        g_new = grad_dirichlet_energy(mesh, areas, u, v, theta)
        J_new = build_jacobian(mesh, alpha, u, v, s, phi, theta)
        F_new = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)
        JT_lambda_new = J_new.T @ lambda_

        merit = np.linalg.norm(g_new + JT_lambda_new) + np.linalg.norm(F_new)
        energy = dirichlet_energy(mesh, areas, u, v, theta)

        history.append((merit, energy))

        if callback:
            callback(iteration, u, v, theta, merit)

        if merit < tol:
            break

    return u, v, theta, history
