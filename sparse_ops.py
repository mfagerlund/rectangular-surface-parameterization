"""
Sparse matrix operators for Corman-Crane rectangular parameterization.

Implements:
- Cotan-Laplacian L (Eq. 1 from supplement)
- Discrete divergence operator (Eq. 2)
- Gradient operator G
- Constraint system (Algorithms 5-8)
"""

import numpy as np
from scipy import sparse
from typing import Tuple

from mesh import TriangleMesh
from geometry import compute_corner_angles, compute_cotan_weights, compute_halfedge_cotan_weights


def build_laplacian(mesh: TriangleMesh, alpha: np.ndarray = None) -> sparse.csr_matrix:
    """
    Build cotan-Laplacian matrix L.

    From Supplement Eq. 1:
    L[i,j] = -w_ij     for edge ij in E
    L[i,i] = sum_k w_ik

    where w_ij = (1/2)(cot(alpha_ij^k) + cot(alpha_ij^l))

    Returns:
        L: |V| x |V| sparse matrix
    """
    if alpha is None:
        alpha = compute_corner_angles(mesh)

    n_vertices = mesh.n_vertices
    cotan_weights = compute_cotan_weights(mesh, alpha)

    # Build sparse matrix using triplet format
    rows = []
    cols = []
    data = []

    for e in range(mesh.n_edges):
        i, j = mesh.edge_vertices[e]
        w = cotan_weights[e]

        # Off-diagonal entries
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-w, -w])

        # Diagonal contributions
        rows.extend([i, j])
        cols.extend([i, j])
        data.extend([w, w])

    L = sparse.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    return L


def build_divergence(mesh: TriangleMesh, alpha: np.ndarray = None) -> sparse.csr_matrix:
    """
    Build discrete divergence operator.

    From Supplement Eq. 2:
    div[i, ij] = s_ij * w_ij

    where s_ij = +1 if edge is oriented from i to j, -1 otherwise.
    We use the convention that edge e = (i,j) has i < j.

    Returns:
        div: |V| x |E| sparse matrix
    """
    if alpha is None:
        alpha = compute_corner_angles(mesh)

    n_vertices = mesh.n_vertices
    n_edges = mesh.n_edges
    cotan_weights = compute_cotan_weights(mesh, alpha)

    rows = []
    cols = []
    data = []

    for e in range(n_edges):
        i, j = mesh.edge_vertices[e]  # i < j by construction
        w = cotan_weights[e]

        # s_ij = +1 for vertex i (edge goes from i)
        # s_ij = -1 for vertex j (edge goes to j)
        rows.extend([i, j])
        cols.extend([e, e])
        data.extend([w, -w])

    div = sparse.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_edges))
    return div


def build_gradient(mesh: TriangleMesh) -> sparse.csr_matrix:
    """
    Build gradient operator G.

    (Gf)_e = f_j - f_i for edge e = (i, j)

    Returns:
        G: |E| x |V| sparse matrix
    """
    n_vertices = mesh.n_vertices
    n_edges = mesh.n_edges

    rows = []
    cols = []
    data = []

    for e in range(n_edges):
        i, j = mesh.edge_vertices[e]

        rows.extend([e, e])
        cols.extend([i, j])
        data.extend([-1.0, 1.0])  # f_j - f_i

    G = sparse.csr_matrix((data, (rows, cols)), shape=(n_edges, n_vertices))
    return G


def build_mass_matrix_vertices(mesh: TriangleMesh, areas: np.ndarray = None) -> sparse.csr_matrix:
    """
    Build diagonal mass matrix at vertices (barycentric area).

    M[i,i] = (1/3) * sum of areas of incident faces

    Returns:
        M: |V| x |V| diagonal sparse matrix
    """
    if areas is None:
        from geometry import compute_face_areas
        areas = compute_face_areas(mesh)

    n_vertices = mesh.n_vertices
    mass = np.zeros(n_vertices)

    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        area_contrib = areas[f] / 3
        mass[v0] += area_contrib
        mass[v1] += area_contrib
        mass[v2] += area_contrib

    return sparse.diags(mass)


def build_mass_matrix_corners(mesh: TriangleMesh, alpha: np.ndarray = None) -> sparse.csr_matrix:
    """
    Build diagonal mass matrix at corners (cotan weight).

    From supplement: M[c,c] = (1/2) * cot(alpha_c)

    Returns:
        M: |C| x |C| diagonal sparse matrix
    """
    if alpha is None:
        alpha = compute_corner_angles(mesh)

    # Mass = (1/2) * cot(alpha) at each corner
    mass = 0.5 / np.tan(alpha + 1e-30)
    mass = np.maximum(mass, 1e-10)  # ensure positive

    return sparse.diags(mass)


def verify_laplacian(mesh: TriangleMesh, L: sparse.csr_matrix) -> Tuple[bool, str]:
    """
    Verify Laplacian properties:
    1. Symmetric
    2. Row sums = 0
    3. L = -div @ G

    Returns:
        (is_valid, message)
    """
    errors = []

    # Check symmetry
    diff = L - L.T
    sym_err = sparse.linalg.norm(diff)
    if sym_err > 1e-10:
        errors.append(f"Not symmetric (error={sym_err:.2e})")

    # Check row sums
    row_sums = np.array(L.sum(axis=1)).flatten()
    max_row_err = np.max(np.abs(row_sums))
    if max_row_err > 1e-10:
        errors.append(f"Row sums not zero (max={max_row_err:.2e})")

    # Check L = -div @ G
    alpha = compute_corner_angles(mesh)
    div = build_divergence(mesh, alpha)
    G = build_gradient(mesh)
    L_check = -div @ G
    diff = L - L_check
    fact_err = sparse.linalg.norm(diff)
    if fact_err > 1e-10:
        errors.append(f"L != -div@G (error={fact_err:.2e})")

    if errors:
        return False, "; ".join(errors)
    return True, "Laplacian verified"


# ============================================================================
# Constraint system (Algorithms 5-8)
# ============================================================================

def compute_residual(
    i: int, j: int, k: int,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    eta: float,
    mesh: TriangleMesh
) -> float:
    """
    Algorithm 6: Compute residual contribution from dual edge in triangle ijk.

    Args:
        i, j, k: vertex indices of triangle
        alpha: corner angles
        u, v: log scale factors at vertices
        s: sign bits at corners
        eta: angle of current frame relative to edge ij
        mesh: triangle mesh (for face lookup)

    Returns:
        Residual contribution
    """
    # Find the face containing vertices i, j, k
    # For now, we pass corner indices directly

    # Get corner indices for face ijk
    # We need to find which face has these vertices
    # This is called with (i,j,k) from a known face, so we compute corners

    # Actually, the algorithm passes vertex indices i,j,k
    # We need to find the corners in the face

    # For the residual, we need:
    # - alpha at corners of the triangle
    # - v values at corners (v[i], v[j], v[k] with sign bits)

    # This function is called from BuildSystem with known face structure
    # Let's assume we're given face-local information

    # For simplicity, we'll implement this differently
    # See build_constraint_system below

    pass  # Implemented inline in build_constraint_system


def build_constraint_system(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    omega0: np.ndarray
) -> np.ndarray:
    """
    Algorithm 5: BuildSystem

    Build the constraint residual F(u, v, theta) = 0.

    Args:
        mesh: Triangle mesh
        alpha: |C| corner angles
        u, v: |V| log scale factors
        s: |C| sign bits
        phi: |H| reference frame angles
        theta: |F| frame rotation angles
        omega0: |E| reference frame rotation across edge

    Returns:
        rho: |E| constraint residual
    """
    n_edges = mesh.n_edges
    n_faces = mesh.n_faces

    rho = np.zeros(n_edges)

    # For each edge, compute left-hand side (line 2)
    for e in range(n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue  # boundary edge

        f0 = he0 // 3  # face containing halfedge he0
        f1 = he1 // 3  # face containing halfedge he1

        # theta difference minus omega0
        rho[e] = (theta[f0] - theta[f1]) - omega0[e]

    # For each face, add right-hand side contributions (lines 3-6)
    for f in range(n_faces):
        v0, v1, v2 = mesh.faces[f]

        # Corner indices
        c0 = 3 * f + 0  # corner at v0
        c1 = 3 * f + 1  # corner at v1
        c2 = 3 * f + 2  # corner at v2

        # Halfedge indices
        he_01 = 3 * f + 0  # halfedge v0->v1
        he_12 = 3 * f + 1  # halfedge v1->v2
        he_20 = 3 * f + 2  # halfedge v2->v0

        # Frame angles relative to each edge
        eta_01 = phi[he_01] + theta[f]
        eta_12 = eta_01 - (np.pi - alpha[c1])
        eta_20 = eta_12 - (np.pi - alpha[c2])

        # Add residual contributions from each halfedge
        # Line 4: edge v0-v1
        e = mesh.halfedge_to_edge[he_01]
        rho[e] += _residual_contribution(v0, v1, v2, c0, c1, c2,
                                          alpha, u, v, s, eta_01, mesh)

        # Line 5: edge v1-v2
        e = mesh.halfedge_to_edge[he_12]
        rho[e] += _residual_contribution(v1, v2, v0, c1, c2, c0,
                                          alpha, u, v, s, eta_12, mesh)

        # Line 6: edge v2-v0
        e = mesh.halfedge_to_edge[he_20]
        rho[e] += _residual_contribution(v2, v0, v1, c2, c0, c1,
                                          alpha, u, v, s, eta_20, mesh)

    return rho


def _residual_contribution(
    i: int, j: int, k: int,
    ci: int, cj: int, ck: int,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    eta: float,
    mesh: TriangleMesh
) -> float:
    """
    Algorithm 6: Residual contribution from edge ij in triangle ijk.

    Line 1-9 of Algorithm 6.
    """
    # Values at corners (with sign bits)
    v_ci = s[ci] * v[i]
    v_cj = s[cj] * v[j]
    v_ck = s[ck] * v[k]

    # Residual computation
    rho = u[j] - u[i]  # conformal part (line 2)
    rho += np.cos(2 * eta) * (v_cj - v_ci)  # non-conformal part (line 3)
    rho += np.sin(2 * eta) * (1 / np.tan(alpha[cj] + 1e-30)) * (v_ck - v_ci)  # line 4
    rho += np.sin(2 * eta) * (1 / np.tan(alpha[ci] + 1e-30)) * (v_ck - v_cj)  # line 5

    # Cotan weight (line 6)
    rho *= 0.5 / np.tan(alpha[ck] + 1e-30)

    # Account for edge orientation (line 7-8)
    if i > j:
        rho = -rho

    return rho


def build_jacobian(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray
) -> sparse.csr_matrix:
    """
    Algorithm 7-8: BuildJacobian

    Build Jacobian of constraint system.

    Returns:
        J: |E| x (2|V| + |F|) sparse matrix
        Columns: [u (|V|), v (|V|), theta (|F|)]
    """
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges

    # Column indices: u is 0:|V|, v is |V|:2|V|, theta is 2|V|:2|V|+|F|
    u_offset = 0
    v_offset = n_vertices
    theta_offset = 2 * n_vertices

    rows_u = []
    cols_u = []
    data_u = []

    rows_v = []
    cols_v = []
    data_v = []

    rows_theta = []
    cols_theta = []
    data_theta = []

    # Jacobian of left-hand side (theta terms, lines 3-5)
    for e in range(n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # d(rho_e)/d(theta_f0) = +1
        # d(rho_e)/d(theta_f1) = -1
        rows_theta.extend([e, e])
        cols_theta.extend([f0, f1])
        data_theta.extend([1.0, -1.0])

    # Jacobian of right-hand side (lines 6-13)
    for f in range(n_faces):
        v0, v1, v2 = mesh.faces[f]

        c0 = 3 * f + 0
        c1 = 3 * f + 1
        c2 = 3 * f + 2

        he_01 = 3 * f + 0
        he_12 = 3 * f + 1
        he_20 = 3 * f + 2

        eta_01 = phi[he_01] + theta[f]
        eta_12 = eta_01 - (np.pi - alpha[c1])
        eta_20 = eta_12 - (np.pi - alpha[c2])

        # Edge v0-v1
        e = mesh.halfedge_to_edge[he_01]
        _add_jacobian_contribution(e, v0, v1, v2, c0, c1, c2, f,
                                    alpha, u, v, s, eta_01, mesh,
                                    rows_u, cols_u, data_u,
                                    rows_v, cols_v, data_v,
                                    rows_theta, cols_theta, data_theta)

        # Edge v1-v2
        e = mesh.halfedge_to_edge[he_12]
        _add_jacobian_contribution(e, v1, v2, v0, c1, c2, c0, f,
                                    alpha, u, v, s, eta_12, mesh,
                                    rows_u, cols_u, data_u,
                                    rows_v, cols_v, data_v,
                                    rows_theta, cols_theta, data_theta)

        # Edge v2-v0
        e = mesh.halfedge_to_edge[he_20]
        _add_jacobian_contribution(e, v2, v0, v1, c2, c0, c1, f,
                                    alpha, u, v, s, eta_20, mesh,
                                    rows_u, cols_u, data_u,
                                    rows_v, cols_v, data_v,
                                    rows_theta, cols_theta, data_theta)

    # Build sparse matrices for each block
    Ju = sparse.csr_matrix((data_u, (rows_u, cols_u)),
                           shape=(n_edges, n_vertices))
    Jv = sparse.csr_matrix((data_v, (rows_v, cols_v)),
                           shape=(n_edges, n_vertices))
    Jtheta = sparse.csr_matrix((data_theta, (rows_theta, cols_theta)),
                               shape=(n_edges, n_faces))

    # Concatenate: J = [Ju, Jv, Jtheta]
    J = sparse.hstack([Ju, Jv, Jtheta])
    return J.tocsr()


def _add_jacobian_contribution(
    e: int,
    i: int, j: int, k: int,
    ci: int, cj: int, ck: int,
    f: int,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    s: np.ndarray,
    eta: float,
    mesh: TriangleMesh,
    rows_u, cols_u, data_u,
    rows_v, cols_v, data_v,
    rows_theta, cols_theta, data_theta
):
    """
    Algorithm 8: Add Jacobian contribution from edge ij in triangle ijk.
    """
    # Cotan weight (line 1)
    w = 0.5 / np.tan(alpha[ck] + 1e-30)

    # Account for edge orientation (line 2-3)
    if i > j:
        w = -w

    # Jacobian w.r.t. u (lines 4-6)
    rows_u.extend([e, e])
    cols_u.extend([i, j])
    data_u.extend([-w, +w])

    # Jacobian w.r.t. v (lines 7-10)
    cot_cj = 1 / np.tan(alpha[cj] + 1e-30)
    cot_ci = 1 / np.tan(alpha[ci] + 1e-30)

    # d/dv[i]
    dvi = -w * s[ci] * (np.cos(2 * eta) + np.sin(2 * eta) * cot_cj)
    rows_v.append(e)
    cols_v.append(i)
    data_v.append(dvi)

    # d/dv[j]
    dvj = +w * s[cj] * (np.cos(2 * eta) - np.sin(2 * eta) * cot_ci)
    rows_v.append(e)
    cols_v.append(j)
    data_v.append(dvj)

    # d/dv[k]
    dvk = +w * s[ck] * np.sin(2 * eta) * (cot_cj + cot_ci)
    rows_v.append(e)
    cols_v.append(k)
    data_v.append(dvk)

    # Jacobian w.r.t. theta (lines 11-15)
    v_ci = s[ci] * v[i]
    v_cj = s[cj] * v[j]
    v_ck = s[ck] * v[k]

    dtheta = -2 * w * np.sin(2 * eta) * (v_cj - v_ci)
    dtheta += 2 * w * np.cos(2 * eta) * cot_cj * (v_ck - v_ci)
    dtheta += 2 * w * np.cos(2 * eta) * cot_ci * (v_ck - v_cj)

    rows_theta.append(e)
    cols_theta.append(f)
    data_theta.append(dtheta)


def build_hessian(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lambda_: np.ndarray,
    s: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray
) -> sparse.csr_matrix:
    """
    Algorithm 9-10: BuildHessian

    Build Hessian of Lagrangian (constraint Hessian weighted by multipliers).

    Returns:
        D: (2|V| + |F|) x (2|V| + |F|) sparse matrix
    """
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces

    # The Hessian has structure:
    # D = [[0, 0, 0],
    #      [0, 0, Hv^T],
    #      [0, Hv, Htheta]]

    rows_v = []
    cols_v = []
    data_v = []

    rows_theta = []
    cols_theta = []
    data_theta = []

    for f in range(n_faces):
        v0, v1, v2 = mesh.faces[f]

        c0 = 3 * f + 0
        c1 = 3 * f + 1
        c2 = 3 * f + 2

        he_01 = 3 * f + 0
        he_12 = 3 * f + 1
        he_20 = 3 * f + 2

        eta_01 = phi[he_01] + theta[f]
        eta_12 = eta_01 - (np.pi - alpha[c1])
        eta_20 = eta_12 - (np.pi - alpha[c2])

        # Edge v0-v1
        e = mesh.halfedge_to_edge[he_01]
        _add_hessian_contribution(e, v0, v1, v2, c0, c1, c2, f,
                                   alpha, u, v, lambda_, s, eta_01, mesh,
                                   rows_v, cols_v, data_v,
                                   rows_theta, cols_theta, data_theta)

        # Edge v1-v2
        e = mesh.halfedge_to_edge[he_12]
        _add_hessian_contribution(e, v1, v2, v0, c1, c2, c0, f,
                                   alpha, u, v, lambda_, s, eta_12, mesh,
                                   rows_v, cols_v, data_v,
                                   rows_theta, cols_theta, data_theta)

        # Edge v2-v0
        e = mesh.halfedge_to_edge[he_20]
        _add_hessian_contribution(e, v2, v0, v1, c2, c0, c1, f,
                                   alpha, u, v, lambda_, s, eta_20, mesh,
                                   rows_v, cols_v, data_v,
                                   rows_theta, cols_theta, data_theta)

    # Build sparse blocks
    # Hv: |V| x |F| (derivative of Jacobian w.r.t. v, differentiated by theta)
    Hv = sparse.csr_matrix((data_v, (rows_v, cols_v)),
                           shape=(n_vertices, n_faces))

    # Htheta: |F| x |F| (second derivative w.r.t. theta)
    Htheta = sparse.csr_matrix((data_theta, (rows_theta, cols_theta)),
                               shape=(n_faces, n_faces))

    # Assemble full Hessian
    # D has blocks: [[0, 0, 0], [0, 0, Hv], [0, Hv^T, Htheta]]
    # where Hv is |V|x|F| (mixed derivative d^2/dv dtheta)
    Z_VV = sparse.csr_matrix((n_vertices, n_vertices))
    Z_VF = sparse.csr_matrix((n_vertices, n_faces))

    D = sparse.bmat([
        [Z_VV, Z_VV, Z_VF],
        [Z_VV, Z_VV, Hv],        # (v, theta) block: |V| x |F|
        [Z_VF.T, Hv.T, Htheta]   # (theta, v) block: |F| x |V|, (theta, theta): |F| x |F|
    ])

    return D.tocsr()


def _add_hessian_contribution(
    e: int,
    i: int, j: int, k: int,
    ci: int, cj: int, ck: int,
    f: int,
    alpha: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lambda_: np.ndarray,
    s: np.ndarray,
    eta: float,
    mesh: TriangleMesh,
    rows_v, cols_v, data_v,
    rows_theta, cols_theta, data_theta
):
    """
    Algorithm 10: Add Hessian contribution from edge ij in triangle ijk.
    """
    w = 0.5 / np.tan(alpha[ck] + 1e-30)
    if i > j:
        w = -w

    lam = lambda_[e]
    cot_cj = 1 / np.tan(alpha[cj] + 1e-30)
    cot_ci = 1 / np.tan(alpha[ci] + 1e-30)

    # Hessian w.r.t. v and theta (lines 5-7)
    # d^2/dv[i]dtheta
    dvi_dtheta = 2 * s[ci] * lam * w * (np.sin(2 * eta) - np.cos(2 * eta) * cot_cj)
    rows_v.append(i)
    cols_v.append(f)
    data_v.append(dvi_dtheta)

    # d^2/dv[j]dtheta
    dvj_dtheta = 2 * s[cj] * lam * w * (-np.sin(2 * eta) - np.cos(2 * eta) * cot_ci)
    rows_v.append(j)
    cols_v.append(f)
    data_v.append(dvj_dtheta)

    # d^2/dv[k]dtheta
    dvk_dtheta = 2 * s[ck] * lam * w * np.cos(2 * eta) * (cot_cj + cot_ci)
    rows_v.append(k)
    cols_v.append(f)
    data_v.append(dvk_dtheta)

    # Hessian w.r.t. theta, theta (lines 9-12)
    v_ci = s[ci] * v[i]
    v_cj = s[cj] * v[j]
    v_ck = s[ck] * v[k]

    d2theta = -4 * lam * w * np.cos(2 * eta) * (v_cj - v_ci)
    d2theta -= 4 * lam * w * np.sin(2 * eta) * cot_cj * (v_ck - v_ci)
    d2theta -= 4 * lam * w * np.sin(2 * eta) * cot_ci * (v_ck - v_cj)

    rows_theta.append(f)
    cols_theta.append(f)
    data_theta.append(d2theta)
