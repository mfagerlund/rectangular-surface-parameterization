"""
UV coordinate recovery (Algorithm 11 from supplement).

Recovers final parameterization coordinates from the optimization solution.

IMPORTANT: This implementation follows Algorithm 11's corner indexing convention:
- "Corner ij^k" means the corner OPPOSITE edge ij (i.e., at vertex k)
- Each edge ij in face (i,j,k) contributes one row to the system
- The row is indexed by the opposite corner (vertex k in that face)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, lsqr
from typing import Tuple

from mesh import TriangleMesh
from geometry import compute_corner_angles, compute_edge_lengths


def recover_parameterization(
    mesh: TriangleMesh,
    Gamma: np.ndarray,
    zeta: np.ndarray,
    ell: np.ndarray,
    alpha: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    u: np.ndarray,
    v: np.ndarray
) -> np.ndarray:
    """
    Algorithm 11: RecoverParameterization

    Recover UV coordinates from optimization solution using Poisson solve.

    Args:
        mesh: Triangle mesh
        Gamma: |E| cut edge indicator
        zeta: |E| quarter-rotation jump
        ell: |E| edge lengths
        alpha: |C| corner angles
        phi: |H| reference frame angles
        theta: |F| frame angles
        u, v: |V| log scale factors

    Returns:
        f: |C| x 2 UV coordinates per corner
    """
    n_corners = mesh.n_corners
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges
    n_halfedges = mesh.n_halfedges

    # Step 1-3: Compute edge scales from log-scales
    # a_ij = exp((u_i + u_j + v_i + v_j)/2)
    # b_ij = exp((u_i + u_j - v_i - v_j)/2)
    a_edge = np.zeros(n_edges)
    b_edge = np.zeros(n_edges)

    for e in range(n_edges):
        i, j = mesh.edge_vertices[e]
        a_edge[e] = np.exp((u[i] + u[j] + v[i] + v[j]) / 2)
        b_edge[e] = np.exp((u[i] + u[j] - v[i] - v[j]) / 2)

    # Step 4-13: Compute target edge vectors μ per halfedge
    # μ is indexed by halfedge (same as corner index)
    mu = np.zeros((n_halfedges, 2))

    for f in range(n_faces):
        v0, v1, v2 = mesh.faces[f]

        # Corner indices (corner c_k is at vertex v_k)
        c0 = 3 * f + 0  # corner at v0
        c1 = 3 * f + 1  # corner at v1
        c2 = 3 * f + 2  # corner at v2

        # Halfedge indices
        he_01 = 3 * f + 0  # v0 -> v1 (opposite corner c2)
        he_12 = 3 * f + 1  # v1 -> v2 (opposite corner c0)
        he_20 = 3 * f + 2  # v2 -> v0 (opposite corner c1)

        # Frame angle relative to first edge (line 5)
        eta = phi[he_01] + theta[f]

        # Get edge indices
        e01 = mesh.halfedge_to_edge[he_01]
        e12 = mesh.halfedge_to_edge[he_12]
        e20 = mesh.halfedge_to_edge[he_20]

        # Target edge vectors (lines 9-11)
        # μ^k_ij = ℓ_ij * (a_ij * cos(η), b_ij * sin(η))

        # Edge 0->1: η_01 = η
        mu[he_01, 0] = ell[e01] * a_edge[e01] * np.cos(eta)
        mu[he_01, 1] = ell[e01] * b_edge[e01] * np.sin(eta)

        # Edge 1->2: η_12 = η - (π - α_1)
        eta_12 = eta - (np.pi - alpha[c1])
        mu[he_12, 0] = ell[e12] * a_edge[e12] * np.cos(eta_12)
        mu[he_12, 1] = ell[e12] * b_edge[e12] * np.sin(eta_12)

        # Edge 2->0: η_20 = η_12 - (π - α_2)
        eta_20 = eta_12 - (np.pi - alpha[c2])
        mu[he_20, 0] = ell[e20] * a_edge[e20] * np.cos(eta_20)
        mu[he_20, 1] = ell[e20] * b_edge[e20] * np.sin(eta_20)

    # Step 12-13: Build right-hand side b
    # Subtraction formula: b = 0.5 * (R_zeta * mu[he] - mu[he_twin])
    # mu[he] and mu[he_twin] point in opposite directions for the same edge,
    # so subtracting gives a consistent edge vector direction
    b_rhs = np.zeros((n_halfedges, 2))

    for he in range(n_halfedges):
        he_twin = mesh.halfedge_twin[he]
        e = mesh.halfedge_to_edge[he]

        if he_twin == -1:
            # Boundary halfedge - just use μ
            b_rhs[he] = mu[he]
        else:
            # Interior edge - rotate current mu by zeta, subtract twin
            z = zeta[e]
            cos_z, sin_z = np.cos(z), np.sin(z)

            mu_rot = np.array([
                cos_z * mu[he, 0] - sin_z * mu[he, 1],
                sin_z * mu[he, 0] + cos_z * mu[he, 1]
            ])

            b_rhs[he] = 0.5 * (mu_rot - mu[he_twin])

    # Build the linear system following Algorithm 11's indexing
    # For each halfedge he (edge i->j in face with opposite corner k):
    # - Row is indexed by the OPPOSITE corner (corner k)
    # - A[row, corner_j] = +1, A[row, corner_i] = -1
    # - We want: f[corner_j] - f[corner_i] = b[he]

    # Build matrix A: each halfedge contributes one row
    # Row index = halfedge index (which equals corner index in our convention)
    # But Algorithm 11 wants row = opposite corner

    # Create mapping: halfedge -> opposite corner
    he_to_opposite_corner = np.zeros(n_halfedges, dtype=int)
    for f in range(n_faces):
        # he_01 (v0->v1) has opposite corner c2
        he_to_opposite_corner[3*f + 0] = 3*f + 2
        # he_12 (v1->v2) has opposite corner c0
        he_to_opposite_corner[3*f + 1] = 3*f + 0
        # he_20 (v2->v0) has opposite corner c1
        he_to_opposite_corner[3*f + 2] = 3*f + 1

    # Build A matrix with Algorithm 11's convention
    A_rows = []
    A_cols = []
    A_data = []

    # Also build weight matrix W using opposite corner angles
    W_diag = np.zeros(n_halfedges)

    for f in range(n_faces):
        v0, v1, v2 = mesh.faces[f]
        c0, c1, c2 = 3*f, 3*f+1, 3*f+2

        # Halfedge he_01: v0 -> v1, opposite corner c2
        # mu points i->j, so we want f[j] - f[i] = b
        # f[c1] - f[c0] = b[he_01]
        row = c2
        A_rows.extend([row, row])
        A_cols.extend([c0, c1])
        A_data.extend([-1.0, 1.0])  # -1 at source, +1 at target
        W_diag[3*f + 0] = 0.5 / np.tan(alpha[c2] + 1e-10)

        # Halfedge he_12: v1 -> v2, opposite corner c0
        row = c0
        A_rows.extend([row, row])
        A_cols.extend([c1, c2])
        A_data.extend([-1.0, 1.0])
        W_diag[3*f + 1] = 0.5 / np.tan(alpha[c0] + 1e-10)

        # Halfedge he_20: v2 -> v0, opposite corner c1
        row = c1
        A_rows.extend([row, row])
        A_cols.extend([c2, c0])
        A_data.extend([-1.0, 1.0])
        W_diag[3*f + 2] = 0.5 / np.tan(alpha[c1] + 1e-10)

    # Clamp weights to avoid extreme values
    W_diag = np.clip(W_diag, 0.01, 100)
    W = sparse.diags(W_diag)

    # Reindex b_rhs to match row ordering (opposite corner)
    b_reindexed = np.zeros((n_corners, 2))
    for he in range(n_halfedges):
        opp_corner = he_to_opposite_corner[he]
        b_reindexed[opp_corner] = b_rhs[he]

    # Build A as sparse matrix
    A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(n_corners, n_corners))

    # Build corner identification constraints U
    # Corners should match across non-cut edges
    U_rows = []
    U_cols = []
    U_data = []
    constraint_idx = 0

    for e in range(n_edges):
        if Gamma[e] == 1:  # Cut edge - don't identify
            continue

        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        # Get faces and local indices
        f0, local0 = he0 // 3, he0 % 3
        f1, local1 = he1 // 3, he1 % 3

        # he0 goes from vertex at local0 to vertex at (local0+1)%3 in face f0
        # he1 goes from vertex at local1 to vertex at (local1+1)%3 in face f1
        # These are the same edge, so:
        # - corner at start of he0 should match corner at end of he1
        # - corner at end of he0 should match corner at start of he1

        c0_start = 3 * f0 + local0
        c0_end = 3 * f0 + (local0 + 1) % 3
        c1_start = 3 * f1 + local1
        c1_end = 3 * f1 + (local1 + 1) % 3

        # c0_start (vertex at start of he0) = c1_end (vertex at end of he1)
        U_rows.extend([constraint_idx, constraint_idx])
        U_cols.extend([c0_start, c1_end])
        U_data.extend([1.0, -1.0])
        constraint_idx += 1

        # c0_end (vertex at end of he0) = c1_start (vertex at start of he1)
        U_rows.extend([constraint_idx, constraint_idx])
        U_cols.extend([c0_end, c1_start])
        U_data.extend([1.0, -1.0])
        constraint_idx += 1

    # Solve for each coordinate
    f = np.zeros((n_corners, 2))

    for coord in range(2):
        b = b_reindexed[:, coord]

        if constraint_idx > 0:
            U = sparse.csr_matrix((U_data, (U_rows, U_cols)), shape=(constraint_idx, n_corners))

            # Solve: min ||A f - b||^2_W  s.t. U f = 0
            # KKT: [A^T W A, U^T; U, 0] [f; λ] = [A^T W b; 0]
            ATWA = A.T @ W @ A
            ATWb = A.T @ W @ b

            K = sparse.bmat([
                [ATWA, U.T],
                [U, sparse.csr_matrix((constraint_idx, constraint_idx))]
            ])

            rhs = np.concatenate([ATWb, np.zeros(constraint_idx)])

            # Regularize and solve
            K_reg = K + 1e-8 * sparse.eye(K.shape[0])
            sol = spsolve(K_reg.tocsc(), rhs)
            f[:, coord] = sol[:n_corners]
        else:
            # No constraints - simple least squares
            ATWA = A.T @ W @ A
            ATWb = A.T @ W @ b
            ATWA_reg = ATWA + 1e-8 * sparse.eye(n_corners)
            f[:, coord] = spsolve(ATWA_reg.tocsc(), ATWb)

    # Flip y-coordinate to fix global orientation
    # The phi angles in cut_graph.py produce opposite winding, so we flip y
    f[:, 1] = -f[:, 1]

    return f


def compute_uv_quality(mesh: TriangleMesh, corner_uvs: np.ndarray) -> dict:
    """
    Compute quality metrics for the parameterization.
    """
    from geometry import compute_face_areas

    areas_3d = compute_face_areas(mesh)
    n_faces = mesh.n_faces

    areas_uv = np.zeros(n_faces)
    flipped = 0
    angle_errors = []

    for f in range(n_faces):
        c0, c1, c2 = 3*f, 3*f+1, 3*f+2
        uv0, uv1, uv2 = corner_uvs[c0], corner_uvs[c1], corner_uvs[c2]

        # UV area (signed)
        e1 = uv1 - uv0
        e2 = uv2 - uv0
        area_uv_signed = 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])

        if area_uv_signed < 0:
            flipped += 1

        areas_uv[f] = abs(area_uv_signed)

        # Angle error
        for local in range(3):
            ca = 3*f + local
            cb = 3*f + (local + 1) % 3
            cc = 3*f + (local + 2) % 3

            uv_a, uv_b, uv_c = corner_uvs[ca], corner_uvs[cb], corner_uvs[cc]
            e1 = uv_b - uv_a
            e2 = uv_c - uv_a

            len1, len2 = np.linalg.norm(e1), np.linalg.norm(e2)
            if len1 > 1e-10 and len2 > 1e-10:
                cos_angle = np.clip(np.dot(e1, e2) / (len1 * len2), -1, 1)
                angle_uv = np.arccos(cos_angle)
            else:
                angle_uv = 0

            # 3D angle
            va = mesh.faces[f, local]
            vb = mesh.faces[f, (local + 1) % 3]
            vc = mesh.faces[f, (local + 2) % 3]

            e1_3d = mesh.positions[vb] - mesh.positions[va]
            e2_3d = mesh.positions[vc] - mesh.positions[va]

            len1_3d, len2_3d = np.linalg.norm(e1_3d), np.linalg.norm(e2_3d)
            if len1_3d > 1e-10 and len2_3d > 1e-10:
                cos_3d = np.clip(np.dot(e1_3d, e2_3d) / (len1_3d * len2_3d), -1, 1)
                angle_3d = np.arccos(cos_3d)
            else:
                angle_3d = 0

            angle_errors.append(abs(angle_uv - angle_3d))

    total_area_3d = np.sum(areas_3d)
    total_area_uv = np.sum(areas_uv)

    return {
        'flipped_count': flipped,
        'flipped_fraction': flipped / n_faces if n_faces > 0 else 0,
        'angle_error_mean': np.mean(angle_errors) if angle_errors else 0,
        'angle_error_max': np.max(angle_errors) if angle_errors else 0,
        'total_area_3d': total_area_3d,
        'total_area_uv': total_area_uv
    }


def normalize_uvs(corner_uvs: np.ndarray) -> np.ndarray:
    """Normalize UV coordinates to [0, 1] range."""
    uvs = corner_uvs.copy()
    uvs -= uvs.min(axis=0)
    max_range = uvs.max()
    if max_range > 1e-10:
        uvs /= max_range
    return uvs
