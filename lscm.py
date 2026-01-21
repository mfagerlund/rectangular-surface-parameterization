"""
Least Squares Conformal Mapping (LSCM) - baseline parameterization.

This is a simpler alternative to verify the mesh/UV pipeline works.
LSCM minimizes angle distortion and always produces a valid (no flips) UV layout
for disk-topology meshes.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh
from typing import Tuple

from mesh import TriangleMesh
from geometry import compute_corner_angles


def lscm_parameterize(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute LSCM parameterization.

    Returns:
        corner_uvs: |C| x 2 UV coordinates per corner
    """
    n_faces = mesh.n_faces
    n_vertices = mesh.n_vertices

    # Build the LSCM matrix
    # For each triangle, we have a 2x6 matrix M_t such that
    # M_t @ [u0, v0, u1, v1, u2, v2]^T = conformal residual

    # Actually, let's use the simpler formulation:
    # Minimize sum_t || (z1-z0) * conj(p2-p0) - (z2-z0) * conj(p1-p0) ||^2
    # where z_i = u_i + i*v_i are UV coords and p_i are 3D positions projected to 2D

    # For each face, compute local 2D coordinates
    A_real_rows = []
    A_real_cols = []
    A_real_data = []
    A_imag_rows = []
    A_imag_cols = []
    A_imag_data = []

    for f in range(n_faces):
        v0, v1, v2 = mesh.faces[f]
        p0 = mesh.positions[v0]
        p1 = mesh.positions[v1]
        p2 = mesh.positions[v2]

        # Project triangle to 2D using local frame
        e01 = p1 - p0
        e02 = p2 - p0

        # Local x = direction of e01
        len01 = np.linalg.norm(e01)
        if len01 < 1e-10:
            continue
        x_dir = e01 / len01

        # Local y = perpendicular in triangle plane
        normal = np.cross(e01, e02)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-10:
            continue
        normal = normal / normal_len
        y_dir = np.cross(normal, x_dir)

        # Local 2D coordinates
        x0, y0 = 0.0, 0.0
        x1, y1 = len01, 0.0
        x2, y2 = np.dot(e02, x_dir), np.dot(e02, y_dir)

        # Complex numbers for local coords
        # w0 = x0 + i*y0 = 0
        # w1 = x1 + i*y1 = len01
        # w2 = x2 + i*y2

        # LSCM condition: (z1-z0)*conj(w2-w0) = (z2-z0)*conj(w1-w0)
        # => z1*conj(w2) - z0*conj(w2) = z2*conj(w1) - z0*conj(w1)
        # => z1*conj(w2) - z2*conj(w1) + z0*(conj(w1) - conj(w2)) = 0

        # conj(w1) = x1 - i*y1 = len01
        # conj(w2) = x2 - i*y2

        # Coefficient for z0: conj(w1) - conj(w2) = (x1 - x2) + i*(y2 - y1)
        c0_re = x1 - x2
        c0_im = y2 - y1

        # Coefficient for z1: conj(w2) = x2 - i*y2
        c1_re = x2
        c1_im = -y2

        # Coefficient for z2: -conj(w1) = -x1 + i*y1 = -len01
        c2_re = -x1
        c2_im = y1

        # Expand z_i = u_i + i*v_i:
        # c0*(u0 + i*v0) + c1*(u1 + i*v1) + c2*(u2 + i*v2) = 0
        # Real part: c0_re*u0 - c0_im*v0 + c1_re*u1 - c1_im*v1 + c2_re*u2 - c2_im*v2 = 0
        # Imag part: c0_im*u0 + c0_re*v0 + c1_im*u1 + c1_re*v1 + c2_im*u2 + c2_re*v2 = 0

        # Real equation
        A_real_rows.extend([f, f, f, f, f, f])
        A_real_cols.extend([v0, n_vertices + v0, v1, n_vertices + v1, v2, n_vertices + v2])
        A_real_data.extend([c0_re, -c0_im, c1_re, -c1_im, c2_re, -c2_im])

        # Imaginary equation
        A_imag_rows.extend([f, f, f, f, f, f])
        A_imag_cols.extend([v0, n_vertices + v0, v1, n_vertices + v1, v2, n_vertices + v2])
        A_imag_data.extend([c0_im, c0_re, c1_im, c1_re, c2_im, c2_re])

    # Combine into single matrix
    A_real = sparse.csr_matrix((A_real_data, (A_real_rows, A_real_cols)),
                                shape=(n_faces, 2 * n_vertices))
    A_imag = sparse.csr_matrix((A_imag_data, (A_imag_rows, A_imag_cols)),
                                shape=(n_faces, 2 * n_vertices))

    A = sparse.vstack([A_real, A_imag])

    # Solve min ||A x||^2 with constraints to fix scale and position
    # We pin two vertices to avoid trivial solution

    # Find two vertices that are far apart (for better conditioning)
    dists = np.linalg.norm(mesh.positions[:, np.newaxis, :] - mesh.positions[np.newaxis, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)

    # Pin vertex i at (0, 0) and vertex j at (1, 0)
    # This means: u[i] = 0, v[i] = 0, u[j] = 1, v[j] = 0

    # Build constraint matrix
    C_rows = [0, 1, 2, 3]
    C_cols = [i, n_vertices + i, j, n_vertices + j]
    C_data = [1.0, 1.0, 1.0, 1.0]
    C = sparse.csr_matrix((C_data, (C_rows, C_cols)), shape=(4, 2 * n_vertices))

    c_rhs = np.array([0.0, 0.0, 1.0, 0.0])

    # Solve: min ||Ax||^2 s.t. Cx = c_rhs
    # KKT: [A^T A, C^T; C, 0] [x; λ] = [0; c_rhs]
    ATA = A.T @ A
    K = sparse.bmat([
        [ATA, C.T],
        [C, sparse.csr_matrix((4, 4))]
    ])
    rhs = np.concatenate([np.zeros(2 * n_vertices), c_rhs])

    # Add small regularization
    K_reg = K + 1e-10 * sparse.eye(K.shape[0])

    sol = spsolve(K_reg.tocsc(), rhs)
    x = sol[:2 * n_vertices]

    u = x[:n_vertices]
    v = x[n_vertices:]

    # Convert vertex UVs to corner UVs
    corner_uvs = np.zeros((mesh.n_corners, 2))
    for f in range(n_faces):
        for local in range(3):
            c = 3 * f + local
            vertex = mesh.faces[f, local]
            corner_uvs[c, 0] = u[vertex]
            corner_uvs[c, 1] = v[vertex]

    return corner_uvs


def normalize_uvs(corner_uvs: np.ndarray) -> np.ndarray:
    """Normalize UVs to [0,1] range."""
    uvs = corner_uvs.copy()
    uvs -= uvs.min(axis=0)
    max_range = max(uvs[:, 0].max(), uvs[:, 1].max())
    if max_range > 1e-10:
        uvs /= max_range
    return uvs
