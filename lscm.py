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
    Compute LSCM parameterization using eigenvalue approach.

    Returns:
        corner_uvs: |C| x 2 UV coordinates per corner
    """
    n_faces = mesh.n_faces
    n_vertices = mesh.n_vertices

    # Check if mesh is closed (no boundary)
    n_boundary = sum(1 for e in range(mesh.n_edges)
                     if mesh.edge_to_halfedge[e, 0] == -1 or mesh.edge_to_halfedge[e, 1] == -1)

    # For closed meshes, we need to "cut" by excluding one face
    excluded_face = -1
    if n_boundary == 0:
        excluded_face = 0

    active_faces = [f for f in range(n_faces) if f != excluded_face]
    n_active = len(active_faces)

    # Build the full LSCM matrix (all vertices)
    A_real_rows = []
    A_real_cols = []
    A_real_data = []
    A_imag_rows = []
    A_imag_cols = []
    A_imag_data = []

    for row_idx, f in enumerate(active_faces):
        v0, v1, v2 = mesh.faces[f]
        p0, p1, p2 = mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]

        e01 = p1 - p0
        e02 = p2 - p0

        len01 = np.linalg.norm(e01)
        if len01 < 1e-10:
            continue
        x_dir = e01 / len01

        normal = np.cross(e01, e02)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-10:
            continue
        normal = normal / normal_len
        y_dir = np.cross(normal, x_dir)

        x2, y2 = np.dot(e02, x_dir), np.dot(e02, y_dir)

        # Coefficients for c0*z0 + c1*z1 + c2*z2 = 0
        coeffs = [
            (len01 - x2, y2),   # c0
            (x2, -y2),          # c1
            (-len01, 0.0)       # c2
        ]
        verts = [v0, v1, v2]

        for k, vtx in enumerate(verts):
            c_re, c_im = coeffs[k]

            # Real: c_re * u - c_im * v
            A_real_rows.extend([row_idx, row_idx])
            A_real_cols.extend([vtx, n_vertices + vtx])
            A_real_data.extend([c_re, -c_im])

            # Imag: c_im * u + c_re * v
            A_imag_rows.extend([row_idx, row_idx])
            A_imag_cols.extend([vtx, n_vertices + vtx])
            A_imag_data.extend([c_im, c_re])

    A_real = sparse.csr_matrix((A_real_data, (A_real_rows, A_real_cols)),
                                shape=(n_active, 2 * n_vertices))
    A_imag = sparse.csr_matrix((A_imag_data, (A_imag_rows, A_imag_cols)),
                                shape=(n_active, 2 * n_vertices))
    A = sparse.vstack([A_real, A_imag])

    # Solve using eigenvalue decomposition
    # The null space of A^T A gives the conformal maps
    # We want the smallest non-trivial eigenvectors
    ATA = A.T @ A

    # Find two vertices far apart for pinning (to fix scale/translation)
    if n_vertices > 1000:
        sample = np.random.choice(n_vertices, min(100, n_vertices), replace=False)
        dists = np.linalg.norm(mesh.positions[sample, np.newaxis, :] -
                               mesh.positions[sample][np.newaxis, :, :], axis=2)
        i_s, j_s = np.unravel_index(np.argmax(dists), dists.shape)
        pin1, pin2 = sample[i_s], sample[j_s]
    else:
        dists = np.linalg.norm(mesh.positions[:, np.newaxis, :] -
                               mesh.positions[np.newaxis, :, :], axis=2)
        pin1, pin2 = np.unravel_index(np.argmax(dists), dists.shape)

    if pin1 == pin2:
        pin2 = (pin1 + 1) % n_vertices

    # Add soft constraints to fix translation and scale
    # pin1 at (0, 0), pin2 at (1, 0)
    weight = 1e6  # Large weight for soft constraints

    # Add constraint rows
    C_rows = [0, 1, 2, 3]
    C_cols = [pin1, n_vertices + pin1, pin2, n_vertices + pin2]
    C_data = [weight, weight, weight, weight]
    C = sparse.csr_matrix((C_data, (C_rows, C_cols)), shape=(4, 2 * n_vertices))
    c_rhs = np.array([0.0, 0.0, weight, 0.0])

    # Use eigenvalue decomposition to find the non-trivial conformal map
    # This is more robust than soft constraints for closed meshes
    try:
        eigenvalues, eigenvectors = eigsh(ATA + 1e-10 * sparse.eye(ATA.shape[0]),
                                           k=6, which='SM')

        # First 2 eigenvectors are constant (translation null space)
        # Eigenvectors 2,3 give the conformal map coordinates
        # We need to find ones where BOTH u and v have non-zero variation
        u = None
        v = None

        for i in range(2, len(eigenvalues)):
            ev = eigenvectors[:, i]
            u_ev = ev[:n_vertices]
            v_ev = ev[n_vertices:]

            if np.std(u_ev) > 0.01 and np.std(v_ev) > 0.01:
                u = u_ev.copy()
                v = v_ev.copy()
                break

        if u is None:
            # Fallback: use eigenvectors 2 for u and 3 for v
            u = eigenvectors[:n_vertices, 2]
            v = eigenvectors[n_vertices:, 3]

        # Apply rigid transformation to match pinning constraints
        # pin1 -> (0, 0), pin2 -> (1, 0)
        # First translate so pin1 is at origin
        u = u - u[pin1]
        v = v - v[pin1]

        # Then rotate and scale so pin2 is at (1, 0)
        dx = u[pin2]
        dy = v[pin2]
        dist = np.sqrt(dx*dx + dy*dy)

        if dist > 1e-10:
            # Rotation angle to align pin2 with x-axis
            cos_theta = dx / dist
            sin_theta = dy / dist

            # Rotate all points
            u_new = cos_theta * u + sin_theta * v
            v_new = -sin_theta * u + cos_theta * v

            # Scale so pin2 is at distance 1
            u = u_new / dist
            v = v_new / dist

    except Exception as e:
        # Fallback to soft constraint solution
        CTC = C.T @ C
        CTc = C.T @ c_rhs
        M = ATA + CTC
        M_reg = M + 1e-8 * sparse.eye(M.shape[0])
        x = spsolve(M_reg.tocsc(), CTc)
        u = x[:n_vertices]
        v = x[n_vertices:]

    # Convert to corner UVs
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
