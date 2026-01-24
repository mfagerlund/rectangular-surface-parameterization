

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Tuple


# function [k21,Reduction] = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

def reduction_from_ff2d(
    Src,
    param,
    ang: np.ndarray,
    omega: np.ndarray,
    Edge_jump: sp.spmatrix,
    v2t: sp.spmatrix
) -> Tuple[np.ndarray, sp.csr_matrix]:
    """
    Build reduction matrix from cross field for 2D parameterization.

    Computes the jump indices k21 per edge from the cross field and builds
    a reduction matrix that incorporates sign flips based on accumulated jumps.

    Parameters
    ----------
    Src : mesh data structure
        Contains ne, nf, nv, T
    param : parameter structure
        Contains E2T, ide_int, para_trans
    ang : ndarray (nf,)
        Frame angle per face
    omega : ndarray (ne,)
        Field rotation per edge
    Edge_jump : sparse matrix
        Edge jump accumulation matrix from reduce_corner_var_2d[_cut]
    v2t : sparse matrix
        Vertex to corner mapping matrix

    Returns
    -------
    k21 : ndarray (ne,)
        Jump indices per edge (values 1-4)
    Reduction : sparse matrix
        Block diagonal reduction matrix combining ut and vt reductions
    """
    # k21 = ones(Src.ne,1);
    # [~,k21i] = min(abs(exp(1i*ang(param.E2T(param.ide_int,2))+(0:3)*1i*pi/2+1i*(omega(param.ide_int)-param.para_trans(param.ide_int))) - exp(1i*ang(param.E2T(param.ide_int,1)))), [], 2);
    # k21(param.ide_int) = k21i;

    k21 = np.ones(Src.ne, dtype=int)

    # Get face indices for interior edges (E2T is 0-indexed in Python)
    # E2T[e, 0] = face on one side, E2T[e, 1] = face on other side
    face1 = param.E2T[param.ide_int, 0]
    face2 = param.E2T[param.ide_int, 1]

    # Compute angle difference
    ang_diff = omega[param.ide_int] - param.para_trans[param.ide_int]

    # For each interior edge, find which rotation (0, 1, 2, 3) * pi/2 best aligns
    # the cross field vectors across the edge
    # MATLAB: exp(1i*ang2 + k*1i*pi/2 + 1i*ang_diff) should match exp(1i*ang1)
    # We find k that minimizes |exp(1i*ang2 + k*1i*pi/2 + 1i*ang_diff) - exp(1i*ang1)|

    ang1 = ang[face1]  # Angle on face 1
    ang2 = ang[face2]  # Angle on face 2

    # Create rotation options: k = 0, 1, 2, 3
    k_options = np.array([0, 1, 2, 3])

    # Compute |exp(i*ang2 + i*k*pi/2 + i*ang_diff) - exp(i*ang1)| for each k
    # Shape: (n_interior_edges, 4)
    rotated = np.exp(1j * ang2[:, np.newaxis] +
                     1j * k_options * np.pi / 2 +
                     1j * ang_diff[:, np.newaxis])
    target = np.exp(1j * ang1)[:, np.newaxis]
    diffs = np.abs(rotated - target)

    # Find k that gives minimum difference (MATLAB min returns 1-indexed, so k21i is 1-4)
    k21i = np.argmin(diffs, axis=1) + 1  # Convert to 1-based like MATLAB

    k21[param.ide_int] = k21i

    # k21T = mod(reshape(Edge_jump*(k21-1), Src.nf,[]), 4);

    # Edge_jump is (3*nf, ne), k21-1 is (ne,)
    # Result is (3*nf,) which we reshape to (nf, 3)
    k21T_flat = Edge_jump @ (k21 - 1)
    k21T = np.mod(k21T_flat.reshape((Src.nf, 3), order='F'), 4)

    # s = (-1).^k21T;

    s = np.power(-1.0, k21T)

    # v2t_smooth = sparse(reshape((1:3*Src.nf)', [Src.nf,3]), Src.T, 1, 3*Src.nf, Src.nv);
    # Reduction = blkdiag(v2t_smooth, spdiags(s(:), 0, 3*Src.nf, 3*Src.nf)*v2t);

    # v2t_smooth: maps vertex values to corner values without jumps
    # Row indices: corner indices (0 to 3*nf-1)
    # Col indices: vertex indices from T
    row_idx = np.arange(3 * Src.nf).reshape((Src.nf, 3), order='F')
    v2t_smooth = sp.coo_matrix(
        (np.ones(3 * Src.nf), (row_idx.ravel('F'), Src.T.ravel('F'))),
        shape=(3 * Src.nf, Src.nv)
    ).tocsr()

    # Sign diagonal matrix
    s_diag = sp.diags(s.ravel('F'), 0, shape=(3 * Src.nf, 3 * Src.nf), format='csr')

    # Build block diagonal reduction matrix
    # First block: v2t_smooth for ut (no sign flips)
    # Second block: s_diag * v2t for vt (with sign flips)
    Reduction = sp.block_diag([v2t_smooth, s_diag @ v2t], format='csr')

    return k21, Reduction
