

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, Union

from Orthotropic.omega_from_scale import omega_from_scale


# function [F,Jf,Hf] = oracle_integrability_condition(mesh, param, dec, omega, ut, vt, ang, lambda, Reduction, ide_free)
#

def oracle_integrability_condition(
    mesh,
    param,
    dec,
    omega: np.ndarray,
    ut: np.ndarray,
    vt: np.ndarray,
    ang: np.ndarray,
    lam: np.ndarray,
    Reduction: sp.spmatrix,
    ide_free: Optional[np.ndarray] = None,
    compute_hessian: bool = False
) -> Union[Tuple[np.ndarray, sp.spmatrix], Tuple[np.ndarray, sp.spmatrix, sp.spmatrix]]:
    """
    Compute the integrability condition per edge and its derivative.

    Parameters
    ----------
    mesh : mesh data structure
        Contains nf, ne, T2E, cot_corner_angle
    param : parameter structure
        Contains ide_free, ide_hard, ide_bound, ang_basis
    dec : DEC structure
        Contains d0d
    omega : ndarray (ne,)
        Field rotation per edge
    ut : ndarray (nf, 3)
        Scale factor u at triangle corners
    vt : ndarray (nf, 3)
        Scale factor v at triangle corners
    ang : ndarray (nf,)
        Frame angle per face
    lam : ndarray
        Lagrange multipliers (named 'lam' to avoid Python keyword 'lambda')
    Reduction : sparse matrix
        Reduction matrix from omega_from_scale
    ide_free : ndarray, optional
        Free edge indices. Defaults to param.ide_free
    compute_hessian : bool
        Whether to compute the Hessian (3rd return value)

    Returns
    -------
    F : ndarray
        Integrability condition evaluated on edges defined by ide_free
    Jf : sparse matrix
        Jacobian of F wrt u, v, theta
    Hf : sparse matrix (only if compute_hessian=True)
        Second order derivative needed for Newton method
    """
    # if ~exist('ide_free','var')
    #     ide_free = param.ide_free;
    # end

    if ide_free is None:
        ide_free = param.ide_free

    # d0d = dec.d0d;
    # d0d(param.ide_hard,:) = 0;
    # d0d(param.ide_bound,:) = 0;

    # Copy d0d and zero out hard and boundary edge rows
    d0d = dec.d0d.copy().tolil()
    if hasattr(param, 'ide_hard') and len(param.ide_hard) > 0:
        for idx in param.ide_hard:
            d0d[idx, :] = 0
    if hasattr(param, 'ide_bound') and len(param.ide_bound) > 0:
        for idx in param.ide_bound:
            d0d[idx, :] = 0
    d0d = d0d.tocsr()

    # [O,Or,dO] = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction);

    O, Or, dO = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction, compute_derivative=True)

    # F = O(ide_free,:)*[ut(:); vt(:)] - omega(ide_free);

    # Flatten ut and vt in column-major order (MATLAB style)
    uv_vec = np.concatenate([np.ravel(ut, 'F'), np.ravel(vt, 'F')])
    F = O[ide_free, :] @ uv_vec - omega[ide_free]

    # Jf = [Or(ide_free,:), dO(ide_free,:) - d0d(ide_free,:)];

    Jf = sp.hstack([Or[ide_free, :], dO[ide_free, :] - d0d[ide_free, :]])

    # if nargout >= 3

    if not compute_hessian:
        return F, Jf

    #     cot_ang = mesh.cot_corner_angle;
    #     cos_2ff1 = cos(2*ang + 2*param.ang_basis(:,1));
    #     sin_2ff1 = sin(2*ang + 2*param.ang_basis(:,1));
    #     cos_2ff2 = cos(2*ang + 2*param.ang_basis(:,2));
    #     sin_2ff2 = sin(2*ang + 2*param.ang_basis(:,2));
    #     cos_2ff3 = cos(2*ang + 2*param.ang_basis(:,3));
    #     sin_2ff3 = sin(2*ang + 2*param.ang_basis(:,3));

    cot_ang = mesh.cot_corner_angle
    cos_2ff1 = np.cos(2 * ang + 2 * param.ang_basis[:, 0])
    sin_2ff1 = np.sin(2 * ang + 2 * param.ang_basis[:, 0])
    cos_2ff2 = np.cos(2 * ang + 2 * param.ang_basis[:, 1])
    sin_2ff2 = np.sin(2 * ang + 2 * param.ang_basis[:, 1])
    cos_2ff3 = np.cos(2 * ang + 2 * param.ang_basis[:, 2])
    sin_2ff3 = np.sin(2 * ang + 2 * param.ang_basis[:, 2])

    #     vt1  = vt(:,1);
    #     vt2  = vt(:,2);
    #     vt3  = vt(:,3);

    vt1 = vt[:, 0]
    vt2 = vt[:, 1]
    vt3 = vt[:, 2]

    #     lambda_full = zeros(mesh.num_edges,1);
    #     lambda_full(ide_free) = lambda(1:length(ide_free));
    #     le = sign(mesh.T2E).*lambda_full(abs(mesh.T2E));

    lambda_full = np.zeros(mesh.num_edges)
    lambda_full[ide_free] = lam[:len(ide_free)]

    # T2E uses signed encoding: decode with abs(T2E)-1 for index, sign(T2E) for sign
    T2E_idx = np.abs(mesh.T2E) - 1  # Convert to 0-based edge indices
    T2E_sign = np.sign(mesh.T2E)
    le = T2E_sign * lambda_full[T2E_idx]

    #     dthS = 2*le(:,1).*cot_ang(:,3).*(-sin_2ff1.*(cot_ang(:,2).*(vt3 - vt1) + cot_ang(:,1).*(vt3 - vt2)) - cos_2ff1.*(vt2 - vt1)) + ...
    #            2*le(:,2).*cot_ang(:,1).*(-sin_2ff2.*(cot_ang(:,3).*(vt1 - vt2) + cot_ang(:,2).*(vt1 - vt3)) - cos_2ff2.*(vt3 - vt2)) + ...
    #            2*le(:,3).*cot_ang(:,2).*(-sin_2ff3.*(cot_ang(:,1).*(vt2 - vt3) + cot_ang(:,3).*(vt2 - vt1)) - cos_2ff3.*(vt1 - vt3));
    #     D2_th = spdiags(dthS, 0, mesh.num_faces, mesh.num_faces);

    dthS = (2 * le[:, 0] * cot_ang[:, 2] *
            (-sin_2ff1 * (cot_ang[:, 1] * (vt3 - vt1) + cot_ang[:, 0] * (vt3 - vt2)) - cos_2ff1 * (vt2 - vt1)) +
            2 * le[:, 1] * cot_ang[:, 0] *
            (-sin_2ff2 * (cot_ang[:, 2] * (vt1 - vt2) + cot_ang[:, 1] * (vt1 - vt3)) - cos_2ff2 * (vt3 - vt2)) +
            2 * le[:, 2] * cot_ang[:, 1] *
            (-sin_2ff3 * (cot_ang[:, 0] * (vt2 - vt3) + cot_ang[:, 2] * (vt2 - vt1)) - cos_2ff3 * (vt1 - vt3)))
    D2_th = sp.diags(dthS, 0, shape=(mesh.num_faces, mesh.num_faces), format='csr')

    #     I = repmat((1:mesh.num_faces)', [1,3]);
    #     J = reshape((1:3*mesh.num_faces)', [mesh.num_faces,3]);
    #     dvS1 = le(:,1).*cot_ang(:,3).*(-cos_2ff1.*cot_ang(:,2) + sin_2ff1) + ...
    #            le(:,2).*cot_ang(:,1).*cos_2ff2.*(cot_ang(:,3) + cot_ang(:,2)) + ...
    #            le(:,3).*cot_ang(:,2).*(-cos_2ff3.*cot_ang(:,3) - sin_2ff3);
    #     dvS2 = le(:,1).*cot_ang(:,3).*(-cos_2ff1.*cot_ang(:,1) - sin_2ff1) + ...
    #            le(:,2).*cot_ang(:,1).*(-cos_2ff2.*cot_ang(:,3) + sin_2ff2) + ...
    #            le(:,3).*cot_ang(:,2).*cos_2ff3.*(cot_ang(:,1) + cot_ang(:,3));
    #     dvS3 = le(:,1).*cot_ang(:,3).*cos_2ff1.*(cot_ang(:,2) + cot_ang(:,1)) + ...
    #            le(:,2).*cot_ang(:,1).*(-cos_2ff2.*cot_ang(:,2) - sin_2ff2) + ...
    #            le(:,3).*cot_ang(:,2).*(-cos_2ff3.*cot_ang(:,1) + sin_2ff3);
    #     D_vth = sparse(I, J, [dvS1, dvS2, dvS3], mesh.num_faces, 3*mesh.num_faces);

    # Build row/column indices for sparse matrix
    # I: nf x 3 matrix where each row is [i, i, i] (0-indexed face indices)
    # J: nf x 3 matrix where each row is [i, nf+i, 2*nf+i] (0-indexed corner indices)
    I_rows = np.tile(np.arange(mesh.num_faces).reshape(-1, 1), (1, 3))
    J_cols = np.arange(3 * mesh.num_faces).reshape((mesh.num_faces, 3), order='F')

    dvS1 = (le[:, 0] * cot_ang[:, 2] * (-cos_2ff1 * cot_ang[:, 1] + sin_2ff1) +
            le[:, 1] * cot_ang[:, 0] * cos_2ff2 * (cot_ang[:, 2] + cot_ang[:, 1]) +
            le[:, 2] * cot_ang[:, 1] * (-cos_2ff3 * cot_ang[:, 2] - sin_2ff3))
    dvS2 = (le[:, 0] * cot_ang[:, 2] * (-cos_2ff1 * cot_ang[:, 0] - sin_2ff1) +
            le[:, 1] * cot_ang[:, 0] * (-cos_2ff2 * cot_ang[:, 2] + sin_2ff2) +
            le[:, 2] * cot_ang[:, 1] * cos_2ff3 * (cot_ang[:, 0] + cot_ang[:, 2]))
    dvS3 = (le[:, 0] * cot_ang[:, 2] * cos_2ff1 * (cot_ang[:, 1] + cot_ang[:, 0]) +
            le[:, 1] * cot_ang[:, 0] * (-cos_2ff2 * cot_ang[:, 1] - sin_2ff2) +
            le[:, 2] * cot_ang[:, 1] * (-cos_2ff3 * cot_ang[:, 0] + sin_2ff3))

    dvS = np.column_stack([dvS1, dvS2, dvS3])
    D_vth = sp.coo_matrix(
        (dvS.ravel('F'), (I_rows.ravel('F'), J_cols.ravel('F'))),
        shape=(mesh.num_faces, 3 * mesh.num_faces)
    ).tocsr()

    #     assert(max(abs(D_vth*vt(:) - dO'*lambda_full)) < 1e-6, 'Second derivative of constraints is invalid.');

    vt_vec = np.ravel(vt, 'F')
    assert np.max(np.abs(D_vth @ vt_vec - dO.T @ lambda_full)) < 1e-6, \
        'Second derivative of constraints is invalid.'

    #     Hf = [sparse(3*mesh.num_faces,3*mesh.num_faces), sparse(3*mesh.num_faces,3*mesh.num_faces), sparse(3*mesh.num_faces,mesh.num_faces);...
    #           sparse(3*mesh.num_faces,3*mesh.num_faces), sparse(3*mesh.num_faces,3*mesh.num_faces),                  D_vth';...
    #             sparse(mesh.num_faces,3*mesh.num_faces),                     D_vth,                  D2_th];

    n3f = 3 * mesh.num_faces
    nf = mesh.num_faces

    # Build Hf as block matrix:
    # [0_{3nf x 3nf}, 0_{3nf x 3nf}, 0_{3nf x nf}]
    # [0_{3nf x 3nf}, 0_{3nf x 3nf}, D_vth'      ]
    # [0_{nf x 3nf},  D_vth,         D2_th       ]
    zero_3nf_3nf = sp.csr_matrix((n3f, n3f))
    zero_3nf_nf = sp.csr_matrix((n3f, nf))
    zero_nf_3nf = sp.csr_matrix((nf, n3f))

    Hf = sp.bmat([
        [zero_3nf_3nf, zero_3nf_3nf, zero_3nf_nf],
        [zero_3nf_3nf, zero_3nf_3nf, D_vth.T],
        [zero_nf_3nf, D_vth, D2_th]
    ], format='csr')

    #     Red = blkdiag(Reduction, speye(mesh.num_faces,mesh.num_faces));
    #     Hf = Red'*Hf*Red;
    #     Hf = (Hf + Hf')/2;
    # end

    Red = sp.block_diag([Reduction, sp.eye(mesh.num_faces)], format='csr')
    Hf = Red.T @ Hf @ Red
    Hf = (Hf + Hf.T) / 2

    return F, Jf, Hf
