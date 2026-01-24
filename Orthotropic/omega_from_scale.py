

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple


def omega_from_scale(
    mesh,
    param,
    dec,
    ut: np.ndarray,
    vt: np.ndarray,
    ang: np.ndarray,
    Reduction: Optional[sp.spmatrix] = None,
    compute_derivative: bool = False
) -> Tuple[sp.spmatrix, Optional[sp.spmatrix], Optional[sp.spmatrix]]:
    """
    Build frame rotation from scale factors ut and vt defined at corners.

    Args:
        mesh: Mesh data structure with T2E, ne, nf, cot_corner_angle
        param: Parameter structure with ang_basis
        dec: DEC structure with star1p, d0p_tri
        ut: Scale factors u at corners (nf x 3)
        vt: Scale factors v at corners (nf x 3)
        ang: Frame angle per face (nf,)
        Reduction: Optional reduction matrix for reduced variables
        compute_derivative: If True, compute dO (derivative wrt ang)

    Returns:
        O: (ne x 6*nf sparse matrix) O @ [ut.ravel('F'); vt.ravel('F')] is the frame rotation omega
        Or: (ne x 2*nv sparse matrix) linear map from scale factors to omega in reduced variables
            (only if Reduction is provided)
        dO: (ne x nf sparse matrix) derivative of omega wrt ang
            (only if compute_derivative is True)
    """
    # function [O,Or,dO] = omega_from_scale(mesh, param, dec, ut, vt, ang, Reduction)


    # cot_ang = mesh.cot_corner_angle;
    cot_ang = mesh.cot_corner_angle  # (nf x 3)

    # cos_2ff1 = cos(2*ang + 2*param.ang_basis(:,1));
    # sin_2ff1 = sin(2*ang + 2*param.ang_basis(:,1));
    # cos_2ff2 = cos(2*ang + 2*param.ang_basis(:,2));
    # sin_2ff2 = sin(2*ang + 2*param.ang_basis(:,2));
    # cos_2ff3 = cos(2*ang + 2*param.ang_basis(:,3));
    # sin_2ff3 = sin(2*ang + 2*param.ang_basis(:,3));
    cos_2ff1 = np.cos(2 * ang + 2 * param.ang_basis[:, 0])
    sin_2ff1 = np.sin(2 * ang + 2 * param.ang_basis[:, 0])
    cos_2ff2 = np.cos(2 * ang + 2 * param.ang_basis[:, 1])
    sin_2ff2 = np.sin(2 * ang + 2 * param.ang_basis[:, 1])
    cos_2ff3 = np.cos(2 * ang + 2 * param.ang_basis[:, 2])
    sin_2ff3 = np.sin(2 * ang + 2 * param.ang_basis[:, 2])

    # I = mesh.T2E(:,[1 1 1 2 2 2 3 3 3]);
    # T2E is (nf x 3), signed edge indices encoded as (edge_idx + 1) * sign
    # We need to extract edge indices and signs
    T2E = mesh.T2E  # (nf x 3), signed 1-based encoding
    ne = mesh.num_edges
    nf = mesh.num_faces

    # Decode signed edge indices: edge_idx = abs(T2E) - 1, sign = sign(T2E)
    edge_idx = np.abs(T2E) - 1  # (nf x 3)
    edge_sign = np.sign(T2E)  # (nf x 3)

    # J = repmat(reshape((1:3*mesh.num_faces)', [mesh.num_faces,3]), [1,3]);
    # Creates (nf x 9) matrix with columns: [corner1, corner1, corner1, corner2, corner2, corner2, corner3, corner3, corner3]
    # In MATLAB, corners are 1:3*nf reshaped to (nf,3)
    # Corner indices: corners 0,1,2 of face f are at f*3+0, f*3+1, f*3+2 (row-major)
    # But MATLAB uses column-major, so corner j of face i is at j*nf + i (0-indexed)
    # In column-major (Fortran order): corner 0 of all faces, then corner 1, then corner 2
    # corner_indices[i,j] = j * nf + i (0-indexed)
    corner_indices = np.arange(3 * nf).reshape((nf, 3), order='F')  # (nf x 3)

    # S = 0.5*sign(I).*[cot_ang(:,3).*[-cos_2ff1 - sin_2ff1.*cot_ang(:,2)      , cos_2ff1 - sin_2ff1.*cot_ang(:,1)      , sin_2ff1.*(cot_ang(:,1) + cot_ang(:,2))], ...
    #                   cot_ang(:,1).*[ sin_2ff2.*(cot_ang(:,2) + cot_ang(:,3)),-cos_2ff2 - sin_2ff2.*cot_ang(:,3)      , cos_2ff2 - sin_2ff2.*cot_ang(:,2)      ], ...
    #                   cot_ang(:,2).*[ cos_2ff3 - sin_2ff3.*cot_ang(:,3)      , sin_2ff3.*(cot_ang(:,3) + cot_ang(:,1)),-cos_2ff3 - sin_2ff3.*cot_ang(:,1)      ]];

    # Build the S matrix (nf x 9) corresponding to 9 columns:
    # Columns correspond to: edge1 with corners 1,2,3, edge2 with corners 1,2,3, edge3 with corners 1,2,3
    # Edge 1 contributions (first 3 columns)
    S1_c1 = cot_ang[:, 2] * (-cos_2ff1 - sin_2ff1 * cot_ang[:, 1])
    S1_c2 = cot_ang[:, 2] * (cos_2ff1 - sin_2ff1 * cot_ang[:, 0])
    S1_c3 = cot_ang[:, 2] * (sin_2ff1 * (cot_ang[:, 0] + cot_ang[:, 1]))

    # Edge 2 contributions (next 3 columns)
    S2_c1 = cot_ang[:, 0] * (sin_2ff2 * (cot_ang[:, 1] + cot_ang[:, 2]))
    S2_c2 = cot_ang[:, 0] * (-cos_2ff2 - sin_2ff2 * cot_ang[:, 2])
    S2_c3 = cot_ang[:, 0] * (cos_2ff2 - sin_2ff2 * cot_ang[:, 1])

    # Edge 3 contributions (last 3 columns)
    S3_c1 = cot_ang[:, 1] * (cos_2ff3 - sin_2ff3 * cot_ang[:, 2])
    S3_c2 = cot_ang[:, 1] * (sin_2ff3 * (cot_ang[:, 2] + cot_ang[:, 0]))
    S3_c3 = cot_ang[:, 1] * (-cos_2ff3 - sin_2ff3 * cot_ang[:, 0])

    # Build I, J, S arrays for Dv_tri sparse matrix (ne x 3*nf)
    # I = mesh.T2E(:,[1 1 1 2 2 2 3 3 3]) -> repeat each edge column 3 times
    # Each face contributes 9 entries
    I_arr = np.column_stack([
        edge_idx[:, 0], edge_idx[:, 0], edge_idx[:, 0],
        edge_idx[:, 1], edge_idx[:, 1], edge_idx[:, 1],
        edge_idx[:, 2], edge_idx[:, 2], edge_idx[:, 2]
    ]).ravel()  # (9*nf,)

    # Signs for each entry
    signs_arr = np.column_stack([
        edge_sign[:, 0], edge_sign[:, 0], edge_sign[:, 0],
        edge_sign[:, 1], edge_sign[:, 1], edge_sign[:, 1],
        edge_sign[:, 2], edge_sign[:, 2], edge_sign[:, 2]
    ]).ravel()  # (9*nf,)

    # J indices: corner indices repeated per edge
    J_arr = np.column_stack([
        corner_indices[:, 0], corner_indices[:, 1], corner_indices[:, 2],  # edge 1
        corner_indices[:, 0], corner_indices[:, 1], corner_indices[:, 2],  # edge 2
        corner_indices[:, 0], corner_indices[:, 1], corner_indices[:, 2]   # edge 3
    ]).ravel()  # (9*nf,)

    # S values
    S_arr = 0.5 * signs_arr * np.column_stack([
        S1_c1, S1_c2, S1_c3,
        S2_c1, S2_c2, S2_c3,
        S3_c1, S3_c2, S3_c3
    ]).ravel()  # (9*nf,)

    # Dv_tri = sparse(abs(I), J, S, mesh.num_edges, 3*mesh.num_faces);
    Dv_tri = sp.coo_matrix((S_arr, (I_arr, J_arr)), shape=(ne, 3 * nf)).tocsr()

    # O = [-dec.star1p*dec.d0p_tri, Dv_tri];
    # O is (ne x 6*nf): first 3*nf columns for ut, last 3*nf columns for vt
    O = sp.hstack([-dec.star1p @ dec.d0p_tri, Dv_tri])

    # if exist('Reduction','var') && ~isempty(Reduction)
    #     Or = O*Reduction;
    # end
    Or = None
    if Reduction is not None:
        Or = O @ Reduction

    # if nargout >= 3
    dO = None
    if compute_derivative:
        # vt1  = vt(:,1);
        # vt2  = vt(:,2);
        # vt3  = vt(:,3);
        vt1 = vt[:, 0]
        vt2 = vt[:, 1]
        vt3 = vt[:, 2]

        # I = mesh.T2E;
        # J = repmat((1:mesh.num_faces)', [1,3]);
        # S = sign(I).*[cot_ang(:,3).*(cos_2ff1.*(cot_ang(:,2).*(vt3 - vt1) + cot_ang(:,1).*(vt3 - vt2)) - sin_2ff1.*(vt2 - vt1)), ...
        #               cot_ang(:,1).*(cos_2ff2.*(cot_ang(:,3).*(vt1 - vt2) + cot_ang(:,2).*(vt1 - vt3)) - sin_2ff2.*(vt3 - vt2)), ...
        #               cot_ang(:,2).*(cos_2ff3.*(cot_ang(:,1).*(vt2 - vt3) + cot_ang(:,3).*(vt2 - vt1)) - sin_2ff3.*(vt1 - vt3))];
        # dO = sparse(abs(I), J, S, mesh.num_edges, mesh.num_faces);

        dO_S1 = cot_ang[:, 2] * (cos_2ff1 * (cot_ang[:, 1] * (vt3 - vt1) + cot_ang[:, 0] * (vt3 - vt2)) - sin_2ff1 * (vt2 - vt1))
        dO_S2 = cot_ang[:, 0] * (cos_2ff2 * (cot_ang[:, 2] * (vt1 - vt2) + cot_ang[:, 1] * (vt1 - vt3)) - sin_2ff2 * (vt3 - vt2))
        dO_S3 = cot_ang[:, 1] * (cos_2ff3 * (cot_ang[:, 0] * (vt2 - vt3) + cot_ang[:, 2] * (vt2 - vt1)) - sin_2ff3 * (vt1 - vt3))

        # Apply signs
        dO_S1 = edge_sign[:, 0] * dO_S1
        dO_S2 = edge_sign[:, 1] * dO_S2
        dO_S3 = edge_sign[:, 2] * dO_S3

        # Build sparse matrix: each face contributes to 3 edges
        dO_I = np.concatenate([edge_idx[:, 0], edge_idx[:, 1], edge_idx[:, 2]])
        dO_J = np.concatenate([np.arange(nf), np.arange(nf), np.arange(nf)])
        dO_S = np.concatenate([dO_S1, dO_S2, dO_S3])

        dO = sp.coo_matrix((dO_S, (dO_I, dO_J)), shape=(ne, nf)).tocsr()

    return O, Or, dO
