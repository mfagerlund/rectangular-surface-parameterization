# === ISSUES ===
# - quadprog: use cvxpy or scipy.optimize for quadratic programming with equality constraints
# - blkdiag: use scipy.sparse.block_diag
# - accumarray: use np.bincount or np.add.at
# - dot(A, B, 2): use np.sum(A * B, axis=1) for row-wise dot product
# - complex(): numpy uses 1j for imaginary unit
# === END ISSUES ===

# function [Xp,mu] = parametrization_from_scales(Src, SrcCut, dec_cut, param, ang, omega, ut, vt, Align, Rot)
#
# if (exist('Align','var') && ~isempty(Align)) || (exist('Rot','var') && ~isempty(Rot))
#     ifseamless_const = true;
# else
#     ifseamless_const = false;
# end
# if ifseamless_const
#     if ~exist('Align','var')
#         Align = sparse(2*SrcCut.nv,0);
#     end
#     if ~exist('Rot','var')
#         Rot = sparse(2*SrcCut.nv,0);
#     end
# end
#
# % Integrated scales per edge
# u1_int = ut(:,[1 2 3])/2 + ut(:,[2 3 1])/2;
# u2_int = vt(:,[1 2 3])/2 + vt(:,[2 3 1])/2;
#
# expu = [exp(u1_int + u2_int),      zeros(Src.nf,3), ...
#              zeros(Src.nf,3), exp(u1_int - u2_int)];
#
# % Average frame on edge
# omega(param.ide_bound) = 0;
# e1 = exp(1i*ang);
# e1_edge = [e1, e1, e1];
# e1_edge = (e1_edge + exp(-1i*omega(abs(Src.T2E)).*sign(Src.T2E)).*e1_edge)/2;
# e2_edge = 1i*e1_edge;
#
# % Edge of cut mesh on local basis
# edge = dec_cut.d0p*SrcCut.X;
# edge1 = edge(abs(SrcCut.T2E(:,1)),:);
# edge2 = edge(abs(SrcCut.T2E(:,2)),:);
# edge3 = edge(abs(SrcCut.T2E(:,3)),:);
# edge_tri_cut = [complex(dot(param.e1r, edge1, 2), dot(param.e2r, edge1, 2)), complex(dot(param.e1r, edge2, 2), dot(param.e2r, edge2, 2)), complex(dot(param.e1r, edge3, 2), dot(param.e2r, edge3, 2))];
#
# % Deformed edges inside triangle (ie \mu_{ij}^k)
# sigma1_tri = real(conj(e1_edge).*edge_tri_cut);
# sigma2_tri = real(conj(e2_edge).*edge_tri_cut);
# mu1_tri = expu(:,1:3).*sigma1_tri + expu(:,4:6)  .*sigma2_tri;
# mu2_tri = expu(:,7:9).*sigma1_tri + expu(:,10:12).*sigma2_tri;
#
# % New edge vector (ie \mu_{ij})
# mu = zeros(SrcCut.ne,2);
# mu(:,1) = accumarray(abs(SrcCut.T2E(:)), mu1_tri(:))./accumarray(abs(SrcCut.T2E(:)), 1);
# mu(:,2) = accumarray(abs(SrcCut.T2E(:)), mu2_tri(:))./accumarray(abs(SrcCut.T2E(:)), 1);
#
# % Integration with/out seamless constraints
# W = dec_cut.W + 1e-5*dec_cut.star0p;
# W = (W + W')/2;
# div_dX = dec_cut.d1d*dec_cut.star1p*mu;
#
# if ifseamless_const
#     Xp = quadprog(blkdiag(W,W),-div_dX(:), [], [], [Align; Rot], zeros(size(Align,1)+size(Rot,1),1));
#     Xp = reshape(Xp, [SrcCut.nv,2]);
# else
#     Xp = W\div_dX;
# end

import numpy as np
from scipy.sparse import csr_matrix, block_diag, vstack
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Preprocess.dec_tri import DEC
from Preprocess.MeshInfo import MeshInfo


def solve_qp_equality(H: csr_matrix, Aeq: csr_matrix, beq: np.ndarray) -> np.ndarray:
    """
    Solve quadratic program with equality constraints using KKT system.

    min 0.5 * x' * H * x + f' * x
    s.t. Aeq * x = beq

    With f = 0, this reduces to solving the KKT system:
    [H   Aeq'] [x]   [0  ]
    [Aeq  0  ] [λ] = [beq]

    Parameters
    ----------
    H : sparse matrix (n, n)
        Quadratic cost matrix (positive semi-definite).
    Aeq : sparse matrix (m, n)
        Equality constraint matrix.
    beq : ndarray (m,)
        Equality constraint right-hand side.

    Returns
    -------
    x : ndarray (n,)
        Optimal solution.
    """
    from scipy.sparse import hstack, csr_matrix as sparse_csr
    from scipy.sparse.linalg import spsolve, lsqr

    n = H.shape[0]
    m = Aeq.shape[0]

    if m == 0:
        # No constraints, just solve H * x = 0 (trivial solution x = 0)
        return np.zeros(n)

    # Build KKT system
    # [H   Aeq'] [x]   [0  ]
    # [Aeq  0  ] [λ] = [beq]

    zeros_mm = sparse_csr((m, m))

    KKT = vstack([
        hstack([H, Aeq.T]),
        hstack([Aeq, zeros_mm])
    ]).tocsr()

    rhs = np.concatenate([np.zeros(n), beq])

    # Add small regularization for numerical stability
    KKT = KKT + 1e-10 * sparse_csr(np.eye(n + m))

    # Solve KKT system
    try:
        solution = spsolve(KKT, rhs)
    except Exception:
        # Fall back to least squares if direct solve fails
        result = lsqr(KKT, rhs)
        solution = result[0]

    # Extract x (first n components)
    x = solution[:n]

    return x


def parametrization_from_scales(
    Src: MeshInfo,
    SrcCut: MeshInfo,
    dec_cut: DEC,
    param,
    ang: np.ndarray,
    omega: np.ndarray,
    ut: np.ndarray,
    vt: np.ndarray,
    Align: Optional[csr_matrix] = None,
    Rot: Optional[csr_matrix] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute parameterization from scale factors.

    This integrates the deformed edge vectors to obtain UV coordinates,
    optionally enforcing seamless constraints for quad meshing.

    Parameters
    ----------
    Src : MeshInfo
        Original mesh data structure.
    SrcCut : MeshInfo
        Cut mesh (disk topology).
    dec_cut : DEC
        DEC operators for the cut mesh.
    param : object
        Parameter structure with ide_bound, e1r, e2r, T2E.
    ang : ndarray (nf,)
        Cross-field angle per face.
    omega : ndarray (ne,)
        Field rotation per edge.
    ut : ndarray (nf, 3)
        Scale factor u at each triangle corner.
    vt : ndarray (nf, 3)
        Scale factor v at each triangle corner.
    Align : sparse matrix, optional
        Alignment constraints for boundary/hard edges.
    Rot : sparse matrix, optional
        Rotation constraints for seamless matching.

    Returns
    -------
    Xp : ndarray (nv_cut, 2)
        UV coordinates for each vertex of the cut mesh.
    mu : ndarray (ne_cut, 2)
        Deformed edge vectors.
    """

    # if (exist('Align','var') && ~isempty(Align)) || (exist('Rot','var') && ~isempty(Rot))
    #     ifseamless_const = true;
    # else
    #     ifseamless_const = false;
    # end

    ifseamless_const = False
    if Align is not None and Align.shape[0] > 0:
        ifseamless_const = True
    if Rot is not None and Rot.shape[0] > 0:
        ifseamless_const = True

    # if ifseamless_const
    #     if ~exist('Align','var')
    #         Align = sparse(2*SrcCut.nv,0);
    #     end
    #     if ~exist('Rot','var')
    #         Rot = sparse(2*SrcCut.nv,0);
    #     end
    # end

    if ifseamless_const:
        if Align is None:
            Align = csr_matrix((0, 2 * SrcCut.nv))
        if Rot is None:
            Rot = csr_matrix((0, 2 * SrcCut.nv))

    # % Integrated scales per edge
    # u1_int = ut(:,[1 2 3])/2 + ut(:,[2 3 1])/2;
    # u2_int = vt(:,[1 2 3])/2 + vt(:,[2 3 1])/2;

    # Average scale factors at edge midpoints
    # MATLAB [1 2 3] -> Python [0, 1, 2]
    # MATLAB [2 3 1] -> Python [1, 2, 0]
    u1_int = ut[:, [0, 1, 2]] / 2 + ut[:, [1, 2, 0]] / 2  # shape (nf, 3)
    u2_int = vt[:, [0, 1, 2]] / 2 + vt[:, [1, 2, 0]] / 2  # shape (nf, 3)

    # expu = [exp(u1_int + u2_int),      zeros(Src.nf,3), ...
    #              zeros(Src.nf,3), exp(u1_int - u2_int)];

    # Build the deformation tensor components
    # expu has shape (nf, 12): columns 0-2, 3-5, 6-8, 9-11
    # [exp(u1+u2), 0, 0, exp(u1-u2)] for each edge
    exp_sum = np.exp(u1_int + u2_int)   # shape (nf, 3)
    exp_diff = np.exp(u1_int - u2_int)  # shape (nf, 3)
    zeros_nf3 = np.zeros((Src.nf, 3))

    expu = np.column_stack([exp_sum, zeros_nf3, zeros_nf3, exp_diff])  # shape (nf, 12)

    # % Average frame on edge
    # omega(param.ide_bound) = 0;

    omega = omega.copy()
    ide_bound = np.asarray(param.ide_bound)
    if len(ide_bound) > 0:
        omega[ide_bound] = 0

    # e1 = exp(1i*ang);

    e1 = np.exp(1j * ang)  # complex unit vector for frame direction

    # e1_edge = [e1, e1, e1];
    # e1_edge = (e1_edge + exp(-1i*omega(abs(Src.T2E)).*sign(Src.T2E)).*e1_edge)/2;

    e1_edge = np.column_stack([e1, e1, e1])  # shape (nf, 3)

    # T2E is signed 1-based encoding: decode as abs(T2E)-1 for 0-based edge index
    T2E = np.asarray(Src.T2E)
    T2E_abs = np.abs(T2E) - 1  # 0-based edge indices
    T2E_sign = np.sign(T2E)

    # Rotation factor for averaging across edge
    omega_at_edges = omega[T2E_abs]  # shape (nf, 3)
    rotation_factor = np.exp(-1j * omega_at_edges * T2E_sign)  # shape (nf, 3)

    e1_edge = (e1_edge + rotation_factor * e1_edge) / 2  # Average frame on edge

    # e2_edge = 1i*e1_edge;

    e2_edge = 1j * e1_edge  # Perpendicular frame direction

    # % Edge of cut mesh on local basis
    # edge = dec_cut.d0p*SrcCut.X;

    edge = dec_cut.d0p @ SrcCut.X  # shape (ne_cut, 3)

    # edge1 = edge(abs(SrcCut.T2E(:,1)),:);
    # edge2 = edge(abs(SrcCut.T2E(:,2)),:);
    # edge3 = edge(abs(SrcCut.T2E(:,3)),:);

    # SrcCut.T2E is signed 1-based encoding
    T2E_cut = np.asarray(SrcCut.T2E)
    T2E_cut_abs = np.abs(T2E_cut) - 1  # 0-based edge indices

    edge1 = edge[T2E_cut_abs[:, 0], :]  # shape (nf, 3)
    edge2 = edge[T2E_cut_abs[:, 1], :]  # shape (nf, 3)
    edge3 = edge[T2E_cut_abs[:, 2], :]  # shape (nf, 3)

    # edge_tri_cut = [complex(dot(param.e1r, edge1, 2), dot(param.e2r, edge1, 2)), ...]

    # Project edges onto local frame (e1r, e2r are per-face frame vectors)
    e1r = np.asarray(param.e1r)  # shape (nf, 3)
    e2r = np.asarray(param.e2r)  # shape (nf, 3)

    # Row-wise dot product: sum(A * B, axis=1)
    def row_dot(A, B):
        return np.sum(A * B, axis=1)

    edge_tri_cut = np.column_stack([
        row_dot(e1r, edge1) + 1j * row_dot(e2r, edge1),  # complex edge 1
        row_dot(e1r, edge2) + 1j * row_dot(e2r, edge2),  # complex edge 2
        row_dot(e1r, edge3) + 1j * row_dot(e2r, edge3),  # complex edge 3
    ])  # shape (nf, 3), complex

    # % Deformed edges inside triangle (ie \mu_{ij}^k)
    # sigma1_tri = real(conj(e1_edge).*edge_tri_cut);
    # sigma2_tri = real(conj(e2_edge).*edge_tri_cut);

    sigma1_tri = np.real(np.conj(e1_edge) * edge_tri_cut)  # shape (nf, 3)
    sigma2_tri = np.real(np.conj(e2_edge) * edge_tri_cut)  # shape (nf, 3)

    # mu1_tri = expu(:,1:3).*sigma1_tri + expu(:,4:6)  .*sigma2_tri;
    # mu2_tri = expu(:,7:9).*sigma1_tri + expu(:,10:12).*sigma2_tri;

    # MATLAB 1:3 -> Python 0:3, 4:6 -> 3:6, 7:9 -> 6:9, 10:12 -> 9:12
    mu1_tri = expu[:, 0:3] * sigma1_tri + expu[:, 3:6] * sigma2_tri  # shape (nf, 3)
    mu2_tri = expu[:, 6:9] * sigma1_tri + expu[:, 9:12] * sigma2_tri  # shape (nf, 3)

    # % New edge vector (ie \mu_{ij})
    # mu = zeros(SrcCut.ne,2);
    # mu(:,1) = accumarray(abs(SrcCut.T2E(:)), mu1_tri(:))./accumarray(abs(SrcCut.T2E(:)), 1);
    # mu(:,2) = accumarray(abs(SrcCut.T2E(:)), mu2_tri(:))./accumarray(abs(SrcCut.T2E(:)), 1);

    # Average deformed edge vectors per edge (each edge appears in 1 or 2 triangles)
    mu = np.zeros((SrcCut.ne, 2))

    # T2E_cut_abs flattened in column-major order to match MATLAB
    T2E_flat = T2E_cut_abs.flatten('F')  # shape (3*nf,)
    mu1_flat = mu1_tri.flatten('F')  # shape (3*nf,)
    mu2_flat = mu2_tri.flatten('F')  # shape (3*nf,)

    # Accumulate values and counts per edge
    mu_sum1 = np.bincount(T2E_flat, weights=mu1_flat, minlength=SrcCut.ne)
    mu_sum2 = np.bincount(T2E_flat, weights=mu2_flat, minlength=SrcCut.ne)
    mu_count = np.bincount(T2E_flat, minlength=SrcCut.ne)

    # Avoid division by zero (shouldn't happen for valid mesh)
    mu_count = np.maximum(mu_count, 1)

    mu[:, 0] = mu_sum1 / mu_count
    mu[:, 1] = mu_sum2 / mu_count

    # % Integration with/out seamless constraints
    # W = dec_cut.W + 1e-5*dec_cut.star0p;
    # W = (W + W')/2;

    W = dec_cut.W + 1e-5 * dec_cut.star0p
    W = (W + W.T) / 2  # Symmetrize

    # div_dX = dec_cut.d1d*dec_cut.star1p*mu;

    div_dX = dec_cut.d1d @ dec_cut.star1p @ mu  # shape (nv_cut, 2)

    # if ifseamless_const
    #     Xp = quadprog(blkdiag(W,W),-div_dX(:), [], [], [Align; Rot], zeros(size(Align,1)+size(Rot,1),1));
    #     Xp = reshape(Xp, [SrcCut.nv,2]);
    # else
    #     Xp = W\div_dX;
    # end

    if ifseamless_const:
        # Build block diagonal Laplacian for [u; v]
        W_blk = block_diag((W, W), format='csr')

        # Stack constraints
        Aeq = vstack([Align, Rot], format='csr')
        beq = np.zeros(Aeq.shape[0])

        # Flatten div_dX in column-major order (u coordinates first, then v)
        f = -div_dX.flatten('F')  # shape (2*nv_cut,)

        # Solve QP: min 0.5 * x' * W_blk * x + f' * x  s.t. Aeq * x = beq
        # This is equivalent to solving the Poisson equation with constraints
        Xp_flat = solve_qp_with_linear_term(W_blk, f, Aeq, beq)

        # Reshape to (nv_cut, 2)
        Xp = Xp_flat.reshape((SrcCut.nv, 2), order='F')
    else:
        # Simple Poisson solve without constraints
        Xp = spsolve(W.tocsr(), div_dX)

        # Handle case where spsolve returns 1D array for single RHS column
        if Xp.ndim == 1:
            Xp = Xp.reshape(-1, 1)
        if Xp.shape[1] == 1 and div_dX.shape[1] == 2:
            # Solve each column separately
            Xp = np.column_stack([
                spsolve(W.tocsr(), div_dX[:, 0]),
                spsolve(W.tocsr(), div_dX[:, 1])
            ])

    return Xp, mu


def solve_qp_with_linear_term(
    H: csr_matrix,
    f: np.ndarray,
    Aeq: csr_matrix,
    beq: np.ndarray
) -> np.ndarray:
    """
    Solve quadratic program with linear term and equality constraints.

    min 0.5 * x' * H * x + f' * x
    s.t. Aeq * x = beq

    Uses KKT conditions:
    [H   Aeq'] [x]   [-f ]
    [Aeq  0  ] [λ] = [beq]

    Parameters
    ----------
    H : sparse matrix (n, n)
        Quadratic cost matrix.
    f : ndarray (n,)
        Linear cost vector.
    Aeq : sparse matrix (m, n)
        Equality constraint matrix.
    beq : ndarray (m,)
        Equality constraint RHS.

    Returns
    -------
    x : ndarray (n,)
        Optimal solution.
    """
    from scipy.sparse import hstack, csr_matrix as sparse_csr
    from scipy.sparse.linalg import spsolve, lsqr

    n = H.shape[0]
    m = Aeq.shape[0]

    if m == 0:
        # No constraints, solve H * x = -f
        return spsolve(H.tocsr(), -f)

    # Build KKT system
    zeros_mm = sparse_csr((m, m))

    KKT = vstack([
        hstack([H, Aeq.T]),
        hstack([Aeq, zeros_mm])
    ]).tocsr()

    rhs = np.concatenate([-f, beq])

    # Solve KKT system
    try:
        solution = spsolve(KKT, rhs)
    except Exception:
        # Fall back to least squares if direct solve fails
        result = lsqr(KKT, rhs)
        solution = result[0]

    # Extract x (first n components)
    x = solution[:n]

    return x
