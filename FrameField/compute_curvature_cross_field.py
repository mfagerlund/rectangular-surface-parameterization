# === ISSUES ===
# - wrapToPi: implemented as wrap_to_pi using np.arctan2(np.sin(x), np.cos(x))
# - brush_frame_field: imported from brush_frame_field.py
# === END ISSUES ===

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional

from .brush_frame_field import brush_frame_field


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]. Equivalent to MATLAB wrapToPi."""
    return np.arctan2(np.sin(x), np.cos(x))


# function [omega,ang,sing,kappa,Curv] = compute_curvature_cross_field(Src, param, dec, smoothing_iter, alpha)
# % Compute curvature aligned cross field
# % - omega: field rotation
# % -   ang: field angle
# % -  sing: field cingularities
# % - kappa: principal curvatures
# % -  Curv: symmetric curvature tensor

def compute_curvature_cross_field(
    Src,
    param,
    dec,
    smoothing_iter: int,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute curvature aligned cross field.

    Args:
        Src: Mesh data structure with X, E2V, E2T, T, T2E, normal, area, ne, nf
        param: Parameter structure with e1r, e2r, para_trans, ide_int, ide_bound,
               E2T, tri_fix, tri_free, Kt_invisible
        dec: DEC structure with d0d, d1d, star1d
        smoothing_iter: Number of smoothing iterations
        alpha: Smoothing weight parameter

    Returns:
        omega: Field rotation per edge
        ang: Field angle per face
        sing: Field singularities
        kappa: Principal curvatures (nf x 2)
        Curv: Symmetric curvature tensor (nf x 3)
    """

    # %% Compute curvature tensor
    # % "SECOND FUNDAMENTAL MEASURE OF GEOMETRIC SETS AND LOCAL APPROXIMATION OF CURVATURES"
    # % David Cohen-Steiner & Jean-Marie Morvan

    # % Compute dihedral angle
    # comp_angle = @(u,v,n) atan2(dot(cross(u,v,2),n,2), dot(u,v,2));

    def comp_angle(u, v, n):
        """Compute signed angle between vectors u and v with normal n."""
        cross_prod = np.cross(u, v)
        # dot along rows (axis=1)
        sin_angle = np.sum(cross_prod * n, axis=1)
        cos_angle = np.sum(u * v, axis=1)
        return np.arctan2(sin_angle, cos_angle)

    # edge = Src.X(Src.E2V(:,2),:) - Src.X(Src.E2V(:,1),:);
    # edge_length = sqrt(sum(edge.^2,2));
    # edge = edge./edge_length;

    # Edge vectors (0-indexed)
    edge = Src.X[Src.E2V[:, 1], :] - Src.X[Src.E2V[:, 0], :]
    edge_length = np.sqrt(np.sum(edge**2, axis=1))
    edge = edge / edge_length[:, np.newaxis]

    # Eedge = [edge(:,1).*edge(:,1), edge(:,1).*edge(:,2), edge(:,1).*edge(:,3), ...
    #          edge(:,2).*edge(:,1), edge(:,2).*edge(:,2), edge(:,2).*edge(:,3), ...
    #          edge(:,3).*edge(:,1), edge(:,3).*edge(:,2), edge(:,3).*edge(:,3)];

    # Outer product of edge with itself, stored as 9-column matrix
    Eedge = np.column_stack([
        edge[:, 0] * edge[:, 0], edge[:, 0] * edge[:, 1], edge[:, 0] * edge[:, 2],
        edge[:, 1] * edge[:, 0], edge[:, 1] * edge[:, 1], edge[:, 1] * edge[:, 2],
        edge[:, 2] * edge[:, 0], edge[:, 2] * edge[:, 1], edge[:, 2] * edge[:, 2]
    ])

    # ide_int = all(Src.E2T(:,1:2) ~= 0,2);
    # dihedral_angle = zeros(Src.ne,1);
    # dihedral_angle(ide_int) = Src.E2T(ide_int,4).*comp_angle(Src.normal(Src.E2T(ide_int,1),:), Src.normal(Src.E2T(ide_int,2),:), edge(ide_int,:));

    # Interior edges: both adjacent triangles exist (not -1 in 0-indexed)
    ide_int = np.all(Src.E2T[:, 0:2] >= 0, axis=1)
    dihedral_angle = np.zeros(Src.ne)

    # For interior edges, compute dihedral angle
    # Src.E2T[:, 3] contains orientation sign, Src.E2T[:, 0:2] are triangle indices
    int_idx = np.where(ide_int)[0]
    t1 = Src.E2T[int_idx, 0]
    t2 = Src.E2T[int_idx, 1]
    sign_e = Src.E2T[int_idx, 3]

    dihedral_angle[int_idx] = sign_e * comp_angle(
        Src.normal[t1, :],
        Src.normal[t2, :],
        edge[int_idx, :]
    )

    # % Compute curvature tensor
    # Curv = zeros(Src.nf,4);
    # J = [0,-1; 1, 0];
    # K = zeros(Src.nf,1);

    Curv = np.zeros((Src.nf, 4))
    J = np.array([[0, -1], [1, 0]])
    K = np.zeros(Src.nf)

    # for i = 1:Src.nf
    #     idt = i;
    #     idt = find(any(ismember(Src.T, Src.T(idt,:)),2));
    #     ide = unique(abs(Src.T2E(idt,:)));
    #     E = [param.e1r(i,:)', param.e2r(i,:)'];
    #
    #     A = sum(dihedral_angle(ide).*edge_length(ide).*Eedge(ide,:))/Src.area(i);
    #     A = reshape(A, [3,3]);
    #     A = (A + A')/2;
    #     A = J'*E'*A*E*J;
    #     A = (A + A')/2;
    #
    #     Curv(i,:) = A(:);
    #     K(i) = det(A);
    # end

    for i in range(Src.nf):
        # Find triangles sharing vertices with triangle i
        # idt = find(any(ismember(Src.T, Src.T(idt,:)),2));
        face_verts = Src.T[i, :]  # Vertices of face i
        # Find all faces that share any vertex with face i
        idt = np.where(np.any(np.isin(Src.T, face_verts), axis=1))[0]

        # Get unique edge indices from these triangles
        # Src.T2E uses signed 1-based encoding, decode to get edge indices
        t2e_vals = Src.T2E[idt, :].ravel()
        ide = np.unique(np.abs(t2e_vals) - 1)  # Decode: abs(x)-1 gives 0-based edge index

        # Local reference frame: E is 3x2 matrix with e1r and e2r as columns
        E = np.column_stack([param.e1r[i, :], param.e2r[i, :]])

        # Sum weighted outer products
        weighted_sum = np.sum(
            dihedral_angle[ide, np.newaxis] * edge_length[ide, np.newaxis] * Eedge[ide, :],
            axis=0
        ) / Src.area[i]

        # Reshape to 3x3 matrix (MATLAB's reshape is column-major)
        A = weighted_sum.reshape((3, 3), order='F')
        A = (A + A.T) / 2

        # Project to local frame and rotate by J
        A = J.T @ E.T @ A @ E @ J
        A = (A + A.T) / 2

        # Store as flattened 2x2 matrix (column-major)
        Curv[i, :] = A.ravel(order='F')
        K[i] = np.linalg.det(A)

    # Curv = Curv(:,[1 2 4]);

    # Keep only unique elements of symmetric matrix: [a11, a12, a22]
    # In column-major: [0]=a11, [1]=a21, [2]=a12, [3]=a22
    # MATLAB indices [1,2,4] -> Python [0,1,3] (a11, a21=a12, a22)
    Curv = Curv[:, [0, 1, 3]]

    # %% Extract principal directions
    # dir_min = zeros(Src.nf,1); % principal direction
    # kappa = zeros(Src.nf,2); % principal curvature
    # for i = 1:Src.nf
    #     A = [Curv(i,1), Curv(i,2); Curv(i,2), Curv(i,3)];
    #
    #     [V,D] = eig(A);
    #
    #     dir_min(i) = complex(V(1,1), V(2,1));
    #     kappa(i,:) = diag(D);
    # end

    dir_min = np.zeros(Src.nf, dtype=complex)
    kappa = np.zeros((Src.nf, 2))

    for i in range(Src.nf):
        # Reconstruct symmetric 2x2 matrix
        A = np.array([[Curv[i, 0], Curv[i, 1]],
                      [Curv[i, 1], Curv[i, 2]]])

        # Compute eigenvalues and eigenvectors
        D, V = np.linalg.eig(A)

        # Sort by eigenvalue (MATLAB's eig returns sorted by magnitude for symmetric)
        # For consistency, we take the first eigenvector as dir_min
        idx_sort = np.argsort(D)
        D = D[idx_sort]
        V = V[:, idx_sort]

        # Principal direction as complex number
        dir_min[i] = complex(V[0, 0], V[1, 0])
        kappa[i, :] = D

    # %% Frame field smoothing
    # z = (dir_min./abs(dir_min)).^4; % Init cross field from principal direction
    # z_fix = ones(length(param.tri_fix),1); % reference frame is algned with constraint edge by construction
    # z(param.tri_fix) = z_fix; % alignment constraints

    # Initialize cross field from principal directions
    z = (dir_min / np.abs(dir_min))**4
    z_fix = np.ones(len(param.tri_fix), dtype=complex)
    z[param.tri_fix] = z_fix

    # if smoothing_iter > 0
    if smoothing_iter > 0:
        # % Build connection Laplacian
        # I = [param.ide_int,param.ide_int];
        # J = param.E2T(param.ide_int,1:2);
        # rot = param.para_trans(param.ide_int);
        # S = [exp(1i*4*rot/2); -exp(-1i*4*rot/2)];
        # d0d_cplx = sparse(I, J, S, Src.ne, Src.nf);
        # Wcon = d0d_cplx'*dec.star1d*d0d_cplx;
        # Wcon = (Wcon + Wcon')/2;

        # Build connection Laplacian
        ide_int_param = param.ide_int
        n_int = len(ide_int_param)

        # Row indices: each interior edge appears twice
        I = np.concatenate([ide_int_param, ide_int_param])
        # Column indices: the two adjacent triangles
        J = np.concatenate([param.E2T[ide_int_param, 0], param.E2T[ide_int_param, 1]])

        rot = param.para_trans[ide_int_param]
        # Values: complex exponentials for connection
        S = np.concatenate([np.exp(1j * 4 * rot / 2), -np.exp(-1j * 4 * rot / 2)])

        d0d_cplx = sp.csr_matrix((S, (I, J)), shape=(Src.ne, Src.nf))
        Wcon = d0d_cplx.conj().T @ dec.star1d @ d0d_cplx
        Wcon = (Wcon + Wcon.conj().T) / 2

        # % Screen smoothing
        # w = (abs(K) + 1e-3); % Gaussian curvature weight
        # M = spdiags(alpha*w.*Src.area, 0, Src.nf, Src.nf); % Modified mass matrix
        # A = Wcon + M;

        w = np.abs(K) + 1e-3
        M = sp.diags(alpha * w * Src.area, 0, shape=(Src.nf, Src.nf), format='csr')
        A_mat = Wcon + M

        # for i = 1:smoothing_iter
        #     if ~isempty(param.tri_fix)
        #         z(param.tri_free) = A(param.tri_free,param.tri_free)\(M(param.tri_free,param.tri_free)*z(param.tri_free) - A(param.tri_free,param.tri_fix)*z_fix);
        #     else
        #         z = A\(M*z);
        #     end
        #     z = z./abs(z);
        # end

        for _ in range(smoothing_iter):
            if len(param.tri_fix) > 0:
                # Extract submatrices
                A_ff = A_mat[param.tri_free, :][:, param.tri_free]
                A_fc = A_mat[param.tri_free, :][:, param.tri_fix]
                M_ff = M[param.tri_free, :][:, param.tri_free]

                # Solve for free triangles
                rhs = M_ff @ z[param.tri_free] - A_fc @ z_fix
                z[param.tri_free] = spsolve(A_ff.tocsr(), rhs)
            else:
                z = spsolve(A_mat.tocsr(), M @ z)

            # Normalize to unit circle
            z = z / np.abs(z)

    # %% Extract angles in reference basis
    # % Compute frames rotation
    # ang = angle(z)/4;
    # omega = wrapToPi(4*(dec.d0d*ang + param.para_trans))/4;
    # omega(param.ide_bound) = 0;
    # sing = (dec.d1d*(param.para_trans - omega) + param.Kt_invisible)/(2*pi);

    ang = np.angle(z) / 4
    omega = wrap_to_pi(4 * (dec.d0d @ ang + param.para_trans)) / 4
    omega[param.ide_bound] = 0
    sing = (dec.d1d @ (param.para_trans - omega) + param.Kt_invisible) / (2 * np.pi)

    # % Brush frame field
    # ang = brush_frame_field(param, omega, param.tri_fix, ang(param.tri_fix));

    ang = brush_frame_field(param, omega, param.tri_fix, ang[param.tri_fix])

    # %% Match curvature with closest frame direction
    # z_min = (dir_min./abs(dir_min)).^2;
    # z_max = 1i*z_min;
    # [~,id] = min(abs([z_min, z_max] - exp(2*1i*ang)), [], 2);
    # kappa = [kappa(:,1).*(id == 1) + kappa(:,2).*(id == 2), ...
    #          kappa(:,1).*(id == 2) + kappa(:,2).*(id == 1)];

    z_min = (dir_min / np.abs(dir_min))**2
    z_max = 1j * z_min

    # Compare both directions with the frame angle
    target = np.exp(2j * ang)
    diff_min = np.abs(z_min - target)
    diff_max = np.abs(z_max - target)

    # Find which direction is closer (1 = min, 2 = max)
    # id = 1 if z_min is closer, id = 2 if z_max is closer
    id_arr = np.where(diff_min <= diff_max, 1, 2)

    # Swap kappa columns based on which direction aligns with frame
    kappa_new = np.zeros_like(kappa)
    kappa_new[:, 0] = np.where(id_arr == 1, kappa[:, 0], kappa[:, 1])
    kappa_new[:, 1] = np.where(id_arr == 1, kappa[:, 1], kappa[:, 0])
    kappa = kappa_new

    return omega, ang, sing, kappa, Curv
