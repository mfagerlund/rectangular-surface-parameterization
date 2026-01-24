

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class DistortionMetrics:
    """Distortion metrics per triangle."""
    ang_param: np.ndarray = field(default_factory=lambda: np.array([]))  # Angular parameterization error
    detJ: np.ndarray = field(default_factory=lambda: np.array([]))       # Jacobian determinant
    area: np.ndarray = field(default_factory=lambda: np.array([]))       # Area scaling (s1*s2)
    conf: np.ndarray = field(default_factory=lambda: np.array([]))       # Conformal distortion (s1/s2)
    orth: np.ndarray = field(default_factory=lambda: np.array([]))       # Orthogonality angle
    cheb: np.ndarray = field(default_factory=lambda: np.array([]))       # Chebyshev distortion
    cheb_ang: np.ndarray = field(default_factory=lambda: np.array([]))   # Chebyshev angle distortion


# function [disto,ut,theta,u_tri] = extract_scale_from_param(Xp, X, T, param, T_cut, ang)

def extract_scale_from_param(
    Xp: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    param,
    T_cut: np.ndarray,
    ang: Optional[np.ndarray] = None
) -> Tuple[DistortionMetrics, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Extract scale and distortion information from parameterization.

    Computes various distortion metrics by analyzing the Jacobian of the
    parameterization map from 3D surface to 2D UV space.

    Args:
        Xp: Parameterized vertex positions (#V_cut, 2 or 3) - UV coordinates
        X: Original vertex positions (#V, 3)
        T: Face indices into X (#F, 3), 0-indexed
        param: Parameter structure with e1r, e2r (local frame per triangle)
        T_cut: Face indices into Xp after cutting (#F, 3), 0-indexed
        ang: Cross-field angles per face (optional)

    Returns:
        disto: DistortionMetrics dataclass with per-face metrics
        ut: Per-vertex scaling (if ang provided), shape (#F, 6)
        theta: Angle alignment error (if ang provided)
        u_tri: Per-face scaling (#F, 2)
    """
    # nf = size(T,1);

    nf = T.shape[0]

    # u_tri = zeros(nf,2);
    # disto.ang_param = zeros(nf,1);
    # disto.detJ = zeros(nf,1);
    # disto.area = zeros(nf,1);
    # disto.conf = zeros(nf,1);
    # disto.orth = zeros(nf,1);
    # disto.cheb = zeros(nf,1);
    # disto.cheb_ang = zeros(nf,1);

    # Compute distortion
    u_tri = np.zeros((nf, 2))
    disto = DistortionMetrics(
        ang_param=np.zeros(nf),
        detJ=np.zeros(nf),
        area=np.zeros(nf),
        conf=np.zeros(nf),
        orth=np.zeros(nf),
        cheb=np.zeros(nf),
        cheb_ang=np.zeros(nf)
    )

    # for i = 1:nf

    for i in range(nf):
        # Jac = inv([param.e1r(i,:); param.e2r(i,:)]*[X(T(i,1),:) - X(T(i,2),:); X(T(i,1),:) - X(T(i,3),:)]');
        # Jac = [Xp(T_cut(i,1),1:2) - Xp(T_cut(i,2),1:2); Xp(T_cut(i,1),1:2) - Xp(T_cut(i,3),1:2)]'*Jac;

        # Build local frame matrix [e1r; e2r] (2x3)
        local_frame = np.vstack([param.e1r[i, :], param.e2r[i, :]])

        # Edge vectors in 3D: (v0-v1), (v0-v2)
        # T is 0-indexed
        edge_mat = np.vstack([
            X[T[i, 0], :] - X[T[i, 1], :],
            X[T[i, 0], :] - X[T[i, 2], :]
        ])

        # local_frame @ edge_mat.T gives 2x2 matrix representing edges in local coords
        local_edges = local_frame @ edge_mat.T  # 2x2

        # Jacobian inverse of local frame
        Jac_inv = np.linalg.inv(local_edges)

        # Edge vectors in UV space: (uv0-uv1), (uv0-uv2)
        uv_edges = np.vstack([
            Xp[T_cut[i, 0], 0:2] - Xp[T_cut[i, 1], 0:2],
            Xp[T_cut[i, 0], 0:2] - Xp[T_cut[i, 2], 0:2]
        ])

        # Full Jacobian: maps from local 2D tangent space to UV space
        Jac = uv_edges.T @ Jac_inv  # 2x2

        # [U,S,V] = svd(Jac);
        # s = diag(S);

        U, s, Vh = np.linalg.svd(Jac)
        V = Vh.T

        # Q = U*V';
        # if det(Q) < 0
        #     d = [1,1]; [~,id] = min(s); d(id) =-1;
        #     Q = U*diag(d)*V';
        # end

        Q = U @ V.T
        if np.linalg.det(Q) < 0:
            d = np.array([1.0, 1.0])
            min_idx = np.argmin(s)
            d[min_idx] = -1
            Q = U @ np.diag(d) @ V.T

        # disto.ang_param(i) = atan2(Q(1,2), Q(1,1));

        disto.ang_param[i] = np.arctan2(Q[0, 1], Q[0, 0])

        # d = diag(U*S*U');
        # u_tri(i,1) = log(d(1)*d(2))/2;
        # u_tri(i,2) = log(d(1)/d(2))/2;

        S_mat = np.diag(s)
        d_diag = np.diag(U @ S_mat @ U.T)
        u_tri[i, 0] = np.log(d_diag[0] * d_diag[1]) / 2
        u_tri[i, 1] = np.log(d_diag[0] / d_diag[1]) / 2

        # if exist('ang','var') && abs(Q(1,:)*[cos(ang(i)); sin(ang(i))]) < 0.5
        #     u_tri(i,2) =-u_tri(i,2);
        # end

        if ang is not None and np.abs(Q[0, :] @ np.array([np.cos(ang[i]), np.sin(ang[i])])) < 0.5:
            u_tri[i, 1] = -u_tri[i, 1]

        # disto.area(i) = s(1)*s(2);
        # disto.conf(i) = s(1)/s(2);
        # disto.detJ(i) = det(Jac);
        # disto.orth(i) = acos((Jac(1,:)*Jac(2,:)')/(norm(Jac(1,:))*norm(Jac(2,:))));

        disto.area[i] = s[0] * s[1]
        disto.conf[i] = s[0] / s[1] if s[1] > 1e-12 else np.inf
        disto.detJ[i] = np.linalg.det(Jac)

        norm1 = np.linalg.norm(Jac[0, :])
        norm2 = np.linalg.norm(Jac[1, :])
        if norm1 > 1e-12 and norm2 > 1e-12:
            cos_angle = np.clip(np.dot(Jac[0, :], Jac[1, :]) / (norm1 * norm2), -1, 1)
            disto.orth[i] = np.arccos(cos_angle)
        else:
            disto.orth[i] = 0

        # u = (Jac\[1;1])/sqrt(2); v = (Jac\[-1;1])/sqrt(2);
        # disto.cheb(i) = ((norm(u) - 1)^2 + (norm(v) - 1)^2);
        # disto.cheb_ang(i) = acos((u'*v)/(norm(u)*norm(v)))*180/pi;

        try:
            u = np.linalg.solve(Jac, np.array([1, 1])) / np.sqrt(2)
            v = np.linalg.solve(Jac, np.array([-1, 1])) / np.sqrt(2)

            disto.cheb[i] = (np.linalg.norm(u) - 1)**2 + (np.linalg.norm(v) - 1)**2

            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u > 1e-12 and norm_v > 1e-12:
                cos_uv = np.clip(np.dot(u, v) / (norm_u * norm_v), -1, 1)
                disto.cheb_ang[i] = np.arccos(cos_uv) * 180 / np.pi
            else:
                disto.cheb_ang[i] = 0
        except np.linalg.LinAlgError:
            disto.cheb[i] = np.inf
            disto.cheb_ang[i] = 0

    # if any(disto.detJ <= 0)
    #     warning([num2str(sum(disto.detJ <= 0)), ' negative determinant.']);
    # end

    n_negative = np.sum(disto.detJ <= 0)
    if n_negative > 0:
        warnings.warn(f'{n_negative} negative determinant.')

    # if nargout > 1
    #     theta = angle(exp(4*1i*(disto.ang_param - ang)))/4;
    #
    #     u = accumarray(T(:), [u_tri(:,1); u_tri(:,1); u_tri(:,1)])./accumarray(T(:), 1);
    #     v = accumarray(T_cut(:), [u_tri(:,2); u_tri(:,2); u_tri(:,2)])./accumarray(T_cut(:), 1);
    #     ut = [u(T), v(T)];
    # end

    ut = None
    theta = None

    if ang is not None:
        # theta = angle(exp(4*1i*(disto.ang_param - ang)))/4;
        theta = np.angle(np.exp(4j * (disto.ang_param - ang))) / 4

        # Average on vertices using accumarray pattern
        # T(:) flattens column-major in MATLAB, we use ravel('F')
        T_flat = T.ravel('F')  # Column-major flattening
        T_cut_flat = T_cut.ravel('F')

        # Replicate u_tri[:,0] three times (once per vertex of each face)
        u_tri_1_rep = np.tile(u_tri[:, 0], 3)
        u_tri_2_rep = np.tile(u_tri[:, 1], 3)

        # Number of vertices
        nv = int(np.max(T)) + 1
        nv_cut = int(np.max(T_cut)) + 1

        # u = accumarray(T(:), values) / accumarray(T(:), 1) = weighted average
        u_sum = np.zeros(nv)
        u_count = np.zeros(nv)
        np.add.at(u_sum, T_flat, u_tri_1_rep)
        np.add.at(u_count, T_flat, 1)
        u = np.divide(u_sum, u_count, where=u_count > 0, out=np.zeros_like(u_sum))

        v_sum = np.zeros(nv_cut)
        v_count = np.zeros(nv_cut)
        np.add.at(v_sum, T_cut_flat, u_tri_2_rep)
        np.add.at(v_count, T_cut_flat, 1)
        v = np.divide(v_sum, v_count, where=v_count > 0, out=np.zeros_like(v_sum))

        # ut = [u(T), v(T)];
        # In MATLAB this creates (#F, 6) with [u(T(:,1)), u(T(:,2)), u(T(:,3)), v(T(:,1)), v(T(:,2)), v(T(:,3))]
        # Note: MATLAB uses v(T) which indexes v by T, but v is defined over T_cut vertices
        # This may be intentional - mapping back through the original connectivity

        # Validate index ranges to catch T/T_cut mismatches
        max_T_idx = int(np.max(T))
        if max_T_idx >= len(u):
            raise ValueError(
                f"T references vertex index {max_T_idx} but u only has {len(u)} entries. "
                f"T/T_cut mismatch detected."
            )
        if max_T_idx >= len(v):
            raise ValueError(
                f"T references vertex index {max_T_idx} but v only has {len(v)} entries "
                f"(from T_cut with {nv_cut} vertices). T/T_cut mismatch detected."
            )

        ut = np.column_stack([
            u[T[:, 0]], u[T[:, 1]], u[T[:, 2]],
            v[T[:, 0]], v[T[:, 1]], v[T[:, 2]]
        ])

    return disto, ut, theta, u_tri
