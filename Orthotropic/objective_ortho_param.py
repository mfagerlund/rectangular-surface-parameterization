

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Tuple


def objective_ortho_param(
    energy_type: str,
    weight,
    mesh,
    dec,
    param,
    angn: np.ndarray,
    ut: np.ndarray,
    vt: np.ndarray,
    Reduction: sp.spmatrix
) -> Tuple[float, sp.spmatrix, np.ndarray]:
    """
    Compute objective function, Hessian, and gradient for orthotropic parameterization.

    Args:
        energy_type: Type of energy ('distortion', 'chebyshev', or 'alignment')
        weight: Weight parameters with fields:
            - w_conf_ar: conformal/area weight for 'distortion'
            - w_gradv: gradient weight (optional)
            - ang_dir, aspect_ratio, w_ratio, w_ang: for 'alignment'
        mesh: Mesh data structure with nv, nf, area
        dec: DEC structure with W_tri, star0p_tri, star0d
        param: Parameter structure with tri_fix
        angn: Frame angle per face (nf,)
        ut: Scale factors u at corners (nf x 3)
        vt: Scale factors v at corners (nf x 3)
        Reduction: Reduction matrix (6*nf x 2*nv)

    Returns:
        fct: Objective function value
        H: Hessian matrix (2*nv + nf x 2*nv + nf)
        df: Gradient vector (2*nv + nf,)
    """
    # function [fct,H,df] = objective_ortho_param(energy_type, weight, mesh, dec, param, angn, ut, vt, Reduction)

    # Wt = dec.W_tri;
    # AeT = dec.star0p_tri;
    # Iso = blkdiag(AeT, AeT);
    # Conf = blkdiag(0*AeT, AeT);
    # Ar = blkdiag(AeT, 0*AeT);
    # integral_tri = ones(3,1)/3;
    Wt = dec.W_tri
    AeT = dec.star0p_tri
    zero_AeT = sp.csr_matrix(AeT.shape)

    # Block diagonal matrices (not always needed, built on demand)
    # Iso = blkdiag(AeT, AeT)  # Isometric energy weight
    # Conf = blkdiag(0*AeT, AeT)  # Conformal energy weight
    # Ar = blkdiag(AeT, 0*AeT)  # Area energy weight
    integral_tri = np.ones(3) / 3

    nv = mesh.num_vertices
    nf = mesh.num_faces

    # H = sparse(2*mesh.num_vertices+mesh.num_faces,2*mesh.num_vertices+mesh.num_faces);
    # Initialize H (will be built at the end)

    # if strcmp(energy_type, 'distortion')
    if energy_type == 'distortion':
        # H_u = (1 - weight.w_conf_ar)*AeT;
        # H_v =       weight.w_conf_ar*AeT;
        # H_ang = sparse(mesh.num_faces, mesh.num_faces);
        H_u = (1 - weight.w_conf_ar) * AeT
        H_v = weight.w_conf_ar * AeT
        H_ang = sp.csr_matrix((nf, nf))

        # H_uv = blkdiag(H_u, H_v);
        H_uv = sp.block_diag([H_u, H_v])

        # df = [Reduction'*[H_u*ut(:); H_v*vt(:)]; H_ang*angn];
        # ut(:) and vt(:) flatten in column-major order
        ut_flat = ut.ravel('F')
        vt_flat = vt.ravel('F')

        Hu_ut = H_u @ ut_flat
        Hv_vt = H_v @ vt_flat
        df_uv = Reduction.T @ np.concatenate([Hu_ut, Hv_vt])
        df_ang = H_ang @ angn
        df = np.concatenate([df_uv, df_ang])

        # fct = [ut(:); vt(:)]'*[H_u*ut(:); H_v*vt(:)];
        uv_flat = np.concatenate([ut_flat, vt_flat])
        Huv_uv = np.concatenate([Hu_ut, Hv_vt])
        fct = uv_flat @ Huv_uv

    # elseif strcmp(energy_type, 'chebyshev')
    elif energy_type == 'chebyshev':
        # err_diag = log(exp(- 2*ut*integral_tri - 2*vt*integral_tri)/2 + exp(- 2*ut*integral_tri + 2*vt*integral_tri)/2);

        # ut @ integral_tri: per-face average of ut corners
        # ut is (nf x 3), integral_tri is (3,)
        ut_avg = ut @ integral_tri  # (nf,)
        vt_avg = vt @ integral_tri  # (nf,)

        err_diag = np.log(
            np.exp(-2 * ut_avg - 2 * vt_avg) / 2 +
            np.exp(-2 * ut_avg + 2 * vt_avg) / 2
        )

        # da = -2*ones(mesh.num_faces,1)/3;
        # db = (2/3)*(exp(4*vt*integral_tri) - 1)./(exp(4*vt*integral_tri) + 1);
        da = -2 * np.ones(nf) / 3
        exp_4vt = np.exp(4 * vt_avg)
        db = (2 / 3) * (exp_4vt - 1) / (exp_4vt + 1)

        # I = repmat((1:mesh.num_faces)', [1,3]);
        # J = reshape((1:3*mesh.num_faces)', [mesh.num_faces,3]);
        # Da = sparse(I, J, [da,da,da], mesh.num_faces, 3*mesh.num_faces);
        # Db = sparse(I, J, [db,db,db], mesh.num_faces, 3*mesh.num_faces);

        # Build sparse Da and Db matrices
        # Each row i has entries at columns corresponding to corners of face i
        # Column-major corner indexing: corner j of face i is at j*nf + i
        face_idx = np.arange(nf)
        corner_cols = np.arange(3 * nf).reshape((nf, 3), order='F')

        I_idx = np.tile(face_idx.reshape(-1, 1), (1, 3)).ravel()
        J_idx = corner_cols.ravel()

        Da_vals = np.tile(da.reshape(-1, 1), (1, 3)).ravel()
        Db_vals = np.tile(db.reshape(-1, 1), (1, 3)).ravel()

        Da = sp.coo_matrix((Da_vals, (I_idx, J_idx)), shape=(nf, 3 * nf)).tocsr()
        Db = sp.coo_matrix((Db_vals, (I_idx, J_idx)), shape=(nf, 3 * nf)).tocsr()

        # daa = zeros(mesh.num_faces,1);
        # dab = zeros(mesh.num_faces,1);
        # dbb = (16/9)*exp(4*vt*integral_tri)./(exp(4*vt*integral_tri) + 1).^2;
        daa = np.zeros(nf)
        dab = np.zeros(nf)
        dbb = (16 / 9) * exp_4vt / (exp_4vt + 1) ** 2

        # Haa = sparse([I,mesh.num_faces+I,2*mesh.num_faces+I], [J,J,J], repmat(daa.*err_diag.*mesh.area, [1,9]), 3*mesh.num_faces, 3*mesh.num_faces);
        # Hab = sparse([I,mesh.num_faces+I,2*mesh.num_faces+I], [J,J,J], repmat(dab.*err_diag.*mesh.area, [1,9]), 3*mesh.num_faces, 3*mesh.num_faces);
        # Hbb = sparse([I,mesh.num_faces+I,2*mesh.num_faces+I], [J,J,J], repmat(dbb.*err_diag.*mesh.area, [1,9]), 3*mesh.num_faces, 3*mesh.num_faces);

        # Build Haa, Hab, Hbb matrices (3*nf x 3*nf)
        # Row indices: [I, nf+I, 2*nf+I] for each of 3 corner columns -> (nf x 9) entries
        # But actually MATLAB code has [I, nf+I, 2*nf+I] as row indices
        # and [J, J, J] meaning corner columns repeated 3 times

        # Each face contributes a 3x3 block (one for each corner pair)
        # This creates a diagonal block structure

        def build_second_order_matrix(d_vals):
            """Build 3*nf x 3*nf second-order matrix."""
            weighted_d = d_vals * err_diag * mesh.area

            # For each face f, we have a 3x3 block with constant value weighted_d[f]
            # Block (i,j) of face f: row = i*nf + f, col = j*nf + f
            I_full = []
            J_full = []
            S_full = []

            for i in range(3):
                for j in range(3):
                    I_full.append(i * nf + face_idx)
                    J_full.append(j * nf + face_idx)
                    S_full.append(weighted_d)

            I_full = np.concatenate(I_full)
            J_full = np.concatenate(J_full)
            S_full = np.concatenate(S_full)

            return sp.coo_matrix((S_full, (I_full, J_full)), shape=(3 * nf, 3 * nf)).tocsr()

        Haa = build_second_order_matrix(daa)
        Hab = build_second_order_matrix(dab)
        Hbb = build_second_order_matrix(dbb)

        # H_uv = [Da, Db]'*dec.star0d*[Da, Db] + [Haa, Hab; Hab', Hbb];
        # [Da, Db] is (nf x 6*nf), [Da, Db]' @ star0d @ [Da, Db] is (6*nf x 6*nf)
        Da_Db = sp.hstack([Da, Db])  # (nf x 6*nf)
        H_uv_1 = Da_Db.T @ dec.star0d @ Da_Db  # (6*nf x 6*nf)

        # [Haa, Hab; Hab', Hbb] - block matrix
        H_uv_2 = sp.bmat([[Haa, Hab], [Hab.T, Hbb]])

        H_uv = H_uv_1 + H_uv_2

        # H_ang = sparse(mesh.num_faces, mesh.num_faces);
        H_ang = sp.csr_matrix((nf, nf))

        # df = [Reduction'*([Da, Db]'*dec.star0d*err_diag); zeros(mesh.num_faces,1)];
        df_uv = Reduction.T @ (Da_Db.T @ dec.star0d @ err_diag)
        df_ang = np.zeros(nf)
        df = np.concatenate([df_uv, df_ang])

        # fct = err_diag'*dec.star0d*err_diag;
        fct = err_diag @ dec.star0d @ err_diag

    # elseif strcmp(energy_type, 'alignment')
    elif energy_type == 'alignment':
        # ang_dir = weight.ang_dir;
        ang_dir = weight.ang_dir

        # log_aspect_ratio = log(weight.aspect_ratio)/2;
        # diff = [ut(:); vt(:)] - repmat(log_aspect_ratio, [6,1]);
        # H_uv = weight.w_ratio*Conf;
        # df_uv = Reduction'*H_uv*diff;
        # fct = diff'*H_uv*diff;

        ut_flat = ut.ravel('F')
        vt_flat = vt.ravel('F')
        uv_flat = np.concatenate([ut_flat, vt_flat])

        # Handle both scalar and per-face aspect_ratio
        # MATLAB: repmat(log_aspect_ratio, [6,1]) broadcasts scalar to all 6*nf elements
        # If aspect_ratio is per-face (nf,), we need to tile it to match [ut(:); vt(:)]
        aspect_ratio = np.atleast_1d(weight.aspect_ratio)
        log_aspect_ratio = np.log(aspect_ratio) / 2

        if log_aspect_ratio.size == 1:
            # Scalar case: broadcast to all elements
            diff = uv_flat - log_aspect_ratio[0]
        elif log_aspect_ratio.size == nf:
            # Per-face case: tile to match corner structure
            # ut_flat has 3*nf elements (nf faces x 3 corners, column-major)
            # vt_flat has 3*nf elements
            # Need to repeat each face's value 3 times for its corners
            # Column-major: corners are at indices [f, f+nf, f+2*nf] for face f
            log_ar_tiled = np.tile(log_aspect_ratio, 6)  # [ar0..ar_{nf-1}] x 6
            diff = uv_flat - log_ar_tiled
        else:
            raise ValueError(
                f"aspect_ratio must be scalar or (nf,) array, got shape {aspect_ratio.shape}"
            )

        # Conf = blkdiag(0*AeT, AeT)
        Conf = sp.block_diag([zero_AeT, AeT])
        H_uv = weight.w_ratio * Conf

        df_uv = Reduction.T @ (H_uv @ diff)
        fct = diff @ (H_uv @ diff)

        # H_ang = weight.w_ang*dec.star0d;
        # df_ang = H_ang*(angn - ang_dir);
        # fct = fct + (angn - ang_dir)'*(H_ang*(angn - ang_dir));
        H_ang = weight.w_ang * dec.star0d
        ang_diff = angn - ang_dir
        df_ang = H_ang @ ang_diff
        fct = fct + ang_diff @ (H_ang @ ang_diff)

        # df = [df_uv; df_ang];
        df = np.concatenate([df_uv, df_ang])

    # else
    #     error('This energy does not exist.');
    else:
        raise ValueError(f"Energy type '{energy_type}' does not exist.")

    # if isfield(weight,'w_gradv') && (weight.w_gradv > 0)
    #     H_uv = H_uv + weight.w_gradv*blkdiag(0*Wt, Wt);
    #     fct = fct + weight.w_gradv*vt(:)'*Wt*vt(:);
    #     df = df + [weight.w_gradv*Reduction'*blkdiag(0*Wt, Wt)*[ut(:); vt(:)]; zeros(mesh.num_faces,1)];
    # end
    if hasattr(weight, 'w_gradv') and weight.w_gradv > 0:
        zero_Wt = sp.csr_matrix(Wt.shape)
        gradv_block = sp.block_diag([zero_Wt, Wt])

        H_uv = H_uv + weight.w_gradv * gradv_block

        ut_flat = ut.ravel('F')
        vt_flat = vt.ravel('F')
        uv_flat = np.concatenate([ut_flat, vt_flat])

        fct = fct + weight.w_gradv * (vt_flat @ (Wt @ vt_flat))

        df_gradv_uv = weight.w_gradv * Reduction.T @ (gradv_block @ uv_flat)
        df_gradv_ang = np.zeros(nf)
        df = df + np.concatenate([df_gradv_uv, df_gradv_ang])

    # H = blkdiag(Reduction'*H_uv*Reduction, H_ang);
    # H = (H + H')/2;
    H_reduced = Reduction.T @ H_uv @ Reduction
    H = sp.block_diag([H_reduced, H_ang])
    H = (H + H.T) / 2

    return fct, H, df
