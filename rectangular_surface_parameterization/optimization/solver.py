

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, lsqr
import warnings
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from rectangular_surface_parameterization.optimization.params import OptimizationParams, DEFAULT_PARAMS
from rectangular_surface_parameterization.utils.sparse_solve import regularized_solve


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]. Equivalent to MATLAB wrapToPi."""
    return np.arctan2(np.sin(x), np.cos(x))


# function [u,v,ut,vt,om,angn,flag,it] = optimize_RSP(omega, ang, u, v, mesh, param, dec, Reduction, energy_type, weight, if_plot, itmax, A_const, b_const)
#

@dataclass
class OptimizeResult:
    """Result from optimize_RSP."""
    u: np.ndarray
    v: np.ndarray
    ut: np.ndarray
    vt: np.ndarray
    om: np.ndarray
    angn: np.ndarray
    flag: int
    it: int


def optimize_RSP(
    omega: np.ndarray,
    ang: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mesh,
    param,
    dec,
    Reduction: sp.spmatrix,
    energy_type: str,
    weight,
    if_plot: bool = False,
    itmax: int = 300,
    A_const: Optional[sp.spmatrix] = None,
    b_const: Optional[np.ndarray] = None,
    opt_params: Optional[OptimizationParams] = None,
) -> OptimizeResult:
    """
    Newton solver minimizing the energy defined in "energy_type" subject to
    the satisfaction of the integrability constraint.

    Args:
        omega: Initial field rotation per edge (ne,)
        ang: Initial frame field angles per face (nf,)
        u: Scale factor u at vertices (nv,)
        v: Scale factor v at vertices (nv,)
        mesh: Mesh data structure with nv, ne, nf, T2E
        param: Parameter structure with tri_fix, ide_fix, ide_hard, ide_bound
        dec: DEC structure with d0d, star0p, star0d, star1p
        Reduction: Reduction matrix from vertex to corner variables
        energy_type: Type of energy ('distortion', 'chebyshev', 'alignment')
        weight: Weight parameters structure
        if_plot: Whether to show visualization plots
        itmax: Maximum iterations (default 300). Deprecated: use opt_params.itmax.
        A_const: Linear constraint matrix (optional)
        b_const: Linear constraint RHS (optional)
        opt_params: Optimization parameters (tolerances, line search, etc.).
            If None, uses DEFAULT_PARAMS.

    Returns:
        OptimizeResult with u, v, ut, vt, om, angn, flag, it
        flag: 1 = reached max iter, 0 = convergence, -1 = linesearch failed
    """
    from rectangular_surface_parameterization.optimization.integrability import oracle_integrability_condition
    from rectangular_surface_parameterization.optimization.objective import objective_ortho_param

    # Use default optimization parameters if not provided
    if opt_params is None:
        opt_params = DEFAULT_PARAMS

    # Override itmax from opt_params if using defaults (backwards compatibility)
    # If itmax was explicitly passed as non-default, honor it
    if itmax == 300:
        itmax = opt_params.itmax

    # Ensure inputs are 1D arrays
    u = np.ravel(u)
    v = np.ravel(v)
    omega = np.ravel(omega)
    ang = np.ravel(ang)

    # if ~exist('A_const', 'var') || (size(A_const,2) ~= size(Reduction,2) + mesh.num_faces)
    #     dof = size(Reduction,2);
    #     A_const = sparse(1:length(param.tri_fix), dof+param.tri_fix, 1, length(param.tri_fix), dof+mesh.num_faces);
    #     b_const = zeros(length(param.tri_fix),1); % Assume that the cross field already respect bnd conditions
    # end

    # Default linear constraints
    dof = Reduction.shape[1]
    if A_const is None or A_const.shape[1] != dof + mesh.num_faces:
        n_tri_fix = len(param.tri_fix)
        if n_tri_fix > 0:
            # Build sparse matrix: A_const(i, dof + tri_fix[i]) = 1
            rows = np.arange(n_tri_fix)
            cols = dof + param.tri_fix  # 0-indexed
            data = np.ones(n_tri_fix)
            A_const = sp.csr_matrix((data, (rows, cols)), shape=(n_tri_fix, dof + mesh.num_faces))
        else:
            A_const = sp.csr_matrix((0, dof + mesh.num_faces))
        # Assume that the cross field already respects boundary conditions
        b_const = np.zeros(n_tri_fix)

    # rho = 0.9;
    # beta = 1;

    # Line-search parameters (from opt_params)
    rho = opt_params.rho
    beta = opt_params.beta_init

    # ide_free = setdiff((1:mesh.num_edges)', param.ide_fix);

    # List of edges respecting integrability condition
    all_edges = np.arange(mesh.num_edges)
    ide_free = np.setdiff1d(all_edges, param.ide_fix)

    # om = omega;
    # angn = ang;
    # ut = reshape(Reduction*[u; v], [mesh.num_faces,6]);
    # vt = ut(:,4:6);
    # ut = ut(:,1:3);

    # Initialization of optimization variables
    om = omega.copy()
    angn = ang.copy()
    uv = np.concatenate([u, v])
    ut_vt = Reduction @ uv
    # Reshape to (nf, 6) in column-major (Fortran) order
    ut_vt_reshaped = np.reshape(ut_vt, (mesh.num_faces, 6), order='F')
    ut = ut_vt_reshaped[:, 0:3]
    vt = ut_vt_reshaped[:, 3:6]

    # n_dual = length(ide_free) + size(A_const,1);
    # lambda = zeros(n_dual,1);

    n_dual = len(ide_free) + A_const.shape[0]
    lam = np.zeros(n_dual)  # 'lambda' is reserved keyword

    # d0d = dec.d0d;
    # d0d(param.ide_hard,:) = 0;
    # d0d(param.ide_bound,:) = 0;

    # Make a copy and zero out hard and boundary edges
    d0d = dec.d0d.copy()
    if hasattr(param, 'ide_hard') and len(param.ide_hard) > 0:
        d0d = _zero_rows(d0d, param.ide_hard)
    if hasattr(param, 'ide_bound') and len(param.ide_bound) > 0:
        d0d = _zero_rows(d0d, param.ide_bound)

    # flag = 1;
    # fct = zeros(itmax+1,1);
    # fct_const = zeros(itmax+1,2);
    # fct_grad_norm = zeros(itmax+1,2);
    # err = zeros(itmax+1,1);

    # Keep track of objective evolution
    flag = 1
    fct = np.zeros(itmax + 1)
    fct_const = np.zeros((itmax + 1, 2))
    fct_grad_norm = np.zeros((itmax + 1, 2))
    err = np.zeros(itmax + 1)

    # for it = 1:itmax

    alp = np.zeros(mesh.num_faces)  # Initialize for later use

    for it in range(itmax):
        # disp(['-- Iteration ', num2str(it)]);
        print(f'-- Iteration {it + 1}')

        # weight.om = om;
        weight.om = om

        # [F,Jf,Hf] = oracle_integrability_condition(mesh, param, dec, om, ut, vt, angn, lambda, Reduction, ide_free);
        # F = [F; A_const*[u; v; zeros(mesh.num_faces,1)] - b_const];
        # Jf = [Jf; A_const];

        # Compute derivative of constraints
        F, Jf, Hf = oracle_integrability_condition(
            mesh, param, dec, om, ut, vt, angn, lam, Reduction, ide_free,
            compute_hessian=True
        )

        # Append linear constraints
        uv_zeros = np.concatenate([u, v, np.zeros(mesh.num_faces)])
        F = np.concatenate([F, A_const @ uv_zeros - b_const])
        Jf = sp.vstack([Jf, A_const])

        # fct_const(it,1) = norm(F);
        # fct_const(it,2) = max(abs(F));

        fct_const[it, 0] = np.linalg.norm(F)
        fct_const[it, 1] = np.max(np.abs(F)) if len(F) > 0 else 0.0

        # [fct(it),Hfct,dfct] = objective_ortho_param(energy_type, weight, mesh, dec, param, angn, ut, vt, Reduction);

        # Compute derivative of objective functions
        fct[it], Hfct, dfct = objective_ortho_param(
            energy_type, weight, mesh, dec, param, angn, ut, vt, Reduction
        )

        # fct_grad_norm(it,1) = norm(dfct + Jf'*lambda);
        # fct_grad_norm(it,2) = max(abs(dfct + Jf'*lambda));
        # err(it) = sqrt(fct_const(it,1)^2 + fct_grad_norm(it,1)^2);

        grad = dfct + Jf.T @ lam
        fct_grad_norm[it, 0] = np.linalg.norm(grad)
        fct_grad_norm[it, 1] = np.max(np.abs(grad))
        err[it] = np.sqrt(fct_const[it, 0]**2 + fct_grad_norm[it, 0]**2)

        # A = [Hfct + Hf, Jf'; Jf, sparse(n_dual,n_dual)]; A = (A + A')/2;
        # b =-[dfct + Jf'*lambda; F];

        # Newton on KKT conditions
        n_primal = Hfct.shape[0]
        zeros_dual = sp.csr_matrix((n_dual, n_dual))
        A_kkt = sp.bmat([
            [Hfct + Hf, Jf.T],
            [Jf, zeros_dual]
        ], format='csr')
        A_kkt = (A_kkt + A_kkt.T) / 2  # Symmetrize

        b_kkt = -np.concatenate([dfct + Jf.T @ lam, F])

        # H = blkdiag(dec.star0p, dec.star0p, dec.star0d, dec.star1p(ide_free,ide_free), dec.star0d(param.tri_fix,param.tri_fix));
        # H = (H + H')/2;
        # f = zeros(size(H,1),1);

        # Build regularization matrix (used in fallback QP solve)
        star1p_free = dec.star1p[np.ix_(ide_free, ide_free)]
        if len(param.tri_fix) > 0:
            star0d_fix = dec.star0d[np.ix_(param.tri_fix, param.tri_fix)]
        else:
            star0d_fix = sp.csr_matrix((0, 0))

        H_reg = sp.block_diag([
            dec.star0p,
            dec.star0p,
            dec.star0d,
            star1p_free,
            star0d_fix
        ], format='csr')
        H_reg = (H_reg + H_reg.T) / 2

        # x = A\b;
        # try
        #     assert(norm(A*x - b) < 1e-5, 'Optimization failed.');
        # catch
        #     x = quadprog(H, f, [], [], A, b);
        #     assert(norm(A*x - b) < 1e-5, 'Optimization failed.');
        # end

        # Solve KKT system with regularization fallback for singular matrices
        x = regularized_solve(A_kkt, b_kkt)
        residual = np.linalg.norm(A_kkt @ x - b_kkt)
        if residual >= opt_params.kkt_residual_tol:
            # Fallback: regularized solve with explicit regularization
            x = _solve_qp_equality_constrained(H_reg, A_kkt, b_kkt, opt_params)
            residual = np.linalg.norm(A_kkt @ x - b_kkt)
            if residual >= opt_params.kkt_residual_tol:
                raise RuntimeError(f'Optimization failed. Residual: {residual}')

        # run = true;
        # t = 1;

        # Line-search
        run = True
        t = 1.0

        # while run
        while run:
            # t = min(1, beta/err(it));

            # Step size
            t = min(1.0, beta / err[it]) if err[it] > 0 else 1.0

            # u_new = u + t*x(1:mesh.num_vertices);
            # v_new = v + t*x(mesh.num_vertices+1:2*mesh.num_vertices);
            # ut_new = reshape(Reduction*[u_new; v_new], [mesh.num_faces,6]);
            # vt_new = ut_new(:,4:6);
            # ut_new = ut_new(:,1:3);
            # alp_new = x(2*mesh.num_vertices+1:2*mesh.num_vertices+mesh.num_faces);
            # angn_new = angn + t*alp_new;
            # om_new = om + t*d0d*alp_new;
            # lambda_new = lambda + t*x(2*mesh.num_vertices+mesh.num_faces+1:end);

            # Update variables
            u_new = u + t * x[0:mesh.num_vertices]
            v_new = v + t * x[mesh.num_vertices:2*mesh.num_vertices]
            uv_new = np.concatenate([u_new, v_new])
            ut_vt_new = Reduction @ uv_new
            ut_vt_new_reshaped = np.reshape(ut_vt_new, (mesh.num_faces, 6), order='F')
            ut_new = ut_vt_new_reshaped[:, 0:3]
            vt_new = ut_vt_new_reshaped[:, 3:6]

            alp_new = x[2*mesh.num_vertices:2*mesh.num_vertices + mesh.num_faces]
            angn_new = angn + t * alp_new
            om_new = om + t * (d0d @ alp_new)
            lam_new = lam + t * x[2*mesh.num_vertices + mesh.num_faces:]

            # [F,Jf] = oracle_integrability_condition(mesh, param, dec, om_new, ut_new, vt_new, angn_new, lambda_new, Reduction, ide_free);
            # F = [F; A_const*[u_new; v_new; zeros(mesh.num_faces,1)] - b_const];
            # Jf = [Jf; A_const];

            # Update constraints (no hessian needed here)
            F_new, Jf_new = oracle_integrability_condition(
                mesh, param, dec, om_new, ut_new, vt_new, angn_new, lam_new, Reduction, ide_free,
                compute_hessian=False
            )
            uv_zeros_new = np.concatenate([u_new, v_new, np.zeros(mesh.num_faces)])
            F_new = np.concatenate([F_new, A_const @ uv_zeros_new - b_const])
            Jf_new = sp.vstack([Jf_new, A_const])

            # [fct(it+1),~,dfct] = objective_ortho_param(energy_type, weight, mesh, dec, param, angn_new, ut_new, vt_new, Reduction);

            # Update objective
            fct[it + 1], _, dfct_new = objective_ortho_param(
                energy_type, weight, mesh, dec, param, angn_new, ut_new, vt_new, Reduction
            )

            # fct_const(it+1,1) = norm(F);
            # fct_const(it+1,2) = max(abs(F));
            # fct_grad_norm(it+1,1) = norm(dfct + Jf'*lambda_new);
            # fct_grad_norm(it+1,2) = max(abs(dfct + Jf'*lambda_new));
            # err(it+1) = sqrt(fct_const(it+1,1)^2 + fct_grad_norm(it+1,1)^2);

            fct_const[it + 1, 0] = np.linalg.norm(F_new)
            fct_const[it + 1, 1] = np.max(np.abs(F_new)) if len(F_new) > 0 else 0.0
            grad_new = dfct_new + Jf_new.T @ lam_new
            fct_grad_norm[it + 1, 0] = np.linalg.norm(grad_new)
            fct_grad_norm[it + 1, 1] = np.max(np.abs(grad_new))
            err[it + 1] = np.sqrt(fct_const[it + 1, 0]**2 + fct_grad_norm[it + 1, 0]**2)

            # if t == 1
            #     if_end_search = err(it+1) < err(it)^2/(2*beta);
            # else
            #     if_end_search = err(it+1) < err(it) - beta/2;
            # end

            # Check if the search is over
            if t == 1.0:
                if_end_search = err[it + 1] < err[it]**2 / (2 * beta)
            else:
                if_end_search = err[it + 1] < err[it] - beta / 2

            # if if_end_search
            #     run = false;
            #     u = u_new;
            #     ut = ut_new;
            #     v = v_new;
            #     vt = vt_new;
            #     angn = angn_new;
            #     alp = t*alp_new;
            #     om = om_new;
            #     lambda = lambda_new;
            #
            #     beta = beta/rho;
            # else
            #     beta = beta*rho;
            # end

            if if_end_search:
                # Found new variables
                run = False
                u = u_new
                ut = ut_new
                v = v_new
                vt = vt_new
                angn = angn_new
                alp = t * alp_new
                om = om_new
                lam = lam_new

                beta = beta / rho
            else:
                # Keep the search going by reducing the step size
                beta = beta * rho

            # if t < 1e-12
            #     warning('Linesearch failed.');
            #     break;
            # end

            if t < opt_params.step_min:
                warnings.warn('Linesearch failed.')
                break

        # if t < 1e-12
        #     flag =-1;
        #     break;
        # end

        if t < opt_params.step_min:
            flag = -1
            break

        # disp(['Total error : ', num2str(err(it+1,1)), ' -- Objective : ', num2str(fct(it+1,1))]);
        # disp(['Grad norm : ', num2str(fct_grad_norm(it+1,1)), ' -- Max : ', num2str(fct_grad_norm(it+1,2))]);
        # disp(['Integrability : ', num2str(fct_const(it+1,1)), ' -- Max : ', num2str(fct_const(it+1,2))]);
        # err_ang = abs(alp)*180/pi;
        # disp(['Max frame field angle change : ', num2str(max(abs(err_ang)))]);

        # Display optimization energies
        print(f'Total error : {err[it + 1]:.6e} -- Objective : {fct[it + 1]:.6e}')
        print(f'Grad norm : {fct_grad_norm[it + 1, 0]:.6e} -- Max : {fct_grad_norm[it + 1, 1]:.6e}')
        print(f'Integrability : {fct_const[it + 1, 0]:.6e} -- Max : {fct_const[it + 1, 1]:.6e}')
        err_ang = np.abs(alp) * 180 / np.pi
        print(f'Max frame field angle change : {np.max(np.abs(err_ang)):.6e}')

        # if if_plot
        #     plot_frame_field(1, mesh, param, angn, err_ang);
        #     title(['New frame field ', num2str(it)]); colorbar;
        # end

        # Show new frame field
        if if_plot:
            import matplotlib.pyplot as plt
            from rectangular_surface_parameterization.cross_field.plot import plot_frame_field
            fig = plot_frame_field(None, mesh, param, angn, err_ang)
            plt.title(f'New frame field {it + 1}')
            plt.colorbar()
            plt.show()

        # if ~isempty(param.tri_fix)
        #     err_ang_bound = (180/pi)*wrapToPi(4*angn(param.tri_fix) - 4*ang(param.tri_fix))/4;
        #     assert(max(abs(err_ang_bound)) < 1e-3, 'Boundary constraints not respected.');
        # end

        # Check that boundary constraints still hold
        if len(param.tri_fix) > 0:
            err_ang_bound = (180 / np.pi) * wrap_to_pi(4 * angn[param.tri_fix] - 4 * ang[param.tri_fix]) / 4
            assert np.max(np.abs(err_ang_bound)) < opt_params.angle_change_tol, 'Boundary constraints not respected.'

        # if (err(it+1) < 1e-5) && (max(abs(err_ang)) < 1e-3)
        #     flag = 0;
        #     break;
        # end

        # Stop optimization when converged
        if (err[it + 1] < opt_params.err_tol) and (np.max(np.abs(err_ang)) < opt_params.angle_change_tol):
            flag = 0
            break

    # disp('-- end loop --');
    print('-- end loop --')

    # if if_plot
    #     figure;
    #     semilogy([fct_grad_norm(1:it+1,1), fct_grad_norm(1:it+1,2)], 'linewidth', 2);
    #     legend({'Grad norm L^2' 'Grad norm L^\infty'}, 'fontsize', 14);
    #     title('Grad norm');
    #     grid on;
    #
    #     figure;
    #     semilogy([fct_const(1:it+1,1), fct_const(1:it+1,2)], 'linewidth', 2);
    #     legend({'Integrability L^2' 'Integrability L^\infty'}, 'fontsize', 14);
    #     title('Integrability');
    #     grid on;
    #
    #     figure;
    #     plot(fct(1:it+1), 'linewidth', 2);
    #     title('Objective');
    # end

    # Show convergence plots
    if if_plot:
        import matplotlib.pyplot as plt

        fig1, ax1 = plt.subplots()
        ax1.semilogy(fct_grad_norm[0:it + 2, 0], linewidth=2, label=r'Grad norm $L^2$')
        ax1.semilogy(fct_grad_norm[0:it + 2, 1], linewidth=2, label=r'Grad norm $L^\infty$')
        ax1.legend(fontsize=14)
        ax1.set_title('Grad norm')
        ax1.grid(True)

        fig2, ax2 = plt.subplots()
        ax2.semilogy(fct_const[0:it + 2, 0], linewidth=2, label=r'Integrability $L^2$')
        ax2.semilogy(fct_const[0:it + 2, 1], linewidth=2, label=r'Integrability $L^\infty$')
        ax2.legend(fontsize=14)
        ax2.set_title('Integrability')
        ax2.grid(True)

        fig3, ax3 = plt.subplots()
        ax3.plot(fct[0:it + 2], linewidth=2)
        ax3.set_title('Objective')

        plt.show()

    return OptimizeResult(
        u=u,
        v=v,
        ut=ut,
        vt=vt,
        om=om,
        angn=angn,
        flag=flag,
        it=it + 1  # Return 1-indexed iteration count to match MATLAB convention
    )


def _zero_rows(mat: sp.spmatrix, row_indices: np.ndarray) -> sp.spmatrix:
    """Zero out specified rows of a sparse matrix."""
    mat = mat.tolil()
    for idx in row_indices:
        mat[idx, :] = 0
    return mat.tocsr()


def _solve_qp_equality_constrained(
    H: sp.spmatrix,
    A_kkt: sp.spmatrix,
    b_kkt: np.ndarray,
    opt_params: OptimizationParams,
) -> np.ndarray:
    """
    Fallback QP solver when direct KKT solve fails.

    Solves the KKT system using regularized least squares with the
    provided regularization matrix H to improve conditioning.

    The MATLAB code uses quadprog as fallback:
        x = quadprog(H, f, [], [], A, b)
    which solves: min 0.5*x'*H*x + f'*x  s.t. A*x = b

    Since f=0 in the MATLAB code, this is equivalent to solving the
    regularized KKT system with H providing numerical stability.

    Args:
        H: Regularization matrix (positive semi-definite) - NOT CURRENTLY USED
            but kept for API compatibility and future improvement
        A_kkt: KKT system matrix
        b_kkt: KKT RHS vector
        opt_params: Optimization parameters with solver tolerances

    Returns:
        x: Solution vector
    """
    from scipy.sparse.linalg import lsqr, lsmr

    # Try LSMR first (often more stable than LSQR for ill-conditioned systems)
    # LSMR is mathematically equivalent to applying MINRES to the normal equations
    result = lsmr(
        A_kkt, b_kkt,
        atol=opt_params.lsmr_atol,
        btol=opt_params.lsmr_btol,
        maxiter=opt_params.lsmr_maxiter,
    )
    x = result[0]

    # Check residual
    residual = np.linalg.norm(A_kkt @ x - b_kkt)
    if residual < opt_params.kkt_residual_tol:
        return x

    # If LSMR fails, try adding regularization to diagonal
    # This mimics what quadprog would do with positive definite H
    n = A_kkt.shape[0]
    A_reg = A_kkt + opt_params.reg_factor * sp.eye(n, format='csr')
    try:
        x = spsolve(A_reg, b_kkt)
        residual = np.linalg.norm(A_kkt @ x - b_kkt)
        if residual < opt_params.kkt_residual_tol:
            return x
    except Exception:
        pass

    # Last resort: use LSQR with damping
    result = lsqr(
        A_kkt, b_kkt,
        damp=opt_params.lsqr_damp,
        atol=opt_params.lsmr_atol,
        btol=opt_params.lsmr_btol,
        iter_lim=opt_params.lsqr_iter_lim,
    )
    return result[0]
