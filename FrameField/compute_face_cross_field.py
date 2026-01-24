

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigsh, lsqr
from typing import Tuple, Optional
import warnings


def regularized_solve(A, b, reg=1e-10):
    """
    Solve Ax = b with regularization fallback for singular matrices.

    MATLAB vs scipy difference:
        MATLAB's backslash operator (\\) automatically handles singular matrices
        by switching to least-squares or QR factorization. Python's spsolve
        uses strict LU decomposition that fails on singular matrices.

    This function provides MATLAB-like robustness by:
    1. Trying direct solve first (fast path)
    2. Adding small diagonal regularization if singular
    3. Falling back to least-squares (lsqr) as last resort

    Args:
        A: Sparse matrix (can be singular)
        b: Right-hand side vector
        reg: Regularization strength (default 1e-10)

    Returns:
        Solution vector x that minimizes ||Ax - b||
    """
    A_csr = A.tocsr() if not sp.isspmatrix_csr(A) else A

    # Suppress singular matrix warnings since we handle them
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Matrix is exactly singular')
        warnings.filterwarnings('ignore', 'Matrix is singular')
        x = spsolve(A_csr, b)

    if not np.any(np.isnan(x)):
        return x

    # Add regularization and retry
    n = A.shape[0]
    A_reg = A_csr + reg * sp.eye(n, format='csr')
    x = spsolve(A_reg, b)

    if not np.any(np.isnan(x)):
        return x

    # If still singular, use least-squares
    result = lsqr(A_csr, b, atol=1e-10, btol=1e-10)
    return result[0]


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]. Equivalent to MATLAB wrapToPi."""
    return np.arctan2(np.sin(x), np.cos(x))


# function [omega,ang,sing] = compute_face_cross_field(mesh, param, dec, smoothing_iter)

def compute_face_cross_field(
    mesh,
    param,
    dec,
    smoothing_iter: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smooth cross field.

    Args:
        mesh: Mesh data structure with ne, nf, nv, X
        param: Parameter structure with tri_fix, K, idx_fix_plus, Vp2V,
               ide_fix, d1d, para_trans, ide_int, E2T, ide_bound, Kt_invisible
        dec: DEC structure with star1d, star2d, d1d, star0d, d0d
        smoothing_iter: Number of heat flow smoothing iterations

    Returns:
        omega: Field rotation per edge
        ang: Field angle per face
        sing: Field singularities per vertex
    """
    # power = 4;
    power = 4

    # if ~isempty(param.tri_fix)
    #     ifcadff = true;
    # else
    #     ifcadff = false;
    # end

    # Change connection
    # If hardedge or boundary take care of acute angles
    # cf "Frame Fields for CAD models", Advances in Visual Computing, 2021
    # https://inria.hal.science/hal-03537852/
    if hasattr(param, 'tri_fix') and param.tri_fix is not None and len(param.tri_fix) > 0:
        ifcadff = True
    else:
        ifcadff = False

    omega_cadff = None

    # if ifcadff
    #     tol = pi/16;
    #
    #     id = param.K(param.idx_fix_plus) > pi/2;
    #     idx = param.Vp2V(param.idx_fix_plus(id));
    #     idxp = param.Vp2V(param.idx_fix_plus);
    #
    #     d = (mesh.vertices(idx,1)' - mesh.vertices(idxp,1)).^2 + (mesh.vertices(idx,2)' - mesh.vertices(idxp,2)).^2 + (mesh.vertices(idx,3)' - mesh.vertices(idxp,3)).^2;
    #     K_new = param.K(param.idx_fix_plus);
    #     K_new((K_new >-tol) & (K_new < tol) & any(d < 1e-3*repmat(max(d,[],1), [size(d,1),1]),2)) = 0;
    #     K_new(id) = pi/2;
    #
    #     H = dec.star1d + 1e-3*(dec.d1d'*dec.star2d*dec.d1d); H = (H' + H)/2;
    #     A = sparse(1:length(param.ide_fix), param.ide_fix, 0, length(param.ide_fix), mesh.num_edges);
    #     b = zeros(length(param.ide_fix),1);
    #     omega_cadff = quadprog(H,[], [], [], [A; param.d1d(param.idx_fix_plus,:)], [b; param.K(param.idx_fix_plus) - K_new], [], [], []);
    # end

    # Compute new connection transforming acute angles into right angle
    if ifcadff:
        tol = np.pi / 16

        # Find indices of acute corner angle
        # param.idx_fix_plus are vertex indices in the extended vertex list
        idx_fix_plus = param.idx_fix_plus
        K_at_fix = param.K[idx_fix_plus]
        id_acute = K_at_fix > np.pi / 2  # Boolean mask for acute corners

        # Get vertex positions
        idx = param.Vp2V[idx_fix_plus[id_acute]]  # Vertices with acute angles
        idxp = param.Vp2V[idx_fix_plus]  # All fixed vertices

        # Deform Gaussian curvature
        # Compute squared distances between acute vertices and all fixed vertices
        if len(idx) > 0 and len(idxp) > 0:
            # d[i,j] = squared distance from idx[j] to idxp[i]
            d = ((mesh.vertices[idx, 0].reshape(1, -1) - mesh.vertices[idxp, 0].reshape(-1, 1)) ** 2 +
                 (mesh.vertices[idx, 1].reshape(1, -1) - mesh.vertices[idxp, 1].reshape(-1, 1)) ** 2 +
                 (mesh.vertices[idx, 2].reshape(1, -1) - mesh.vertices[idxp, 2].reshape(-1, 1)) ** 2)

            K_new = K_at_fix.copy()
            # Zero out curvature for vertices that are close to acute corners and have small curvature
            # MATLAB: max(d,[],1) = column-wise max (axis=0 in numpy)
            # This gives the max distance for each acute vertex (column), then we check
            # if any vertex in idxp is within 0.1% of that max for any acute vertex
            max_d = np.max(d, axis=0, keepdims=True)
            close_mask = np.any(d < 1e-3 * max_d, axis=1)
            small_K_mask = (K_new > -tol) & (K_new < tol)
            K_new[close_mask & small_K_mask] = 0
            K_new[id_acute] = np.pi / 2
        else:
            K_new = K_at_fix.copy()
            K_new[id_acute] = np.pi / 2

        # Find new connection via quadratic programming
        # H = dec.star1d + 1e-3*(dec.d1d'*dec.star2d*dec.d1d); H = (H' + H)/2;
        H = dec.star1d + 1e-3 * (dec.d1d.T @ dec.star2d @ dec.d1d)
        H = (H + H.T) / 2

        # A = sparse(1:length(param.ide_fix), param.ide_fix, 0, length(param.ide_fix), mesh.num_edges);
        n_ide_fix = len(param.ide_fix)
        # Note: MATLAB sparse(i,j,0,...) creates a zero matrix - just used for size
        # The constraint matrix is [A; param.d1d(param.idx_fix_plus,:)]
        A_zero = sp.csr_matrix((n_ide_fix, mesh.num_edges))

        # b = zeros(length(param.ide_fix),1);
        b_zero = np.zeros(n_ide_fix)

        # Equality constraint: [A; param.d1d(param.idx_fix_plus,:)] * omega = [b; param.K(param.idx_fix_plus) - K_new]
        Aeq = sp.vstack([A_zero, param.d1d[idx_fix_plus, :]])
        beq = np.concatenate([b_zero, param.K[idx_fix_plus] - K_new])

        # Solve QP: min 0.5 * x' * H * x  s.t. Aeq * x = beq
        omega_cadff = solve_qp_equality(H, Aeq, beq)

    # I = [param.ide_int,param.ide_int];
    # J = param.E2T(param.ide_int,1:2);
    # if ifcadff
    #     rot = param.para_trans(param.ide_int) - omega_cadff(param.ide_int);
    # else
    #     rot = param.para_trans(param.ide_int);
    # end
    # S = [exp(1i*power*rot/2); -exp(-1i*power*rot/2)];
    # d0d_cplx = sparse(I, J, S, mesh.num_edges, mesh.num_faces);
    # Wcon = d0d_cplx'*dec.star1d*d0d_cplx;
    # Wcon = (Wcon + Wcon')/2;

    # Compute cross field
    # Build connection Laplacian
    ide_int = param.ide_int
    n_int = len(ide_int)

    # I = [param.ide_int, param.ide_int] - row indices (edges)
    I = np.concatenate([ide_int, ide_int])

    # J = param.E2T(param.ide_int, 1:2) - column indices (faces)
    # MATLAB E2T is 1-indexed, Python is 0-indexed
    J = param.E2T[ide_int, :2].ravel('F')  # Column-major flattening

    # Compute rotation
    if ifcadff:
        rot = param.para_trans[ide_int] - omega_cadff[ide_int]
    else:
        rot = param.para_trans[ide_int]

    # S = [exp(1i*power*rot/2); -exp(-1i*power*rot/2)];
    S = np.concatenate([np.exp(1j * power * rot / 2), -np.exp(-1j * power * rot / 2)])

    # d0d_cplx = sparse(I, J, S, mesh.num_edges, mesh.num_faces);
    # MATLAB sparse(I, J, S) accumulates duplicates by default
    d0d_cplx = sp.csr_matrix((S, (I, J)), shape=(mesh.num_edges, mesh.num_faces))

    # Wcon = d0d_cplx'*dec.star1d*d0d_cplx;
    # Wcon = (Wcon + Wcon')/2;
    Wcon = d0d_cplx.conj().T @ dec.star1d @ d0d_cplx
    Wcon = (Wcon + Wcon.conj().T) / 2

    # tri_fix = param.tri_fix;
    # z_fix = ones(length(tri_fix),1); % reference frame is algned with constraint edge by construction
    # tri_free = setdiff((1:mesh.num_faces)', tri_fix);

    # Set constraints
    tri_fix = param.tri_fix if hasattr(param, 'tri_fix') and param.tri_fix is not None else np.array([], dtype=int)
    z_fix = np.ones(len(tri_fix), dtype=complex)  # Reference frame aligned with constraint edge
    all_tri = np.arange(mesh.num_faces)
    tri_free = np.setdiff1d(all_tri, tri_fix)

    # z = zeros(mesh.num_faces,1);
    # z(tri_fix) = z_fix;
    # if ~isempty(tri_fix) % If boundaries: solve Poisson problem
    #     z(tri_free) =-Wcon(tri_free,tri_free)\(Wcon(tri_free,tri_fix)*z_fix);
    #     D = eigs(Wcon, dec.star0d, 5, 'sm');
    #     dt = 20*real(D(2));
    # else % If no boundaries: compute smallest eigenvector
    #     [P,D] = eigs(Wcon, dec.star0d, 5, 'sm');
    #     z = P(:,1);
    #     dt = 20*real(D(1,1));
    # end
    # z = z./abs(z);

    # Compute initial cross field
    z = np.zeros(mesh.num_faces, dtype=complex)
    if len(tri_fix) > 0:
        z[tri_fix] = z_fix

    if len(tri_fix) > 0:
        # If boundaries: solve Poisson problem
        # z(tri_free) = -Wcon(tri_free,tri_free) \ (Wcon(tri_free,tri_fix)*z_fix)
        Wcon_ff = Wcon[np.ix_(tri_free, tri_free)]
        Wcon_fc = Wcon[np.ix_(tri_free, tri_fix)]
        rhs = Wcon_fc @ z_fix
        z[tri_free] = -regularized_solve(Wcon_ff, rhs)

        # D = eigs(Wcon, dec.star0d, 5, 'sm');
        # dt = 20*real(D(2));
        # Generalized eigenvalue problem: Wcon * v = lambda * star0d * v
        try:
            eigenvalues = eigsh(Wcon, k=5, M=dec.star0d, sigma=0, which='LM', return_eigenvectors=False)
            eigenvalues = np.sort(np.real(eigenvalues))
            dt = 20 * eigenvalues[1] if len(eigenvalues) > 1 else 20 * eigenvalues[0]
        except Exception:
            # Fallback if eigsh fails
            dt = 0.1

    else:
        # If no boundaries: compute smallest eigenvector
        # [P,D] = eigs(Wcon, dec.star0d, 5, 'sm');
        # z = P(:,1);
        # dt = 20*real(D(1,1));
        try:
            eigenvalues, eigenvectors = eigsh(Wcon, k=5, M=dec.star0d, sigma=0, which='LM')
            # Sort by eigenvalue
            idx_sort = np.argsort(np.real(eigenvalues))
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors = eigenvectors[:, idx_sort]
            z = eigenvectors[:, 0]
            dt = 20 * np.real(eigenvalues[0])
        except Exception:
            # Fallback
            z = np.ones(mesh.num_faces, dtype=complex)
            dt = 0.1

    # z = z./abs(z);
    z = z / np.abs(z)

    # A = Wcon + dt*dec.star0d;
    # for i = 1:smoothing_iter
    #     if ~isempty(tri_fix)
    #         z(tri_free) = A(tri_free,tri_free)\(dt*dec.star0d(tri_free,tri_free)*z(tri_free) - A(tri_free,tri_fix)*z_fix);
    #     else
    #         z = A\(dt*dec.star0d*z);
    #     end
    #     z = z./abs(z);
    # end
    # assert(all(~isnan(z)), 'NaN vector field.');

    # Smoothing by heat flow
    A = Wcon + dt * dec.star0d

    for _ in range(smoothing_iter):
        if len(tri_fix) > 0:
            # z(tri_free) = A(tri_free,tri_free)\(dt*dec.star0d(tri_free,tri_free)*z(tri_free) - A(tri_free,tri_fix)*z_fix)
            A_ff = A[np.ix_(tri_free, tri_free)]
            A_fc = A[np.ix_(tri_free, tri_fix)]
            star0d_ff = dec.star0d[np.ix_(tri_free, tri_free)]
            rhs = dt * star0d_ff @ z[tri_free] - A_fc @ z_fix
            z[tri_free] = regularized_solve(A_ff, rhs)
        else:
            # z = A\(dt*dec.star0d*z)
            rhs = dt * dec.star0d @ z
            z = regularized_solve(A, rhs)
        # z = z./abs(z);
        z = z / np.abs(z)

    # assert(all(~isnan(z)), 'NaN vector field.');
    assert not np.any(np.isnan(z)), 'NaN vector field.'

    # ang = angle(z)/power;
    # if ifcadff
    #     omega = wrapToPi(power*(dec.d0d*ang + param.para_trans - omega_cadff))/power + omega_cadff;
    # else
    #     omega = wrapToPi(power*(dec.d0d*ang + param.para_trans))/power;
    # end
    # omega(param.ide_bound) = 0;
    # omega(param.ide_fix) = 0;

    # Extract angles in reference basis
    # Compute rotation
    ang = np.angle(z) / power

    if ifcadff:
        omega = wrap_to_pi(power * (dec.d0d @ ang + param.para_trans - omega_cadff)) / power + omega_cadff
    else:
        omega = wrap_to_pi(power * (dec.d0d @ ang + param.para_trans)) / power

    # Zero out boundary and fixed edges
    if hasattr(param, 'ide_bound') and len(param.ide_bound) > 0:
        omega[param.ide_bound] = 0
    if hasattr(param, 'ide_fix') and len(param.ide_fix) > 0:
        omega[param.ide_fix] = 0

    # sing = (dec.d1d*(param.para_trans - omega) + param.Kt_invisible)/(2*pi);

    # Compute singularities
    sing = (dec.d1d @ (param.para_trans - omega) + param.Kt_invisible) / (2 * np.pi)

    # ang = brush_frame_field(param, omega, tri_fix, ang(tri_fix));

    # Brush cross field
    ang_init = ang[tri_fix] if len(tri_fix) > 0 else None
    ang = brush_frame_field(param, omega, tri_fix, ang_init)

    return omega, ang, sing


def solve_qp_equality(H, Aeq, beq) -> np.ndarray:
    """
    Solve quadratic program with equality constraints.

    min 0.5 * x' * H * x
    s.t. Aeq * x = beq

    Uses KKT conditions to solve directly.

    Args:
        H: Quadratic cost matrix (positive semi-definite)
        Aeq: Equality constraint matrix
        beq: Equality constraint RHS

    Returns:
        x: Optimal solution
    """
    from scipy.sparse import vstack, hstack, csr_matrix

    # Convert to sparse if needed
    if not sp.issparse(H):
        H = sp.csr_matrix(H)
    if not sp.issparse(Aeq):
        Aeq = sp.csr_matrix(Aeq)

    n = H.shape[0]
    m = Aeq.shape[0]

    # KKT system:
    # [H   Aeq'] [x]   [0  ]
    # [Aeq  0  ] [λ] = [beq]

    zeros_mm = csr_matrix((m, m))

    KKT = vstack([
        hstack([H, Aeq.T]),
        hstack([Aeq, zeros_mm])
    ])

    rhs = np.concatenate([np.zeros(n), beq])

    # Solve KKT system
    solution = regularized_solve(KKT, rhs)

    # Extract x (first n components)
    x = solution[:n]

    return x


def brush_frame_field(
    param,
    omega: np.ndarray,
    tri_fix: np.ndarray,
    ang_init: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Propagate frame field angles via BFS.

    Args:
        param: Parameter structure with E2T, ide_int, para_trans, tri_bound
        omega: Field rotation per edge
        tri_fix: Fixed triangle indices
        ang_init: Initial angles for fixed triangles (optional)

    Returns:
        ang: Frame angle per face
    """
    # if ~exist('ang_init', 'var')
    #     ang_init = zeros(length(tri_fix),1);
    # end

    if ang_init is None:
        ang_init = np.zeros(len(tri_fix))

    # if ~isempty(tri_fix)
    #     init_tree = tri_fix(1);
    # elseif ~isempty(param.tri_bound)
    #     init_tree = param.tri_bound(1);
    # else
    #     init_tree = 1;
    # end

    # Determine starting face for BFS
    if len(tri_fix) > 0:
        init_tree = tri_fix[0]
    elif hasattr(param, 'tri_bound') and len(param.tri_bound) > 0:
        init_tree = param.tri_bound[0]
    else:
        init_tree = 0  # 0-indexed

    # nf = max(param.E2T(:));
    # ang = zeros(nf,1);
    # ang(tri_fix) = ang_init;

    nf = int(np.max(param.E2T)) + 1  # +1 for 0-indexed
    ang = np.zeros(nf)
    if len(tri_fix) > 0:
        ang[tri_fix] = ang_init

    # ang = breadth_first_search(ang, omega(param.ide_int) - param.para_trans(param.ide_int),
    #                            param.E2T(param.ide_int,:), @(x,y) x+y, init_tree);

    # BFS propagation
    omega_delta = omega[param.ide_int] - param.para_trans[param.ide_int]
    E2T_int = param.E2T[param.ide_int, :]

    ang = breadth_first_search(ang, omega_delta, E2T_int, init_tree)

    return ang


def breadth_first_search(
    x: np.ndarray,
    omega: np.ndarray,
    E2V: np.ndarray,
    init: int = 0
) -> np.ndarray:
    """
    BFS propagation of values through a graph.

    Propagates x[init] through the graph defined by edges E2V,
    updating values using x_j = x_i + sign * omega_ij

    Args:
        x: Values at vertices (modified copy returned)
        omega: Update values for each edge
        E2V: Edge-to-vertex connectivity (edge i connects E2V[i,0] to E2V[i,1])
        init: Starting vertex index

    Returns:
        y: Propagated values
    """
    from collections import deque

    # nv = max(E2V(:));
    # ne = size(E2V,1);

    nv = int(np.max(E2V)) + 1  # +1 for 0-indexed
    ne = E2V.shape[0]

    # assert(size(x,1) == nv, 'Variable has wrong dimension.');
    # assert(size(omega,1) == ne, 'Update has wrong dimension.');

    assert len(x) >= nv, 'Variable has wrong dimension.'
    assert len(omega) == ne, 'Update has wrong dimension.'

    y = x.copy()

    # Q = init;
    # S = -ones(nv,1);
    # S(Q) = 0;
    # level = zeros(nv,1);

    Q = deque([init])
    S = -np.ones(nv, dtype=int)  # Visited vertices (-1 = not visited)
    S[init] = 0

    # while ~isempty(Q)
    #     idx = Q(1);
    #     Q(1) = [];

    while len(Q) > 0:
        idx = Q.popleft()

        # id1 = find(E2V(:,1) == idx);
        # id2 = find(E2V(:,2) == idx);

        id1 = np.where(E2V[:, 0] == idx)[0]
        id2 = np.where(E2V[:, 1] == idx)[0]

        # adjedge = [id1,-ones(size(id1)); id2, ones(size(id2))];
        # adj = [E2V(id1,2)', E2V(id2,1)'];

        # Build adjacent edge info: (edge_index, sign)
        adjedge_idx = np.concatenate([id1, id2])
        adjedge_sign = np.concatenate([-np.ones(len(id1)), np.ones(len(id2))])
        adj = np.concatenate([E2V[id1, 1], E2V[id2, 0]])

        # adjedge(adj == 0,:) = [];
        # adj(adj == 0) = [];
        # In MATLAB 0 means no neighbor; in Python 0 is valid, use -1 for boundary sentinel
        # But here E2T contains face indices from ide_int (interior edges), so no -1 expected

        # for i = 1:length(adj)
        #     if S(adj(i)) == -1
        #         S(adj(i)) = idx;
        #         level(adj(i)) = l;
        #         Q = [Q; adj(i)];
        #         s = adjedge(i,2);
        #         y(adj(i),:) = fun(y(idx,:), s*omega(adjedge(i,1),:));
        #     end
        # end

        for i in range(len(adj)):
            neighbor = int(adj[i])
            if 0 <= neighbor < nv and S[neighbor] == -1:
                S[neighbor] = idx
                Q.append(neighbor)

                s = adjedge_sign[i]
                edge_idx = int(adjedge_idx[i])
                y[neighbor] = y[idx] + s * omega[edge_idx]

    return y
