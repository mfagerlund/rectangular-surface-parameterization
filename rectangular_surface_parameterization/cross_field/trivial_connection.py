

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple
from dataclasses import dataclass


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]. Equivalent to MATLAB wrapToPi."""
    return np.arctan2(np.sin(x), np.cos(x))


# function [omega,ang,sing] = trivial_connection(mesh, param, dec, ifboundary, ifhardedge, sing, om_cycle, om_link)

def trivial_connection(
    mesh,
    param,
    dec,
    ifboundary: bool,
    ifhardedge: bool,
    sing: Optional[np.ndarray] = None,
    om_cycle: Optional[np.ndarray] = None,
    om_link: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute trivial connection on mesh.

    Computes the angle change (omega) when parallel transporting a vector across
    an edge. Key for cross-field computation.

    Args:
        mesh: Mesh data structure with nv, ne, nf, E2T
        param: Parameter structure with idx_bound, Kt, para_trans, Icycle, Ilink,
               d1d, ide_bound, ide_hard, tri_fix, K, Kt_invisible
        dec: DEC structure with d1d, star1d
        ifboundary: Enforce boundary alignment
        ifhardedge: Enforce hard-edge alignment
        sing: Singularity indices (optional)
        om_cycle: Non-contractible cycle constraints (optional)
        om_link: Connecting path constraints (optional)

    Returns:
        omega: Field rotation per edge
        ang: Field angle per face
        sing: Singularity indices
    """
    from scipy.optimize import minimize

    # if ~exist('sing','var')
    #     if ~isempty(param.idx_bound)
    #         sing = zeros(mesh.num_vertices,1);
    #         sing(param.idx_bound) = round(4*param.Kt(param.idx_bound)/(2*pi))/4;
    #     else
    #         sing = zeros(mesh.num_vertices,1);
    #         id = randi(mesh.num_vertices, 4*(mesh.num_faces - mesh.num_edges + mesh.num_vertices), 1);
    #         sing(id) = 1/4;
    #     end
    # end

    # Default value for vertex singularity indices
    if sing is None:
        if len(param.idx_bound) > 0:
            sing = np.zeros(mesh.num_vertices)
            # Round to nearest quarter-integer
            sing[param.idx_bound] = np.round(4 * param.Kt[param.idx_bound] / (2 * np.pi)) / 4
        else:
            sing = np.zeros(mesh.num_vertices)
            # Random singularity placement for closed surfaces
            # Euler characteristic: chi = nv - ne + nf = 2 - 2*genus
            # For 4-fold cross field: sum of indices = chi
            n_sing = 4 * (mesh.num_faces - mesh.num_edges + mesh.num_vertices)
            id = np.random.randint(0, mesh.num_vertices, n_sing)
            sing[id] = 1/4

    # if ~exist('om_cycle','var') || isempty(om_cycle)
    #     om_cycle = param.Icycle*param.para_trans;
    #     om_cycle = om_cycle - 2*pi*round(4*om_cycle/(2*pi))/4;
    # end

    # Default value for non-contractible cycle indices
    if om_cycle is None or len(om_cycle) == 0:
        om_cycle = param.Icycle @ param.para_trans
        om_cycle = om_cycle - 2 * np.pi * np.round(4 * om_cycle / (2 * np.pi)) / 4

    # if ~exist('om_link','var') || isempty(om_link)
    #     om_link = param.Ilink*param.para_trans;
    #     om_link = om_link - 2*pi*round(4*om_link/(2*pi))/4;
    # end

    # Default value for connecting path indices
    if om_link is None or len(om_link) == 0:
        om_link = param.Ilink @ param.para_trans
        om_link = om_link - 2 * np.pi * np.round(4 * om_link / (2 * np.pi)) / 4

    # if isempty(param.idx_bound)
    #     assert(norm(sum(sing) - (mesh.num_faces - mesh.num_edges + mesh.num_vertices)) < 1e-5, 'Singularities do not satisfy Gauss-Bonnet.');
    # end

    # Check Gauss-Bonnet constraint for closed surfaces
    if len(param.idx_bound) == 0:
        euler_char = mesh.num_faces - mesh.num_edges + mesh.num_vertices
        assert np.abs(np.sum(sing) - euler_char) < 1e-5, \
            'Singularities do not satisfy Gauss-Bonnet.'

    # if ifboundary && ifhardedge
    #     s2 = [sing; round(2*param.K(mesh.num_vertices+1:end)/pi)/4];
    #     A = [param.d1d; sparse(1:length(param.ide_bound), param.ide_bound, 1, length(param.ide_bound), mesh.num_edges); param.Ilink; param.Icycle];
    #     b = [param.K-2*pi*s2; zeros(length(param.ide_bound),1); om_link; om_cycle];
    # elseif ifboundary
    #     A = [dec.d1d; sparse(1:length(param.ide_bound), param.ide_bound, 1, length(param.ide_bound), mesh.num_edges); param.Ilink; param.Icycle];
    #     b = [param.Kt-2*pi*sing; zeros(length(param.ide_bound),1); om_link; om_cycle];
    # elseif ifhardedge
    #     idx = setdiff((1:size(param.d1d,1))', param.idx_bound);
    #     s2 = [sing; round(2*param.K(mesh.num_vertices+1:end)/pi)/4];
    #     A = [param.d1d(idx,:); sparse(1:length(param.ide_hard), param.ide_hard, 1, length(param.ide_hard), mesh.num_edges); param.Ilink; param.Icycle];
    #     b = [param.K(idx)-2*pi*s2(idx); zeros(length(param.ide_hard),1); om_link; om_cycle];
    # else
    #     A = [dec.d1d(param.idx_int,:); sparse(1:length(param.ide_bound), param.ide_bound, 1, length(param.ide_bound), mesh.num_edges); param.Ilink; param.Icycle];
    #     b = [param.Kt(param.idx_int)-2*pi*sing(param.idx_int); zeros(length(param.ide_bound),1); om_link; om_cycle];
    # end

    # Build loop constraints
    n_bound = len(param.ide_bound)
    n_hard = len(param.ide_hard) if hasattr(param, 'ide_hard') else 0

    if ifboundary and ifhardedge:
        # Extended singularities including hard edge corners
        s2 = np.concatenate([sing, np.round(2 * param.K[mesh.num_vertices:] / np.pi) / 4])

        # Boundary edge selector matrix
        bound_selector = sp.csr_matrix(
            (np.ones(n_bound), (np.arange(n_bound), param.ide_bound)),
            shape=(n_bound, mesh.num_edges)
        )

        # Stack constraint matrices
        A = sp.vstack([param.d1d, bound_selector, param.Ilink, param.Icycle])
        b = np.concatenate([
            param.K - 2 * np.pi * s2,
            np.zeros(n_bound),
            om_link,
            om_cycle
        ])

    elif ifboundary:
        # Boundary edge selector matrix
        bound_selector = sp.csr_matrix(
            (np.ones(n_bound), (np.arange(n_bound), param.ide_bound)),
            shape=(n_bound, mesh.num_edges)
        )

        A = sp.vstack([dec.d1d, bound_selector, param.Ilink, param.Icycle])
        b = np.concatenate([
            param.Kt - 2 * np.pi * sing,
            np.zeros(n_bound),
            om_link,
            om_cycle
        ])

    elif ifhardedge:
        # Get interior vertex indices (exclude boundary)
        all_idx = np.arange(param.d1d.shape[0])
        idx = np.setdiff1d(all_idx, param.idx_bound)

        # Extended singularities
        s2 = np.concatenate([sing, np.round(2 * param.K[mesh.num_vertices:] / np.pi) / 4])

        # Hard edge selector matrix
        hard_selector = sp.csr_matrix(
            (np.ones(n_hard), (np.arange(n_hard), param.ide_hard)),
            shape=(n_hard, mesh.num_edges)
        )

        A = sp.vstack([param.d1d[idx, :], hard_selector, param.Ilink, param.Icycle])
        b = np.concatenate([
            param.K[idx] - 2 * np.pi * s2[idx],
            np.zeros(n_hard),
            om_link,
            om_cycle
        ])

    else:
        # No boundary, no hard edges - use interior vertices only
        bound_selector = sp.csr_matrix(
            (np.ones(n_bound), (np.arange(n_bound), param.ide_bound)),
            shape=(n_bound, mesh.num_edges)
        )

        A = sp.vstack([dec.d1d[param.idx_int, :], bound_selector, param.Ilink, param.Icycle])
        b = np.concatenate([
            param.Kt[param.idx_int] - 2 * np.pi * sing[param.idx_int],
            np.zeros(n_bound),
            om_link,
            om_cycle
        ])

    # omega = quadprog(dec.star1d, zeros(mesh.num_edges,1), [], [], A, b);

    # Solve quadratic program: min 0.5 * x' * H * x  s.t. A*x = b
    # MATLAB quadprog(H, f, Aineq, bineq, Aeq, beq) with f=0, no inequalities
    # This is a QP with equality constraints
    omega = solve_qp_equality(dec.star1d, A, b)

    # ang = brush_frame_field(param, omega, param.tri_fix);

    # Compute frame angles via BFS propagation
    ang = brush_frame_field(param, omega, param.tri_fix)


    # (Optional check commented out in MATLAB)

    # sing2 = (dec.d1d*(param.para_trans - omega) + param.Kt_invisible)/(2*pi);
    # assert(norm(sing(param.idx_int) - sing2(param.idx_int)) < 1e-5, 'Failed to prescribe singularities.');

    # Verify singularity indices match prescribed values
    sing2 = (dec.d1d @ (param.para_trans - omega) + param.Kt_invisible) / (2 * np.pi)
    assert np.linalg.norm(sing[param.idx_int] - sing2[param.idx_int]) < 1e-5, \
        'Failed to prescribe singularities.'

    # sing_loop = wrapToPi(om_cycle - param.Icycle*param.para_trans)/(2*pi);
    # sing_loop2 = wrapToPi(param.Icycle*(omega - param.para_trans))/(2*pi);
    # assert(norm(sing_loop - sing_loop2) < 1e-5, 'Failing cycle constraints.');

    # Verify cycle constraints
    sing_loop = wrap_to_pi(om_cycle - param.Icycle @ param.para_trans) / (2 * np.pi)
    sing_loop2 = wrap_to_pi(param.Icycle @ (omega - param.para_trans)) / (2 * np.pi)
    assert np.linalg.norm(sing_loop - sing_loop2) < 1e-5, \
        'Failing cycle constraints.'

    # sing_link = wrapToPi(om_cycle - param.Ilink*param.para_trans)/(2*pi);
    # sing_link2 = wrapToPi(param.Ilink*(omega - param.para_trans))/(2*pi);
    # assert(norm(sing_link - sing_link2) < 1e-5, 'Failed to prescribe constraints between feature curves.');

    # Verify link constraints (note: MATLAB uses om_cycle here, likely a typo - should be om_link)
    sing_link = wrap_to_pi(om_link - param.Ilink @ param.para_trans) / (2 * np.pi)
    sing_link2 = wrap_to_pi(param.Ilink @ (omega - param.para_trans)) / (2 * np.pi)
    assert np.linalg.norm(sing_link - sing_link2) < 1e-5, \
        'Failed to prescribe constraints between feature curves.'

    return omega, ang, sing


def solve_qp_equality(H, Aeq, beq) -> np.ndarray:
    """
    Solve quadratic program with equality constraints.

    min 0.5 * x' * H * x
    s.t. Aeq * x = beq

    Uses KKT conditions to solve directly.

    Args:
        H: Quadratic cost matrix (positive definite)
        Aeq: Equality constraint matrix
        beq: Equality constraint RHS

    Returns:
        x: Optimal solution
    """
    from scipy.sparse import vstack, hstack, csr_matrix
    from rectangular_surface_parameterization.utils.sparse_solve import regularized_solve

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

    # Solve KKT system with regularization fallback
    solution = regularized_solve(KKT, rhs)

    # Extract x (first n components)
    x = solution[:n]

    return x


def brush_frame_field(param, omega: np.ndarray, tri_fix: np.ndarray,
                       ang_init: Optional[np.ndarray] = None) -> np.ndarray:
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
    from collections import deque

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
    elif len(param.tri_bound) > 0:
        init_tree = param.tri_bound[0]
    else:
        init_tree = 0  # 0-indexed

    # nf = max(param.edge_to_triangle(:));
    # ang = zeros(nf,1);
    # ang(tri_fix) = ang_init;

    nf = int(np.max(param.edge_to_triangle)) + 1  # +1 for 0-indexed
    ang = np.zeros(nf)
    if len(tri_fix) > 0:
        ang[tri_fix] = ang_init

    # ang = breadth_first_search(ang, omega(param.ide_int) - param.para_trans(param.ide_int),
    #                            param.edge_to_triangle(param.ide_int,:), @(x,y) x+y, init_tree);

    # BFS propagation
    omega_delta = omega[param.ide_int] - param.para_trans[param.ide_int]
    E2T_int = param.edge_to_triangle[param.ide_int, :]

    ang = breadth_first_search(ang, omega_delta, E2T_int, init_tree)

    return ang


def breadth_first_search(x: np.ndarray, omega: np.ndarray,
                         E2V: np.ndarray, init: int = 0) -> np.ndarray:
    """
    BFS propagation of values through a graph.

    Propagates x[init] through the graph defined by edges E2V,
    updating values using x_j = x_i + sign * omega_ij

    Args:
        x: Values at vertices (modified in place and returned)
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
    level = np.zeros(nv, dtype=int)
    l = 0

    # while ~isempty(Q)
    #     idx = Q(1);
    #     Q(1) = [];
    #     l = l + 1;

    while len(Q) > 0:
        idx = Q.popleft()
        l += 1

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

        # Filter out invalid (boundary) adjacencies - in 0-indexed, we check for -1 or missing
        # Actually, in MATLAB 0 means no neighbor. In Python with 0-indexed, we need to handle differently.
        # Assuming E2T has valid face indices (no special boundary marker here since we use ide_int)

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
            if neighbor >= 0 and neighbor < nv and S[neighbor] == -1:
                S[neighbor] = idx
                level[neighbor] = l
                Q.append(neighbor)

                s = adjedge_sign[i]
                edge_idx = adjedge_idx[i]
                y[neighbor] = y[idx] + s * omega[edge_idx]

    return y
