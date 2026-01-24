

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
from collections import deque
from typing import Optional, Tuple

# function [ang,init_tree] = brush_frame_field(param, omega, tri_fix, ang_init)
#
# if ~exist('ang_init', 'var')
#     ang_init = zeros(length(tri_fix),1);
# end
#
# if ~isempty(tri_fix)
#     init_tree = tri_fix(1);
# elseif ~isempty(param.tri_bound)
#     init_tree = param.tri_bound(1);
# else
#     init_tree = 1;
# end
#
# nf = max(param.edge_to_triangle(:));
# ang = zeros(nf,1);
# ang(tri_fix) = ang_init;
# ang = breadth_first_search(ang, omega(param.ide_int) - param.para_trans(param.ide_int), param.edge_to_triangle(param.ide_int,:), @(x,y) x+y, init_tree);
#
# end


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


# function [y,S,level] = breadth_first_search(x, omega, E2V, fun, init)
#
# if ~exist('init', 'var')
#     init = 1;
# end
#
# nv = max(E2V(:));
# ne = size(E2V,1);
#
# assert(size(x,1) == nv, 'Variable has wrong dimension.');
# assert(size(omega,1) == ne, 'Update has wrong dimension.');
# y = x;
#
# Q = init;
# S = -ones(nv,1); % Visited vertices
# S(Q) = 0;
# level = zeros(nv,1);
# l = 0;
# while ~isempty(Q)
#     idx = Q(1);
#     Q(1) = [];
#     l = l + 1;
#
#     id1 = find(E2V(:,1) == idx);
#     id2 = find(E2V(:,2) == idx);
#     adjedge = [id1,-ones(size(id1)); id2, ones(size(id2))];
#     adj = [E2V(id1,2)', E2V(id2,1)'];
#     adjedge(adj == 0,:) = [];
#     adj(adj == 0) = [];
#     for i = 1:length(adj)
#         if S(adj(i)) == -1
#             S(adj(i)) = idx;
#             level(adj(i)) = l;
#             Q = [Q; adj(i)];
#
#             s = adjedge(i,2);
#             y(adj(i),:) = fun(y(idx,:), s*omega(adjedge(i,1),:));
#         end
#     end
# end
# end


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
        x: Values at vertices (modified in place and returned)
        omega: Update values for each edge
        E2V: Edge-to-vertex connectivity (edge i connects E2V[i,0] to E2V[i,1])
        init: Starting vertex index

    Returns:
        y: Propagated values
    """
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
    # S = -ones(nv,1); % Visited vertices
    # S(Q) = 0;
    # level = zeros(nv,1);
    # l = 0;

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
        # adjedge(adj == 0,:) = [];
        # adj(adj == 0) = [];

        # Build adjacent edge info: (edge_index, sign)
        adjedge_idx = np.concatenate([id1, id2])
        adjedge_sign = np.concatenate([-np.ones(len(id1)), np.ones(len(id2))])
        adj = np.concatenate([E2V[id1, 1], E2V[id2, 0]])

        # Filter out invalid (boundary) adjacencies
        # In MATLAB, 0 means no neighbor. In Python with 0-indexed, we use -1
        # However, for the E2T array passed here, faces are indexed from 0
        # and we need to filter based on the specific sentinel used
        valid_mask = adj >= 0
        adjedge_idx = adjedge_idx[valid_mask]
        adjedge_sign = adjedge_sign[valid_mask]
        adj = adj[valid_mask]

        # for i = 1:length(adj)
        #     if S(adj(i)) == -1
        #         S(adj(i)) = idx;
        #         level(adj(i)) = l;
        #         Q = [Q; adj(i)];
        #
        #         s = adjedge(i,2);
        #         y(adj(i),:) = fun(y(idx,:), s*omega(adjedge(i,1),:));
        #     end
        # end

        for i in range(len(adj)):
            neighbor = int(adj[i])
            if neighbor < nv and S[neighbor] == -1:
                S[neighbor] = idx
                level[neighbor] = l
                Q.append(neighbor)

                s = adjedge_sign[i]
                edge_idx = int(adjedge_idx[i])
                y[neighbor] = y[idx] + s * omega[edge_idx]

    return y
