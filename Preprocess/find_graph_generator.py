

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from typing import List, Tuple, Optional


# function [cycle,cocycle] = find_graph_generator(l, T, E2T, E2V, init)
#
# if ~exist('init', 'var')
#     init = 1;
# end
#
# nv = max(T(:));
# ne = size(E2T,1);
# nf = max(E2T(:));

def find_graph_generator(
    l: np.ndarray,
    T: np.ndarray,
    E2T: np.ndarray,
    E2V: np.ndarray,
    init: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Find graph generators (homology basis loops) for surfaces with handles.

    Based on "Greedy Optimal Homotopy and Homology Generators" by Erickson & Whittlesey.

    Parameters
    ----------
    l : ndarray of shape (ne,)
        Edge lengths
    T : ndarray of shape (nf, 3)
        Triangle vertex indices (0-indexed)
    E2T : ndarray of shape (ne, 2) or (ne, 4)
        Edge-to-triangle mapping. Columns 0,1 are adjacent triangles (-1 for boundary)
    E2V : ndarray of shape (ne, 2)
        Edge-to-vertex mapping
    init : int
        Root vertex for spanning tree (0-indexed, default 0)

    Returns
    -------
    cycle : list of ndarray
        Primal cycle generators (vertex sequences)
    cocycle : list of ndarray
        Dual cycle generators (face sequences)
    """
    nv = T.max() + 1  # MATLAB: max(T(:)), but Python is 0-indexed so +1 for count
    ne = E2T.shape[0]
    nf = E2T[:, :2].max() + 1  # Only first two columns are triangle indices


# ide_bound = any(E2T(:,1:2) == 0, 2);
# l = abs(l) + 1e-5;
# w = l;
# w(ide_bound) = 0;

    # Primal graph
    # Set edge weights, zero at boundaries
    # MATLAB uses 0 for "no triangle", Python uses -1
    ide_bound = np.any(E2T[:, :2] < 0, axis=1)
    l = np.abs(l) + 1e-5
    w = l.copy()
    w[ide_bound] = 0


# try
#     G = graph(E2V(:,1), E2V(:,2), w);
#     [Tree, pred] = minspantree(G, 'Type','tree', 'Root',init);
#     pred = pred';
#     EdgeTree = [pred(2:end), (2:nv)'];
# catch
#     warning('Kruskal algorithm from hell.');
#     wAdj = sparse(E2V(:,[1 2]), E2V(:,[2 1]), [w,w], nv, nv);
#     Adj = sparse(E2V(:,[1 2]), E2V(:,[2 1]), ones(nf,2), nv, nv);
#     [~, EdgeTree, ~] = kruskal(Adj, wAdj);
#     EdgeTree = sort(EdgeTree, 2);
#     pred = tree_predecessor(init, EdgeTree);
# end

    # Compute minimal spanning tree on primal graph
    # Build weighted adjacency matrix (symmetric)
    row = np.concatenate([E2V[:, 0], E2V[:, 1]])
    col = np.concatenate([E2V[:, 1], E2V[:, 0]])
    data = np.concatenate([w, w])
    wAdj = csr_matrix((data, (row, col)), shape=(nv, nv))

    # Compute MST using scipy (returns MST as sparse matrix)
    mst = minimum_spanning_tree(wAdj)

    # Extract edges from MST and compute predecessor array via BFS/DFS
    mst_sym = mst + mst.T  # Make symmetric for traversal
    pred = _compute_predecessors_bfs(mst_sym, init, nv)

    # Build EdgeTree from predecessor array
    # EdgeTree[i] = (pred[i], i) for all i where pred[i] >= 0
    tree_edges = []
    for i in range(nv):
        if pred[i] >= 0 and pred[i] != i:
            tree_edges.append(sorted([pred[i], i]))
    EdgeTree = np.array(tree_edges) if tree_edges else np.zeros((0, 2), dtype=int)


# [~,id] = setdiff(sort(E2V,2), sort(EdgeTree,2), 'rows');
# assert(all(~isnan(pred)), 'Multiple connected components.');

    # Find edge indices which are not in the tree
    E2V_sorted = np.sort(E2V, axis=1)
    EdgeTree_sorted = np.sort(EdgeTree, axis=1) if len(EdgeTree) > 0 else np.zeros((0, 2), dtype=int)

    # setdiff for rows: find edges in E2V but not in EdgeTree
    id_not_in_tree = _setdiff_rows(E2V_sorted, EdgeTree_sorted)

    # Check for disconnected components: any vertex (other than root) with pred == -1
    # means it was unreachable from the root during BFS
    disconnected = np.where(pred == -1)[0]
    if len(disconnected) > 0:
        raise ValueError(
            f"Mesh has disconnected components: {len(disconnected)} vertices "
            f"unreachable from root {init}. First few: {disconnected[:5].tolist()}"
        )


# id = setdiff(id, find(ide_bound));

    # Remove boundary edges
    boundary_edges = np.where(ide_bound)[0]
    id_not_in_tree = np.setdiff1d(id_not_in_tree, boundary_edges)


# w = 1./l;

    # Dual graph
    # Edge weights for the dual graph
    w_dual = 1.0 / l


# try
#     Gco = graph(E2T(id,1), E2T(id,2), w(id));
#     [CoTree, copred] = minspantree(Gco, 'Type','forest', 'Method','sparse');
#     copred = copred';
#     EdgeCoTree = [copred(1:end), (1:max(vec(E2T(id,1:2))))'];%CoTree.Edges.EndNodes;
# catch
#     warning('Kruskal algorithm from hell.');
#     wCoAdj = sparse(E2T(id,[1 2]), E2T(id,[2 1]), w, nf, nf);
#     CoAdj = sparse(E2T(id,[1 2]), E2T(id,[2 1]), ones(length(id),2), nf, nf);
#     [~, EdgeCoTree, ~] = kruskal(CoAdj, wCoAdj);
#     copred = tree_predecessor(init, EdgeCoTree);
# end

    # Compute minimal spanning tree on the dual graph of the remaining edges
    if len(id_not_in_tree) > 0:
        # Build dual graph adjacency from edges not in primal tree
        dual_e0 = E2T[id_not_in_tree, 0]
        dual_e1 = E2T[id_not_in_tree, 1]
        dual_w = w_dual[id_not_in_tree]

        # Filter out boundary edges (where triangle index is -1)
        valid = (dual_e0 >= 0) & (dual_e1 >= 0)
        dual_e0 = dual_e0[valid]
        dual_e1 = dual_e1[valid]
        dual_w = dual_w[valid]

        if len(dual_e0) > 0:
            row_dual = np.concatenate([dual_e0, dual_e1])
            col_dual = np.concatenate([dual_e1, dual_e0])
            data_dual = np.concatenate([dual_w, dual_w])
            wCoAdj = csr_matrix((data_dual, (row_dual, col_dual)), shape=(nf, nf))

            # Compute MST on dual graph (forest if disconnected)
            co_mst = minimum_spanning_tree(wCoAdj)
            co_mst_sym = co_mst + co_mst.T

            # Compute predecessor for dual graph (starting from face 0)
            copred = _compute_predecessors_bfs(co_mst_sym, 0, nf)

            # Build EdgeCoTree from dual MST
            co_tree_edges = []
            for i in range(nf):
                if copred[i] >= 0 and copred[i] != i:
                    co_tree_edges.append(sorted([copred[i], i]))
            EdgeCoTree = np.array(co_tree_edges) if co_tree_edges else np.zeros((0, 2), dtype=int)
        else:
            copred = np.full(nf, -1, dtype=int)
            copred[0] = 0
            EdgeCoTree = np.zeros((0, 2), dtype=int)
    else:
        copred = np.full(nf, -1, dtype=int)
        copred[0] = 0
        EdgeCoTree = np.zeros((0, 2), dtype=int)


# [~,idCoEdge] = setdiff(sort(E2T(:,1:2),2), sort(EdgeCoTree,2), 'rows');

    # Find edge indices which are not in the dual tree
    E2T_sorted = np.sort(E2T[:, :2], axis=1)
    EdgeCoTree_sorted = np.sort(EdgeCoTree, axis=1) if len(EdgeCoTree) > 0 else np.zeros((0, 2), dtype=int)
    idCoEdge = _setdiff_rows(E2T_sorted, EdgeCoTree_sorted)


# idGen = intersect(id, idCoEdge);

    # Find edges which are neither in the primal nor in the dual graph
    # These are the generators
    idGen = np.intersect1d(id_not_in_tree, idCoEdge)


# cycle = cell(length(idGen),1);
# for i = 1:length(idGen)
#     left = flipud(predecessors(pred, E2V(idGen(i),1)));
#     right = predecessors(pred, E2V(idGen(i),2));
#     cycle{i} = [left; right(1:end-1)];
# end

    # Build cycle basis
    # Find the primal loops starting from idGen
    cycle = []
    for i in range(len(idGen)):
        edge_idx = idGen[i]
        v0, v1 = E2V[edge_idx, 0], E2V[edge_idx, 1]
        left = _predecessors(pred, v0)[::-1]  # flipud
        right = _predecessors(pred, v1)
        # Combine: left + right[:-1] (exclude last which is common root)
        cycle_i = np.concatenate([left, right[:-1]]) if len(right) > 1 else left
        cycle.append(cycle_i)


# cocycle = cell(length(idGen),1);
# assert(all(~isnan(pred)));
# assert(all(~isnan(copred)));
# for i = 1:length(idGen)
#     left = flipud(predecessors(copred, E2T(idGen(i),1)));
#     right = predecessors(copred, E2T(idGen(i),2));
#     cocycle{i} = [left; right(1:end-1)];
# end

    # Find the dual loops starting from idGen
    cocycle = []
    for i in range(len(idGen)):
        edge_idx = idGen[i]
        f0, f1 = E2T[edge_idx, 0], E2T[edge_idx, 1]
        if f0 >= 0 and f1 >= 0:
            left = _predecessors(copred, f0)[::-1]  # flipud
            right = _predecessors(copred, f1)
            cocycle_i = np.concatenate([left, right[:-1]]) if len(right) > 1 else left
            cocycle.append(cocycle_i)
        else:
            # Boundary edge, skip
            cocycle.append(np.array([], dtype=int))

    return cycle, cocycle


# end
#
# function path = predecessors(pred, i)
# path = i;
# i = pred(path(end));
# while i ~= 0
#     path = [path; i];
#     i = pred(path(end));
# end
# end

def _predecessors(pred: np.ndarray, i: int) -> np.ndarray:
    """
    Trace path from vertex i to root via predecessor array.

    Parameters
    ----------
    pred : ndarray
        Predecessor array where pred[v] is parent of v, pred[root] = root or -1
    i : int
        Starting vertex

    Returns
    -------
    path : ndarray
        Sequence of vertices from i to root (inclusive)
    """
    path = [i]
    current = pred[i]
    # Stop when we reach root (pred[root] == root) or sentinel (-1 or same as current)
    while current >= 0 and current != path[-1]:
        path.append(current)
        if pred[current] == current:
            break
        current = pred[current]
    return np.array(path, dtype=int)


def _compute_predecessors_bfs(adj: csr_matrix, root: int, n: int) -> np.ndarray:
    """
    Compute predecessor array via BFS on sparse adjacency matrix.

    Parameters
    ----------
    adj : csr_matrix
        Symmetric adjacency matrix (weighted or unweighted)
    root : int
        Root vertex for BFS
    n : int
        Number of vertices

    Returns
    -------
    pred : ndarray
        Predecessor array where pred[v] is parent of v in BFS tree
    """
    pred = np.full(n, -1, dtype=int)
    pred[root] = root  # Root's predecessor is itself

    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    queue = [root]

    while queue:
        u = queue.pop(0)
        # Get neighbors from sparse matrix
        row_start = adj.indptr[u]
        row_end = adj.indptr[u + 1]
        neighbors = adj.indices[row_start:row_end]

        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                pred[v] = u
                queue.append(v)

    return pred


def _setdiff_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Find row indices in A that are not in B (set difference for rows).

    Parameters
    ----------
    A : ndarray of shape (m, k)
        First array
    B : ndarray of shape (n, k)
        Second array

    Returns
    -------
    indices : ndarray
        Indices of rows in A that are not present in B
    """
    if len(A) == 0:
        return np.array([], dtype=int)
    if len(B) == 0:
        return np.arange(len(A))

    # Convert rows to tuples for set operations
    A_set = set(map(tuple, A))
    B_set = set(map(tuple, B))

    indices = []
    for i, row in enumerate(A):
        if tuple(row) not in B_set:
            indices.append(i)

    return np.array(indices, dtype=int)
