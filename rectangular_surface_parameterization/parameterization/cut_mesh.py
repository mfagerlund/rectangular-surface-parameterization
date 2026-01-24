
# function [disk_mesh,idx_cut_inv,ide_cut_inv,edge_cut] = cut_mesh(X, T, E2V, E2T, T2E, T2T, idcone, edge_jump_tag)
# 
# nv = size(X,1);
# ne = size(E2V,1);
# nf = size(T,1);
# 
# Q = 1;
# tri_pred =-ones(nf,1); % Visited faces
# tri_pred(Q) = 0;
# visited_edge = false(ne,1);
# visited_edge(edge_jump_tag) = true;
# while ~isempty(Q)
#     idtri = Q(1);
#     Q(1) = [];
# 
#     adj = T2T(idtri,:);
#     adj(adj == 0) = [];
#     adjedge = abs(T2E(idtri,:));
#     [~,ib] = ismember(adj, sum(E2T(adjedge,1:2).*(E2T(adjedge,1:2) ~= idtri),2));
#     adjedge = adjedge(ib);
#     
#     for i = 1:length(adj)
#         if (tri_pred(adj(i)) == -1) && ~visited_edge(adjedge(i))
#             tri_pred(adj(i)) = idtri;
#             Q = [Q; adj(i)];
#             
#             visited_edge(adjedge(i)) = true;
#         end
#     end
# end
# 
# visited_edge(edge_jump_tag) = false; 
# edge_cut =~visited_edge; % Set of current edge in the cut
# deg = accumarray(vec(E2V(edge_cut,:)), 1, [nv,1]); % vertices degree
# 
# deg1 = deg == 1; % set of degree 1 vertices
# while sum(deg1(idcone)) ~= sum(deg1) % while there's degree 1 vertices
#     deg1(idcone(deg1(idcone))) = false;
#     edge_cut(any(deg1(E2V),2)) = false; % remove edges attached to degree 1 vertices from the cut
#     deg = accumarray(vec(E2V(edge_cut,:)), 1, [nv,1]); % recompute vertices degree with the new cut
#     
#     deg1 = deg == 1; % set of degree 1 vertices
# end
# 
# if any(edge_cut) && any(all(E2T(edge_cut,1:2) ~= 0,2))
#     e2v = [T(:,1),T(:,2); T(:,2),T(:,3); T(:,3),T(:,1)];
#     [e2vs,ids] = sort(e2v,2);
#     id_cut = find(ismember(e2vs, E2V(edge_cut,:), 'rows'));
#     
#     [~,ia,ic] = unique(e2vs, 'rows');
#     idtri_bound = ia(accumarray(ic,1) == 1);
#     idtri1 = setdiff(ia(accumarray(ic,1) == 2), id_cut);
#     idtri2 = setdiff((1:size(e2vs,1))', [idtri1; idtri_bound; id_cut]);
#     [~,id1,id2] = intersect(e2vs(idtri1,:), e2vs(idtri2,:), 'rows');
#     
#     Tc = reshape((1:3*nf)', [nf,3]);
#     e2v_cut = [Tc(:,1),Tc(:,2); Tc(:,2),Tc(:,3); Tc(:,3),Tc(:,1)];
#     e2v_cuts = [e2v_cut(:,1).*(ids(:,1) == 1) + e2v_cut(:,2).*(ids(:,1) == 2), e2v_cut(:,1).*(ids(:,2) == 1) + e2v_cut(:,2).*(ids(:,2) == 2)];
#     equiv_vx = [e2v_cuts(idtri1(id1),1), e2v_cuts(idtri2(id2),1) ; e2v_cuts(idtri1(id1),2), e2v_cuts(idtri2(id2),2)];
#     
#     Tc = reshape(union_find(3*nf, equiv_vx), [nf,3]);
#     idx_cut_inv = unique([Tc(:), T(:)], 'rows');
#     idx_cut_inv = idx_cut_inv(:,2);
#     assert(max(Tc(:)) == length(idx_cut_inv), 'Failure to find new indices.');
#     
#     Xc = X(idx_cut_inv,:);
# else
#     idx_cut_inv = (1:nv)';
#     Tc = T;
#     Xc = X;
# end
# 
# disk_mesh = MeshInfo(Xc, Tc);
# chiCut = disk_mesh.num_faces - disk_mesh.num_edges + disk_mesh.num_vertices;
# if chiCut ~= 1
#     warning('Not topological disk after cut.');
# end
# 
# [~,ide_cut_inv] = ismember(sort(idx_cut_inv(disk_mesh.edge_to_vertex),2), E2V, 'rows');
# assert(all(ide_cut_inv ~= 0))
# ids = idx_cut_inv(disk_mesh.edge_to_vertex(:,1)) == E2V(ide_cut_inv,1);
# ide_cut_inv = ids.*ide_cut_inv - (~ids).*ide_cut_inv;
# 
# end
# 
# function x = union_find(n, equiv)
# assert(size(equiv,2) == 1 || size(equiv,2) == 2, 'Argument size invalid.');
# assert(all(equiv(:) > 0), 'Wrong indexes.');
# 
# parent = (1:n)';
# if size(equiv,2) == 2
#     assert(all(equiv(:) <= n), 'Wrong indexes.');
#     
#     for i = 1:size(equiv,1)
#         parent = union_tree(equiv(i,1), equiv(i,2), parent);
#     end
# elseif size(equiv,2) == 1
#     neq = max(equiv);
#     for i = 1:neq
#         ind = mod(find(equiv == i)-1,n)+1;
#         nind = length(ind);
#         for j = 1:nind
#             parent = union_tree(ind(j), ind(mod(j,nind)+1), parent);
#         end
#     end
# else
#     error('Argument size invalid.');
# end
# 
# x = parent;
# for i = 1:n
#     [x(i),~] = find_root(i, parent);
# end
# 
# nset = unique(x);
# uniq_id = zeros(max(nset),1);
# uniq_id(nset) = 1:length(nset);
# x = uniq_id(x);
# end
# 
# function parent = union_tree(x, y, parent)
# [x_root,x_size] = find_root(x, parent);
# [y_root,y_size] = find_root(y, parent);
# 
# if x_root ~= y_root
#     if x_size > y_size
#         parent(y_root) = x_root;
#     else
#         parent(x_root) = y_root;
#     end
# end
# end
# 
# function [root,s] = find_root(x, parent)
# root = x;
# s = 0;
# while root ~= parent(root)
#     root = parent(root);
#     s = s + 1;
# end
# end


# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from collections import deque


@dataclass
class MeshInfo:
    vertices: np.ndarray
    triangles: np.ndarray
    edge_to_vertex: np.ndarray
    edge_to_triangle: np.ndarray
    T2E: np.ndarray
    triangle_to_triangle: np.ndarray
    num_vertices: int
    num_edges: int
    num_faces: int


def _rows_view(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view([("", a.dtype)] * a.shape[1]).reshape(-1)


def _detect_one_based(T: np.ndarray, E2V: np.ndarray) -> bool:
    return T.size > 0 and E2V.size > 0 and T.min() >= 1 and E2V.min() >= 1


def _shift_signed_indices(arr: np.ndarray, one_based: bool) -> np.ndarray:
    if not one_based:
        return arr.copy()
    out = arr.copy()
    sign = np.sign(out)
    out = sign * (np.abs(out) - 1)
    return out


def _shift_positive_or_zero(arr: np.ndarray, one_based: bool) -> np.ndarray:
    if not one_based:
        return arr.copy()
    out = arr.copy()
    mask = out > 0
    out[mask] = out[mask] - 1
    out[~mask] = -1
    return out


def _edge_jump_tag_to_bool(edge_jump_tag, n_edges: int, one_based: bool) -> np.ndarray:
    if isinstance(edge_jump_tag, np.ndarray) and edge_jump_tag.dtype == bool:
        if edge_jump_tag.shape != (n_edges,):
            raise ValueError("edge_jump_tag boolean shape mismatch")
        return edge_jump_tag.copy()
    indices = np.asarray(edge_jump_tag, dtype=int).reshape(-1)
    if indices.size == 0:
        return np.zeros(n_edges, dtype=bool)
    if one_based:
        indices = indices - 1
    mask = np.zeros(n_edges, dtype=bool)
    mask[indices] = True
    return mask


def _build_meshinfo(X: np.ndarray, T: np.ndarray) -> MeshInfo:
    T = np.asarray(T, dtype=int)
    X = np.asarray(X)
    nf = T.shape[0]
    nv = X.shape[0]

    edge_dict = {}
    T2E = np.zeros((nf, 3), dtype=int)

    for f in range(nf):
        for local in range(3):
            i = int(T[f, local])
            j = int(T[f, (local + 1) % 3])
            key = (i, j) if i < j else (j, i)
            if key not in edge_dict:
                edge_dict[key] = len(edge_dict)
            T2E[f, local] = edge_dict[key]

    ne = len(edge_dict)
    E2V = np.zeros((ne, 2), dtype=int)
    for (i, j), e in edge_dict.items():
        E2V[e] = [i, j]

    E2T = np.full((ne, 2), -1, dtype=int)
    for f in range(nf):
        for local in range(3):
            e = T2E[f, local]
            if E2T[e, 0] == -1:
                E2T[e, 0] = f
            else:
                E2T[e, 1] = f

    T2T = np.full((nf, 3), -1, dtype=int)
    for f in range(nf):
        for local in range(3):
            e = T2E[f, local]
            a, b = E2T[e]
            if a == f:
                T2T[f, local] = b
            else:
                T2T[f, local] = a

    return MeshInfo(
        vertices=X,
        triangles=T,
        edge_to_vertex=E2V,
        edge_to_triangle=E2T,
        T2E=T2E,
        triangle_to_triangle=T2T,
        num_vertices=nv,
        num_edges=ne,
        num_faces=nf,
    )


def _union_find(n: int, equiv: np.ndarray) -> np.ndarray:
    equiv = np.asarray(equiv, dtype=int)
    if equiv.ndim != 2 or equiv.shape[1] not in (1, 2):
        raise AssertionError("Argument size invalid.")
    if equiv.size and np.any(equiv < 0):
        raise AssertionError("Wrong indexes.")

    parent = np.arange(n, dtype=int)
    size = np.ones(n, dtype=int)

    def find_root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_tree(x: int, y: int) -> None:
        rx = find_root(x)
        ry = find_root(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    if equiv.shape[1] == 2:
        if equiv.size and np.any(equiv >= n):
            raise AssertionError("Wrong indexes.")
        for x, y in equiv:
            union_tree(int(x), int(y))
    else:
        neq = int(np.max(equiv)) if equiv.size else 0
        for i in range(neq + 1):
            inds = np.where(equiv[:, 0] == i)[0]
            if inds.size <= 1:
                continue
            for j in range(inds.size - 1):
                union_tree(int(inds[j]), int(inds[j + 1]))

    roots = np.array([find_root(i) for i in range(n)], dtype=int)
    uniq = np.unique(roots)
    remap = np.zeros(uniq.max() + 1, dtype=int)
    remap[uniq] = np.arange(len(uniq))
    return remap[roots]


def cut_mesh(X, T, E2V, E2T, T2E, T2T, idcone, edge_jump_tag):
    """
    Python translation of cut_mesh.m.

    Notes:
    - Inputs are expected to be zero-based arrays. If 1-based indexing is
      detected, indices are shifted to zero-based for processing.
    - ide_cut_inv is returned as signed 1-based edge indices to preserve sign
      information (edge 0 cannot be negated in 0-based encoding).
    """
    X = np.asarray(X)
    T = np.asarray(T, dtype=int)
    E2V = np.asarray(E2V, dtype=int)
    E2T = np.asarray(E2T, dtype=int)
    T2E = np.asarray(T2E, dtype=int)
    T2T = np.asarray(T2T, dtype=int)
    idcone = np.asarray(idcone, dtype=int).reshape(-1)

    one_based = _detect_one_based(T, E2V)

    if one_based:
        T = T - 1
        E2V = E2V - 1
        E2T = _shift_positive_or_zero(E2T, one_based)
        T2E = _shift_signed_indices(T2E, one_based)
        T2T = _shift_positive_or_zero(T2T, one_based)
        if idcone.size:
            idcone = idcone - 1

    nv = X.shape[0]
    ne = E2V.shape[0]
    nf = T.shape[0]

    edge_jump_tag = _edge_jump_tag_to_bool(edge_jump_tag, ne, one_based)

    visited_edge = np.zeros(ne, dtype=bool)
    visited_edge[edge_jump_tag] = True

    tri_pred = -np.ones(nf, dtype=int)
    seed = 0
    tri_pred[seed] = seed

    # Detect if T2E uses 1-based encoding (signed: edge_idx = abs(val) - 1)
    is_t2e_one_based = np.max(np.abs(T2E)) == ne

    queue = deque([seed])
    bfs_edges_marked = 0
    while queue:
        idtri = queue.popleft()

        adj = T2T[idtri]
        adj = adj[adj >= 0]
        if adj.size == 0:
            continue

        # T2E may be signed 1-based (edge+1)*sign, need to convert to 0-based
        adjedge_raw = T2E[idtri]
        if is_t2e_one_based:
            adjedge = np.abs(adjedge_raw).astype(int) - 1
        else:
            adjedge = np.abs(adjedge_raw).astype(int)
        adjedge = adjedge[adjedge >= 0]
        if adjedge.size == 0:
            continue

        # For each adjacent face, find the connecting edge by checking E2T
        for face in adj:
            # Find which edge of idtri connects to face
            for e in adjedge:
                t0, t1 = E2T[e, 0], E2T[e, 1]
                if (t0 == idtri and t1 == face) or (t1 == idtri and t0 == face):
                    # Found the edge connecting idtri to face
                    if tri_pred[face] == -1 and not visited_edge[e]:
                        tri_pred[face] = idtri
                        visited_edge[e] = True
                        queue.append(int(face))
                        bfs_edges_marked += 1
                    break

    visited_edge[edge_jump_tag] = False
    edge_cut = ~visited_edge

    deg = np.zeros(nv, dtype=int)
    if np.any(edge_cut):
        np.add.at(deg, E2V[edge_cut].ravel(), 1)

    deg1 = deg == 1
    is_cone = np.zeros(nv, dtype=bool)
    if idcone.size:
        is_cone[idcone] = True

    while np.sum(deg1[is_cone]) != np.sum(deg1):
        deg1_non_cone = deg1 & ~is_cone
        if not np.any(deg1_non_cone):
            break
        edge_cut[np.any(deg1_non_cone[E2V], axis=1)] = False
        deg[:] = 0
        if np.any(edge_cut):
            np.add.at(deg, E2V[edge_cut].ravel(), 1)
        deg1 = deg == 1

    if np.any(edge_cut) and np.any(np.all(E2T[edge_cut, :2] != -1, axis=1)):
        e2v = np.vstack([
            T[:, [0, 1]],
            T[:, [1, 2]],
            T[:, [2, 0]],
        ])
        ids = np.argsort(e2v, axis=1)
        e2vs = np.take_along_axis(e2v, ids, axis=1)

        cut_rows = _rows_view(np.sort(E2V[edge_cut], axis=1))
        e2vs_rows = _rows_view(e2vs)
        id_cut = np.where(np.isin(e2vs_rows, cut_rows))[0]

        uniq_rows, ia, ic = np.unique(e2vs_rows, return_index=True, return_inverse=True)
        counts = np.bincount(ic)

        idtri_bound = ia[counts == 1]
        idtri1 = np.setdiff1d(ia[counts == 2], id_cut, assume_unique=False)

        all_ids = np.arange(e2vs.shape[0])
        remove_ids = np.unique(np.concatenate([idtri1, idtri_bound, id_cut]))
        idtri2 = np.setdiff1d(all_ids, remove_ids, assume_unique=False)

        rows1 = _rows_view(e2vs[idtri1])
        rows2 = _rows_view(e2vs[idtri2])
        _, id1, id2 = np.intersect1d(rows1, rows2, return_indices=True)

        Tc = np.arange(3 * nf, dtype=int).reshape(nf, 3)
        e2v_cut = np.vstack([
            Tc[:, [0, 1]],
            Tc[:, [1, 2]],
            Tc[:, [2, 0]],
        ])

        e2v_cuts = np.column_stack([
            np.where(ids[:, 0] == 0, e2v_cut[:, 0], e2v_cut[:, 1]),
            np.where(ids[:, 1] == 0, e2v_cut[:, 0], e2v_cut[:, 1]),
        ])

        equiv_vx = np.vstack([
            np.column_stack([e2v_cuts[idtri1[id1], 0], e2v_cuts[idtri2[id2], 0]]),
            np.column_stack([e2v_cuts[idtri1[id1], 1], e2v_cuts[idtri2[id2], 1]]),
        ])

        Tc = _union_find(3 * nf, equiv_vx).reshape(nf, 3)

        n_new = int(Tc.max()) + 1
        idx_cut_inv = np.full(n_new, -1, dtype=int)
        for new_idx, orig_idx in zip(Tc.ravel(), T.ravel()):
            if idx_cut_inv[new_idx] == -1:
                idx_cut_inv[new_idx] = int(orig_idx)

        if np.any(idx_cut_inv < 0):
            raise AssertionError("Failure to find new indices.")

        Xc = X[idx_cut_inv]
    else:
        idx_cut_inv = np.arange(nv, dtype=int)
        Tc = T.copy()
        Xc = X.copy()

    disk_mesh = _build_meshinfo(Xc, Tc)
    chi_cut = disk_mesh.num_faces - disk_mesh.num_edges + disk_mesh.num_vertices
    if chi_cut != 1:
        print("Warning: Not topological disk after cut.")

    orig_edge_map = {tuple(row): i for i, row in enumerate(E2V)}
    mapped_edges = np.sort(idx_cut_inv[disk_mesh.edge_to_vertex], axis=1)
    ide_cut_inv = np.array([orig_edge_map.get(tuple(row), -1) for row in mapped_edges], dtype=int)
    if np.any(ide_cut_inv < 0):
        raise AssertionError("Failed to map cut edges to original edges.")

    ids = idx_cut_inv[disk_mesh.edge_to_vertex[:, 0]] == E2V[ide_cut_inv, 0]
    ide_cut_inv = np.where(ids, ide_cut_inv + 1, -(ide_cut_inv + 1))

    if one_based:
        idx_cut_inv = idx_cut_inv + 1

    return disk_mesh, idx_cut_inv, ide_cut_inv, edge_cut
