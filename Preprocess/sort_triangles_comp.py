# === ISSUES ===
# - vec(): MATLAB vec() flattens column-major; use .ravel(order='F') or .flatten()
# - circshift: use np.roll
# - intersect(..., 'rows', 'stable'): need custom implementation preserving order
# - ismember: use np.isin
# === END ISSUES ===

import numpy as np
from typing import Tuple, Optional


# function [tri_ord,edge_ord,sign_edge] = sort_triangles_comp(idx, T, E2T, T2T, E2V, T2E)
# % sort ring triangles around idx

def sort_triangles_comp(
    idx: int,
    T: np.ndarray,
    E2T: np.ndarray,
    T2T: np.ndarray,
    E2V: np.ndarray,
    T2E: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sort ring triangles around vertex idx in consistent order.

    Parameters
    ----------
    idx : int
        Vertex index (0-indexed).
    T : ndarray (nf, 3)
        Triangle vertex indices (0-indexed).
    E2T : ndarray (ne, 4)
        Edge-to-triangle connectivity. [tri0, tri1, sign0, sign1].
        tri=-1 means no neighbor (boundary).
    T2T : ndarray (nf, 3)
        Triangle-to-triangle adjacency.
    E2V : ndarray (ne, 2)
        Edge-to-vertex connectivity.
    T2E : ndarray (nf, 3), optional
        Triangle-to-edge connectivity (signed).

    Returns
    -------
    tri_ord : ndarray (n,)
        Sorted triangle indices around vertex. Contains -1 at end if boundary vertex.
    edge_ord : ndarray (n,) or None
        Sorted edge indices (only if requested).
    sign_edge : ndarray (n,) or None
        Edge signs (only if requested).
    """

    # ifbound = any(vec(E2V(any(E2T(:,1:2) == 0, 2),:) == idx));
    #
    # In MATLAB, E2T(:,1:2) == 0 checks for boundary edges (no neighbor = 0).
    # In Python, we use -1 for no neighbor, so check E2T[:, 0:2] == -1.
    # Find boundary edges: edges where either adjacent triangle is -1 (or 0 in MATLAB).
    boundary_edge_mask = np.any(E2T[:, 0:2] == -1, axis=1)
    boundary_edges = E2V[boundary_edge_mask, :]  # vertices of boundary edges

    # Check if idx is on any boundary edge
    ifbound = np.any(boundary_edges == idx)

    tri_start = None

    # if ifbound
    #     edi = E2T(any(E2V == idx, 2),:);
    #
    #     m = edi(:,1:2) == 0;
    #
    #     vxi = E2V(any(E2V == idx, 2),:);
    #     vxi = vxi(any(m ~= 0,2),:);
    #     vxi = sum(vxi .* (vxi ~= idx),2);
    #     vxi = [idx*ones(2,1), vxi];
    #
    #     tri_start = edi(:,1:2).*edi(:,3:4).*m(:,[2 1]);
    #     tri_start = abs(tri_start(tri_start ~= 0));
    #
    #     if1 = false; deg1 = sum(T(tri_start(1),:) ~= 0);
    #     if2 = false; deg2 = sum(T(tri_start(2),:) ~= 0);
    #     for k = 1:size(T,2)
    #         i = mod(k-1, deg1) + 1;
    #         j = mod(k  , deg1) + 1;
    #         if1 = if1 || all(T(tri_start(1),[i j]) == vxi(1,:));
    #         i = mod(k-1, deg2) + 1;
    #         j = mod(k  , deg2) + 1;
    #         if2 = if2 || all(T(tri_start(2),[i j]) == vxi(2,:));
    #     end
    #
    #     if if1
    #         tri_start = tri_start(1);
    #     elseif if2
    #         tri_start = tri_start(2);
    #     else
    #         error('Could not find a staring triangle.');
    #     end

    if ifbound:
        # Find all edges incident to vertex idx
        edges_at_idx_mask = np.any(E2V == idx, axis=1)
        edi = E2T[edges_at_idx_mask, :]  # E2T rows for edges incident to idx

        # m marks boundary edges: where tri0 or tri1 is -1 (MATLAB uses 0)
        m = edi[:, 0:2] == -1

        # Get the vertices of edges at idx, filter to boundary edges
        vxi_all = E2V[edges_at_idx_mask, :]
        boundary_at_idx_mask = np.any(m, axis=1)
        vxi = vxi_all[boundary_at_idx_mask, :]

        # Get the other vertex (not idx) for each boundary edge
        # vxi = sum(vxi .* (vxi ~= idx),2) gives the other vertex
        other_v = np.sum(vxi * (vxi != idx), axis=1)
        vxi = np.column_stack([np.full(len(other_v), idx), other_v])

        # tri_start = edi(:,1:2).*edi(:,3:4).*m(:,[2 1]);
        # This extracts the triangle index from the non-boundary side
        # m[:, [1, 0]] swaps columns of m
        #
        # FIX: MATLAB uses 0 for "no triangle" (1-based indices), so multiplication works.
        # Python uses 0-indexed triangles with -1 for "no neighbor", so multiplying by
        # the mask causes triangle 0 to be dropped (0 * anything = 0).
        # Use np.where instead of multiplication.
        edi_boundary = edi[boundary_at_idx_mask, :]
        m_boundary = m[boundary_at_idx_mask, :]

        # For each boundary edge, we want the triangle on the NON-boundary side.
        # m_boundary[:, 0] is True when edi_boundary[:, 0] == -1 (no triangle on side 0)
        # m_boundary[:, 1] is True when edi_boundary[:, 1] == -1 (no triangle on side 1)
        # When side 1 is boundary (-1), select side 0 triangle; vice versa.
        tri_start_candidates = []
        for i in range(len(edi_boundary)):
            if m_boundary[i, 1]:  # Side 1 is boundary (-1), use side 0
                tri_idx = edi_boundary[i, 0]
                if tri_idx != -1:
                    tri_start_candidates.append(tri_idx)
            elif m_boundary[i, 0]:  # Side 0 is boundary (-1), use side 1
                tri_idx = edi_boundary[i, 1]
                if tri_idx != -1:
                    tri_start_candidates.append(tri_idx)
        tri_start_candidates = np.array(tri_start_candidates, dtype=int)

        # Need exactly 2 boundary edges for a proper boundary vertex
        if len(tri_start_candidates) < 2:
            raise ValueError("Boundary vertex does not have 2 boundary edges.")

        # Check which starting triangle has correct winding
        # deg1 = sum(T(tri_start(1),:) ~= 0) - degree check (for possibly mixed polygon mesh)
        # For triangles, deg = 3 always
        if1 = False
        if2 = False
        tri1 = int(tri_start_candidates[0])
        tri2 = int(tri_start_candidates[1])
        deg1 = np.sum(T[tri1, :] >= 0)  # In Python, valid vertices are >= 0
        deg2 = np.sum(T[tri2, :] >= 0)

        for k in range(1, T.shape[1] + 1):  # k = 1 to size(T,2) in MATLAB
            # MATLAB: i = mod(k-1, deg1) + 1; j = mod(k, deg1) + 1
            # Python: i = (k-1) % deg1; j = k % deg1 (0-indexed)
            i1 = (k - 1) % deg1
            j1 = k % deg1
            if np.all(T[tri1, [i1, j1]] == vxi[0, :]):
                if1 = True

            i2 = (k - 1) % deg2
            j2 = k % deg2
            if np.all(T[tri2, [i2, j2]] == vxi[1, :]):
                if2 = True

        if if1:
            tri_start = tri1
        elif if2:
            tri_start = tri2
        else:
            raise ValueError("Could not find a starting triangle.")

    # trii = find(any(T == idx, 2));

    # Find all triangles containing vertex idx
    trii = np.where(np.any(T == idx, axis=1))[0]

    # if ifbound
    #     trii = circshift(trii, [1-find(trii == tri_start),0]);
    # end

    if ifbound and tri_start is not None:
        # Rotate trii so that tri_start is first
        start_pos = np.where(trii == tri_start)[0]
        if len(start_pos) > 0:
            trii = np.roll(trii, -start_pos[0])

    # tri_ord = zeros(length(trii),1);
    # tri_ord(1) = trii(1);
    # tovisit = trii(2:end);
    # k = 1;

    tri_ord = np.full(len(trii), -1, dtype=int)  # Use -1 sentinel (0 is valid index)
    tri_ord[0] = trii[0]
    tovisit = list(trii[1:])
    k = 0  # 0-indexed

    # while ~isempty(tovisit)
    #     [t,~,id] = intersect(T2T(tri_ord(k),:), tovisit);
    #
    #     % pick a consistent ordering
    #     if length(t) == 2
    #         if k == 1
    #             t1 = T(t(1),:);
    #             t1 = circshift(t1, [0,1-find(t1 == idx)]);
    #
    #             if all(ismember(t1(1:2), T(trii(1),:)))
    #                 t = t(1);
    #                 id = id(1);
    #             else
    #                 t = t(2);
    #                 id = id(2);
    #             end
    #         else
    #             error('Problem when sorting triangles.');
    #         end
    #     end
    #
    #     k = k + 1;
    #     tri_ord(k) = t;
    #     tovisit(id) = [];
    # end

    while len(tovisit) > 0:
        # Find intersection of neighbors of current triangle with tovisit
        neighbors = T2T[tri_ord[k], :]
        # intersect returns values in both arrays
        t_list = []
        id_list = []
        for i, tv in enumerate(tovisit):
            if tv in neighbors:
                t_list.append(tv)
                id_list.append(i)

        if len(t_list) == 0:
            break

        t = t_list
        id_ = id_list

        # Pick a consistent ordering if 2 options
        if len(t) == 2:
            if k == 0:  # k == 1 in MATLAB (1-indexed)
                t1_verts = T[t[0], :].copy()
                # circshift to put idx first
                idx_pos = np.where(t1_verts == idx)[0]
                if len(idx_pos) > 0:
                    t1_verts = np.roll(t1_verts, -idx_pos[0])

                # Check if first two vertices of t1 are both in trii[0]
                if np.all(np.isin(t1_verts[0:2], T[trii[0], :])):
                    t = [t[0]]
                    id_ = [id_[0]]
                else:
                    t = [t[1]]
                    id_ = [id_[1]]
            else:
                raise ValueError("Problem when sorting triangles.")

        k = k + 1
        tri_ord[k] = t[0]
        # Remove visited triangle from tovisit
        tovisit.pop(id_[0])

    # Validate: check for unfilled entries (sentinel -1 still present, excluding boundary marker)
    unfilled = np.sum(tri_ord == -1)
    if unfilled > 0:
        raise ValueError(f"Triangle ordering incomplete: {unfilled} unfilled entries for vertex {idx}")

    # if ifbound
    #     tri_ord = [tri_ord; 0];
    # end

    if ifbound:
        # Append -1 (Python's "no neighbor" marker, MATLAB uses 0)
        tri_ord = np.append(tri_ord, -1)

    # % sort edges
    # if nargout >= 2

    edge_ord = None
    sign_edge = None

    # if length(tri_ord) > 2
    #     path = [tri_ord, circshift(tri_ord, [1,0])];
    #     paths = sort(path,2);
    #     if exist('T2E','var')
    #         ide = unique(abs(T2E(trii,:)));
    #         [~,~,edge_ord] = intersect(paths, sort(E2T(ide,1:2),2), 'rows', 'stable');
    #         edge_ord = ide(edge_ord);
    #     else
    #         [~,~,edge_ord] = intersect(paths, sort(E2T(:,1:2),2), 'rows', 'stable');
    #     end
    #     sign_edge = (path(:,1) == E2T(edge_ord,1)) .* E2T(edge_ord,3) + (path(:,1) == E2T(edge_ord,2)) .* E2T(edge_ord,4);
    #
    #     if length(edge_ord) ~= length(tri_ord)
    #         assert(size(unique(paths, 'rows'),1) == size(paths,1), 'Non manifold mesh.');
    #         error('Could not find edge ordering.');
    #     end
    # else
    #     path = [tri_ord, circshift(tri_ord, [1,0])];
    #     edge_ord = find((E2T(:,1) == tri_ord(1)) & (E2T(:,2) == tri_ord(2)));
    #     sign_edge = (path(:,1) == E2T(edge_ord,1)) .* E2T(edge_ord,3) + (path(:,1) == E2T(edge_ord,2)) .* E2T(edge_ord,4);
    # end

    # Count actual triangles (excluding boundary marker -1)
    n_triangles = np.sum(tri_ord != -1)

    if n_triangles > 2 or (n_triangles == 2 and not ifbound):
        # Multiple triangles: compute edge ordering around the ring
        # path = [tri_ord, circshift(tri_ord, [1,0])]
        # circshift with [1,0] shifts rows down by 1
        path = np.column_stack([tri_ord, np.roll(tri_ord, 1)])
        paths_sorted = np.sort(path, axis=1)

        if T2E is not None:
            # ide = unique(abs(T2E(trii,:)))
            # T2E uses 1-based signed encoding, so abs(T2E)-1 gives 0-based edge indices
            ide = np.unique(np.abs(T2E[trii, :].ravel()) - 1)
            # Remove -1 if present (no edge marker, from edges that were index 0 in broken encoding)
            ide = ide[ide >= 0]

            # Sort E2T rows for comparison
            E2T_subset = E2T[ide, 0:2]
            E2T_subset_sorted = np.sort(E2T_subset, axis=1)

            # Find edge_ord: for each row in paths_sorted, find matching row in E2T_subset_sorted
            edge_ord = np.full(len(tri_ord), -1, dtype=int)  # Use -1 sentinel (0 is valid index)
            for i, p_row in enumerate(paths_sorted):
                for j, e_row in enumerate(E2T_subset_sorted):
                    if np.all(p_row == e_row):
                        edge_ord[i] = ide[j]
                        break

            # Validate: check for unfilled edge entries
            unfilled = np.sum(edge_ord == -1)
            if unfilled > 0:
                raise ValueError(f"Edge ordering incomplete: {unfilled} edges not matched for vertex {idx}")
        else:
            # Sort all E2T rows for comparison
            E2T_sorted = np.sort(E2T[:, 0:2], axis=1)

            edge_ord = np.full(len(tri_ord), -1, dtype=int)  # Use -1 sentinel (0 is valid index)
            for i, p_row in enumerate(paths_sorted):
                matches = np.where(np.all(E2T_sorted == p_row, axis=1))[0]
                if len(matches) > 0:
                    edge_ord[i] = matches[0]

            # Validate: check for unfilled edge entries
            unfilled = np.sum(edge_ord == -1)
            if unfilled > 0:
                raise ValueError(f"Edge ordering incomplete: {unfilled} edges not matched for vertex {idx}")

        # sign_edge = (path(:,1) == E2T(edge_ord,1)) .* E2T(edge_ord,3) + (path(:,1) == E2T(edge_ord,2)) .* E2T(edge_ord,4)
        sign_edge = (
            (path[:, 0] == E2T[edge_ord, 0]).astype(int) * E2T[edge_ord, 2] +
            (path[:, 0] == E2T[edge_ord, 1]).astype(int) * E2T[edge_ord, 3]
        )

        if len(edge_ord) != len(tri_ord):
            # Check for non-manifold
            unique_paths = np.unique(paths_sorted, axis=0)
            assert unique_paths.shape[0] == paths_sorted.shape[0], "Non manifold mesh."
            raise ValueError("Could not find edge ordering.")

    elif n_triangles == 2 and ifbound:
        # Boundary vertex with exactly 2 triangles: edge ordering is just the edge between them
        # tri_ord = [t1, t2, -1], we want edge between t1 and t2
        t1, t2 = tri_ord[0], tri_ord[1]
        matches = np.where(
            ((E2T[:, 0] == t1) & (E2T[:, 1] == t2)) |
            ((E2T[:, 0] == t2) & (E2T[:, 1] == t1))
        )[0]

        if len(matches) > 0:
            # Only return the single valid edge (between t1 and t2)
            # The boundary "edges" (t2 to -1, -1 to t1) are not real edges
            edge_ord = np.array([matches[0]], dtype=int)

            sign_val = (
                int(t1 == E2T[matches[0], 0]) * E2T[matches[0], 2] +
                int(t1 == E2T[matches[0], 1]) * E2T[matches[0], 3]
            )
            sign_edge = np.array([sign_val], dtype=int)
        else:
            edge_ord = np.array([], dtype=int)
            sign_edge = np.array([], dtype=int)

    elif n_triangles == 1:
        # Single triangle (boundary vertex with only 1 triangle)
        # No ring of edges to sort - edges are incident to the vertex, not between triangles
        # Return empty arrays (not None) for consistency with the API
        edge_ord = np.array([], dtype=int)
        sign_edge = np.array([], dtype=int)

    else:
        # n_triangles == 0 or other edge case
        edge_ord = np.array([], dtype=int)
        sign_edge = np.array([], dtype=int)

    return tri_ord, edge_ord, sign_edge
