

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np

from rectangular_surface_parameterization.core.signed_edge_array import SignedEdgeArray


# function [E2V, T2E, E2T, T2T] = connectivity(T)

def connectivity(T):
    """
    Computes the adjacency properties given a list of triangles.

    Parameters
    ----------
    T : ndarray (nf, 3)
        Triangle vertex indices (0-indexed)

    Returns
    -------
    E2V : ndarray (ne, 2)
        Edge to vertices mapping (each row is [v0, v1] sorted), 0-indexed
    T2E : SignedEdgeArray (nf, 3)
        Triangle to edge mapping with orientation signs.
        Use T2E.indices for 0-based edge indices, T2E.signs for orientations.
    E2T : ndarray (ne, 4)
        Edge to triangles: [tri0, tri1, sign0, sign1] where sign indicates orientation.
        tri values are 0-indexed, -1 means no neighbor (boundary).
    T2T : ndarray (nf, 3)
        Triangle to triangle neighbors, 0-indexed. -1 means no neighbor (boundary).
    """

    # nf = size(T,1);
    # nv = max(T(:));

    nf = T.shape[0]
    nv = T.max() + 1  # +1 because 0-indexed

    # E2V = [T(:,1) T(:,2) ; T(:,2) T(:,3) ; T(:,3) T(:,1)];
    # [E2V,id] = sort(E2V,2);
    # [E2V,ia,ic] = unique(E2V,'rows');

    # Build all directed edges (each triangle contributes 3 edges)
    E2V_unsorted = np.vstack([
        np.column_stack([T[:, 0], T[:, 1]]),  # edge 0: v0 -> v1
        np.column_stack([T[:, 1], T[:, 2]]),  # edge 1: v1 -> v2
        np.column_stack([T[:, 2], T[:, 0]])   # edge 2: v2 -> v0
    ])  # shape: (3*nf, 2)

    # Sort each edge so smaller vertex comes first, track which column was first
    sort_idx = np.argsort(E2V_unsorted, axis=1)
    E2V_sorted = np.take_along_axis(E2V_unsorted, sort_idx, axis=1)

    # Get unique edges
    E2V, ia, ic = np.unique(E2V_sorted, axis=0, return_index=True, return_inverse=True)
    ne = E2V.shape[0]

    # edgeSg = id(:,1) - id(:,2);
    # t2es = [edgeSg(1:nf), edgeSg((nf+1):(2*nf)), edgeSg((2*nf+1):end)];
    # edgeSg = edgeSg(ia);

    # Edge sign: +1 if original order matched sorted order, -1 if swapped
    # In MATLAB: id(:,1) - id(:,2) where id is sort indices
    # id(:,1)=1, id(:,2)=2 means no swap -> sign = -1
    # id(:,1)=2, id(:,2)=1 means swap -> sign = +1
    edge_sg_all = sort_idx[:, 0] - sort_idx[:, 1]  # 0-1=-1 or 1-0=+1
    t2es = np.column_stack([
        edge_sg_all[:nf],
        edge_sg_all[nf:2*nf],
        edge_sg_all[2*nf:]
    ])
    edge_sg = edge_sg_all[ia]

    # e1 = ic(1:nf);
    # e2 = ic((nf+1):(2*nf));
    # e3 = ic((2*nf+1):end);
    # T2E = [e1 e2 e3];

    e1 = ic[:nf]
    e2 = ic[nf:2*nf]
    e3 = ic[2*nf:]
    T2E = np.column_stack([e1, e2, e3])  # unsigned for now

    # bound = find(accumarray(T2E(:), 1) == 1);
    # nB = length(bound);

    # Count how many times each edge appears (1 = boundary, 2 = interior)
    edge_counts = np.bincount(T2E.ravel(), minlength=ne)
    bound = np.where(edge_counts == 1)[0]
    nB = len(bound)

    # [~, idx] = sort([e1; e2; e3; bound]);
    # idx(idx <= 3*nf) = mod(idx(idx <= 3*nf)-1, nf)+1;
    # E2T = reshape(idx', [2, size(E2V,1)])';
    # E2T(E2T > nf) = 0;
    # E2T = [E2T, edgeSg, -edgeSg];

    # Build E2T: for each edge, find the two triangles (or one if boundary)
    # MATLAB approach: sort edge indices and use the sorted indices to find triangles
    all_edges = np.concatenate([e1, e2, e3, bound])
    idx = np.argsort(all_edges)

    # Convert sorted indices to triangle indices
    # Indices 0 to 3*nf-1 correspond to triangles, indices >= 3*nf are boundary markers
    idx_tri = idx.copy()
    mask = idx < 3 * nf
    idx_tri[mask] = idx[mask] % nf  # triangle index (0-indexed)

    # Reshape to (ne, 2) - each edge has at most 2 adjacent triangles
    E2T_tri = idx_tri.reshape((ne, 2))

    # Mark boundary edges (where index came from boundary marker)
    E2T_tri[E2T_tri >= nf] = -1  # Use -1 for no neighbor (will convert to 0 below for MATLAB compat)

    # For compatibility with MATLAB code that uses 0 for no neighbor:
    E2T_tri[E2T_tri == -1] = -1  # Keep as -1 internally, or use 0 if needed

    # Actually let's use -1 consistently for "no neighbor" in Python
    # But MATLAB uses 0, so for E2T output we'll use -1 (Python convention)
    E2T = np.column_stack([E2T_tri, edge_sg, -edge_sg])

    # T2T = [E2T(T2E(:,1),1), E2T(T2E(:,2),1), E2T(T2E(:,3),1), E2T(T2E(:,1),2), E2T(T2E(:,2),2), E2T(T2E(:,3),2)];
    # T2T = sort((T2T ~= repmat((1:nf)', [1,6])).*T2T, 2);
    # T2T = T2T(:,4:6);

    # Get neighbor triangles for each triangle
    # For each triangle's edges, look up the adjacent triangles
    T2T_raw = np.column_stack([
        E2T[T2E[:, 0], 0], E2T[T2E[:, 1], 0], E2T[T2E[:, 2], 0],
        E2T[T2E[:, 0], 1], E2T[T2E[:, 1], 1], E2T[T2E[:, 2], 1]
    ])

    # Filter out self-references (triangle shouldn't be its own neighbor)
    tri_indices = np.arange(nf).reshape(-1, 1)
    T2T_filtered = np.where(T2T_raw != tri_indices, T2T_raw, -1)

    # Sort each row and take the last 3 columns (the actual neighbors)
    T2T_sorted = np.sort(T2T_filtered, axis=1)
    T2T = T2T_sorted[:, 3:6]  # shape: (nf, 3)

    # T2E = T2E.*t2es;

    # Apply signs to T2E. Previously used 1-based encoding to avoid edge-0 sign loss.
    # Now using SignedEdgeArray which handles the encoding internally.
    T2E_signed = SignedEdgeArray.from_edges_and_signs(T2E, t2es)


    # (V2T computation is commented out in MATLAB, skipping)

    return E2V, T2E_signed, E2T, T2T


def check_mesh_connected(mesh) -> bool:
    """
    Check if a mesh is a single connected component.

    Uses BFS on the vertex adjacency graph to verify all vertices
    are reachable from vertex 0.

    Parameters
    ----------
    mesh : MeshInfo
        Mesh with E2V edge-to-vertex mapping

    Returns
    -------
    bool
        True if mesh is connected, False otherwise
    """
    from collections import deque

    nv = mesh.num_vertices
    E2V = mesh.edge_to_vertex

    # Build vertex adjacency list
    adj = [[] for _ in range(nv)]
    for e in range(E2V.shape[0]):
        v0, v1 = E2V[e, 0], E2V[e, 1]
        adj[v0].append(v1)
        adj[v1].append(v0)

    # BFS from vertex 0
    visited = np.zeros(nv, dtype=bool)
    queue = deque([0])
    visited[0] = True
    count = 1

    while queue:
        v = queue.popleft()
        for neighbor in adj[v]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)

    return count == nv
