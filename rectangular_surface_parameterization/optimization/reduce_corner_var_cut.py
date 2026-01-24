

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List, Optional, Union

from rectangular_surface_parameterization.preprocessing.sort_triangles import sort_triangles


# function [Edge_jump,v2t,base_tri] = reduce_corner_var_2d_cut(mesh, ide_cut)

def reduce_corner_var_2d_cut(
    mesh,
    ide_cut: Optional[Union[np.ndarray, None]] = None
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Reduce corner variables to vertex variables with cut edges.

    Similar to reduce_corner_var_2d but handles cut edges that split vertices
    into multiple copies in the parameterization.

    Parameters
    ----------
    mesh : mesh data structure
        Contains nv, nf, ne, T, E2T, T2T, E2V, T2E
    ide_cut : ndarray, optional
        Boolean array or index array marking cut edges.
        If None or empty, defaults to all False (no cuts).

    Returns
    -------
    Edge_jump : sparse matrix (3*nf, ne)
        Edge jump accumulation matrix
    v2t : sparse matrix (3*nf, nv_total)
        Vertex to corner mapping matrix (nv_total >= nv due to cut vertex copies)
    base_tri : ndarray (3*nf,)
        Base triangle for each corner
    """
    # if ~exist('ide_cut','var') || isempty(ide_cut)
    #     ide_cut = false(mesh.num_edges,1);
    # end
    # if ~islogical(ide_cut)
    #     ide_cutb = false(mesh.num_edges,1);
    #     ide_cutb(ide_cut) = true;
    #     ide_cut = ide_cutb;
    # end

    if ide_cut is None or len(ide_cut) == 0:
        ide_cut = np.zeros(mesh.num_edges, dtype=bool)
    elif not ide_cut.dtype == bool:
        # Convert index array to boolean mask
        ide_cutb = np.zeros(mesh.num_edges, dtype=bool)
        ide_cutb[ide_cut] = True
        ide_cut = ide_cutb

    # Make a mutable copy since we modify ide_cut for boundary edges
    ide_cut = ide_cut.copy()

    # base_tri = zeros(mesh.num_faces,3);
    # path_vx = cell(mesh.num_vertices,1);
    # path_edge = cell(mesh.num_vertices,1);
    # Tc = reshape((1:3*mesh.num_faces)', [mesh.num_faces,3]);
    # nv = mesh.num_vertices;

    base_tri = np.zeros(3 * mesh.num_faces, dtype=int)
    # Use dict instead of fixed-size list since we may add more vertices
    path_vx: dict = {}
    path_edge: dict = {}

    # Tc: corner indices, reshaped so Tc[f, c] gives corner index for face f, corner c
    Tc = np.arange(3 * mesh.num_faces).reshape((mesh.num_faces, 3), order='F')
    nv = mesh.num_vertices

    # for i = 1:mesh.num_vertices
    for i in range(mesh.num_vertices):
        # [tri_ord,edge_ord,sign_edge] = sort_triangles(i, mesh.triangles, mesh.edge_to_triangle, mesh.triangle_to_triangle, mesh.edge_to_vertex, mesh.T2E);
        # edge_ord = edge_ord.*sign_edge;

        tri_ord, edge_ord, sign_edge = sort_triangles(i, mesh.triangles, mesh.edge_to_triangle, mesh.triangle_to_triangle, mesh.edge_to_vertex, mesh.T2E)
        # Create signed edge indices using 1-based encoding: (idx+1)*sign
        edge_ord_signed = (edge_ord + 1) * sign_edge

        # Convert to lists for easier manipulation
        tri_ord = list(tri_ord)
        edge_ord_signed = list(edge_ord_signed)

        # if tri_ord(end) == 0
        #     ifbound = true;
        #
        #     ide_cut(abs(edge_ord([1 end]))) = false;
        #     tri_ord(end) = [];
        #     edge_ord(end) = [];
        # else
        #     ifbound = false;
        #
        #     id = ide_cut(abs(edge_ord));
        #     if any(id)
        #         tri_ord = circshift(tri_ord, [1-find(id, 1, 'first'),0]);
        #         edge_ord = circshift(edge_ord, [1-find(id, 1, 'first'),0]);
        #
        #         id = ide_cut(abs(edge_ord));
        #         assert(id(1))
        #     end
        # end

        # Check for boundary vertex (sentinel value -1 in Python)
        if tri_ord[-1] == -1:
            ifbound = True

            # Boundary edges are not cut edges - decode signed edge index
            # Only process if we have edges (single-triangle vertices have no inter-triangle edges)
            if len(edge_ord_signed) > 0:
                first_edge = np.abs(edge_ord_signed[0]) - 1
                last_edge = np.abs(edge_ord_signed[-1]) - 1
                ide_cut[first_edge] = False
                ide_cut[last_edge] = False

            # Remove boundary sentinel
            tri_ord = tri_ord[:-1]
            edge_ord_signed = edge_ord_signed[:-1]
        else:
            ifbound = False

            # Get cut edge status for each edge around vertex
            edge_indices = [np.abs(e) - 1 for e in edge_ord_signed]
            id_cut = ide_cut[edge_indices]

            if np.any(id_cut):
                # Rotate so first edge is a cut edge
                first_cut = np.argmax(id_cut)  # find first True
                tri_ord = np.roll(tri_ord, -first_cut).tolist()
                edge_ord_signed = np.roll(edge_ord_signed, -first_cut).tolist()

                # Verify rotation worked
                edge_indices = [np.abs(e) - 1 for e in edge_ord_signed]
                id_cut = ide_cut[edge_indices]
                assert id_cut[0], "First edge should be cut after rotation"

        # id = ide_cut(abs(edge_ord));
        # n = sum(id);

        edge_indices = [np.abs(e) - 1 for e in edge_ord_signed]
        id_cut = ide_cut[edge_indices]
        n = np.sum(id_cut)

        # p = 0;
        # flag = zeros(length(edge_ord),1);
        # for j = 1:length(edge_ord)
        #     if id(j)
        #         p = p + 1;
        #     end
        #     flag(j) = p;
        # end
        # if ~ifbound
        #     flag(flag == n) = 0;
        #     n = max(0,n-1);
        # end

        p = 0
        flag = np.zeros(len(edge_ord_signed), dtype=int)
        for j in range(len(edge_ord_signed)):
            if id_cut[j]:
                p += 1
            flag[j] = p

        if not ifbound:
            flag[flag == n] = 0
            n = max(0, n - 1)

        # for j = 0:n
        for j in range(n + 1):
            # idt = tri_ord(flag == j);
            # idvx = sum(Tc(idt,:) .* (mesh.triangles(idt,:) == i), 2);
            # assert(~isempty(idvx))

            idt = np.array(tri_ord)[flag == j]
            if len(idt) == 0:
                continue

            mask = (mesh.triangles[idt, :] == i)
            idvx = np.sum(Tc[idt, :] * mask, axis=1)
            assert len(idvx) > 0, "idvx should not be empty"

            # if j == 0
            #     path_vx{i} = [idvx, i*ones(length(idvx),1)];
            # else
            #     path_vx{nv+j} = [idvx, (nv+j)*ones(length(idvx),1)];
            # end

            if j == 0:
                path_vx[i] = np.column_stack([idvx, np.full(len(idvx), i)])
            else:
                # New vertex index: nv + (j-1) since j starts at 1 for first new vertex
                # but new vertex indices should start at nv (the current vertex count)
                new_vx_idx = nv + (j - 1)
                path_vx[new_vx_idx] = np.column_stack([idvx, np.full(len(idvx), new_vx_idx)])

            # I = repelem(idvx(2:end), (1:length(idvx)-1)');
            # J = zeros(size(I));
            # d = edge_ord(flag == j);
            # d(1) = [];
            # id = 1;
            # for k = 1:length(idvx)-1
            #     J(id) = d(1:k);
            #     id = id(end) + (1:k+1);
            # end

            if len(idvx) > 1:
                counts = np.arange(1, len(idvx))
                I = np.repeat(idvx[1:], counts)
                J = np.zeros(len(I), dtype=int)

                d = np.array(edge_ord_signed)[flag == j]
                d = d[1:]  # Remove first element

                idx = 0
                for k in range(1, len(idvx)):
                    J[idx:idx + k] = d[:k]
                    idx += k

                # if j == 0
                #     path_edge{i} = [I, J];
                #     if isempty(path_edge{i})
                #         path_edge{i} = zeros(0,2);
                #     end
                # else
                #     path_edge{nv+j} = [I, J];
                #     if isempty(path_edge{nv+j})
                #         path_edge{nv+j} = zeros(0,2);
                #     end
                # end

                if j == 0:
                    path_edge[i] = np.column_stack([I, J])
                else:
                    new_vx_idx = nv + (j - 1)
                    path_edge[new_vx_idx] = np.column_stack([I, J])
            else:
                if j == 0:
                    path_edge[i] = np.zeros((0, 2), dtype=int)
                else:
                    new_vx_idx = nv + (j - 1)
                    path_edge[new_vx_idx] = np.zeros((0, 2), dtype=int)

            # base_tri(idvx) = idt(1);

            base_tri[idvx] = idt[0]

        # if n >= 1
        #     nv = nv + n;
        # end

        if n >= 1:
            nv = nv + n

    # I = cell2mat(path_edge);
    # Edge_jump = sparse(I(:,1), abs(I(:,2)), sign(I(:,2)), 3*mesh.num_faces, mesh.num_edges);

    path_edge_valid = [pe for pe in path_edge.values() if pe is not None and len(pe) > 0]
    if len(path_edge_valid) > 0:
        I_edge = np.vstack(path_edge_valid)
        row_idx = I_edge[:, 0]
        col_idx = np.abs(I_edge[:, 1]) - 1  # Convert back to 0-based
        vals = np.sign(I_edge[:, 1])
        Edge_jump = sp.coo_matrix((vals, (row_idx, col_idx)),
                                  shape=(3 * mesh.num_faces, mesh.num_edges)).tocsr()
    else:
        Edge_jump = sp.csr_matrix((3 * mesh.num_faces, mesh.num_edges))

    # I = cell2mat(path_vx);
    # v2t = sparse(I(:,1), I(:,2), 1, 3*mesh.num_faces, nv);

    path_vx_valid = [pv for pv in path_vx.values() if pv is not None and len(pv) > 0]
    if len(path_vx_valid) > 0:
        I_vx = np.vstack(path_vx_valid)
        row_idx = I_vx[:, 0]
        col_idx = I_vx[:, 1]
        vals = np.ones(len(row_idx))
        v2t = sp.coo_matrix((vals, (row_idx, col_idx)),
                            shape=(3 * mesh.num_faces, nv)).tocsr()
    else:
        v2t = sp.csr_matrix((3 * mesh.num_faces, nv))

    return Edge_jump, v2t, base_tri
