

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List

from rectangular_surface_parameterization.preprocessing.sort_triangles import sort_triangles


# function [Edge_jump,v2t,base_tri] = reduce_corner_var_2d(mesh)

def reduce_corner_var_2d(mesh, allow_open_mesh: bool = False) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """
    Reduce corner variables to vertex variables without cut edges.

    Builds a reduction matrix that maps vertex variables to corner variables
    by accumulating edge jumps around each vertex.

    WARNING: This function is designed for CLOSED meshes only. For meshes with
    boundaries, use reduce_corner_var_2d_cut instead, which properly handles
    cut edges and boundary vertices.

    Parameters
    ----------
    mesh : mesh data structure
        Contains nv, nf, ne, T, E2T, T2T, E2V, T2E
    allow_open_mesh : bool, default False
        If False, raises ValueError for meshes with boundary edges.
        Set to True only if you understand the limitations on open meshes.

    Returns
    -------
    Edge_jump : sparse matrix (3*nf, ne)
        Edge jump accumulation matrix
    v2t : sparse matrix (3*nf, nv)
        Vertex to corner mapping matrix
    base_tri : ndarray (nv,)
        Base triangle for each vertex

    Raises
    ------
    ValueError
        If mesh has boundary edges and allow_open_mesh is False.
    """
    # Check for boundary edges (E2T[:, 1] == -1 indicates boundary)
    if hasattr(mesh, 'E2T'):
        has_boundary = np.any(mesh.edge_to_triangle[:, 1] < 0)
        if has_boundary and not allow_open_mesh:
            raise ValueError(
                "reduce_corner_var_2d is designed for closed meshes only. "
                "For meshes with boundaries, use reduce_corner_var_2d_cut instead, "
                "or set allow_open_mesh=True if you understand the limitations."
            )
    # base_tri = zeros(mesh.num_vertices,1);
    # path_vx = cell(mesh.num_vertices,1);
    # path_edge = cell(mesh.num_vertices,1);
    # Tc = reshape((1:3*mesh.num_faces)', [mesh.num_faces,3]);

    base_tri = np.zeros(mesh.num_vertices, dtype=int)
    path_vx: List[np.ndarray] = [None] * mesh.num_vertices
    path_edge: List[np.ndarray] = [None] * mesh.num_vertices

    # Tc: corner indices, reshaped so Tc[f, c] gives corner index for face f, corner c
    # MATLAB: reshape((1:3*nf)', [nf,3]) - column-major
    # This gives [0,nf,2nf; 1,nf+1,2nf+1; ...] in 0-indexed
    Tc = np.arange(3 * mesh.num_faces).reshape((mesh.num_faces, 3), order='F')

    # for i = 1:mesh.num_vertices
    for i in range(mesh.num_vertices):
        # [tri_ord,edge_ord,sign_edge] = sort_triangles(i, mesh.triangles, mesh.edge_to_triangle, mesh.triangle_to_triangle, mesh.edge_to_vertex, mesh.T2E);
        # edge_ord = edge_ord.*sign_edge;

        tri_ord, edge_ord, sign_edge = sort_triangles(i, mesh.triangles, mesh.edge_to_triangle, mesh.triangle_to_triangle, mesh.edge_to_vertex, mesh.T2E)
        # Create signed edge indices using 1-based encoding: (idx+1)*sign
        edge_ord_signed = (edge_ord + 1) * sign_edge

        # base_tri(i) = tri_ord(1);

        base_tri[i] = tri_ord[0]

        # if tri_ord(end) == 0
        # else
        # end

        # In Python, boundary is indicated by tri_ord[-1] == -1 (sentinel value)
        if tri_ord[-1] == -1:
            # Boundary vertex case
            # idvx = sum(Tc(tri_ord(1:end-1),:) .* (mesh.triangles(tri_ord(1:end-1),:) == i), 2);
            # path_vx{i} = [idvx, i*ones(length(idvx),1)];

            tri_valid = tri_ord[:-1]  # Exclude the boundary sentinel
            # Find corner indices where vertex i appears in each triangle
            mask = (mesh.triangles[tri_valid, :] == i)
            idvx = np.sum(Tc[tri_valid, :] * mask, axis=1)

            path_vx[i] = np.column_stack([idvx, np.full(len(idvx), i)])

            # I = repelem(idvx(2:end), (1:length(idvx)-1)');
            # J = zeros(size(I));
            # d = edge_ord(2:end-1);
            # id = 1;
            # for k = 1:length(idvx)-1
            #     J(id) = d(1:k);
            #     id = id(end) + (1:k+1);
            # end
            # path_edge{i} = [I, J];

            if len(idvx) > 1:
                # Build I: repeat idvx[1:] with counts 1, 2, 3, ...
                counts = np.arange(1, len(idvx))
                I = np.repeat(idvx[1:], counts)
                J = np.zeros(len(I), dtype=int)

                d = edge_ord_signed[1:-1]  # edge_ord(2:end-1) in MATLAB (boundary has extra sentinel)
                idx = 0
                for k in range(1, len(idvx)):
                    J[idx:idx + k] = d[:k]
                    idx += k

                path_edge[i] = np.column_stack([I, J])
            else:
                path_edge[i] = np.zeros((0, 2), dtype=int)
        else:
            # Interior vertex case
            # idvx = sum(Tc(tri_ord,:) .* (mesh.triangles(tri_ord,:) == i), 2);
            # path_vx{i} = [idvx, i*ones(length(idvx),1)];

            mask = (mesh.triangles[tri_ord, :] == i)
            idvx = np.sum(Tc[tri_ord, :] * mask, axis=1)

            path_vx[i] = np.column_stack([idvx, np.full(len(idvx), i)])

            # I = repelem(idvx(2:end), (1:length(idvx)-1)');
            # J = zeros(size(I));
            # d = edge_ord(2:end);
            # id = 1;
            # for k = 1:length(idvx)-1
            #     J(id) = d(1:k);
            #     id = id(end) + (1:k+1);
            # end
            # path_edge{i} = [I, J];

            if len(idvx) > 1:
                counts = np.arange(1, len(idvx))
                I = np.repeat(idvx[1:], counts)
                J = np.zeros(len(I), dtype=int)

                d = edge_ord_signed[1:]  # edge_ord(2:end) in MATLAB
                idx = 0
                for k in range(1, len(idvx)):
                    J[idx:idx + k] = d[:k]
                    idx += k

                path_edge[i] = np.column_stack([I, J])
            else:
                path_edge[i] = np.zeros((0, 2), dtype=int)

    # I = cell2mat(path_edge);
    # Edge_jump = sparse(I(:,1), abs(I(:,2)), sign(I(:,2)), 3*mesh.num_faces, mesh.num_edges);

    # Concatenate all path_edge arrays
    path_edge_valid = [pe for pe in path_edge if pe is not None and len(pe) > 0]
    if len(path_edge_valid) > 0:
        I_edge = np.vstack(path_edge_valid)
        # Decode signed edge indices: abs(x)-1 for 0-based index, sign(x) for sign
        row_idx = I_edge[:, 0]
        col_idx = np.abs(I_edge[:, 1]) - 1  # Convert back to 0-based
        vals = np.sign(I_edge[:, 1])
        Edge_jump = sp.coo_matrix((vals, (row_idx, col_idx)),
                                  shape=(3 * mesh.num_faces, mesh.num_edges)).tocsr()
    else:
        Edge_jump = sp.csr_matrix((3 * mesh.num_faces, mesh.num_edges))

    # I = cell2mat(path_vx);
    # v2t = sparse(I(:,1), I(:,2), 1, 3*mesh.num_faces, mesh.num_vertices);

    path_vx_valid = [pv for pv in path_vx if pv is not None and len(pv) > 0]
    if len(path_vx_valid) > 0:
        I_vx = np.vstack(path_vx_valid)
        row_idx = I_vx[:, 0]
        col_idx = I_vx[:, 1]
        vals = np.ones(len(row_idx))
        v2t = sp.coo_matrix((vals, (row_idx, col_idx)),
                            shape=(3 * mesh.num_faces, mesh.num_vertices)).tocsr()
    else:
        v2t = sp.csr_matrix((3 * mesh.num_faces, mesh.num_vertices))

    return Edge_jump, v2t, base_tri
