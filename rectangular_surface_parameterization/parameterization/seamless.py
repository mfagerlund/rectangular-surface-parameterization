
# function [disk_mesh,dec_cut,Align,Rot] = mesh_to_disk_seamless(Src, param, ang, sing, k21, ifseamless_const, ifboundary, ifhardedge)
#
# idcone = param.idx_int(abs(sing(param.idx_int)) > 0.1); % Cone indices
# [disk_mesh,~,ide_cut_inv,~] = cut_mesh(Src.vertices, Src.triangles, Src.edge_to_vertex, Src.edge_to_triangle, Src.T2E, Src.triangle_to_triangle, idcone, k21 ~= 1);
# dec_cut = dec_tri(disk_mesh);
#
# if ifseamless_const && nargout > 2
#     edge_bound_cut = find(any(disk_mesh.edge_to_triangle(:,1:2) == 0, 2));
#     edge_bound_cut = [edge_bound_cut, sum(disk_mesh.edge_to_triangle(edge_bound_cut,1:2),2)];
#     edge_bound_cut = edge_bound_cut(~ismember(abs(ide_cut_inv(edge_bound_cut(:,1))), param.ide_bound),:);
#     [~,id] = sort(abs(ide_cut_inv(edge_bound_cut(:,1))));
#     ide_cut = ide_cut_inv(edge_bound_cut(id,1));
#     assert(all(abs(ide_cut(1:2:end-1)) == abs(ide_cut(2:2:end))), 'Cut failed.');
#     ide_cut_cor = [abs(ide_cut(1:2:end-1)), sign(ide_cut(1:2:end-1)).*edge_bound_cut(id(1:2:end-1),1), sign(ide_cut(2:2:end)).*edge_bound_cut(id(2:2:end),1)];
#     tri_cut_cor = [edge_bound_cut(id(1:2:end-1),2), edge_bound_cut(id(2:2:end),2)];
#     ide_cut_cor(:,2:3) = (tri_cut_cor(:,1) == param.edge_to_triangle(ide_cut_cor(:,1),1)).*ide_cut_cor(:,[2 3]) + (tri_cut_cor(:,2) == param.edge_to_triangle(ide_cut_cor(:,1),1)).*ide_cut_cor(:,[3 2]);
#
#     R0 = eye(2,2);
#     R1 = [0,-1; 1,0];
#     R2 = R1^2;
#     R3 = R1^3;
#     R = [R0(:), R1(:), R2(:), R3(:)]';
#     R = matrix_vector_multiplication(R(k21(ide_cut_cor(:,1)),:));
#     I1 = sparse(1:size(ide_cut_cor,1), abs(ide_cut_cor(:,2)), sign(ide_cut_cor(:,2)), size(ide_cut_cor,1), disk_mesh.num_edges);
#     I2 = sparse(1:size(ide_cut_cor,1), abs(ide_cut_cor(:,3)), sign(ide_cut_cor(:,3)), size(ide_cut_cor,1), disk_mesh.num_edges);
#     Rot = blkdiag(I1*dec_cut.d0p,I1*dec_cut.d0p) - R*blkdiag(I2*dec_cut.d0p,I2*dec_cut.d0p);
#
#     Align = sparse(0,2*disk_mesh.num_vertices);
#     if (ifboundary && ~isempty(param.ide_bound)) || (ifhardedge && ~isempty(param.ide_hard))
#         ide_fix_cut = find(ismember(abs(ide_cut_inv), param.ide_fix));
#         tri_fix_cut = vec(disk_mesh.edge_to_triangle(ide_fix_cut,1:2));
#         id = tri_fix_cut ~= 0;
#         tri_fix_cut = tri_fix_cut(id);
#         [~,ia] = ismember(param.tri_fix, tri_fix_cut);
#         assert(length(ia) == length(tri_fix_cut))
#         ide_fix_cut = [ide_fix_cut; ide_fix_cut];
#         ide_fix_cut = ide_fix_cut(id);
#         ide_fix_cut = ide_fix_cut(ia);
#
#         dir_fix = round(abs(wrapToPi(2*ang(param.tri_fix))/pi) + 1);
#         Align = sparse(1:length(ide_fix_cut), ide_fix_cut + (2 - dir_fix)*disk_mesh.num_edges, 1, length(ide_fix_cut), 2*disk_mesh.num_edges)*blkdiag(dec_cut.d0p, dec_cut.d0p);
#     end
# else
#     Align = sparse(0,2*disk_mesh.num_vertices);
#     Rot = sparse(0,2*disk_mesh.num_vertices);
# end


# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
from scipy.sparse import csr_matrix, block_diag
from typing import Tuple, Optional
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .cut_mesh import cut_mesh, MeshInfo as CutMeshInfo
from .matrix_ops import matrix_vector_multiplication
from rectangular_surface_parameterization.preprocessing.dec import dec_tri, DEC
from rectangular_surface_parameterization.core.mesh_info import MeshInfo


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]. Equivalent to MATLAB wrapToPi."""
    return np.arctan2(np.sin(x), np.cos(x))


def mesh_to_disk_seamless(
    Src: MeshInfo,
    param,
    ang: np.ndarray,
    sing: np.ndarray,
    k21: np.ndarray,
    ifseamless_const: bool,
    ifboundary: bool,
    ifhardedge: bool
) -> Tuple[MeshInfo, DEC, csr_matrix, csr_matrix]:
    """
    Cut mesh to disk topology and build seamless constraints.

    This function:
    1. Identifies cone singularities (vertices where |sing| > 0.1)
    2. Cuts the mesh along edges with non-trivial k21 (rotation mismatch)
    3. Builds seamless constraints for the parameterization

    Parameters
    ----------
    Src : MeshInfo
        Source mesh data structure.
    param : object
        Parameter structure with idx_int, ide_bound, ide_fix, E2T, tri_fix, ide_hard.
    ang : ndarray (nf,)
        Cross-field angle per face.
    sing : ndarray (nv,)
        Singularity indices per vertex.
    k21 : ndarray (ne,)
        Rotation indices for seamless constraints (1-4, where 1 = identity).
    ifseamless_const : bool
        Whether to build seamless constraints.
    ifboundary : bool
        Whether to enforce boundary alignment.
    ifhardedge : bool
        Whether to enforce hard edge alignment.

    Returns
    -------
    disk_mesh : MeshInfo
        Cut mesh (disk topology).
    dec_cut : DEC
        DEC operators for the cut mesh.
    Align : sparse matrix
        Alignment constraints for boundary/hard edges.
    Rot : sparse matrix
        Rotation constraints for seamless matching across cuts.
    """

    # idcone = param.idx_int(abs(sing(param.idx_int)) > 0.1); % Cone indices

    # Interior vertices with non-zero singularity index
    # param.idx_int is 0-indexed array of interior vertex indices
    idx_int = np.asarray(param.idx_int)
    sing_at_int = np.abs(sing[idx_int])
    idcone = idx_int[sing_at_int > 0.1]

    # [disk_mesh,~,ide_cut_inv,~] = cut_mesh(Src.vertices, Src.triangles, Src.edge_to_vertex, Src.edge_to_triangle, Src.T2E, Src.triangle_to_triangle, idcone, k21 ~= 1);

    # edge_jump_tag: edges where k21 != 1 (non-trivial rotation)
    # k21 is 1-4 where 1 = identity rotation
    edge_jump_tag = (k21 != 1)

    SrcCut_raw, idx_cut_inv, ide_cut_inv, edge_cut = cut_mesh(
        Src.vertices, Src.triangles, Src.edge_to_vertex, Src.edge_to_triangle, Src.T2E, Src.triangle_to_triangle, idcone, edge_jump_tag
    )

    # Convert CutMeshInfo to full MeshInfo for dec_tri
    # IMPORTANT: mesh_info() creates a new edge ordering, so we must recompute ide_cut_inv
    # to match the new ordering. We do this after building disk_mesh.
    from rectangular_surface_parameterization.core.mesh_info import mesh_info
    disk_mesh = mesh_info(SrcCut_raw.vertices, SrcCut_raw.triangles)

    # Recompute ide_cut_inv for disk_mesh's edge ordering
    # Map each cut mesh edge to its original edge via vertex mapping
    orig_edge_map = {tuple(sorted(row)): i for i, row in enumerate(Src.edge_to_vertex)}
    mapped_edges = np.sort(idx_cut_inv[disk_mesh.edge_to_vertex], axis=1)
    ide_cut_inv = np.array([orig_edge_map.get(tuple(row), -1) for row in mapped_edges], dtype=int)
    if np.any(ide_cut_inv < 0):
        raise AssertionError("Failed to map cut edges to original edges.")
    # Add sign to preserve orientation (1-based signed encoding)
    ids = idx_cut_inv[disk_mesh.edge_to_vertex[:, 0]] == Src.edge_to_vertex[ide_cut_inv, 0]
    ide_cut_inv = np.where(ids, ide_cut_inv + 1, -(ide_cut_inv + 1))

    # dec_cut = dec_tri(disk_mesh);

    dec_cut = dec_tri(disk_mesh)

    # if ifseamless_const && nargout > 2

    if ifseamless_const:
        # edge_bound_cut = find(any(disk_mesh.edge_to_triangle(:,1:2) == 0, 2));

        # Find boundary edges in cut mesh (edges with only one adjacent face)
        # E2T uses -1 for no neighbor in Python (0 in MATLAB)
        # disk_mesh.edge_to_triangle has shape (ne, 4) with columns [t0, t1, s0, s1]
        E2T_cut = disk_mesh.edge_to_triangle[:, :2]  # First two columns are face indices
        is_boundary = np.any(E2T_cut == -1, axis=1)
        edge_bound_cut_idx = np.where(is_boundary)[0]

        # edge_bound_cut = [edge_bound_cut, sum(disk_mesh.edge_to_triangle(edge_bound_cut,1:2),2)];

        # Get the single adjacent face for each boundary edge
        # sum of [t0, t1] where one is -1 gives the valid face index (minus -1)
        adj_faces = E2T_cut[edge_bound_cut_idx, 0] + E2T_cut[edge_bound_cut_idx, 1] + 1
        # Actually, we need the non-(-1) face. Let's be more careful:
        adj_faces = np.where(
            E2T_cut[edge_bound_cut_idx, 0] >= 0,
            E2T_cut[edge_bound_cut_idx, 0],
            E2T_cut[edge_bound_cut_idx, 1]
        )

        edge_bound_cut = np.column_stack([edge_bound_cut_idx, adj_faces])

        # edge_bound_cut = edge_bound_cut(~ismember(abs(ide_cut_inv(edge_bound_cut(:,1))), param.ide_bound),:);

        # ide_cut_inv is signed 1-based edge index mapping cut edges to original edges
        # We need to decode: edge_idx = abs(ide_cut_inv) - 1
        ide_cut_inv_at_bound = ide_cut_inv[edge_bound_cut[:, 0]]
        orig_edge_idx = np.abs(ide_cut_inv_at_bound) - 1

        # Filter out edges that are on the original mesh boundary
        ide_bound = np.asarray(param.ide_bound)
        is_orig_boundary = np.isin(orig_edge_idx, ide_bound)
        edge_bound_cut = edge_bound_cut[~is_orig_boundary]

        # [~,id] = sort(abs(ide_cut_inv(edge_bound_cut(:,1))));

        ide_cut_at_cut_bound = ide_cut_inv[edge_bound_cut[:, 0]]
        sort_idx = np.argsort(np.abs(ide_cut_at_cut_bound))

        # ide_cut = ide_cut_inv(edge_bound_cut(id,1));

        ide_cut = ide_cut_at_cut_bound[sort_idx]

        # assert(all(abs(ide_cut(1:2:end-1)) == abs(ide_cut(2:2:end))), 'Cut failed.');

        # Cut edges must come in pairs (each internal cut creates two boundary edges).
        # An odd count indicates a malformed cut graph.
        if len(ide_cut) % 2 != 0:
            raise ValueError(
                f"Cut edge pairing failed: found {len(ide_cut)} cut boundary edges (odd count). "
                f"Each internal cut edge should create exactly 2 boundary edges. "
                f"This indicates a malformed cut graph."
            )

        # Check that pairs of cut edges map to the same original edge
        ide_cut_odd = np.abs(ide_cut[0::2])
        ide_cut_even = np.abs(ide_cut[1::2])
        if not np.all(ide_cut_odd == ide_cut_even):
            mismatches = np.where(ide_cut_odd != ide_cut_even)[0]
            raise ValueError(
                f"Cut failed: {len(mismatches)} edge pairs don't match. "
                f"First mismatch at pair {mismatches[0]}: edges {ide_cut_odd[mismatches[0]]} vs {ide_cut_even[mismatches[0]]}"
            )

        # ide_cut_cor = [abs(ide_cut(1:2:end-1)), sign(ide_cut(1:2:end-1)).*edge_bound_cut(id(1:2:end-1),1), sign(ide_cut(2:2:end)).*edge_bound_cut(id(2:2:end),1)];

        n_pairs = len(ide_cut) // 2
        idx_odd = sort_idx[0::2][:n_pairs]
        idx_even = sort_idx[1::2][:n_pairs]

        # Original edge index (0-based)
        orig_edge = np.abs(ide_cut[0::2][:n_pairs]) - 1

        # Signed cut edge indices (1-based encoding for sign preservation)
        sign_odd = np.sign(ide_cut[0::2][:n_pairs])
        sign_even = np.sign(ide_cut[1::2][:n_pairs])
        cut_edge_odd = (edge_bound_cut[idx_odd, 0] + 1) * sign_odd  # 1-based signed
        cut_edge_even = (edge_bound_cut[idx_even, 0] + 1) * sign_even  # 1-based signed

        ide_cut_cor = np.column_stack([orig_edge, cut_edge_odd, cut_edge_even])

        # tri_cut_cor = [edge_bound_cut(id(1:2:end-1),2), edge_bound_cut(id(2:2:end),2)];

        tri_cut_cor = np.column_stack([
            edge_bound_cut[idx_odd, 1],
            edge_bound_cut[idx_even, 1]
        ])

        # ide_cut_cor(:,2:3) = (tri_cut_cor(:,1) == param.edge_to_triangle(ide_cut_cor(:,1),1)).*ide_cut_cor(:,[2 3]) + (tri_cut_cor(:,2) == param.edge_to_triangle(ide_cut_cor(:,1),1)).*ide_cut_cor(:,[3 2]);

        # Reorder cut edges so first edge is on same side as param.edge_to_triangle[orig_edge, 0]
        # param.edge_to_triangle is the original mesh E2T, shape (ne, 4) or (ne, 2+)
        E2T_orig = np.asarray(param.edge_to_triangle)
        t0_orig = E2T_orig[orig_edge, 0]

        # Check which cut triangle matches the first original triangle
        cond1 = (tri_cut_cor[:, 0] == t0_orig)
        cond2 = (tri_cut_cor[:, 1] == t0_orig)

        new_col2 = np.where(cond1, ide_cut_cor[:, 1], np.where(cond2, ide_cut_cor[:, 2], ide_cut_cor[:, 1]))
        new_col3 = np.where(cond1, ide_cut_cor[:, 2], np.where(cond2, ide_cut_cor[:, 1], ide_cut_cor[:, 2]))
        ide_cut_cor[:, 1] = new_col2
        ide_cut_cor[:, 2] = new_col3

        # R0 = eye(2,2);
        # R1 = [0,-1; 1,0];
        # R2 = R1^2;
        # R3 = R1^3;
        # R = [R0(:), R1(:), R2(:), R3(:)]';

        R0 = np.eye(2)
        R1 = np.array([[0, -1], [1, 0]])
        R2 = R1 @ R1
        R3 = R1 @ R1 @ R1

        # Stack flattened rotation matrices as rows
        # IMPORTANT: Use column-major (Fortran) order to match MATLAB's R(:) convention.
        # MATLAB's matrix_vector_multiplication effectively applies R^T when given
        # column-major flattened matrices. Python must do the same for consistency.
        R_all = np.array([
            R0.ravel('F'),  # k21=1: identity
            R1.ravel('F'),  # k21=2: 90 degree rotation
            R2.ravel('F'),  # k21=3: 180 degree rotation
            R3.ravel('F'),  # k21=4: 270 degree rotation
        ])

        # R = matrix_vector_multiplication(R(k21(ide_cut_cor(:,1)),:));

        # k21 values are 1-4, so we need to index R_all with k21-1
        k21_at_cut = k21[orig_edge]  # k21 values for the cut edges
        R_selected = R_all[k21_at_cut - 1]  # shape (n_pairs, 4)
        R = matrix_vector_multiplication(R_selected)

        # I1 = sparse(1:size(ide_cut_cor,1), abs(ide_cut_cor(:,2)), sign(ide_cut_cor(:,2)), size(ide_cut_cor,1), disk_mesh.num_edges);
        # I2 = sparse(1:size(ide_cut_cor,1), abs(ide_cut_cor(:,3)), sign(ide_cut_cor(:,3)), size(ide_cut_cor,1), disk_mesh.num_edges);

        # Decode signed 1-based edge indices
        edge1_idx = np.abs(ide_cut_cor[:, 1]).astype(int) - 1  # 0-based
        edge1_sign = np.sign(ide_cut_cor[:, 1])
        edge2_idx = np.abs(ide_cut_cor[:, 2]).astype(int) - 1  # 0-based
        edge2_sign = np.sign(ide_cut_cor[:, 2])

        row_idx = np.arange(n_pairs)
        I1 = csr_matrix(
            (edge1_sign, (row_idx, edge1_idx)),
            shape=(n_pairs, disk_mesh.num_edges)
        )
        I2 = csr_matrix(
            (edge2_sign, (row_idx, edge2_idx)),
            shape=(n_pairs, disk_mesh.num_edges)
        )

        # Rot = blkdiag(I1*dec_cut.d0p,I1*dec_cut.d0p) - R*blkdiag(I2*dec_cut.d0p,I2*dec_cut.d0p);

        I1_d0p = I1 @ dec_cut.d0p
        I2_d0p = I2 @ dec_cut.d0p

        # blkdiag creates block diagonal matrix
        blk1 = block_diag((I1_d0p, I1_d0p), format='csr')
        blk2 = block_diag((I2_d0p, I2_d0p), format='csr')

        Rot = blk1 - R @ blk2

        # Align = sparse(0,2*disk_mesh.num_vertices);

        Align = csr_matrix((0, 2 * disk_mesh.num_vertices))

        # if (ifboundary && ~isempty(param.ide_bound)) || (ifhardedge && ~isempty(param.ide_hard))

        ide_bound = np.asarray(param.ide_bound)
        ide_hard = np.asarray(param.ide_hard) if hasattr(param, 'ide_hard') else np.array([])
        ide_fix = np.asarray(param.ide_fix) if hasattr(param, 'ide_fix') else np.array([])
        tri_fix = np.asarray(param.tri_fix) if hasattr(param, 'tri_fix') else np.array([])

        if (ifboundary and len(ide_bound) > 0) or (ifhardedge and len(ide_hard) > 0):
            # ide_fix_cut = find(ismember(abs(ide_cut_inv), param.ide_fix));

            # Find cut edges that map to fixed edges
            orig_edges_from_cut = np.abs(ide_cut_inv) - 1  # 0-based original edge indices
            is_fix_edge = np.isin(orig_edges_from_cut, ide_fix)
            ide_fix_cut = np.where(is_fix_edge)[0]

            if len(ide_fix_cut) > 0:
                # tri_fix_cut = vec(disk_mesh.edge_to_triangle(ide_fix_cut,1:2));

                # Get triangles adjacent to fixed cut edges
                tri_fix_cut_2d = disk_mesh.edge_to_triangle[ide_fix_cut, :2]  # shape (n_fix, 2)
                tri_fix_cut = tri_fix_cut_2d.flatten('F')  # column-major flatten

                # id = tri_fix_cut ~= 0;
                # In Python, boundary marker is -1
                id_mask = tri_fix_cut != -1

                # tri_fix_cut = tri_fix_cut(id);
                tri_fix_cut = tri_fix_cut[id_mask]

                # [~,ia] = ismember(param.tri_fix, tri_fix_cut);

                # MATLAB ismember(A, B) returns:
                #   lia: logical array, lia[k] = true if A[k] is in B
                #   locb: index array, locb[k] = first index of A[k] in B (0 if not found)
                # Here we need locb, which gives the index mapping.
                tri_fix_param = np.asarray(tri_fix)

                # Build lookup dict: value -> first occurrence index in tri_fix_cut
                tri_fix_cut_dict = {}
                for i, t in enumerate(tri_fix_cut):
                    if t not in tri_fix_cut_dict:  # First occurrence only (MATLAB semantics)
                        tri_fix_cut_dict[t] = i

                # For each element in tri_fix_param, find its index in tri_fix_cut
                # MATLAB uses 0 for not-found, which causes error when used for indexing
                ia = np.array([tri_fix_cut_dict.get(t, -1) for t in tri_fix_param], dtype=int)

                # assert(length(ia) == length(tri_fix_cut))
                # MATLAB's ismember returns ia with length(param.tri_fix), so this checks equality
                assert len(tri_fix_param) == len(tri_fix_cut), \
                    f"ismember length mismatch: {len(tri_fix_param)} vs {len(tri_fix_cut)}"

                # MATLAB would error on ide_fix_cut(ia) if any ia element is 0
                # We match that behavior by erroring if any element is not found
                if np.any(ia < 0):
                    not_found = tri_fix_param[ia < 0]
                    raise IndexError(
                        f"ismember: {np.sum(ia < 0)} elements from param.tri_fix not found in "
                        f"tri_fix_cut. First few: {not_found[:5].tolist()}. "
                        "MATLAB would error on indexing with 0."
                    )

                # ide_fix_cut = [ide_fix_cut; ide_fix_cut];
                # ide_fix_cut = ide_fix_cut(id);
                # ide_fix_cut = ide_fix_cut(ia);

                ide_fix_cut_doubled = np.concatenate([ide_fix_cut, ide_fix_cut])
                ide_fix_cut_masked = ide_fix_cut_doubled[id_mask]
                ide_fix_cut_final = ide_fix_cut_masked[ia]

                if len(ide_fix_cut_final) > 0:
                    # dir_fix = round(abs(wrapToPi(2*ang(param.tri_fix))/pi) + 1);
                    # MATLAB uses param.tri_fix directly (not filtered)
                    ang_at_fix = ang[tri_fix_param]  # tri_fix_param == param.tri_fix here (no filtering)
                    dir_fix = np.round(np.abs(wrap_to_pi(2 * ang_at_fix) / np.pi) + 1).astype(int)

                    # Align = sparse(1:length(ide_fix_cut), ide_fix_cut + (2 - dir_fix)*disk_mesh.num_edges, 1, length(ide_fix_cut), 2*disk_mesh.num_edges)*blkdiag(dec_cut.d0p, dec_cut.d0p);

                    n_fix = len(ide_fix_cut_final)
                    # Column indices: ide_fix_cut + (2 - dir_fix) * ne
                    # This selects either the u or v component
                    col_idx = ide_fix_cut_final + (2 - dir_fix[:n_fix]) * disk_mesh.num_edges

                    row_idx = np.arange(n_fix)
                    selector = csr_matrix(
                        (np.ones(n_fix), (row_idx, col_idx)),
                        shape=(n_fix, 2 * disk_mesh.num_edges)
                    )

                    d0p_blk = block_diag((dec_cut.d0p, dec_cut.d0p), format='csr')
                    Align = selector @ d0p_blk

    else:
        # Align = sparse(0,2*disk_mesh.num_vertices);
        # Rot = sparse(0,2*disk_mesh.num_vertices);

        Align = csr_matrix((0, 2 * disk_mesh.num_vertices))
        Rot = csr_matrix((0, 2 * disk_mesh.num_vertices))

    return disk_mesh, dec_cut, Align, Rot
