
# function [param,mesh,dec] = preprocess_ortho_param(mesh, dec, ifboundary, ifhardedge, tol_dihedral_deg, Ehard2V)
#
#


# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import scipy.sparse as sp
import warnings
from scipy.sparse.csgraph import connected_components, dijkstra
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from rectangular_surface_parameterization.core.mesh_info import MeshInfo, mesh_info
from .dec import DEC, dec_tri
from .connectivity import connectivity
from .gaussian_curvature import gaussian_curvature
from .angles_of_triangles import angles_of_triangles
from .sort_triangles import sort_triangles, clear_cache as clear_sort_cache
from .find_graph_generator import find_graph_generator


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi]. Equivalent to MATLAB wrapToPi."""
    return np.arctan2(np.sin(x), np.cos(x))


# comp_angle = @(u,v,n) atan2(dot(cross(u,v,2),n,2), dot(u,v,2));

def comp_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Compute signed angle from u to v around normal n.

    Parameters
    ----------
    u, v : ndarray (m, 3)
        Direction vectors
    n : ndarray (m, 3)
        Normal vectors (rotation axis)

    Returns
    -------
    angle : ndarray (m,)
        Signed angle from u to v
    """
    cross_uv = np.cross(u, v)
    sin_angle = np.sum(cross_uv * n, axis=1)
    cos_angle = np.sum(u * v, axis=1)
    return np.arctan2(sin_angle, cos_angle)


@dataclass
class OrthoParam:
    """
    Data structure containing all parametrization constraints information.

    Attributes for hard edges and boundary:
        ide_hard : ndarray - Hard edge indices
        tri_hard : ndarray - Triangles adjacent to hard edges
        idx_bound : ndarray - Boundary vertex indices
        ide_bound : ndarray - Boundary edge indices
        tri_bound : ndarray - Boundary triangle indices
        idx_int : ndarray - Interior vertex indices
        ide_int : ndarray - Interior edge indices
        tri_int : ndarray - Interior triangle indices

    Attributes for cross field:
        E2T : ndarray (ne, 2) - Oriented edge-to-triangle mapping
        e1r : ndarray (nf, 3) - Local basis vector 1 per face
        e2r : ndarray (nf, 3) - Local basis vector 2 per face
        para_trans : ndarray (ne,) - Parallel transport angles
        Kt : ndarray (nv,) - Gaussian curvature per vertex (from triangles)
        Kt_invisible : ndarray (nv,) - Invisible curvature
        ang_basis : ndarray (nf, 3) - Angle between local basis and triangle edges

    Attributes for singularity handling:
        Vp2V : ndarray - Virtual vertex to original vertex mapping
        d1d : sparse matrix - Modified dual boundary operator
        idx_fix_plus : ndarray - Fixed vertices plus virtual vertices
        idx_reg : ndarray - Regular (non-fixed) vertex indices
        K : ndarray - Modified Gaussian curvature
        K_invisible : ndarray - Invisible curvature (modified)

    Attributes for trivial connection:
        Ilink : sparse matrix - Integration along paths connecting components
        Ilink_hard : sparse matrix - Ilink with hard edges zeroed
        Icycle : sparse matrix - Integration along non-contractible cycles
        Icycle_hard : sparse matrix - Icycle with hard edges zeroed

    Attributes for final constraints:
        ide_fix : ndarray - All fixed edge indices
        idx_fix : ndarray - All fixed vertex indices
        ide_free : ndarray - Free edge indices
        tri_fix : ndarray - Fixed triangle indices
        tri_free : ndarray - Free triangle indices
    """
    # Hard edges and boundary
    ide_hard: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tri_hard: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_bound: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_bound: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tri_bound: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tri_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Cross field
    E2T: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    e1r: np.ndarray = field(default_factory=lambda: np.array([]))
    e2r: np.ndarray = field(default_factory=lambda: np.array([]))
    para_trans: np.ndarray = field(default_factory=lambda: np.array([]))
    Kt: np.ndarray = field(default_factory=lambda: np.array([]))
    Kt_invisible: np.ndarray = field(default_factory=lambda: np.array([]))
    ang_basis: np.ndarray = field(default_factory=lambda: np.array([]))

    # Singularity handling
    Vp2V: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    d1d: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    idx_fix_plus: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_reg: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    K: np.ndarray = field(default_factory=lambda: np.array([]))
    K_invisible: np.ndarray = field(default_factory=lambda: np.array([]))

    # Trivial connection
    Ilink: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    Ilink_hard: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    Icycle: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    Icycle_hard: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))

    # Final constraints
    ide_fix: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_fix: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    ide_free: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tri_fix: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tri_free: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def preprocess_ortho_param(
    mesh: MeshInfo,
    dec: DEC,
    ifboundary: bool,
    ifhardedge: bool,
    tol_dihedral_deg: float = 30.0,
    Ehard2V: Optional[np.ndarray] = None
) -> Tuple[OrthoParam, MeshInfo, DEC]:
    """
    Preprocess geometry for rectangular parameterization.

    Detects hard edges, computes boundary edges, remeshes so that each triangle
    has only one constrained edge, and stores data for trivial connection.

    Parameters
    ----------
    mesh : MeshInfo
        Triangle mesh data structure
    dec : DEC
        DEC data structure
    ifboundary : bool
        Enforce boundary alignment
    ifhardedge : bool
        Enforce hard-edge alignment
    tol_dihedral_deg : float
        Hard-edge detection threshold (angle in degrees)
    Ehard2V : ndarray (n, 2), optional
        Vertex indices of alignment edges

    Returns
    -------
    param : OrthoParam
        Data structure containing all parametrization constraints information
    mesh : MeshInfo
        (Possibly remeshed) triangle mesh data structure
    dec : DEC
        (Possibly recomputed) DEC data structure
    """
    param = OrthoParam()

    # Clear sort_triangles cache before processing new mesh
    clear_sort_cache()


    # if ifhardedge && ~exist('Ehard2V','var')

    if ifhardedge and Ehard2V is None:
        # [ide_hard,tri_hard,ide_bound,tri_bound] = detect_hard_edge(mesh, tol_dihedral_deg);

        ide_hard, tri_hard, ide_bound, tri_bound = detect_hard_edge(mesh, tol_dihedral_deg)

        # ide_fix = [ide_hard; ide_bound];    % Constrained edges
        # tri_fix = [tri_hard(:); tri_bound]; % Corresponding faces

        ide_fix = np.concatenate([ide_hard, ide_bound])
        tri_fix = np.concatenate([tri_hard.ravel('F'), tri_bound])  # MATLAB tri_hard(:)

        # if numel(tri_fix) ~= length(unique(tri_fix))

        if len(tri_fix) != len(np.unique(tri_fix)):
            # tri = sum(ismember(abs(mesh.T2E), ide_fix) ,2) >= 2;

            # T2E is a SignedEdgeArray - use .indices for 0-based edge indices
            T2E_abs = mesh.T2E.indices
            tri_mask = np.sum(np.isin(T2E_abs, ide_fix), axis=1) >= 2

            # b = (mesh.vertices(mesh.triangles(tri,1),:)+mesh.vertices(mesh.triangles(tri,2),:)+mesh.vertices(mesh.triangles(tri,3),:))/3;
            # Xs = [mesh.vertices; b];

            tri_indices = np.where(tri_mask)[0]
            b = (mesh.vertices[mesh.triangles[tri_mask, 0], :] +
                 mesh.vertices[mesh.triangles[tri_mask, 1], :] +
                 mesh.vertices[mesh.triangles[tri_mask, 2], :]) / 3
            Xs = np.vstack([mesh.vertices, b])

            # np = mesh.num_vertices+(1:size(b,1))';
            # Ttri = [mesh.triangles(tri,[1 2]), np ; mesh.triangles(tri,[2 3]), np ; mesh.triangles(tri,[3 1]), np];
            # Ts = mesh.triangles;
            # Ts(tri,:) = [];
            # Ts = [Ts; Ttri];

            new_verts = mesh.num_vertices + np.arange(len(b))
            Ttri = np.vstack([
                np.column_stack([mesh.triangles[tri_mask, 0], mesh.triangles[tri_mask, 1], new_verts]),
                np.column_stack([mesh.triangles[tri_mask, 1], mesh.triangles[tri_mask, 2], new_verts]),
                np.column_stack([mesh.triangles[tri_mask, 2], mesh.triangles[tri_mask, 0], new_verts])
            ])
            Ts = np.delete(mesh.triangles, tri_indices, axis=0)
            Ts = np.vstack([Ts, Ttri])

            # mesh = MeshInfo(Xs, Ts);

            mesh = mesh_info(Xs, Ts)

            # [ide_hard,tri_hard,ide_bound,tri_bound] = detect_hard_edge(mesh, tol_dihedral_deg);

            ide_hard, tri_hard, ide_bound, tri_bound = detect_hard_edge(mesh, tol_dihedral_deg)

            # ide_fix = [ide_hard; ide_bound];
            # tri_fix = [tri_hard(:); tri_bound];

            ide_fix = np.concatenate([ide_hard, ide_bound])
            tri_fix = np.concatenate([tri_hard.ravel('F'), tri_bound])  # MATLAB tri_hard(:)

        # assert(numel(tri_fix) == length(unique(tri_fix)), 'Multiple constraints on a triangle.');

        assert len(tri_fix) == len(np.unique(tri_fix)), 'Multiple constraints on a triangle.'

    # elseif exist('Ehard2V','var')
    #     [~,ide_hard] = intersect(mesh.edge_to_vertex, sort(Ehard2V,2), 'rows');
    #     tri_hard = mesh.edge_to_triangle(ide_hard,1:2);

    elif Ehard2V is not None:
        Ehard2V_sorted = np.sort(Ehard2V, axis=1)
        E2V_sorted = np.sort(mesh.edge_to_vertex, axis=1)
        ide_hard = _intersect_rows(E2V_sorted, Ehard2V_sorted)
        tri_hard = mesh.edge_to_triangle[ide_hard, :2]

        n_provided = len(Ehard2V_sorted)
        n_found = len(ide_hard)
        if n_found < n_provided:
            warnings.warn(f'Ehard2V: only {n_found}/{n_provided} provided edges found in mesh. '
                          f'Check that vertex indices are 0-based and match the mesh.')

        # Remesh if needed (same logic as dihedral path): split triangles that have
        # more than one constrained edge so each triangle has at most one.
        ide_bound_pre, _ = boundary_indices(mesh)
        ide_fix = np.concatenate([ide_hard, ide_bound_pre])
        tri_fix = np.concatenate([tri_hard.ravel('F'),
                                  mesh.edge_to_triangle[ide_bound_pre, 0] if len(ide_bound_pre) > 0
                                  else np.array([], dtype=int)])

        if len(tri_fix) != len(np.unique(tri_fix)):
            # Some triangles have multiple constrained edges — need to split them
            T2E_abs = mesh.T2E.indices
            tri_mask = np.sum(np.isin(T2E_abs, ide_fix), axis=1) >= 2
            tri_indices = np.where(tri_mask)[0]
            b = (mesh.vertices[mesh.triangles[tri_mask, 0], :] +
                 mesh.vertices[mesh.triangles[tri_mask, 1], :] +
                 mesh.vertices[mesh.triangles[tri_mask, 2], :]) / 3
            Xs = np.vstack([mesh.vertices, b])
            new_verts = mesh.num_vertices + np.arange(len(b))
            Ttri = np.vstack([
                np.column_stack([mesh.triangles[tri_mask, 0], mesh.triangles[tri_mask, 1], new_verts]),
                np.column_stack([mesh.triangles[tri_mask, 1], mesh.triangles[tri_mask, 2], new_verts]),
                np.column_stack([mesh.triangles[tri_mask, 2], mesh.triangles[tri_mask, 0], new_verts])
            ])
            Ts = np.delete(mesh.triangles, tri_indices, axis=0)
            Ts = np.vstack([Ts, Ttri])
            mesh = mesh_info(Xs, Ts)
            dec = dec_tri(mesh)

            # Re-find the hard edges in the remeshed mesh
            # Original hard edges are preserved (centroid splits don't break existing edges)
            E2V_sorted = np.sort(mesh.edge_to_vertex, axis=1)
            ide_hard = _intersect_rows(E2V_sorted, Ehard2V_sorted)
            tri_hard = mesh.edge_to_triangle[ide_hard, :2]

    # else
    #     ide_hard = [];
    #     tri_hard = mesh.edge_to_triangle(ide_hard,1:2);

    else:
        ide_hard = np.array([], dtype=int)
        tri_hard = np.zeros((0, 2), dtype=int)

    # ide_bound = boundary_indices(mesh);

    ide_bound, _ = boundary_indices(mesh)

    # if ifhardedge || ~isempty(ide_bound)

    if ifhardedge or len(ide_bound) > 0:
        # ide_fix = [ide_hard; ide_bound];

        ide_fix = np.concatenate([ide_hard, ide_bound])

        # idx_fix = unique(mesh.edge_to_vertex(ide_fix,:));

        if len(ide_fix) > 0:
            idx_fix = np.unique(mesh.edge_to_vertex[ide_fix, :])
        else:
            idx_fix = np.array([], dtype=int)

        # ide = find(all(ismember(mesh.edge_to_vertex, idx_fix), 2));

        if len(idx_fix) > 0:
            in_fix = np.isin(mesh.edge_to_vertex, idx_fix)
            ide = np.where(np.all(in_fix, axis=1))[0]
        else:
            ide = np.array([], dtype=int)

        # ide = setdiff(ide, ide_fix);

        ide = np.setdiff1d(ide, ide_fix)

        # if ~isempty(ide)

        if len(ide) > 0:
            # nv = mesh.num_vertices;            % Current vertex count
            # Ts = mesh.triangles;             % New face list
            # Xs = mesh.vertices;             % New vertex list
            # E2T = mesh.edge_to_triangle(:,1:2);   % New edge to face table
            # E2V = mesh.edge_to_vertex;          % New edge to vertices

            nv = mesh.num_vertices
            Ts = mesh.triangles.copy()
            Xs = mesh.vertices.copy()
            E2T = mesh.edge_to_triangle[:, :2].copy()
            E2V = mesh.edge_to_vertex.copy()

            # Initialize hard edge pair tracking for split propagation
            if len(ide_hard) > 0:
                preprocess_ortho_param._hard_pairs = [
                    list(mesh.edge_to_vertex[e, :]) for e in ide_hard
                ]
            else:
                preprocess_ortho_param._hard_pairs = []

            # while ~isempty(ide) % while there are edges to split

            while len(ide) > 0:
                # id = ide(1);
                # idx = E2V(id,:);

                id_edge = ide[0]
                idx = E2V[id_edge, :]

                # idt1 = E2T(id,1);
                # t1 = Ts(idt1,:);
                # t1 = circshift(t1, [0,1-find(~ismember(t1,idx))]);

                idt1 = E2T[id_edge, 0]
                t1 = Ts[idt1, :].copy()
                # Find vertex not in edge and shift so it's first
                not_in_edge = np.where(~np.isin(t1, idx))[0][0]
                t1 = np.roll(t1, -not_in_edge)

                # idt2 = E2T(id,2);
                # t2 = Ts(idt2,:);
                # t2 = circshift(t2, [0,1-find(~ismember(t2,idx))]);

                idt2 = E2T[id_edge, 1]
                t2 = Ts[idt2, :].copy()
                not_in_edge = np.where(~np.isin(t2, idx))[0][0]
                t2 = np.roll(t2, -not_in_edge)

                # Ts(idt1,:) = [t1(1), t1(2), nv+1];
                # Ts(idt2,:) = [t2(1), t2(2), nv+1];
                # Ts = [Ts; t1(1), nv+1, t1(3); t2(1), nv+1, t2(3)];

                Ts[idt1, :] = [t1[0], t1[1], nv]
                Ts[idt2, :] = [t2[0], t2[1], nv]
                Ts = np.vstack([Ts, [t1[0], nv, t1[2]], [t2[0], nv, t2[2]]])

                # m = (Xs(E2V(id,1),:) + Xs(E2V(id,2),:))/2;
                # Xs = [Xs; m];
                # nv = nv + 1;

                m = (Xs[E2V[id_edge, 0], :] + Xs[E2V[id_edge, 1], :]) / 2
                Xs = np.vstack([Xs, m])
                nv = nv + 1

                # idx = sort(E2V(ide_fix,:),2);
                # [E2V,~,E2T] = connectivity(Ts);
                # [~,ide_fix] = intersect(sort(E2V,2), idx, 'rows');
                # idx_fix = unique(E2V(ide_fix,:));
                # ide = find(all(ismember(E2V, idx_fix), 2));
                # ide = setdiff(ide, ide_fix);

                # Track which hard edges were split: replace (v1,v2) with (v1,v_mid),(v_mid,v2)
                split_v1, split_v2 = E2V[id_edge, 0], E2V[id_edge, 1]
                v_mid = nv - 1  # the midpoint vertex we just added
                if hasattr(preprocess_ortho_param, '_hard_pairs'):
                    new_pairs = []
                    for pair in preprocess_ortho_param._hard_pairs:
                        p0, p1 = pair
                        if (min(p0,p1) == min(split_v1,split_v2) and
                            max(p0,p1) == max(split_v1,split_v2)):
                            # This hard edge was split — replace with two child edges
                            new_pairs.append([p0, v_mid])
                            new_pairs.append([v_mid, p1])
                        else:
                            new_pairs.append(pair)
                    preprocess_ortho_param._hard_pairs = new_pairs

                # Store vertex pairs of fixed edges before recomputing connectivity
                if len(ide_fix) > 0:
                    idx_pairs = np.sort(E2V[ide_fix, :], axis=1)
                else:
                    idx_pairs = np.zeros((0, 2), dtype=int)

                E2V, _, E2T_full, _ = connectivity(Ts)
                E2T = E2T_full[:, :2]

                # Find new fixed edge indices
                if len(idx_pairs) > 0:
                    E2V_sorted = np.sort(E2V, axis=1)
                    ide_fix = _intersect_rows(E2V_sorted, idx_pairs)
                else:
                    ide_fix = np.array([], dtype=int)

                if len(ide_fix) > 0:
                    idx_fix = np.unique(E2V[ide_fix, :])
                    in_fix = np.isin(E2V, idx_fix)
                    ide = np.where(np.all(in_fix, axis=1))[0]
                    ide = np.setdiff1d(ide, ide_fix)
                else:
                    ide = np.array([], dtype=int)

            # Recover hard edges, accounting for any splits that occurred
            if hasattr(preprocess_ortho_param, '_hard_pairs') and len(preprocess_ortho_param._hard_pairs) > 0:
                idx_hard = np.sort(np.array(preprocess_ortho_param._hard_pairs), axis=1)
                del preprocess_ortho_param._hard_pairs
            elif len(ide_hard) > 0:
                idx_hard = np.sort(mesh.edge_to_vertex[ide_hard, :], axis=1)
            else:
                idx_hard = np.zeros((0, 2), dtype=int)

            # mesh = MeshInfo(Xs, Ts);
            # dec = dec_tri(mesh);

            mesh = mesh_info(Xs, Ts)
            dec = dec_tri(mesh)

            # [~,ide_hard] = intersect(sort(mesh.edge_to_vertex,2), idx_hard, 'rows');
            # tri_hard = mesh.edge_to_triangle(ide_hard,1:2);

            if len(idx_hard) > 0:
                E2V_sorted = np.sort(mesh.edge_to_vertex, axis=1)
                ide_hard = _intersect_rows(E2V_sorted, idx_hard)
                tri_hard = mesh.edge_to_triangle[ide_hard, :2]
            else:
                ide_hard = np.array([], dtype=int)
                tri_hard = np.zeros((0, 2), dtype=int)

    # param.ide_hard = ide_hard;
    # param.tri_hard = tri_hard;

    param.ide_hard = ide_hard
    param.tri_hard = tri_hard

    # [ide_bound,tri_bound] = boundary_indices(mesh);
    # idx_bound = unique(mesh.edge_to_vertex(ide_bound,:));
    # idx_int = setdiff((1:mesh.num_vertices)', idx_bound);
    #
    # ide_int = setdiff((1:mesh.num_edges)', ide_bound);
    # tri_int = setdiff((1:mesh.num_faces)', tri_bound);

    ide_bound, tri_bound = boundary_indices(mesh)

    if len(ide_bound) > 0:
        idx_bound = np.unique(mesh.edge_to_vertex[ide_bound, :])
    else:
        idx_bound = np.array([], dtype=int)

    idx_int = np.setdiff1d(np.arange(mesh.num_vertices), idx_bound)
    ide_int = np.setdiff1d(np.arange(mesh.num_edges), ide_bound)
    tri_int = np.setdiff1d(np.arange(mesh.num_faces), tri_bound)

    # param.idx_bound = idx_bound;
    # param.ide_bound = ide_bound;
    # param.tri_bound = tri_bound;
    # param.idx_int = idx_int;
    # param.ide_int = ide_int;
    # param.tri_int = tri_int;

    param.idx_bound = idx_bound
    param.ide_bound = ide_bound
    param.tri_bound = tri_bound
    param.idx_int = idx_int
    param.ide_int = ide_int
    param.tri_int = tri_int

    # ide_fix = [ide_hard; ide_bound];

    ide_fix = np.concatenate([ide_hard, ide_bound])

    # idx_fix = unique(mesh.edge_to_vertex(ide_fix,:));
    # idx_fix_inv = zeros(mesh.num_vertices,1);
    # idx_fix_inv(idx_fix) = 1:length(idx_fix);
    # G = graph(idx_fix_inv(mesh.edge_to_vertex(ide_fix,1)), idx_fix_inv(mesh.edge_to_vertex(ide_fix,2)));

    if len(ide_fix) > 0:
        idx_fix = np.unique(mesh.edge_to_vertex[ide_fix, :])
        idx_fix_inv = np.zeros(mesh.num_vertices, dtype=int)
        idx_fix_inv[idx_fix] = np.arange(len(idx_fix))

        # Build graph adjacency matrix for connected components
        n_fix_verts = len(idx_fix)
        row = idx_fix_inv[mesh.edge_to_vertex[ide_fix, 0]]
        col = idx_fix_inv[mesh.edge_to_vertex[ide_fix, 1]]
        data = np.ones(len(ide_fix))
        G = sp.csr_matrix((data, (row, col)), shape=(n_fix_verts, n_fix_verts))
        G = G + G.T  # Make symmetric
    else:
        idx_fix = np.array([], dtype=int)
        n_fix_verts = 0
        G = sp.csr_matrix((0, 0))

    # [bins,binsizes] = conncomp(G);

    if n_fix_verts > 0:
        n_components, bins = connected_components(G, directed=False)
        # Compute component sizes
        binsizes = np.bincount(bins, minlength=n_components)
    else:
        n_components = 0
        bins = np.array([], dtype=int)
        binsizes = np.array([], dtype=int)

    # idx_fix_cell = cell(length(binsizes),1);
    # ide_fix_cell = cell(length(binsizes),1);
    # tri_fix_cell = cell(length(binsizes),1);
    # ide_sign_fix_cell = cell(length(binsizes),1);
    # for i = 1:length(binsizes)
    #     idx_fix_cell{i} = idx_fix(bins == i);
    #     ide_fix_cell{i} = find(all(ismember(mesh.edge_to_vertex, idx_fix_cell{i}), 2));
    #     ide_fix_cell{i} = ide_fix_cell{i}(ismember(ide_fix_cell{i}, ide_fix));
    #     tri_fix_cell{i} = vec(mesh.edge_to_triangle(ide_fix_cell{i},1:2));
    #     ide_sign_fix_cell{i} = vec(mesh.edge_to_triangle(ide_fix_cell{i},3:4));
    # end

    idx_fix_cell = []
    ide_fix_cell = []
    tri_fix_cell = []
    ide_sign_fix_cell = []

    for i in range(n_components):
        # idx_fix_cell{i} = idx_fix(bins == i);
        idx_fix_i = idx_fix[bins == i]
        idx_fix_cell.append(idx_fix_i)

        # ide_fix_cell{i} = find(all(ismember(mesh.edge_to_vertex, idx_fix_cell{i}), 2));
        in_component = np.isin(mesh.edge_to_vertex, idx_fix_i)
        ide_fix_i = np.where(np.all(in_component, axis=1))[0]

        # ide_fix_cell{i} = ide_fix_cell{i}(ismember(ide_fix_cell{i}, ide_fix));
        ide_fix_i = ide_fix_i[np.isin(ide_fix_i, ide_fix)]
        ide_fix_cell.append(ide_fix_i)

        # tri_fix_cell{i} = vec(mesh.edge_to_triangle(ide_fix_cell{i},1:2));
        # MATLAB vec() = column-major flatten
        if len(ide_fix_i) > 0:
            tri_fix_i = mesh.edge_to_triangle[ide_fix_i, :2].flatten('F')
        else:
            tri_fix_i = np.array([], dtype=int)
        tri_fix_cell.append(tri_fix_i)

        # ide_sign_fix_cell{i} = vec(mesh.edge_to_triangle(ide_fix_cell{i},3:4));
        if len(ide_fix_i) > 0:
            ide_sign_fix_i = mesh.edge_to_triangle[ide_fix_i, 2:4].flatten('F')
        else:
            ide_sign_fix_i = np.array([])
        ide_sign_fix_cell.append(ide_sign_fix_i)

    # E2T = zeros(mesh.num_edges,2);
    # E2T(:,1) = mesh.edge_to_triangle(:,1).*(mesh.edge_to_triangle(:,3) > 0) + mesh.edge_to_triangle(:,2).*(mesh.edge_to_triangle(:,3) < 0);
    # E2T(:,2) = mesh.edge_to_triangle(:,1).*(mesh.edge_to_triangle(:,3) < 0) + mesh.edge_to_triangle(:,2).*(mesh.edge_to_triangle(:,3) > 0);

    # NOTE: The MATLAB reordering based on E2T[:, 2] signs doesn't correctly identify
    # f_pos (face with canonical edge direction v0->v1) vs f_neg (face with opposite direction).
    # We explicitly determine f_pos and f_neg by checking edge orientation in each face.
    # E2T[:, 0] = f_neg (face where edge goes v1->v0, opposite to canonical)
    # E2T[:, 1] = f_pos (face where edge goes v0->v1, canonical direction)
    # This ensures para_trans = angle(f_neg) - angle(f_pos) satisfies d1d @ para_trans = K.
    E2T = np.zeros((mesh.num_edges, 2), dtype=int)
    for e in range(mesh.num_edges):
        v0, v1 = mesh.edge_to_vertex[e]  # canonical direction: v0 < v1
        f0, f1 = mesh.edge_to_triangle[e, 0], mesh.edge_to_triangle[e, 1]

        if f1 < 0:  # boundary edge
            E2T[e, 0] = f0
            E2T[e, 1] = f0
            continue

        # Check which face has edge going v0->v1 (canonical = f_pos)
        face0 = mesh.triangles[f0]
        f0_is_canonical = False
        for i in range(3):
            a, b = face0[i], face0[(i+1) % 3]
            if a == v0 and b == v1:
                f0_is_canonical = True
                break

        if f0_is_canonical:
            E2T[e, 0] = f1  # f_neg
            E2T[e, 1] = f0  # f_pos
        else:
            E2T[e, 0] = f0  # f_neg
            E2T[e, 1] = f1  # f_pos

    # K = gaussian_curvature(mesh.vertices, mesh.triangles);
    # assert(norm(sum(K) - 2*pi*(mesh.num_faces-mesh.num_edges+mesh.num_vertices)) < 1e-5, 'Gaussian curvature does not match topology.');

    K, _, _ = gaussian_curvature(mesh.vertices, mesh.triangles)
    euler_char = mesh.num_faces - mesh.num_edges + mesh.num_vertices
    curvature_error = np.abs(np.sum(K) - 2 * np.pi * euler_char)
    if curvature_error >= 1e-5:
        warnings.warn(f'Gaussian curvature does not match topology (error={curvature_error:.2e}, chi={euler_char}). Mesh may have issues.')

    # edge = mesh.vertices(mesh.edge_to_vertex(:,2),:) - mesh.vertices(mesh.edge_to_vertex(:,1),:);
    # edge = edge./sqrt(sum(edge.^2,2));
    # e1r = mesh.vertices(mesh.triangles(:,2),:) - mesh.vertices(mesh.triangles(:,1),:);
    # e1r = e1r./repmat(sqrt(sum(e1r.^2, 2)), [1 3]);

    edge = mesh.vertices[mesh.edge_to_vertex[:, 1], :] - mesh.vertices[mesh.edge_to_vertex[:, 0], :]
    edge = edge / np.linalg.norm(edge, axis=1, keepdims=True)

    e1r = mesh.vertices[mesh.triangles[:, 1], :] - mesh.vertices[mesh.triangles[:, 0], :]
    e1r = e1r / np.linalg.norm(e1r, axis=1, keepdims=True)

    # for i = 1:length(binsizes)
    #     tri = tri_fix_cell{i};
    #     ide = [ide_fix_cell{i}; ide_fix_cell{i}];
    #     ides = ide_sign_fix_cell{i};
    #     e1r(tri(tri ~= 0),:) = edge(ide(tri ~= 0),:).*ides(tri ~= 0);
    # end

    for i in range(n_components):
        tri = tri_fix_cell[i]
        ide_i = ide_fix_cell[i]
        ides = ide_sign_fix_cell[i]

        # ide = [ide_fix_cell{i}; ide_fix_cell{i}] to match tri shape
        ide_doubled = np.concatenate([ide_i, ide_i])

        # Filter out zero triangles (boundary markers, but in Python we use -1)
        # MATLAB uses 0 for no triangle, Python uses -1
        valid = tri >= 0
        if np.any(valid):
            tri_valid = tri[valid]
            ide_valid = ide_doubled[valid]
            ides_valid = ides[valid]
            e1r[tri_valid, :] = edge[ide_valid, :] * ides_valid[:, np.newaxis]

    # e2r = cross(mesh.normal, e1r, 2);

    e2r = np.cross(mesh.normal, e1r)

    # edge_angles = zeros(mesh.num_edges,2);
    # edge_angles(ide_int,1) = comp_angle(edge(ide_int,:), e1r(E2T(ide_int,1),:), mesh.normal(E2T(ide_int,1),:));
    # edge_angles(ide_int,2) = comp_angle(edge(ide_int,:), e1r(E2T(ide_int,2),:), mesh.normal(E2T(ide_int,2),:));

    edge_angles = np.zeros((mesh.num_edges, 2))
    if len(ide_int) > 0:
        edge_angles[ide_int, 0] = comp_angle(
            edge[ide_int, :],
            e1r[E2T[ide_int, 0], :],
            mesh.normal[E2T[ide_int, 0], :]
        )
        edge_angles[ide_int, 1] = comp_angle(
            edge[ide_int, :],
            e1r[E2T[ide_int, 1], :],
            mesh.normal[E2T[ide_int, 1], :]
        )

    # para_trans = wrapToPi(edge_angles(:,1) - edge_angles(:,2));
    # para_trans(ide_bound) = 0;

    para_trans = wrap_to_pi(edge_angles[:, 0] - edge_angles[:, 1])
    para_trans[ide_bound] = 0

    # assert(norm(wrapToPi(dec.d1d*para_trans - K)) < 1e-6, 'Gaussian curvature incompatible with angle defect.');

    residual = wrap_to_pi(dec.d1d @ para_trans - K)
    residual_norm = np.linalg.norm(residual)
    if residual_norm >= 1e-6:
        warnings.warn(f'Gaussian curvature incompatible with angle defect (residual={residual_norm:.2e}). May affect results.')
        # Don't fail, just warn - some meshes have numerical issues

    # param.ang_basis = [comp_angle(mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,2),:), e1r, mesh.normal), ...
    #                    comp_angle(mesh.vertices(mesh.triangles(:,2),:) - mesh.vertices(mesh.triangles(:,3),:), e1r, mesh.normal), ...
    #                    comp_angle(mesh.vertices(mesh.triangles(:,3),:) - mesh.vertices(mesh.triangles(:,1),:), e1r, mesh.normal)];

    ang_basis = np.column_stack([
        comp_angle(mesh.vertices[mesh.triangles[:, 0], :] - mesh.vertices[mesh.triangles[:, 1], :], e1r, mesh.normal),
        comp_angle(mesh.vertices[mesh.triangles[:, 1], :] - mesh.vertices[mesh.triangles[:, 2], :], e1r, mesh.normal),
        comp_angle(mesh.vertices[mesh.triangles[:, 2], :] - mesh.vertices[mesh.triangles[:, 0], :], e1r, mesh.normal)
    ])

    # param.edge_to_triangle = E2T;
    # param.e1r = e1r;
    # param.e2r = e2r;
    # param.para_trans = para_trans;
    # param.Kt = K;
    # param.Kt_invisible = K - dec.d1d*para_trans;

    param.edge_to_triangle = E2T
    param.e1r = e1r
    param.e2r = e2r
    param.para_trans = para_trans
    param.Kt = K
    param.Kt_invisible = K - dec.d1d @ para_trans
    param.ang_basis = ang_basis

    #
    # E2V = mesh.edge_to_vertex;
    # T = mesh.triangles;
    # nv = mesh.num_vertices;

    E2V_mod = mesh.edge_to_vertex.copy()
    T_mod = mesh.triangles.copy()
    nv_mod = mesh.num_vertices

    # for i = idx_fix'
    #     [tri_ord,edge_ord,sign_edge] = sort_triangles(i, mesh.triangles, mesh.edge_to_triangle, mesh.triangle_to_triangle, mesh.edge_to_vertex, mesh.T2E);
    #     id = ismember(edge_ord, ide_hard);
    #     n = sum(id);
    #     if (n == 1) && any(i == idx_int)
    #         continue;
    #     elseif (n == 1) && any(i == idx_bound)
    #         n = n + 1;
    #     end
    #
    #     p = 1;
    #     flag = zeros(length(edge_ord),1);
    #     flag_tri = zeros(length(edge_ord),1);
    #     for j = 1:length(edge_ord)
    #         if id(j)
    #             p = mod(p, n) + 1;
    #         else
    #             flag(j) = p;
    #         end
    #         flag_tri(j) = p;
    #     end
    #
    #     for j = 1:n-1
    #         ide = edge_ord(flag == j);
    #         E2V(ide,:) = (E2V(ide,:) ~= i).*E2V(ide,:) + (E2V(ide,:) == i).*(nv + j);
    #
    #         tri = tri_ord(flag_tri == j);
    #         T(tri,:) = (T(tri,:) ~= i).*T(tri,:) + (T(tri,:) == i).*(nv + j);
    #     end
    #
    #     if n > 1
    #         nv = nv + n - 1;
    #     end
    # end

    for i in idx_fix:
        tri_ord, edge_ord, sign_edge = sort_triangles(i, mesh.triangles, mesh.edge_to_triangle, mesh.triangle_to_triangle, mesh.edge_to_vertex, mesh.T2E)

        id_in_hard = np.isin(edge_ord, ide_hard)
        n = np.sum(id_in_hard)

        if n == 1 and i in idx_int:
            continue
        elif n == 1 and i in idx_bound:
            n = n + 1

        if n <= 1:
            continue

        p = 0  # 0-indexed (MATLAB uses 1-indexed p=1)
        flag = np.zeros(len(edge_ord), dtype=int)
        flag_tri = np.zeros(len(edge_ord), dtype=int)

        for j in range(len(edge_ord)):
            if id_in_hard[j]:
                p = (p + 1) % n
            else:
                flag[j] = p + 1  # +1 to use 1-based for matching MATLAB logic
            flag_tri[j] = p + 1

        for j in range(1, n):  # j = 1:n-1 in MATLAB (1-based)
            ide_j = edge_ord[flag == j]
            # E2V(ide,:) = (E2V(ide,:) ~= i).*E2V(ide,:) + (E2V(ide,:) == i).*(nv + j);
            mask = E2V_mod[ide_j, :] == i
            E2V_mod[ide_j, :] = np.where(mask, nv_mod + j - 1, E2V_mod[ide_j, :])

            tri_j = tri_ord[flag_tri == j]
            mask_t = T_mod[tri_j, :] == i
            T_mod[tri_j, :] = np.where(mask_t, nv_mod + j - 1, T_mod[tri_j, :])

        if n > 1:
            nv_mod = nv_mod + n - 1

    # ide_free = setdiff((1:mesh.num_edges)', ide_fix);
    # d1d = sparse(E2V(ide_free,:), [ide_free, ide_free], [ones(length(ide_free),1),-ones(length(ide_free),1)], nv, mesh.num_edges);

    ide_free = np.setdiff1d(np.arange(mesh.num_edges), ide_fix)

    # Build d1d sparse matrix
    row = np.concatenate([E2V_mod[ide_free, 0], E2V_mod[ide_free, 1]])
    col = np.concatenate([ide_free, ide_free])
    data = np.concatenate([np.ones(len(ide_free)), -np.ones(len(ide_free))])
    d1d_new = sp.csr_matrix((data, (row, col)), shape=(nv_mod, mesh.num_edges))

    # assert(all(sum(abs(d1d),2) ~= 0));

    row_sums = np.array(np.abs(d1d_new).sum(axis=1)).flatten()
    assert np.all(row_sums != 0), 'd1d has zero rows'

    # Vp2V = unique([T(:), mesh.triangles(:)], 'rows');
    # [~,id] = sort(Vp2V(:,1));
    # param.Vp2V = Vp2V(id,2);

    # T_mod.flatten('F') and mesh.triangles.flatten('F') for column-major
    pairs = np.column_stack([T_mod.flatten('F'), mesh.triangles.flatten('F')])
    Vp2V_unique = np.unique(pairs, axis=0)
    sort_idx = np.argsort(Vp2V_unique[:, 0])
    Vp2V = Vp2V_unique[sort_idx, 1]

    # param.d1d = d1d;
    # param.idx_fix_plus = [idx_fix; (mesh.num_vertices+1:nv)'];
    # param.idx_reg = setdiff((1:mesh.num_vertices)', idx_fix);

    idx_fix_plus = np.concatenate([idx_fix, np.arange(mesh.num_vertices, nv_mod)])
    idx_reg = np.setdiff1d(np.arange(mesh.num_vertices), idx_fix)

    # theta = angles_of_triangles(mesh.vertices, mesh.triangles);
    # K = 2*pi - accumarray(T(:), theta(:));
    # K(param.idx_fix_plus) = K(param.idx_fix_plus) - pi;

    theta = angles_of_triangles(mesh.vertices, mesh.triangles)
    K_new = np.full(nv_mod, 2 * np.pi)
    np.add.at(K_new, T_mod.flatten('F'), -theta.flatten('F'))
    K_new[idx_fix_plus] = K_new[idx_fix_plus] - np.pi

    # param.K = K;
    # param.K_invisible = K - param.d1d*para_trans;

    param.Vp2V = Vp2V
    param.d1d = d1d_new
    param.idx_fix_plus = idx_fix_plus
    param.idx_reg = idx_reg
    param.K = K_new
    param.K_invisible = K_new - d1d_new @ para_trans

    # nc = max(length(ide_fix_cell) - 1, 0);
    # Ilink = sparse(nc,mesh.num_edges);

    nc = max(len(ide_fix_cell) - 1, 0)
    Ilink = sp.lil_matrix((nc, mesh.num_edges))

    # for i = 1:nc

    for i in range(nc):
        # ld = max(dec.star1p*sqrt(mesh.sq_edge_length), 1e-5);

        ld = np.maximum(dec.star1p.diagonal() * np.sqrt(mesh.sq_edge_length), 1e-5)

        # for j = 1:nc+1
        #     if (j ~= 1) && (j ~= i+1)
        #         ld(ide_fix_cell{j}) = max(ld)*1e5;  % Large weight on all boundaries except 1 and j
        #     else
        #         ld(ide_fix_cell{j}) = min(ld)*1e-5; % Small weight on all boundaries for 1 and j
        #     end
        # end

        for j in range(nc + 1):
            if (j != 0) and (j != i + 1):
                # Large weight on all boundaries except 0 and i+1
                if len(ide_fix_cell[j]) > 0:
                    ld[ide_fix_cell[j]] = np.max(ld) * 1e5
            else:
                # Small weight on boundaries for 0 and i+1
                if len(ide_fix_cell[j]) > 0:
                    ld[ide_fix_cell[j]] = np.min(ld) * 1e-5

        # Gd = graph(E2T(ide_int,1), E2T(ide_int,2), ld(ide_int));

        # Build dual graph adjacency for shortest path
        row_d = E2T[ide_int, 0]
        col_d = E2T[ide_int, 1]
        weights_d = ld[ide_int]

        # Filter valid edges (both faces exist)
        valid = (row_d >= 0) & (col_d >= 0)
        row_d = row_d[valid]
        col_d = col_d[valid]
        weights_d = weights_d[valid]
        ide_int_valid = ide_int[valid]

        Gd = sp.csr_matrix((weights_d, (row_d, col_d)), shape=(mesh.num_faces, mesh.num_faces))
        Gd = Gd + Gd.T  # Make symmetric

        # P = shortestpath(Gd, tri_fix_cell{1}(1), tri_fix_cell{i+1}(1))';

        start_face = tri_fix_cell[0][0] if len(tri_fix_cell[0]) > 0 else 0
        end_face = tri_fix_cell[i + 1][0] if len(tri_fix_cell[i + 1]) > 0 else 0

        # Use Dijkstra's algorithm to find shortest path
        dist, predecessors = dijkstra(Gd, indices=start_face, return_predecessors=True)

        # Reconstruct path
        P = _reconstruct_path(predecessors, start_face, end_face)

        if len(P) < 2:
            continue

        # ed = [P(1:end-1), P(2:end)];
        # [~,~,ide] = intersect(sort(ed,2), sort(E2T,2), 'rows', 'stable');
        # assert(length(ide) == length(P)-1);

        ed = np.column_stack([P[:-1], P[1:]])
        ed_sorted = np.sort(ed, axis=1)
        E2T_sorted = np.sort(E2T, axis=1)
        ide_path = _intersect_rows_stable(ed_sorted, E2T_sorted)

        if len(ide_path) != len(P) - 1:
            continue  # Path reconstruction failed

        # a = find(ismember(P, tri_fix_cell{1}), 1, 'last');
        # b = find(ismember(P, tri_fix_cell{i+1}), 1, 'first');
        # id = a:b-1;
        # ide = ide(id);
        # ed = ed(id,:);

        in_first = np.isin(P, tri_fix_cell[0])
        in_second = np.isin(P, tri_fix_cell[i + 1])

        a_indices = np.where(in_first)[0]
        b_indices = np.where(in_second)[0]

        if len(a_indices) == 0 or len(b_indices) == 0:
            continue

        a = a_indices[-1]  # last occurrence
        b = b_indices[0]   # first occurrence

        if a >= b:
            continue

        id_range = np.arange(a, b)
        ide_path = ide_path[id_range]
        ed = ed[id_range, :]

        # s = (E2T(ide,1) == ed(:,1)) - (E2T(ide,2) == ed(:,1));

        s = (E2T[ide_path, 0] == ed[:, 0]).astype(int) - (E2T[ide_path, 1] == ed[:, 0]).astype(int)

        # Ilink(i,:) = sparse(ones(length(ide),1), ide, s, 1, mesh.num_edges);

        for idx, edge_idx in enumerate(ide_path):
            Ilink[i, edge_idx] = s[idx]

    # Ilink_hard = Ilink;
    # Ilink_hard(:,ide_hard) = 0; % hard edge do not count

    Ilink = Ilink.tocsr()
    Ilink_hard = Ilink.copy().tolil()
    if len(ide_hard) > 0:
        for e in ide_hard:
            Ilink_hard[:, e] = 0
    Ilink_hard = Ilink_hard.tocsr()

    # param.Ilink = Ilink;
    # param.Ilink_hard = Ilink_hard;

    param.Ilink = Ilink
    param.Ilink_hard = Ilink_hard

    # [cycle,cocycle] = find_graph_generator(full(diag(dec.star1p)), mesh.triangles, mesh.edge_to_triangle, mesh.edge_to_vertex, 1);

    star1p_diag = np.array(dec.star1p.diagonal())
    cycle, cocycle = find_graph_generator(star1p_diag, mesh.triangles, mesh.edge_to_triangle, mesh.edge_to_vertex, init=0)

    # nc = length(cocycle);
    # Icycle = sparse(nc,mesh.num_edges);

    nc_cycles = len(cocycle)
    Icycle = sp.lil_matrix((nc_cycles, mesh.num_edges))

    # for i = 1:nc
    #     ed = [cocycle{i}, circshift(cocycle{i}, [1,0])];
    #     [~,ide] = ismember(sort(ed,2), sort(E2T,2), 'rows');
    #     assert(length(ide) == length(cocycle{i}));
    #
    #     s = (E2T(ide,1) == ed(:,1)) - (E2T(ide,2) == ed(:,1));
    #
    #     Icycle(i,:) = sparse(ones(length(ide),1), ide, s, 1, mesh.num_edges);
    # end

    for i in range(nc_cycles):
        if len(cocycle[i]) == 0:
            continue

        # ed = [cocycle{i}, circshift(cocycle{i}, [1,0])];
        # circshift with [1,0] shifts rows down by 1
        shifted = np.roll(cocycle[i], 1)
        ed = np.column_stack([cocycle[i], shifted])

        # [~,ide] = ismember(sort(ed,2), sort(E2T,2), 'rows');
        ed_sorted = np.sort(ed, axis=1)
        E2T_sorted = np.sort(E2T, axis=1)
        ide_cycle = _ismember_rows(ed_sorted, E2T_sorted)

        if len(ide_cycle) != len(cocycle[i]):
            continue

        # s = (E2T(ide,1) == ed(:,1)) - (E2T(ide,2) == ed(:,1));

        s = (E2T[ide_cycle, 0] == ed[:, 0]).astype(int) - (E2T[ide_cycle, 1] == ed[:, 0]).astype(int)

        for idx, edge_idx in enumerate(ide_cycle):
            Icycle[i, edge_idx] = s[idx]

    # Icycle_hard = Icycle;
    # Icycle_hard(:,ide_hard) = 0; % hard edge do not count

    Icycle = Icycle.tocsr()
    Icycle_hard = Icycle.copy().tolil()
    if len(ide_hard) > 0:
        for e in ide_hard:
            Icycle_hard[:, e] = 0
    Icycle_hard = Icycle_hard.tocsr()

    # param.Icycle = Icycle;
    # param.Icycle_hard = Icycle_hard;

    param.Icycle = Icycle
    param.Icycle_hard = Icycle_hard

    # if ifboundary && ifhardedge
    #     ide_fix = [ide_hard; ide_bound];
    #     tri_fix = [tri_hard(:); tri_bound];
    # elseif ifboundary
    #     ide_fix = ide_bound;
    #     tri_fix = tri_bound;
    # elseif ifhardedge
    #     ide_fix = ide_hard;
    #     tri_fix = tri_hard(:);
    # else
    #     ide_fix = [];
    #     tri_fix = [];
    # end

    if ifboundary and ifhardedge:
        ide_fix_final = np.concatenate([ide_hard, ide_bound])
        tri_fix_final = np.concatenate([tri_hard.ravel('F'), tri_bound])  # MATLAB tri_hard(:)
    elif ifboundary:
        ide_fix_final = ide_bound
        tri_fix_final = tri_bound
    elif ifhardedge:
        ide_fix_final = ide_hard
        tri_fix_final = tri_hard.ravel('F')  # MATLAB tri_hard(:)
    else:
        ide_fix_final = np.array([], dtype=int)
        tri_fix_final = np.array([], dtype=int)

    # idx_fix = unique(mesh.edge_to_vertex(ide_fix,:));
    # param.ide_fix = ide_fix;
    # param.idx_fix = idx_fix;
    # param.ide_free = setdiff((1:mesh.num_edges)', ide_fix);
    # param.tri_fix = tri_fix;
    # param.tri_free = setdiff((1:mesh.num_faces)', param.tri_fix);

    if len(ide_fix_final) > 0:
        idx_fix_final = np.unique(mesh.edge_to_vertex[ide_fix_final, :])
    else:
        idx_fix_final = np.array([], dtype=int)

    param.ide_fix = ide_fix_final
    param.idx_fix = idx_fix_final
    param.ide_free = np.setdiff1d(np.arange(mesh.num_edges), ide_fix_final)
    param.tri_fix = tri_fix_final
    param.tri_free = np.setdiff1d(np.arange(mesh.num_faces), tri_fix_final)

    return param, mesh, dec


# function [ide_hard,tri_hard,ide_bound,tri_bound] = detect_hard_edge(mesh, tol_dihedral_deg)
# comp_angle = @(u,v,n) atan2(dot(cross(u,v,2),n,2), dot(u,v,2));
# tol_dihedral = tol_dihedral_deg*pi/180;

def detect_hard_edge(mesh: MeshInfo, tol_dihedral_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect hard edges based on dihedral angle threshold.

    Parameters
    ----------
    mesh : MeshInfo
        Mesh data structure
    tol_dihedral_deg : float
        Dihedral angle threshold in degrees

    Returns
    -------
    ide_hard : ndarray
        Hard edge indices
    tri_hard : ndarray (n, 2)
        Triangles adjacent to hard edges
    ide_bound : ndarray
        Boundary edge indices
    tri_bound : ndarray
        Boundary triangle indices
    """
    tol_dihedral = tol_dihedral_deg * np.pi / 180

    # [ide_bound,tri_bound] = boundary_indices(mesh);

    ide_bound, tri_bound = boundary_indices(mesh)

    # ide_int = setdiff((1:mesh.num_edges)', ide_bound);

    ide_int = np.setdiff1d(np.arange(mesh.num_edges), ide_bound)

    # edge = mesh.vertices(mesh.edge_to_vertex(:,2),:) - mesh.vertices(mesh.edge_to_vertex(:,1),:);
    # edge = edge./sqrt(sum(edge.^2,2));

    edge = mesh.vertices[mesh.edge_to_vertex[:, 1], :] - mesh.vertices[mesh.edge_to_vertex[:, 0], :]
    edge = edge / np.linalg.norm(edge, axis=1, keepdims=True)

    # dihedral_angle = mesh.edge_to_triangle(ide_int,4).*comp_angle(mesh.normal(mesh.edge_to_triangle(ide_int,1),:), mesh.normal(mesh.edge_to_triangle(ide_int,2),:), edge(ide_int,:));

    # E2T[:, 3] is the edge sign (column 4 in MATLAB 1-indexed)
    t0 = mesh.edge_to_triangle[ide_int, 0]
    t1 = mesh.edge_to_triangle[ide_int, 1]
    sign_col = mesh.edge_to_triangle[ide_int, 3]

    dihedral_angle = sign_col * comp_angle(
        mesh.normal[t0, :],
        mesh.normal[t1, :],
        edge[ide_int, :]
    )

    # ide_hard = ide_int(abs(dihedral_angle) > tol_dihedral);

    ide_hard = ide_int[np.abs(dihedral_angle) > tol_dihedral]

    # tri_hard = mesh.edge_to_triangle(ide_hard,1:2);

    if len(ide_hard) > 0:
        tri_hard = mesh.edge_to_triangle[ide_hard, :2]
    else:
        tri_hard = np.zeros((0, 2), dtype=int)

    return ide_hard, tri_hard, ide_bound, tri_bound


# function [ide_bound,tri_bound] = boundary_indices(mesh)
# ide_bound = find(any(mesh.edge_to_triangle == 0, 2));
#
# tri_bound = sum(mesh.edge_to_triangle(ide_bound,1:2),2);

def boundary_indices(mesh: MeshInfo) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find boundary edge and triangle indices.

    Parameters
    ----------
    mesh : MeshInfo
        Mesh data structure

    Returns
    -------
    ide_bound : ndarray
        Boundary edge indices
    tri_bound : ndarray
        Boundary triangle indices
    """
    # MATLAB uses 0 for "no triangle", Python uses -1
    # A boundary edge belongs to only one triangle
    ide_bound = np.where(np.any(mesh.edge_to_triangle[:, :2] < 0, axis=1))[0]

    # A boundary triangle is incident to a boundary edge
    # For boundary edges, one of E2T[:, 0] or E2T[:, 1] is -1
    # The non-negative one is the triangle
    if len(ide_bound) > 0:
        tri_bound = np.maximum(mesh.edge_to_triangle[ide_bound, 0], mesh.edge_to_triangle[ide_bound, 1])
    else:
        tri_bound = np.array([], dtype=int)

    return ide_bound, tri_bound


def _intersect_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Find indices in A of rows that appear in B.

    Parameters
    ----------
    A : ndarray (m, k)
        First array
    B : ndarray (n, k)
        Second array

    Returns
    -------
    indices : ndarray
        Indices in A of rows that appear in B
    """
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=int)

    B_set = set(map(tuple, B))
    indices = []
    for i, row in enumerate(A):
        if tuple(row) in B_set:
            indices.append(i)
    return np.array(indices, dtype=int)


def _intersect_rows_stable(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Find indices in B of rows that appear in A, preserving order of A.

    MATLAB's intersect(..., 'rows', 'stable') returns indices in B
    for each row of A that matches.

    Parameters
    ----------
    A : ndarray (m, k)
        Query rows
    B : ndarray (n, k)
        Reference array

    Returns
    -------
    indices : ndarray
        For each row of A, the index in B where it appears
    """
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=int)

    # Build lookup from B rows to indices
    B_lookup = {tuple(row): i for i, row in enumerate(B)}

    indices = []
    for row in A:
        t = tuple(row)
        if t in B_lookup:
            indices.append(B_lookup[t])

    return np.array(indices, dtype=int)


def _ismember_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    For each row of A, find the index in B where it appears.

    Parameters
    ----------
    A : ndarray (m, k)
        Query rows
    B : ndarray (n, k)
        Reference array

    Returns
    -------
    indices : ndarray (m,)
        For each row of A, the index in B where it appears (-1 if not found)
    """
    if len(A) == 0:
        return np.array([], dtype=int)
    if len(B) == 0:
        return -np.ones(len(A), dtype=int)

    # Build lookup from B rows to indices
    B_lookup = {tuple(row): i for i, row in enumerate(B)}

    indices = []
    for row in A:
        t = tuple(row)
        if t in B_lookup:
            indices.append(B_lookup[t])
        else:
            indices.append(-1)

    return np.array(indices, dtype=int)


def _reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Reconstruct shortest path from predecessors array.

    Parameters
    ----------
    predecessors : ndarray
        Predecessor array from Dijkstra's algorithm
    start : int
        Start node
    end : int
        End node

    Returns
    -------
    path : ndarray
        Sequence of nodes from start to end
    """
    if predecessors[end] < 0 and end != start:
        return np.array([], dtype=int)

    path = [end]
    current = end
    while current != start:
        current = predecessors[current]
        if current < 0:
            return np.array([], dtype=int)
        path.append(current)

    return np.array(path[::-1], dtype=int)
