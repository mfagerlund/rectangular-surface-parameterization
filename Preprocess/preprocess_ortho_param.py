# === ISSUES ===
# - graph/conncomp: use scipy.sparse.csgraph.connected_components
# - graph/shortestpath: use scipy.sparse.csgraph.dijkstra or shortest_path
# - wrapToPi: use np.arctan2(np.sin(x), np.cos(x))
# - circshift: use np.roll
# - intersect(..., 'rows', 'stable'): need custom row matching preserving order
# - ismember(..., 'rows'): need custom row membership test
# === END ISSUES ===

# function [param,Src,dec] = preprocess_ortho_param(Src, dec, ifboundary, ifhardedge, tol_dihedral_deg, Ehard2V)
# % Preprocess geometry:
# % - Detect hard edges
# % - Compute boundary edges
# % - Remesh so that each triangle has only one constrained edge
# % - Store data for trivial connection (non-contractible cycles, Gaussian curvature, parallel transport)
#
# % Input:
# % - Src: triangle mesh data structure
# % - dec: DEC data structure
# % - ifboundary: (boolean) enforce boundary alignment
# % - ifhardedge: (boolean) enforce hard-edge alignment
# % - tol_dihedral_deg: (double) hard-edge detection threshold (angle in degree)
# % - Ehard2V: (integer array n x 2) vertex indices of alignment edges (optional)
#
# % Output:
# % - param: data structure containing all parametrization constraints information
# % - Src: remeshed triangle mesh data structure
# % - dec: remeshed DEC data structure

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from .MeshInfo import MeshInfo, mesh_info
from .dec_tri import DEC, dec_tri
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
    Src: MeshInfo,
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
    Src : MeshInfo
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
    Src : MeshInfo
        (Possibly remeshed) triangle mesh data structure
    dec : DEC
        (Possibly recomputed) DEC data structure
    """
    param = OrthoParam()

    # Clear sort_triangles cache before processing new mesh
    clear_sort_cache()

    # %% Remeshing: a triangle cannot have two alignment constraints
    # % Check three cases:
    # % 1. alignment constraints are detected by the dihedral angle
    # % 2. alignment constraints is given by vertex indices

    # if ifhardedge && ~exist('Ehard2V','var')

    if ifhardedge and Ehard2V is None:
        # % Compute hard edges
        # [ide_hard,tri_hard,ide_bound,tri_bound] = detect_hard_edge(Src, tol_dihedral_deg);

        ide_hard, tri_hard, ide_bound, tri_bound = detect_hard_edge(Src, tol_dihedral_deg)

        # % List of all constraints (boundary + hard edges)
        # ide_fix = [ide_hard; ide_bound];    % Constrained edges
        # tri_fix = [tri_hard(:); tri_bound]; % Corresponding faces

        ide_fix = np.concatenate([ide_hard, ide_bound])
        tri_fix = np.concatenate([tri_hard.ravel(), tri_bound])

        # % If a face appears twice it is over constrained
        # % -> must be split in 3
        # if numel(tri_fix) ~= length(unique(tri_fix))

        if len(tri_fix) != len(np.unique(tri_fix)):
            # % Find indices of over-constrained triangles
            # tri = sum(ismember(abs(Src.T2E), ide_fix) ,2) >= 2;

            # T2E uses 1-based signed encoding, so abs(T2E)-1 gives 0-based edge indices
            T2E_abs = np.abs(Src.T2E) - 1
            tri_mask = np.sum(np.isin(T2E_abs, ide_fix), axis=1) >= 2

            # % Add the barycenter to the new vertex list Xs
            # b = (Src.X(Src.T(tri,1),:)+Src.X(Src.T(tri,2),:)+Src.X(Src.T(tri,3),:))/3;
            # Xs = [Src.X; b];

            tri_indices = np.where(tri_mask)[0]
            b = (Src.X[Src.T[tri_mask, 0], :] +
                 Src.X[Src.T[tri_mask, 1], :] +
                 Src.X[Src.T[tri_mask, 2], :]) / 3
            Xs = np.vstack([Src.X, b])

            # % Add 3 triangles per constrained triangles to the new triangle list Ts
            # np = Src.nv+(1:size(b,1))';
            # Ttri = [Src.T(tri,[1 2]), np ; Src.T(tri,[2 3]), np ; Src.T(tri,[3 1]), np];
            # Ts = Src.T;
            # Ts(tri,:) = [];
            # Ts = [Ts; Ttri];

            new_verts = Src.nv + np.arange(len(b))
            Ttri = np.vstack([
                np.column_stack([Src.T[tri_mask, 0], Src.T[tri_mask, 1], new_verts]),
                np.column_stack([Src.T[tri_mask, 1], Src.T[tri_mask, 2], new_verts]),
                np.column_stack([Src.T[tri_mask, 2], Src.T[tri_mask, 0], new_verts])
            ])
            Ts = np.delete(Src.T, tri_indices, axis=0)
            Ts = np.vstack([Ts, Ttri])

            # % Recompute connectivity information
            # Src = MeshInfo(Xs, Ts);

            Src = mesh_info(Xs, Ts)

            # % Recompute hard edges
            # [ide_hard,tri_hard,ide_bound,tri_bound] = detect_hard_edge(Src, tol_dihedral_deg);

            ide_hard, tri_hard, ide_bound, tri_bound = detect_hard_edge(Src, tol_dihedral_deg)

            # % New list of all constraints (boundary + hard edges)
            # ide_fix = [ide_hard; ide_bound];
            # tri_fix = [tri_hard(:); tri_bound];

            ide_fix = np.concatenate([ide_hard, ide_bound])
            tri_fix = np.concatenate([tri_hard.ravel(), tri_bound])

        # % Check that the remeshing worked
        # assert(numel(tri_fix) == length(unique(tri_fix)), 'Multiple constraints on a triangle.');

        assert len(tri_fix) == len(np.unique(tri_fix)), 'Multiple constraints on a triangle.'

    # elseif exist('Ehard2V','var')
    #     [~,ide_hard] = intersect(Src.E2V, sort(Ehard2V,2), 'rows');
    #     tri_hard = Src.E2T(ide_hard,1:2);

    elif Ehard2V is not None:
        Ehard2V_sorted = np.sort(Ehard2V, axis=1)
        E2V_sorted = np.sort(Src.E2V, axis=1)
        ide_hard = _intersect_rows(E2V_sorted, Ehard2V_sorted)
        tri_hard = Src.E2T[ide_hard, :2]

    # else
    #     ide_hard = [];
    #     tri_hard = Src.E2T(ide_hard,1:2);

    else:
        ide_hard = np.array([], dtype=int)
        tri_hard = np.zeros((0, 2), dtype=int)

    # %% Remeshing: only constraint edges can have both vertices belonging to constraint edges or a boundary
    # % If this happen, the edge is plitted in two
    # ide_bound = boundary_indices(Src);

    ide_bound, _ = boundary_indices(Src)

    # if ifhardedge || ~isempty(ide_bound)

    if ifhardedge or len(ide_bound) > 0:
        # % Gather edge indices from constrained edges and boundary edges
        # ide_fix = [ide_hard; ide_bound];

        ide_fix = np.concatenate([ide_hard, ide_bound])

        # % Find corresponding vertex indices
        # idx_fix = unique(Src.E2V(ide_fix,:));

        if len(ide_fix) > 0:
            idx_fix = np.unique(Src.E2V[ide_fix, :])
        else:
            idx_fix = np.array([], dtype=int)

        # % Edges indices whose vertices both belongs to constrained vertices
        # ide = find(all(ismember(Src.E2V, idx_fix), 2));

        if len(idx_fix) > 0:
            in_fix = np.isin(Src.E2V, idx_fix)
            ide = np.where(np.all(in_fix, axis=1))[0]
        else:
            ide = np.array([], dtype=int)

        # % Edges indices whose vertices both belongs to constrained vertices but
        # % are not constrained edges themselves
        # ide = setdiff(ide, ide_fix);

        ide = np.setdiff1d(ide, ide_fix)

        # % If ide is not empty, hese edges must be split
        # if ~isempty(ide)

        if len(ide) > 0:
            # nv = Src.nv;            % Current vertex count
            # Ts = Src.T;             % New face list
            # Xs = Src.X;             % New vertex list
            # E2T = Src.E2T(:,1:2);   % New edge to face table
            # E2V = Src.E2V;          % New edge to vertices

            nv = Src.nv
            Ts = Src.T.copy()
            Xs = Src.X.copy()
            E2T = Src.E2T[:, :2].copy()
            E2V = Src.E2V.copy()

            # while ~isempty(ide) % while there are edges to split

            while len(ide) > 0:
                # % Take first edge in the list
                # id = ide(1);
                # idx = E2V(id,:);

                id_edge = ide[0]
                idx = E2V[id_edge, :]

                # % Split the first triangle in two
                # idt1 = E2T(id,1);
                # t1 = Ts(idt1,:);
                # t1 = circshift(t1, [0,1-find(~ismember(t1,idx))]);

                idt1 = E2T[id_edge, 0]
                t1 = Ts[idt1, :].copy()
                # Find vertex not in edge and shift so it's first
                not_in_edge = np.where(~np.isin(t1, idx))[0][0]
                t1 = np.roll(t1, -not_in_edge)

                # % Split the second triangle in two
                # idt2 = E2T(id,2);
                # t2 = Ts(idt2,:);
                # t2 = circshift(t2, [0,1-find(~ismember(t2,idx))]);

                idt2 = E2T[id_edge, 1]
                t2 = Ts[idt2, :].copy()
                not_in_edge = np.where(~np.isin(t2, idx))[0][0]
                t2 = np.roll(t2, -not_in_edge)

                # % Update the triangle list
                # Ts(idt1,:) = [t1(1), t1(2), nv+1];
                # Ts(idt2,:) = [t2(1), t2(2), nv+1];
                # Ts = [Ts; t1(1), nv+1, t1(3); t2(1), nv+1, t2(3)];

                Ts[idt1, :] = [t1[0], t1[1], nv]
                Ts[idt2, :] = [t2[0], t2[1], nv]
                Ts = np.vstack([Ts, [t1[0], nv, t1[2]], [t2[0], nv, t2[2]]])

                # % Middle of the edge is added to the vertex list
                # m = (Xs(E2V(id,1),:) + Xs(E2V(id,2),:))/2;
                # Xs = [Xs; m];
                # nv = nv + 1;

                m = (Xs[E2V[id_edge, 0], :] + Xs[E2V[id_edge, 1], :]) / 2
                Xs = np.vstack([Xs, m])
                nv = nv + 1

                # % Update connectivity and check for new edges to split
                # idx = sort(E2V(ide_fix,:),2);
                # [E2V,~,E2T] = connectivity(Ts);
                # [~,ide_fix] = intersect(sort(E2V,2), idx, 'rows');
                # idx_fix = unique(E2V(ide_fix,:));
                # ide = find(all(ismember(E2V, idx_fix), 2));
                # ide = setdiff(ide, ide_fix);

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

            # % Vertex pair corresponding to constrained edges (needed to find
            # % new edge indices)
            # idx_hard = sort(Src.E2V(ide_hard,:),2);

            if len(ide_hard) > 0:
                idx_hard = np.sort(Src.E2V[ide_hard, :], axis=1)
            else:
                idx_hard = np.zeros((0, 2), dtype=int)

            # % Recompute connectivity and DEC
            # Src = MeshInfo(Xs, Ts);
            # dec = dec_tri(Src);

            Src = mesh_info(Xs, Ts)
            dec = dec_tri(Src)

            # % Update constrained indices
            # [~,ide_hard] = intersect(sort(Src.E2V,2), idx_hard, 'rows');
            # tri_hard = Src.E2T(ide_hard,1:2);

            if len(idx_hard) > 0:
                E2V_sorted = np.sort(Src.E2V, axis=1)
                ide_hard = _intersect_rows(E2V_sorted, idx_hard)
                tri_hard = Src.E2T[ide_hard, :2]
            else:
                ide_hard = np.array([], dtype=int)
                tri_hard = np.zeros((0, 2), dtype=int)

    # % Store hard edges
    # param.ide_hard = ide_hard;
    # param.tri_hard = tri_hard;

    param.ide_hard = ide_hard
    param.tri_hard = tri_hard

    # %% Compute boundary related stuff (mostly needed for, e.g., trivial connections)
    # [ide_bound,tri_bound] = boundary_indices(Src);
    # idx_bound = unique(Src.E2V(ide_bound,:));
    # idx_int = setdiff((1:Src.nv)', idx_bound);
    #
    # ide_int = setdiff((1:Src.ne)', ide_bound);
    # tri_int = setdiff((1:Src.nf)', tri_bound);

    ide_bound, tri_bound = boundary_indices(Src)

    if len(ide_bound) > 0:
        idx_bound = np.unique(Src.E2V[ide_bound, :])
    else:
        idx_bound = np.array([], dtype=int)

    idx_int = np.setdiff1d(np.arange(Src.nv), idx_bound)
    ide_int = np.setdiff1d(np.arange(Src.ne), ide_bound)
    tri_int = np.setdiff1d(np.arange(Src.nf), tri_bound)

    # % Store in structure
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

    # %% Merge boundary and hard edges
    # % group constraints by connected component  (still for trivial connections)
    # ide_fix = [ide_hard; ide_bound];

    ide_fix = np.concatenate([ide_hard, ide_bound])

    # % Compute a graph made of only the constrained edges (still for trivial connections)
    # idx_fix = unique(Src.E2V(ide_fix,:));
    # idx_fix_inv = zeros(Src.nv,1);
    # idx_fix_inv(idx_fix) = 1:length(idx_fix);
    # G = graph(idx_fix_inv(Src.E2V(ide_fix,1)), idx_fix_inv(Src.E2V(ide_fix,2)));

    if len(ide_fix) > 0:
        idx_fix = np.unique(Src.E2V[ide_fix, :])
        idx_fix_inv = np.zeros(Src.nv, dtype=int)
        idx_fix_inv[idx_fix] = np.arange(len(idx_fix))

        # Build graph adjacency matrix for connected components
        n_fix_verts = len(idx_fix)
        row = idx_fix_inv[Src.E2V[ide_fix, 0]]
        col = idx_fix_inv[Src.E2V[ide_fix, 1]]
        data = np.ones(len(ide_fix))
        G = sp.csr_matrix((data, (row, col)), shape=(n_fix_verts, n_fix_verts))
        G = G + G.T  # Make symmetric
    else:
        idx_fix = np.array([], dtype=int)
        n_fix_verts = 0
        G = sp.csr_matrix((0, 0))

    # % Find the connected components of the graph (still for trivial connections)
    # [bins,binsizes] = conncomp(G);

    if n_fix_verts > 0:
        n_components, bins = connected_components(G, directed=False)
        # Compute component sizes
        binsizes = np.bincount(bins, minlength=n_components)
    else:
        n_components = 0
        bins = np.array([], dtype=int)
        binsizes = np.array([], dtype=int)

    # % For each component store: vertex, edge, triangle indices and edge
    # % orientation sign (still for trivial connections)
    # idx_fix_cell = cell(length(binsizes),1);
    # ide_fix_cell = cell(length(binsizes),1);
    # tri_fix_cell = cell(length(binsizes),1);
    # ide_sign_fix_cell = cell(length(binsizes),1);
    # for i = 1:length(binsizes)
    #     idx_fix_cell{i} = idx_fix(bins == i);
    #     ide_fix_cell{i} = find(all(ismember(Src.E2V, idx_fix_cell{i}), 2));
    #     ide_fix_cell{i} = ide_fix_cell{i}(ismember(ide_fix_cell{i}, ide_fix));
    #     tri_fix_cell{i} = vec(Src.E2T(ide_fix_cell{i},1:2));
    #     ide_sign_fix_cell{i} = vec(Src.E2T(ide_fix_cell{i},3:4));
    # end

    idx_fix_cell = []
    ide_fix_cell = []
    tri_fix_cell = []
    ide_sign_fix_cell = []

    for i in range(n_components):
        # idx_fix_cell{i} = idx_fix(bins == i);
        idx_fix_i = idx_fix[bins == i]
        idx_fix_cell.append(idx_fix_i)

        # ide_fix_cell{i} = find(all(ismember(Src.E2V, idx_fix_cell{i}), 2));
        in_component = np.isin(Src.E2V, idx_fix_i)
        ide_fix_i = np.where(np.all(in_component, axis=1))[0]

        # ide_fix_cell{i} = ide_fix_cell{i}(ismember(ide_fix_cell{i}, ide_fix));
        ide_fix_i = ide_fix_i[np.isin(ide_fix_i, ide_fix)]
        ide_fix_cell.append(ide_fix_i)

        # tri_fix_cell{i} = vec(Src.E2T(ide_fix_cell{i},1:2));
        # MATLAB vec() = column-major flatten
        if len(ide_fix_i) > 0:
            tri_fix_i = Src.E2T[ide_fix_i, :2].flatten('F')
        else:
            tri_fix_i = np.array([], dtype=int)
        tri_fix_cell.append(tri_fix_i)

        # ide_sign_fix_cell{i} = vec(Src.E2T(ide_fix_cell{i},3:4));
        if len(ide_fix_i) > 0:
            ide_sign_fix_i = Src.E2T[ide_fix_i, 2:4].flatten('F')
        else:
            ide_sign_fix_i = np.array([])
        ide_sign_fix_cell.append(ide_sign_fix_i)

    # %% Smooth cross field: Face-to-face parallel transport
    # E2T = zeros(Src.ne,2);
    # E2T(:,1) = Src.E2T(:,1).*(Src.E2T(:,3) > 0) + Src.E2T(:,2).*(Src.E2T(:,3) < 0);
    # E2T(:,2) = Src.E2T(:,1).*(Src.E2T(:,3) < 0) + Src.E2T(:,2).*(Src.E2T(:,3) > 0);

    # NOTE: The MATLAB reordering based on E2T[:, 2] signs doesn't correctly identify
    # f_pos (face with canonical edge direction v0->v1) vs f_neg (face with opposite direction).
    # We explicitly determine f_pos and f_neg by checking edge orientation in each face.
    # E2T[:, 0] = f_neg (face where edge goes v1->v0, opposite to canonical)
    # E2T[:, 1] = f_pos (face where edge goes v0->v1, canonical direction)
    # This ensures para_trans = angle(f_neg) - angle(f_pos) satisfies d1d @ para_trans = K.
    E2T = np.zeros((Src.ne, 2), dtype=int)
    for e in range(Src.ne):
        v0, v1 = Src.E2V[e]  # canonical direction: v0 < v1
        f0, f1 = Src.E2T[e, 0], Src.E2T[e, 1]

        if f1 < 0:  # boundary edge
            E2T[e, 0] = f0
            E2T[e, 1] = f0
            continue

        # Check which face has edge going v0->v1 (canonical = f_pos)
        face0 = Src.T[f0]
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

    # % Compute angle defect
    # K = gaussian_curvature(Src.X, Src.T);
    # assert(norm(sum(K) - 2*pi*(Src.nf-Src.ne+Src.nv)) < 1e-5, 'Gaussian curvature does not match topology.');

    K, _, _ = gaussian_curvature(Src.X, Src.T)
    euler_char = Src.nf - Src.ne + Src.nv
    assert np.abs(np.sum(K) - 2 * np.pi * euler_char) < 1e-5, 'Gaussian curvature does not match topology.'

    # % Local basis: e1r aligned with constrained and boundary edges
    # edge = Src.X(Src.E2V(:,2),:) - Src.X(Src.E2V(:,1),:);
    # edge = edge./sqrt(sum(edge.^2,2));
    # e1r = Src.X(Src.T(:,2),:) - Src.X(Src.T(:,1),:);
    # e1r = e1r./repmat(sqrt(sum(e1r.^2, 2)), [1 3]);

    edge = Src.X[Src.E2V[:, 1], :] - Src.X[Src.E2V[:, 0], :]
    edge = edge / np.linalg.norm(edge, axis=1, keepdims=True)

    e1r = Src.X[Src.T[:, 1], :] - Src.X[Src.T[:, 0], :]
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

    # e2r = cross(Src.normal, e1r, 2);

    e2r = np.cross(Src.normal, e1r)

    # % Angle between edge and local basis
    # edge_angles = zeros(Src.ne,2);
    # edge_angles(ide_int,1) = comp_angle(edge(ide_int,:), e1r(E2T(ide_int,1),:), Src.normal(E2T(ide_int,1),:));
    # edge_angles(ide_int,2) = comp_angle(edge(ide_int,:), e1r(E2T(ide_int,2),:), Src.normal(E2T(ide_int,2),:));

    edge_angles = np.zeros((Src.ne, 2))
    if len(ide_int) > 0:
        edge_angles[ide_int, 0] = comp_angle(
            edge[ide_int, :],
            e1r[E2T[ide_int, 0], :],
            Src.normal[E2T[ide_int, 0], :]
        )
        edge_angles[ide_int, 1] = comp_angle(
            edge[ide_int, :],
            e1r[E2T[ide_int, 1], :],
            Src.normal[E2T[ide_int, 1], :]
        )

    # % Parallel transport
    # para_trans = wrapToPi(edge_angles(:,1) - edge_angles(:,2));
    # para_trans(ide_bound) = 0;

    para_trans = wrap_to_pi(edge_angles[:, 0] - edge_angles[:, 1])
    para_trans[ide_bound] = 0

    # assert(norm(wrapToPi(dec.d1d*para_trans - K)) < 1e-6, 'Gaussian curvature incompatible with angle defect.');

    residual = wrap_to_pi(dec.d1d @ para_trans - K)
    assert np.linalg.norm(residual) < 1e-6, 'Gaussian curvature incompatible with angle defect.'

    # % Angle between local basis and triangleedges
    # param.ang_basis = [comp_angle(Src.X(Src.T(:,1),:) - Src.X(Src.T(:,2),:), e1r, Src.normal), ...
    #                    comp_angle(Src.X(Src.T(:,2),:) - Src.X(Src.T(:,3),:), e1r, Src.normal), ...
    #                    comp_angle(Src.X(Src.T(:,3),:) - Src.X(Src.T(:,1),:), e1r, Src.normal)];

    ang_basis = np.column_stack([
        comp_angle(Src.X[Src.T[:, 0], :] - Src.X[Src.T[:, 1], :], e1r, Src.normal),
        comp_angle(Src.X[Src.T[:, 1], :] - Src.X[Src.T[:, 2], :], e1r, Src.normal),
        comp_angle(Src.X[Src.T[:, 2], :] - Src.X[Src.T[:, 0], :], e1r, Src.normal)
    ])

    # % Store stuff
    # param.E2T = E2T;
    # param.e1r = e1r;
    # param.e2r = e2r;
    # param.para_trans = para_trans;
    # param.Kt = K;
    # param.Kt_invisible = K - dec.d1d*para_trans;

    param.E2T = E2T
    param.e1r = e1r
    param.e2r = e2r
    param.para_trans = para_trans
    param.Kt = K
    param.Kt_invisible = K - dec.d1d @ para_trans
    param.ang_basis = ang_basis

    # %% Smooth cross field: take care of acute angle between edge constraints
    # % See: "Frame Fields for CAD models" https://inria.hal.science/hal-03537852/document
    #
    # % Build exterior derivative of dual 1-form (d1d) where constrained edges
    # % are seen as boundaries
    # E2V = Src.E2V;
    # T = Src.T;
    # nv = Src.nv;

    E2V_mod = Src.E2V.copy()
    T_mod = Src.T.copy()
    nv_mod = Src.nv

    # for i = idx_fix'
    #     [tri_ord,edge_ord,sign_edge] = sort_triangles(i, Src.T, Src.E2T, Src.T2T, Src.E2V, Src.T2E);
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
        tri_ord, edge_ord, sign_edge = sort_triangles(i, Src.T, Src.E2T, Src.T2T, Src.E2V, Src.T2E)

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

    # ide_free = setdiff((1:Src.ne)', ide_fix);
    # d1d = sparse(E2V(ide_free,:), [ide_free, ide_free], [ones(length(ide_free),1),-ones(length(ide_free),1)], nv, Src.ne);

    ide_free = np.setdiff1d(np.arange(Src.ne), ide_fix)

    # Build d1d sparse matrix
    row = np.concatenate([E2V_mod[ide_free, 0], E2V_mod[ide_free, 1]])
    col = np.concatenate([ide_free, ide_free])
    data = np.concatenate([np.ones(len(ide_free)), -np.ones(len(ide_free))])
    d1d_new = sp.csr_matrix((data, (row, col)), shape=(nv_mod, Src.ne))

    # assert(all(sum(abs(d1d),2) ~= 0));

    row_sums = np.array(np.abs(d1d_new).sum(axis=1)).flatten()
    assert np.all(row_sums != 0), 'd1d has zero rows'

    # % Store stuff
    # Vp2V = unique([T(:), Src.T(:)], 'rows');
    # [~,id] = sort(Vp2V(:,1));
    # param.Vp2V = Vp2V(id,2);

    # T_mod.flatten('F') and Src.T.flatten('F') for column-major
    pairs = np.column_stack([T_mod.flatten('F'), Src.T.flatten('F')])
    Vp2V_unique = np.unique(pairs, axis=0)
    sort_idx = np.argsort(Vp2V_unique[:, 0])
    Vp2V = Vp2V_unique[sort_idx, 1]

    # param.d1d = d1d;
    # param.idx_fix_plus = [idx_fix; (Src.nv+1:nv)'];
    # param.idx_reg = setdiff((1:Src.nv)', idx_fix);

    idx_fix_plus = np.concatenate([idx_fix, np.arange(Src.nv, nv_mod)])
    idx_reg = np.setdiff1d(np.arange(Src.nv), idx_fix)

    # theta = angles_of_triangles(Src.X, Src.T);
    # K = 2*pi - accumarray(T(:), theta(:));
    # K(param.idx_fix_plus) = K(param.idx_fix_plus) - pi;

    theta = angles_of_triangles(Src.X, Src.T)
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

    # %% Trivial connection: Path between isolated constraints
    # % Build the sparse matrix Ilink accumulating the dual 1-form along each
    # % path connecting the connected component of constraints
    # nc = max(length(ide_fix_cell) - 1, 0);
    # Ilink = sparse(nc,Src.ne);

    nc = max(len(ide_fix_cell) - 1, 0)
    Ilink = sp.lil_matrix((nc, Src.ne))

    # % Compute a shortest path between the first component and all the other
    # for i = 1:nc

    for i in range(nc):
        # % Edge weights
        # ld = max(dec.star1p*sqrt(Src.SqEdgeLength), 1e-5);

        ld = np.maximum(dec.star1p.diagonal() * np.sqrt(Src.SqEdgeLength), 1e-5)

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

        # % Build primal graph (on dual mesh: faces are nodes)
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

        Gd = sp.csr_matrix((weights_d, (row_d, col_d)), shape=(Src.nf, Src.nf))
        Gd = Gd + Gd.T  # Make symmetric

        # % Shortest dual path from a vertex in component 1 and a vertex in
        # % compenent i+1
        # P = shortestpath(Gd, tri_fix_cell{1}(1), tri_fix_cell{i+1}(1))';

        start_face = tri_fix_cell[0][0] if len(tri_fix_cell[0]) > 0 else 0
        end_face = tri_fix_cell[i + 1][0] if len(tri_fix_cell[i + 1]) > 0 else 0

        # Use Dijkstra's algorithm to find shortest path
        dist, predecessors = dijkstra(Gd, indices=start_face, return_predecessors=True)

        # Reconstruct path
        P = _reconstruct_path(predecessors, start_face, end_face)

        if len(P) < 2:
            continue

        # % Find edge indices of the path
        # ed = [P(1:end-1), P(2:end)];
        # [~,~,ide] = intersect(sort(ed,2), sort(E2T,2), 'rows', 'stable');
        # assert(length(ide) == length(P)-1);

        ed = np.column_stack([P[:-1], P[1:]])
        ed_sorted = np.sort(ed, axis=1)
        E2T_sorted = np.sort(E2T, axis=1)
        ide_path = _intersect_rows_stable(ed_sorted, E2T_sorted)

        if len(ide_path) != len(P) - 1:
            continue  # Path reconstruction failed

        # % Remove part of the path on the boundary of 1 and i+1
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

        # % Edge sign
        # s = (E2T(ide,1) == ed(:,1)) - (E2T(ide,2) == ed(:,1));

        s = (E2T[ide_path, 0] == ed[:, 0]).astype(int) - (E2T[ide_path, 1] == ed[:, 0]).astype(int)

        # % Build 1-form integration matrix along the path
        # Ilink(i,:) = sparse(ones(length(ide),1), ide, s, 1, Src.ne);

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

    # %% Trivial connection: Non-contractible cycles
    # % Compute non-constractible loops
    # [cycle,cocycle] = find_graph_generator(full(diag(dec.star1p)), Src.T, Src.E2T, Src.E2V, 1);

    star1p_diag = np.array(dec.star1p.diagonal())
    cycle, cocycle = find_graph_generator(star1p_diag, Src.T, Src.E2T, Src.E2V, init=0)

    # % Build the sparse matrix Icycle accumulating the dual 1-form along each
    # % cycle
    # nc = length(cocycle);
    # Icycle = sparse(nc,Src.ne);

    nc_cycles = len(cocycle)
    Icycle = sp.lil_matrix((nc_cycles, Src.ne))

    # for i = 1:nc
    #     % Find edge indices correpsonding to cycle
    #     ed = [cocycle{i}, circshift(cocycle{i}, [1,0])];
    #     [~,ide] = ismember(sort(ed,2), sort(E2T,2), 'rows');
    #     assert(length(ide) == length(cocycle{i}));
    #
    #     % Find edge sign in the cycle
    #     s = (E2T(ide,1) == ed(:,1)) - (E2T(ide,2) == ed(:,1));
    #
    #     % Build sparce matrix
    #     Icycle(i,:) = sparse(ones(length(ide),1), ide, s, 1, Src.ne);
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

        # % Find edge sign in the cycle
        # s = (E2T(ide,1) == ed(:,1)) - (E2T(ide,2) == ed(:,1));

        s = (E2T[ide_cycle, 0] == ed[:, 0]).astype(int) - (E2T[ide_cycle, 1] == ed[:, 0]).astype(int)

        # % Build sparce matrix
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

    # %% Set constraint list
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
        tri_fix_final = np.concatenate([tri_hard.ravel(), tri_bound])
    elif ifboundary:
        ide_fix_final = ide_bound
        tri_fix_final = tri_bound
    elif ifhardedge:
        ide_fix_final = ide_hard
        tri_fix_final = tri_hard.ravel()
    else:
        ide_fix_final = np.array([], dtype=int)
        tri_fix_final = np.array([], dtype=int)

    # idx_fix = unique(Src.E2V(ide_fix,:));
    # param.ide_fix = ide_fix;
    # param.idx_fix = idx_fix;
    # param.ide_free = setdiff((1:Src.ne)', ide_fix);
    # param.tri_fix = tri_fix;
    # param.tri_free = setdiff((1:Src.nf)', param.tri_fix);

    if len(ide_fix_final) > 0:
        idx_fix_final = np.unique(Src.E2V[ide_fix_final, :])
    else:
        idx_fix_final = np.array([], dtype=int)

    param.ide_fix = ide_fix_final
    param.idx_fix = idx_fix_final
    param.ide_free = np.setdiff1d(np.arange(Src.ne), ide_fix_final)
    param.tri_fix = tri_fix_final
    param.tri_free = np.setdiff1d(np.arange(Src.nf), tri_fix_final)

    return param, Src, dec


# function [ide_hard,tri_hard,ide_bound,tri_bound] = detect_hard_edge(Src, tol_dihedral_deg)
# comp_angle = @(u,v,n) atan2(dot(cross(u,v,2),n,2), dot(u,v,2));
# tol_dihedral = tol_dihedral_deg*pi/180;

def detect_hard_edge(Src: MeshInfo, tol_dihedral_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect hard edges based on dihedral angle threshold.

    Parameters
    ----------
    Src : MeshInfo
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

    # % Find boundary edges
    # [ide_bound,tri_bound] = boundary_indices(Src);

    ide_bound, tri_bound = boundary_indices(Src)

    # % Interior edges
    # ide_int = setdiff((1:Src.ne)', ide_bound);

    ide_int = np.setdiff1d(np.arange(Src.ne), ide_bound)

    # % Compute unit vector edges
    # edge = Src.X(Src.E2V(:,2),:) - Src.X(Src.E2V(:,1),:);
    # edge = edge./sqrt(sum(edge.^2,2));

    edge = Src.X[Src.E2V[:, 1], :] - Src.X[Src.E2V[:, 0], :]
    edge = edge / np.linalg.norm(edge, axis=1, keepdims=True)

    # % Compute angle between normals by using the edge vector for sign
    # dihedral_angle = Src.E2T(ide_int,4).*comp_angle(Src.normal(Src.E2T(ide_int,1),:), Src.normal(Src.E2T(ide_int,2),:), edge(ide_int,:));

    # E2T[:, 3] is the edge sign (column 4 in MATLAB 1-indexed)
    t0 = Src.E2T[ide_int, 0]
    t1 = Src.E2T[ide_int, 1]
    sign_col = Src.E2T[ide_int, 3]

    dihedral_angle = sign_col * comp_angle(
        Src.normal[t0, :],
        Src.normal[t1, :],
        edge[ide_int, :]
    )

    # % Hard edges are angle larger than a threshold
    # ide_hard = ide_int(abs(dihedral_angle) > tol_dihedral);

    ide_hard = ide_int[np.abs(dihedral_angle) > tol_dihedral]

    # % Find adjancent triangles
    # tri_hard = Src.E2T(ide_hard,1:2);

    if len(ide_hard) > 0:
        tri_hard = Src.E2T[ide_hard, :2]
    else:
        tri_hard = np.zeros((0, 2), dtype=int)

    return ide_hard, tri_hard, ide_bound, tri_bound


# function [ide_bound,tri_bound] = boundary_indices(Src)
# % A boundary edge belongs to only one triangle
# ide_bound = find(any(Src.E2T == 0, 2));
#
# % A boundary triangle is incident to a boundary edge
# tri_bound = sum(Src.E2T(ide_bound,1:2),2);

def boundary_indices(Src: MeshInfo) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find boundary edge and triangle indices.

    Parameters
    ----------
    Src : MeshInfo
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
    ide_bound = np.where(np.any(Src.E2T[:, :2] < 0, axis=1))[0]

    # A boundary triangle is incident to a boundary edge
    # For boundary edges, one of E2T[:, 0] or E2T[:, 1] is -1
    # The non-negative one is the triangle
    if len(ide_bound) > 0:
        tri_bound = np.maximum(Src.E2T[ide_bound, 0], Src.E2T[ide_bound, 1])
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
