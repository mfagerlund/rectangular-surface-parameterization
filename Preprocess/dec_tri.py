
# function dec = dec_tri(mesh)
#
# nv = mesh.num_vertices;
# ne = mesh.num_edges;
# nf = mesh.num_faces;
#
# l2 = [sum((mesh.vertices(mesh.triangles(:,2),:) - mesh.vertices(mesh.triangles(:,3),:)).^2, 2), ...
#       sum((mesh.vertices(mesh.triangles(:,3),:) - mesh.vertices(mesh.triangles(:,1),:)).^2, 2), ...
#       sum((mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,2),:)).^2, 2)];
# half_area = l2.*abs(mesh.cot_corner_angle)/4; % force positive area
# half_area = (half_area(:,[2 3 1]) + half_area(:,[3 1 2]))/2;
# half_area = repmat(mesh.area/3, [1,3]);
# vor_area = accumarray(mesh.triangles(:), half_area(:));
# cotweight = accumarray(abs(mesh.T2E(:)), vec(mesh.cot_corner_angle(:,[3 1 2]))/2);
# assert(all(vor_area > 0), 'Negative vertex area.');
#
# if any(cotweight < 1e-5)
#     warning('Non Delaunay tet-mesh: risk of convergence issues!');
#     cotweight = max(cotweight, 1e-5); % clamp to avoid problems
# end
#
# d0p = sparse([1:ne;1:ne]', mesh.E2V, [ones(ne,1),-ones(ne,1)], ne, nv);
# d1p = sparse([1:nf;1:nf;1:nf]', abs(mesh.T2E), sign(mesh.T2E), nf, ne);
# assert(norm(d1p*d0p, 'fro') == 0, 'Assembling DEC: Orinetation problems');
#
# star0p = sparse(1:nv, 1:nv, vor_area, nv, nv);
# star1p = sparse(1:ne, 1:ne, cotweight, ne, ne);
# star2p = sparse(1:nf, 1:nf, 1./mesh.area, nf, nf);
#
# d0d = d1p';
# d1d = d0p';
# assert(norm(d1d*d0d, 'fro') == 0, 'Assembling DEC: Orinetation problems');
#
# star0d = sparse(1:nf, 1:nf, mesh.area, nf, nf);
# star1d = sparse(1:ne, 1:ne, 1./cotweight, ne, ne);
# star2d = sparse(1:nv, 1:nv, 1./vor_area, nv, nv);
#
# dec.d0p = d0p;
# dec.d1p = d1p;
# dec.d0d = d0d;
# dec.d1d = d1d;
#
# dec.star0p = star0p;
# dec.star1p = star1p;
# dec.star2p = star2p;
# dec.star0d = star0d;
# dec.star1d = star1d;
# dec.star2d = star2d;
#
# Reduction_tri = sparse(reshape((1:3*nf)', [nf,3]), mesh.triangles, 1, 3*nf, nv);
# deg_ed = accumarray(abs(mesh.T2E(:)), 1);
# I = abs(mesh.T2E(:,[1 2 3]));
# J = reshape((1:3*nf),[nf,3]);
# S = sign(mesh.T2E(:,[1 2 3]))./deg_ed(abs(mesh.T2E));
# d0p_tri = sparse([I, I], [J, J(:,[2 3 1])], [-S, S], ne, 3*nf);
#
# star0p_tri = sparse(J, J, half_area, 3*nf, 3*nf);
#
# W = d0p'*star1p*d0p;
# W_tri = d0p_tri'*star1p*d0p_tri;
#
# dec.W = (W + W')/2;
# dec.d0p_tri = d0p_tri;
# dec.star0p_tri = star0p_tri;
# dec.W_tri = (W_tri + W_tri')/2;
# dec.Reduction_tri = Reduction_tri;
#
# end


# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
import warnings

from .MeshInfo import MeshInfo


@dataclass
class DEC:
    """
    Discrete Exterior Calculus operators for a triangle mesh.

    Primal operators (on the mesh):
        d0p : sparse (ne, nv) - Gradient operator (vertices to edges)
        d1p : sparse (nf, ne) - Curl operator (edges to faces)
        star0p : sparse (nv, nv) - Hodge star on 0-forms (vertex areas)
        star1p : sparse (ne, ne) - Hodge star on 1-forms (cotangent weights)
        star2p : sparse (nf, nf) - Hodge star on 2-forms (inverse face areas)

    Dual operators (on the dual mesh):
        d0d : sparse (ne, nf) - Dual gradient (d1p.T)
        d1d : sparse (nv, ne) - Dual curl (d0p.T)
        star0d : sparse (nf, nf) - Dual Hodge star on 0-forms (face areas)
        star1d : sparse (ne, ne) - Dual Hodge star on 1-forms (inverse cotangent weights)
        star2d : sparse (nv, nv) - Dual Hodge star on 2-forms (inverse vertex areas)

    Triangle-based operators:
        W : sparse (nv, nv) - Laplacian (cotangent weights)
        d0p_tri : sparse (ne, 3*nf) - Per-corner gradient
        star0p_tri : sparse (3*nf, 3*nf) - Per-corner area weights
        W_tri : sparse (3*nf, 3*nf) - Per-corner Laplacian
        Reduction_tri : sparse (3*nf, nv) - Maps vertices to corners
    """

    d0p: csr_matrix
    d1p: csr_matrix
    d0d: csr_matrix
    d1d: csr_matrix
    star0p: csr_matrix
    star1p: csr_matrix
    star2p: csr_matrix
    star0d: csr_matrix
    star1d: csr_matrix
    star2d: csr_matrix
    W: csr_matrix
    d0p_tri: csr_matrix
    star0p_tri: csr_matrix
    W_tri: csr_matrix
    Reduction_tri: csr_matrix


def dec_tri(mesh: MeshInfo) -> DEC:
    """
    Build Discrete Exterior Calculus operators for a triangle mesh.

    Parameters
    ----------
    mesh : MeshInfo
        Mesh data structure with vertices, triangles, and connectivity.

    Returns
    -------
    DEC
        Data structure containing all DEC operators.
    """

    # nv = mesh.num_vertices;
    # ne = mesh.num_edges;
    # nf = mesh.num_faces;

    nv = mesh.num_vertices
    ne = mesh.num_edges
    nf = mesh.num_faces

    # l2 = [sum((mesh.vertices(mesh.triangles(:,2),:) - mesh.vertices(mesh.triangles(:,3),:)).^2, 2), ...
    #       sum((mesh.vertices(mesh.triangles(:,3),:) - mesh.vertices(mesh.triangles(:,1),:)).^2, 2), ...
    #       sum((mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,2),:)).^2, 2)];

    # Squared edge lengths opposite to each vertex
    # l2[:, 0] = |v1 - v2|^2 (opposite to v0)
    # l2[:, 1] = |v2 - v0|^2 (opposite to v1)
    # l2[:, 2] = |v0 - v1|^2 (opposite to v2)
    X, T = mesh.vertices, mesh.triangles
    l2 = np.column_stack([
        np.sum((X[T[:, 1], :] - X[T[:, 2], :]) ** 2, axis=1),  # opposite to v0
        np.sum((X[T[:, 2], :] - X[T[:, 0], :]) ** 2, axis=1),  # opposite to v1
        np.sum((X[T[:, 0], :] - X[T[:, 1], :]) ** 2, axis=1),  # opposite to v2
    ])

    # half_area = l2.*abs(mesh.cot_corner_angle)/4; % force positive area
    # half_area = (half_area(:,[2 3 1]) + half_area(:,[3 1 2]))/2;
    # half_area = repmat(mesh.area/3, [1,3]);

    # Note: MATLAB code has three assignments to half_area, last one wins
    # Final: half_area = area/3 for each corner (barycentric subdivision)
    half_area = np.tile(mesh.area[:, np.newaxis] / 3, (1, 3))  # shape (nf, 3)

    # vor_area = accumarray(mesh.triangles(:), half_area(:));

    # Accumulate corner areas to vertices
    # T.flatten('F') gives column-major order (MATLAB default)
    # half_area.flatten('F') to match
    T_flat = T.flatten('F')  # shape (3*nf,)
    half_area_flat = half_area.flatten('F')  # shape (3*nf,)
    vor_area = np.bincount(T_flat, weights=half_area_flat, minlength=nv)

    # cotweight = accumarray(abs(mesh.T2E(:)), vec(mesh.cot_corner_angle(:,[3 1 2]))/2);

    # cot_corner_angle[:, i] is the cotangent at vertex i
    # We need cot at the corner OPPOSITE to each edge
    # T2E[f, 0] is edge opposite to vertex 0 -> use cot at vertex 0
    # But MATLAB uses cot_corner_angle(:,[3 1 2]) which is [cot2, cot0, cot1]
    # This is the cotangent at the vertex opposite to the edge
    # MATLAB 1-indexed: column [3,1,2] -> Python 0-indexed: [2,0,1]
    cot_reordered = mesh.cot_corner_angle[:, [2, 0, 1]]  # shape (nf, 3)

    # Flatten in column-major order to match MATLAB vec()
    # T2E uses 1-based signed encoding, so abs(T2E)-1 gives 0-based edge indices
    T2E_flat = (np.abs(mesh.T2E) - 1).flatten('F')  # shape (3*nf,), 0-based edge indices
    cot_flat = cot_reordered.flatten('F') / 2  # shape (3*nf,)

    # Accumulate cotangent weights to edges
    cotweight = np.bincount(T2E_flat, weights=cot_flat, minlength=ne)

    # assert(all(vor_area > 0), 'Negative vertex area.');

    # Check for unreferenced vertices (vor_area = 0) and handle them
    unreferenced = (vor_area == 0)
    n_unreferenced = np.sum(unreferenced)
    if n_unreferenced > 0:
        warnings.warn(f"Mesh has {n_unreferenced} unreferenced vertices. Assigning small area.")
        # Assign a small positive area to unreferenced vertices
        min_positive_area = vor_area[vor_area > 0].min() if np.any(vor_area > 0) else 1e-10
        vor_area[unreferenced] = min_positive_area * 1e-3

    # Check for truly negative areas (shouldn't happen with barycentric subdivision)
    if np.any(vor_area < 0):
        warnings.warn("Negative vertex areas detected. Using absolute values.")
        vor_area = np.abs(vor_area)

    assert np.all(vor_area > 0), "Negative vertex area."

    # if any(cotweight < 1e-5)
    #     warning('Non Delaunay tet-mesh: risk of convergence issues!');
    #     cotweight = max(cotweight, 1e-5); % clamp to avoid problems
    # end

    if np.any(cotweight < 1e-5):
        warnings.warn("Non Delaunay mesh: risk of convergence issues!")
        cotweight = np.maximum(cotweight, 1e-5)

    # d0p = sparse([1:ne;1:ne]', mesh.E2V, [ones(ne,1),-ones(ne,1)], ne, nv);

    # d0p is the gradient operator: for each edge, +1 at end vertex, -1 at start vertex
    # E2V[e] = [v0, v1] -> d0p[e, v1] = +1, d0p[e, v0] = -1
    row_idx = np.concatenate([np.arange(ne), np.arange(ne)])
    col_idx = np.concatenate([mesh.E2V[:, 0], mesh.E2V[:, 1]])
    data = np.concatenate([np.ones(ne), -np.ones(ne)])
    d0p = csr_matrix((data, (row_idx, col_idx)), shape=(ne, nv))

    # d1p = sparse([1:nf;1:nf;1:nf]', abs(mesh.T2E), sign(mesh.T2E), nf, ne);

    # d1p is the curl operator: for each face, +/-1 for each edge based on orientation
    # Note: flatten() uses row-major order, so row_idx must use np.repeat (not np.tile)
    # to get [0,0,0, 1,1,1, ...] matching [T2E[0,0], T2E[0,1], T2E[0,2], T2E[1,0], ...]
    row_idx = np.repeat(np.arange(nf), 3)  # [0,0,0, 1,1,1, ..., nf-1,nf-1,nf-1]
    col_idx = (np.abs(mesh.T2E) - 1).flatten()  # 0-based edge indices
    data = np.sign(mesh.T2E).flatten()  # edge orientations
    d1p = csr_matrix((data, (row_idx, col_idx)), shape=(nf, ne))

    # assert(norm(d1p*d0p, 'fro') == 0, 'Assembling DEC: Orinetation problems');

    assert sparse_norm(d1p @ d0p, 'fro') < 1e-10, "Assembling DEC: Orientation problems"

    # star0p = sparse(1:nv, 1:nv, vor_area, nv, nv);
    # star1p = sparse(1:ne, 1:ne, cotweight, ne, ne);
    # star2p = sparse(1:nf, 1:nf, 1./mesh.area, nf, nf);

    # Hodge star operators (diagonal matrices)
    star0p = diags(vor_area, format='csr')
    star1p = diags(cotweight, format='csr')
    star2p = diags(1.0 / mesh.area, format='csr')

    # d0d = d1p';
    # d1d = d0p';

    # Dual boundary operators are transposes of primal
    d0d = d1p.T.tocsr()
    d1d = d0p.T.tocsr()

    # assert(norm(d1d*d0d, 'fro') == 0, 'Assembling DEC: Orinetation problems');

    assert sparse_norm(d1d @ d0d, 'fro') < 1e-10, "Assembling DEC: Orientation problems"

    # star0d = sparse(1:nf, 1:nf, mesh.area, nf, nf);
    # star1d = sparse(1:ne, 1:ne, 1./cotweight, ne, ne);
    # star2d = sparse(1:nv, 1:nv, 1./vor_area, nv, nv);

    # Dual Hodge star operators
    star0d = diags(mesh.area, format='csr')
    star1d = diags(1.0 / cotweight, format='csr')
    star2d = diags(1.0 / vor_area, format='csr')

    # Reduction_tri = sparse(reshape((1:3*nf)', [nf,3]), mesh.triangles, 1, 3*nf, nv);

    # Reduction_tri maps vertices to corners (3*nf x nv)
    # Reduction_tri[corner, vertex] = 1 if that corner corresponds to that vertex
    # reshape((1:3*nf)', [nf,3]) creates column-major reshape in MATLAB
    # In Python: np.arange(3*nf).reshape((nf, 3), order='F')
    corner_idx = np.arange(3 * nf).reshape((nf, 3), order='F')  # shape (nf, 3)
    row_idx = corner_idx.flatten()  # corner indices
    col_idx = mesh.triangles.flatten()  # vertex indices (row-major is fine here)
    # Actually, need to match MATLAB's column-major flattening
    row_idx = corner_idx.flatten('F')
    col_idx = mesh.triangles.flatten('F')
    data = np.ones(3 * nf)
    Reduction_tri = csr_matrix((data, (row_idx, col_idx)), shape=(3 * nf, nv))

    # deg_ed = accumarray(abs(mesh.T2E(:)), 1);

    # Count how many times each edge appears (edge degree)
    # T2E uses 1-based signed encoding, so abs(T2E)-1 gives 0-based edge indices
    T2E_abs_flat = (np.abs(mesh.T2E) - 1).flatten()
    deg_ed = np.bincount(T2E_abs_flat, minlength=ne)

    # I = abs(mesh.T2E(:,[1 2 3]));

    # MATLAB [1 2 3] -> Python [0, 1, 2] (same order)
    # T2E uses 1-based signed encoding
    I = np.abs(mesh.T2E[:, [0, 1, 2]]) - 1  # shape (nf, 3), 0-based edge indices

    # J = reshape((1:3*nf),[nf,3]);

    # Corner indices reshaped
    J = np.arange(3 * nf).reshape((nf, 3), order='F')  # shape (nf, 3)

    # S = sign(mesh.T2E(:,[1 2 3]))./deg_ed(abs(mesh.T2E));

    # Sign of edge divided by edge degree
    T2E_sign = np.sign(mesh.T2E[:, [0, 1, 2]])  # shape (nf, 3)
    T2E_abs = np.abs(mesh.T2E[:, [0, 1, 2]]) - 1  # shape (nf, 3), 0-based edge indices
    S = T2E_sign / deg_ed[T2E_abs]  # shape (nf, 3)

    # d0p_tri = sparse([I, I], [J, J(:,[2 3 1])], [-S, S], ne, 3*nf);

    # Build d0p_tri: per-corner gradient operator
    # [I, I] means edge indices repeated
    # [J, J(:,[2 3 1])] means corner indices and shifted corner indices
    # MATLAB [2 3 1] -> Python [1, 2, 0]
    J_shifted = J[:, [1, 2, 0]]  # shift columns

    # Flatten all arrays
    I_flat = np.concatenate([I.flatten('F'), I.flatten('F')])
    J_flat = np.concatenate([J.flatten('F'), J_shifted.flatten('F')])
    S_flat = np.concatenate([-S.flatten('F'), S.flatten('F')])

    d0p_tri = csr_matrix((S_flat, (I_flat, J_flat)), shape=(ne, 3 * nf))

    # star0p_tri = sparse(J, J, half_area, 3*nf, 3*nf);

    # Per-corner area weights (diagonal)
    J_flat_diag = J.flatten('F')
    half_area_flat_diag = half_area.flatten('F')
    star0p_tri = csr_matrix(
        (half_area_flat_diag, (J_flat_diag, J_flat_diag)),
        shape=(3 * nf, 3 * nf)
    )

    # W = d0p'*star1p*d0p;
    # W_tri = d0p_tri'*star1p*d0p_tri;

    # Laplacian operators
    W = d0p.T @ star1p @ d0p
    W_tri = d0p_tri.T @ star1p @ d0p_tri

    # dec.W = (W + W')/2;
    # dec.W_tri = (W_tri + W_tri')/2;

    # Symmetrize
    W = (W + W.T) / 2
    W_tri = (W_tri + W_tri.T) / 2

    return DEC(
        d0p=d0p,
        d1p=d1p,
        d0d=d0d,
        d1d=d1d,
        star0p=star0p,
        star1p=star1p,
        star2p=star2p,
        star0d=star0d,
        star1d=star1d,
        star2d=star2d,
        W=W.tocsr(),
        d0p_tri=d0p_tri,
        star0p_tri=star0p_tri,
        W_tri=W_tri.tocsr(),
        Reduction_tri=Reduction_tri,
    )
