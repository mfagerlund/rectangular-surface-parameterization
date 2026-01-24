
# function mesh = MeshInfo(X, T)
#
# mesh.X = X;
# mesh.T = T;
# assert(size(T,2) == 3, 'Not a triangulations.');
# [mesh.E2V, mesh.T2E, mesh.E2T, mesh.T2T] = connectivity(mesh.T);
#
# mesh.nf = size(mesh.T,1);
# mesh.nv = size(mesh.X,1);
# mesh.ne = size(mesh.E2V,1);
#
# mesh.normal = cross(mesh.X(mesh.T(:,1),:) - mesh.X(mesh.T(:,2),:), mesh.X(mesh.T(:,1),:) - mesh.X(mesh.T(:,3),:));
# mesh.area = sqrt(sum(mesh.normal.^2, 2))/2;
# mesh.normal = mesh.normal./repmat(sqrt(sum(mesh.normal.^2, 2)), [1, 3]);
#
# A = sparse(mesh.T, repmat((1:mesh.nf)', [3,1]), repmat(mesh.area, [3,1]), mesh.nv, mesh.nf);
# mesh.Nv = A*mesh.normal;
# mesh.Nv = mesh.Nv./repmat(sqrt(sum(mesh.Nv.^2,2)), [1,3]);
#
# mesh.SqEdgeLength = sum((mesh.X(mesh.E2V(:,1),:) - mesh.X(mesh.E2V(:,2),:)).^2, 2);
#
# mesh.corner_angle = angles_of_triangles(mesh.X, mesh.T);
# mesh.cot_corner_angle = cot(mesh.corner_angle);
#
# end


# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix

from .connectivity import connectivity
from .angles_of_triangles import angles_of_triangles


@dataclass
class MeshInfo:
    """
    Mesh data structure containing geometry and connectivity.

    Attributes
    ----------
    X : ndarray (nv, 3)
        Vertex positions.
    T : ndarray (nf, 3)
        Triangle vertex indices (0-indexed).
    nf : int
        Number of faces (triangles).
    nv : int
        Number of vertices.
    ne : int
        Number of edges.
    E2V : ndarray (ne, 2)
        Edge-to-vertex connectivity. E2V[e] = [v0, v1].
    T2E : ndarray (nf, 3)
        Triangle-to-edge connectivity (signed). T2E[f, i] gives the edge
        opposite to vertex i, with sign indicating orientation.
    E2T : ndarray (ne, 4)
        Edge-to-triangle connectivity. E2T[e] = [t0, t1, s0, s1] where
        t0, t1 are adjacent triangles (0 if boundary) and s0, s1 are signs.
    T2T : ndarray (nf, 3)
        Triangle-to-triangle adjacency.
    normal : ndarray (nf, 3)
        Per-face unit normals.
    area : ndarray (nf,)
        Per-face areas.
    Nv : ndarray (nv, 3)
        Per-vertex unit normals (area-weighted average of face normals).
    SqEdgeLength : ndarray (ne,)
        Squared edge lengths.
    corner_angle : ndarray (nf, 3)
        Corner angles in radians. corner_angle[f, i] is the angle at vertex T[f, i].
    cot_corner_angle : ndarray (nf, 3)
        Cotangent of corner angles.
    """

    X: np.ndarray
    T: np.ndarray
    nf: int
    nv: int
    ne: int
    E2V: np.ndarray
    T2E: np.ndarray
    E2T: np.ndarray
    T2T: np.ndarray
    normal: np.ndarray
    area: np.ndarray
    Nv: np.ndarray
    SqEdgeLength: np.ndarray
    corner_angle: np.ndarray
    cot_corner_angle: np.ndarray


def mesh_info(X: np.ndarray, T: np.ndarray) -> MeshInfo:
    """
    Create a MeshInfo struct from vertices and triangles.

    Parameters
    ----------
    X : ndarray (nv, 3)
        Vertex positions.
    T : ndarray (nf, 3)
        Triangle vertex indices (0-indexed).

    Returns
    -------
    MeshInfo
        Mesh data structure with connectivity and geometric attributes.
    """

    # mesh.X = X;
    # mesh.T = T;
    # assert(size(T,2) == 3, 'Not a triangulations.');

    assert T.shape[1] == 3, "Not a triangulation."

    # [mesh.E2V, mesh.T2E, mesh.E2T, mesh.T2T] = connectivity(mesh.T);

    E2V, T2E, E2T, T2T = connectivity(T)

    # mesh.nf = size(mesh.T,1);
    # mesh.nv = size(mesh.X,1);
    # mesh.ne = size(mesh.E2V,1);

    nf = T.shape[0]
    nv = X.shape[0]
    ne = E2V.shape[0]

    # mesh.normal = cross(mesh.X(mesh.T(:,1),:) - mesh.X(mesh.T(:,2),:), mesh.X(mesh.T(:,1),:) - mesh.X(mesh.T(:,3),:));

    # MATLAB is 1-indexed, Python is 0-indexed
    # T(:,1) -> T[:, 0], T(:,2) -> T[:, 1], T(:,3) -> T[:, 2]
    v0 = X[T[:, 0], :]
    v1 = X[T[:, 1], :]
    v2 = X[T[:, 2], :]
    normal = np.cross(v0 - v1, v0 - v2)

    # mesh.area = sqrt(sum(mesh.normal.^2, 2))/2;

    normal_norms = np.linalg.norm(normal, axis=1)
    area = normal_norms / 2.0

    # mesh.normal = mesh.normal./repmat(sqrt(sum(mesh.normal.^2, 2)), [1, 3]);

    # Normalize face normals (avoid division by zero)
    normal = normal / np.maximum(normal_norms[:, np.newaxis], 1e-12)

    # A = sparse(mesh.T, repmat((1:mesh.nf)', [3,1]), repmat(mesh.area, [3,1]), mesh.nv, mesh.nf);
    # mesh.Nv = A*mesh.normal;

    # Build sparse matrix A of shape (nv, nf) where A[v, f] = area[f] if v is in triangle f
    # MATLAB: sparse(row_indices, col_indices, values, nrows, ncols)
    # The T matrix gives vertex indices, we want to map vertex -> face with area weight
    # MATLAB code: rows = T (flattened), cols = [1:nf, 1:nf, 1:nf], values = [area, area, area]
    row_indices = T.flatten()  # shape (3*nf,) - vertex indices
    col_indices = np.tile(np.arange(nf), 3)  # [0,1,...,nf-1, 0,1,...,nf-1, 0,1,...,nf-1]
    values = np.tile(area, 3)  # area repeated 3 times

    A = csr_matrix((values, (row_indices, col_indices)), shape=(nv, nf))

    # Nv = A * normal gives area-weighted sum of face normals for each vertex
    Nv = A @ normal

    # mesh.Nv = mesh.Nv./repmat(sqrt(sum(mesh.Nv.^2,2)), [1,3]);

    # Normalize vertex normals
    Nv_norms = np.linalg.norm(Nv, axis=1, keepdims=True)
    Nv = Nv / np.maximum(Nv_norms, 1e-12)

    # mesh.SqEdgeLength = sum((mesh.X(mesh.E2V(:,1),:) - mesh.X(mesh.E2V(:,2),:)).^2, 2);

    # MATLAB E2V(:,1) -> Python E2V[:, 0], E2V(:,2) -> E2V[:, 1]
    edge_vec = X[E2V[:, 0], :] - X[E2V[:, 1], :]
    SqEdgeLength = np.sum(edge_vec ** 2, axis=1)

    # mesh.corner_angle = angles_of_triangles(mesh.X, mesh.T);

    corner_angle = angles_of_triangles(X, T)

    # mesh.cot_corner_angle = cot(mesh.corner_angle);

    # MATLAB cot(x) = 1/tan(x)
    cot_corner_angle = 1.0 / np.tan(corner_angle)

    return MeshInfo(
        X=X,
        T=T,
        nf=nf,
        nv=nv,
        ne=ne,
        E2V=E2V,
        T2E=T2E,
        E2T=E2T,
        T2T=T2T,
        normal=normal,
        area=area,
        Nv=Nv,
        SqEdgeLength=SqEdgeLength,
        corner_angle=corner_angle,
        cot_corner_angle=cot_corner_angle,
    )
