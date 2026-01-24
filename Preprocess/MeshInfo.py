
# function mesh = MeshInfo(X, T)
#
# mesh.vertices = X;
# mesh.triangles = T;
# assert(size(T,2) == 3, 'Not a triangulations.');
# [mesh.edge_to_vertex, mesh.T2E, mesh.edge_to_triangle, mesh.triangle_to_triangle] = connectivity(mesh.triangles);
#
# mesh.num_faces = size(mesh.triangles,1);
# mesh.num_vertices = size(mesh.vertices,1);
# mesh.num_edges = size(mesh.edge_to_vertex,1);
#
# mesh.normal = cross(mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,2),:), mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,3),:));
# mesh.area = sqrt(sum(mesh.normal.^2, 2))/2;
# mesh.normal = mesh.normal./repmat(sqrt(sum(mesh.normal.^2, 2)), [1, 3]);
#
# A = sparse(mesh.triangles, repmat((1:mesh.num_faces)', [3,1]), repmat(mesh.area, [3,1]), mesh.num_vertices, mesh.num_faces);
# mesh.vertex_normals = A*mesh.normal;
# mesh.vertex_normals = mesh.vertex_normals./repmat(sqrt(sum(mesh.vertex_normals.^2,2)), [1,3]);
#
# mesh.sq_edge_length = sum((mesh.vertices(mesh.edge_to_vertex(:,1),:) - mesh.vertices(mesh.edge_to_vertex(:,2),:)).^2, 2);
#
# mesh.corner_angle = angles_of_triangles(mesh.vertices, mesh.triangles);
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
from .signed_edge_array import SignedEdgeArray


@dataclass
class MeshInfo:
    """
    Mesh data structure containing geometry and connectivity.

    Attributes
    ----------
    vertices : ndarray (num_vertices, 3)
        Vertex positions.
    triangles : ndarray (num_faces, 3)
        Triangle vertex indices (0-indexed).
    num_faces : int
        Number of faces (triangles).
    num_vertices : int
        Number of vertices.
    num_edges : int
        Number of edges.
    edge_to_vertex : ndarray (ne, 2)
        Edge-to-vertex connectivity. edge_to_vertex[e] = [v0, v1].
    T2E : SignedEdgeArray (nf, 3)
        Triangle-to-edge connectivity with orientation signs.
        Use T2E.indices for 0-based edge indices, T2E.signs for orientations.
    edge_to_triangle : ndarray (ne, 4)
        Edge-to-triangle connectivity. edge_to_triangle[e] = [t0, t1, s0, s1] where
        t0, t1 are adjacent triangles (0 if boundary) and s0, s1 are signs.
    triangle_to_triangle : ndarray (nf, 3)
        Triangle-to-triangle adjacency.
    normal : ndarray (nf, 3)
        Per-face unit normals.
    area : ndarray (nf,)
        Per-face areas.
    vertex_normals : ndarray (nv, 3)
        Per-vertex unit normals (area-weighted average of face normals).
    sq_edge_length : ndarray (ne,)
        Squared edge lengths.
    corner_angle : ndarray (nf, 3)
        Corner angles in radians. corner_angle[f, i] is the angle at vertex T[f, i].
    cot_corner_angle : ndarray (nf, 3)
        Cotangent of corner angles.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    num_faces: int
    num_vertices: int
    num_edges: int
    edge_to_vertex: np.ndarray
    T2E: SignedEdgeArray
    edge_to_triangle: np.ndarray
    triangle_to_triangle: np.ndarray
    normal: np.ndarray
    area: np.ndarray
    vertex_normals: np.ndarray
    sq_edge_length: np.ndarray
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

    # mesh.vertices = X;
    # mesh.triangles = T;
    # assert(size(T,2) == 3, 'Not a triangulations.');

    assert T.shape[1] == 3, "Not a triangulation."

    # [mesh.edge_to_vertex, mesh.T2E, mesh.edge_to_triangle, mesh.triangle_to_triangle] = connectivity(mesh.triangles);

    E2V, T2E, E2T, T2T = connectivity(T)

    # mesh.num_faces = size(mesh.triangles,1);
    # mesh.num_vertices = size(mesh.vertices,1);
    # mesh.num_edges = size(mesh.edge_to_vertex,1);

    nf = T.shape[0]
    nv = X.shape[0]
    ne = E2V.shape[0]

    # mesh.normal = cross(mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,2),:), mesh.vertices(mesh.triangles(:,1),:) - mesh.vertices(mesh.triangles(:,3),:));

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

    # A = sparse(mesh.triangles, repmat((1:mesh.num_faces)', [3,1]), repmat(mesh.area, [3,1]), mesh.num_vertices, mesh.num_faces);
    # mesh.vertex_normals = A*mesh.normal;

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

    # mesh.vertex_normals = mesh.vertex_normals./repmat(sqrt(sum(mesh.vertex_normals.^2,2)), [1,3]);

    # Normalize vertex normals
    Nv_norms = np.linalg.norm(Nv, axis=1, keepdims=True)
    Nv = Nv / np.maximum(Nv_norms, 1e-12)

    # mesh.sq_edge_length = sum((mesh.vertices(mesh.edge_to_vertex(:,1),:) - mesh.vertices(mesh.edge_to_vertex(:,2),:)).^2, 2);

    # MATLAB E2V(:,1) -> Python E2V[:, 0], E2V(:,2) -> E2V[:, 1]
    edge_vec = X[E2V[:, 0], :] - X[E2V[:, 1], :]
    SqEdgeLength = np.sum(edge_vec ** 2, axis=1)

    # mesh.corner_angle = angles_of_triangles(mesh.vertices, mesh.triangles);

    corner_angle = angles_of_triangles(X, T)

    # mesh.cot_corner_angle = cot(mesh.corner_angle);

    # MATLAB cot(x) = 1/tan(x)
    cot_corner_angle = 1.0 / np.tan(corner_angle)

    return MeshInfo(
        vertices=X,
        triangles=T,
        num_faces=nf,
        num_vertices=nv,
        num_edges=ne,
        edge_to_vertex=E2V,
        T2E=T2E,
        edge_to_triangle=E2T,
        triangle_to_triangle=T2T,
        normal=normal,
        area=area,
        vertex_normals=Nv,
        sq_edge_length=SqEdgeLength,
        corner_angle=corner_angle,
        cot_corner_angle=cot_corner_angle,
    )
