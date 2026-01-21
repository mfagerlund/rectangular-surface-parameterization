"""
Geometric quantities for triangle meshes.

Implements computations from Algorithm 1 of the supplement:
- Edge lengths: ell[ij] = |x_i - x_j|
- Corner angles: alpha[jk_i] = angle at vertex i in face ijk
- Triangle areas: A[ijk]
- Face normals: N[ijk]
- Cotan weights: w[ij] = (1/2) * (cot(alpha_ij^k) + cot(alpha_ij^l))
"""

import numpy as np
from typing import Tuple
from mesh import TriangleMesh


def compute_edge_lengths(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute edge lengths.

    Returns:
        edge_lengths: |E| array of edge lengths
    """
    edge_lengths = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        i, j = mesh.edge_vertices[e]
        edge_lengths[e] = np.linalg.norm(mesh.positions[j] - mesh.positions[i])

    return edge_lengths


def compute_halfedge_lengths(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute lengths indexed by halfedge.

    Returns:
        halfedge_lengths: |H| array (ell[he] = length of edge containing he)
    """
    edge_lengths = compute_edge_lengths(mesh)
    return edge_lengths[mesh.halfedge_to_edge]


def compute_corner_angles(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute corner angles alpha[jk_i] at each corner.

    Using the formula: alpha = acos((e1 . e2) / (|e1| |e2|))

    Returns:
        alpha: |C| array of corner angles (in radians)
    """
    alpha = np.zeros(mesh.n_corners, dtype=np.float64)

    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        p0 = mesh.positions[v0]
        p1 = mesh.positions[v1]
        p2 = mesh.positions[v2]

        # Corner 0: angle at v0 (between edges v0->v1 and v0->v2)
        e01 = p1 - p0
        e02 = p2 - p0
        cos_angle = np.clip(
            np.dot(e01, e02) / (np.linalg.norm(e01) * np.linalg.norm(e02) + 1e-30),
            -1.0, 1.0
        )
        alpha[3 * f + 0] = np.arccos(cos_angle)

        # Corner 1: angle at v1 (between edges v1->v2 and v1->v0)
        e12 = p2 - p1
        e10 = p0 - p1
        cos_angle = np.clip(
            np.dot(e12, e10) / (np.linalg.norm(e12) * np.linalg.norm(e10) + 1e-30),
            -1.0, 1.0
        )
        alpha[3 * f + 1] = np.arccos(cos_angle)

        # Corner 2: angle at v2 (between edges v2->v0 and v2->v1)
        e20 = p0 - p2
        e21 = p1 - p2
        cos_angle = np.clip(
            np.dot(e20, e21) / (np.linalg.norm(e20) * np.linalg.norm(e21) + 1e-30),
            -1.0, 1.0
        )
        alpha[3 * f + 2] = np.arccos(cos_angle)

    return alpha


def compute_face_areas(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute triangle areas.

    Returns:
        areas: |F| array of face areas
    """
    areas = np.zeros(mesh.n_faces, dtype=np.float64)

    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        p0 = mesh.positions[v0]
        p1 = mesh.positions[v1]
        p2 = mesh.positions[v2]

        # Area = |cross(p1-p0, p2-p0)| / 2
        cross = np.cross(p1 - p0, p2 - p0)
        areas[f] = np.linalg.norm(cross) / 2.0

    return areas


def compute_face_normals(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute unit face normals.

    Returns:
        normals: |F| x 3 array of unit normals
    """
    normals = np.zeros((mesh.n_faces, 3), dtype=np.float64)

    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        p0 = mesh.positions[v0]
        p1 = mesh.positions[v1]
        p2 = mesh.positions[v2]

        cross = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(cross)
        if norm > 1e-30:
            normals[f] = cross / norm
        else:
            normals[f] = [0, 0, 1]  # degenerate face

    return normals


def compute_vertex_normals(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute area-weighted vertex normals.

    Returns:
        normals: |V| x 3 array of unit normals
    """
    face_normals = compute_face_normals(mesh)
    areas = compute_face_areas(mesh)

    normals = np.zeros((mesh.n_vertices, 3), dtype=np.float64)

    for f in range(mesh.n_faces):
        for local in range(3):
            v = mesh.faces[f, local]
            normals[v] += areas[f] * face_normals[f]

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    normals = normals / norms

    return normals


def compute_cotan_weights(mesh: TriangleMesh, alpha: np.ndarray = None) -> np.ndarray:
    """
    Compute cotan weights for edges.

    w[ij] = (1/2) * (cot(alpha_ij^k) + cot(alpha_ij^l))

    where alpha_ij^k is the angle opposite to edge ij in face ijk.

    Returns:
        weights: |E| array of cotan weights
    """
    if alpha is None:
        alpha = compute_corner_angles(mesh)

    weights = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        # The angle opposite to edge e in face containing he0
        if he0 != -1:
            # he0 is the halfedge ij, the opposite corner is k (local index (local+2)%3)
            face = he0 // 3
            local = he0 % 3
            opposite_corner = 3 * face + (local + 2) % 3
            weights[e] += 0.5 / np.tan(alpha[opposite_corner] + 1e-30)

        # The angle opposite to edge e in face containing he1
        if he1 != -1:
            face = he1 // 3
            local = he1 % 3
            opposite_corner = 3 * face + (local + 2) % 3
            weights[e] += 0.5 / np.tan(alpha[opposite_corner] + 1e-30)

    return weights


def compute_halfedge_cotan_weights(mesh: TriangleMesh, alpha: np.ndarray = None) -> np.ndarray:
    """
    Compute cotan weights per halfedge (just the single opposite angle).

    w[he] = (1/2) * cot(alpha_opposite)

    Returns:
        weights: |H| array
    """
    if alpha is None:
        alpha = compute_corner_angles(mesh)

    weights = np.zeros(mesh.n_halfedges, dtype=np.float64)

    for he in range(mesh.n_halfedges):
        face = he // 3
        local = he % 3
        opposite_corner = 3 * face + (local + 2) % 3
        weights[he] = 0.5 / np.tan(alpha[opposite_corner] + 1e-30)

    return weights


def compute_face_basis(mesh: TriangleMesh, face: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute orthonormal basis (N, T1, T2) for a face, adapted to edge ij.

    From Algorithm 1 lines 4-8:
    - T1 = Unit(x_j - x_i)
    - N = cross product of edges, normalized
    - T2 = N x T1

    Returns:
        N: normal vector
        T1: tangent along first edge
        T2: second tangent (N x T1)
    """
    v0, v1, v2 = mesh.faces[face]
    p0 = mesh.positions[v0]
    p1 = mesh.positions[v1]
    p2 = mesh.positions[v2]

    # Normal
    e01 = p1 - p0
    e02 = p2 - p0
    N = np.cross(e01, e02)
    area_2 = np.linalg.norm(N)
    if area_2 > 1e-30:
        N = N / area_2
    else:
        N = np.array([0.0, 0.0, 1.0])

    # T1 along first edge
    T1 = e01 / (np.linalg.norm(e01) + 1e-30)

    # T2 perpendicular to N and T1
    T2 = np.cross(N, T1)

    return N, T1, T2


def compute_all_face_bases(mesh: TriangleMesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute orthonormal bases for all faces.

    Returns:
        N: |F| x 3 normals
        T1: |F| x 3 first tangents (along edge 0)
        T2: |F| x 3 second tangents
    """
    N = np.zeros((mesh.n_faces, 3), dtype=np.float64)
    T1 = np.zeros((mesh.n_faces, 3), dtype=np.float64)
    T2 = np.zeros((mesh.n_faces, 3), dtype=np.float64)

    for f in range(mesh.n_faces):
        N[f], T1[f], T2[f] = compute_face_basis(mesh, f)

    return N, T1, T2


def compute_cross_field_angles(mesh: TriangleMesh, W: np.ndarray) -> np.ndarray:
    """
    Compute cross field angles xi relative to first edge of each face.

    From Algorithm 1 line 9:
    xi[ijk] = atan2(<T2, W>, <T1, W>)

    Args:
        W: |F| x 3 array of one representative cross field direction per face

    Returns:
        xi: |F| array of angles (in radians)
    """
    xi = np.zeros(mesh.n_faces, dtype=np.float64)
    N, T1, T2 = compute_all_face_bases(mesh)

    for f in range(mesh.n_faces):
        w = W[f]
        # Project W onto the tangent plane and get angle
        xi[f] = np.arctan2(np.dot(T2[f], w), np.dot(T1[f], w))

    return xi


def angle_defect(mesh: TriangleMesh, alpha: np.ndarray = None) -> np.ndarray:
    """
    Compute angle defect (discrete Gaussian curvature) at each vertex.

    K[i] = 2*pi - sum of incident corner angles

    For boundary vertices, it's pi - sum of angles.

    Returns:
        K: |V| array of angle defects
    """
    if alpha is None:
        alpha = compute_corner_angles(mesh)

    K = np.full(mesh.n_vertices, 2 * np.pi, dtype=np.float64)

    # Subtract corner angles
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] -= alpha[c]

    # Adjust for boundary vertices
    for v in range(mesh.n_vertices):
        if mesh.is_boundary_vertex(v):
            # Boundary: K = pi - sum of angles
            K[v] = K[v] - np.pi

    return K


def total_gaussian_curvature(mesh: TriangleMesh, alpha: np.ndarray = None) -> float:
    """
    Compute total Gaussian curvature (should equal 2*pi*chi by Gauss-Bonnet).

    Returns:
        Total Gaussian curvature
    """
    K = angle_defect(mesh, alpha)
    return np.sum(K)


class MeshGeometry:
    """Container for precomputed geometric quantities."""

    def __init__(self, mesh: TriangleMesh):
        self.mesh = mesh

        # Compute all quantities
        self.edge_lengths = compute_edge_lengths(mesh)
        self.halfedge_lengths = self.edge_lengths[mesh.halfedge_to_edge]
        self.alpha = compute_corner_angles(mesh)
        self.areas = compute_face_areas(mesh)
        self.face_normals = compute_face_normals(mesh)
        self.cotan_weights = compute_cotan_weights(mesh, self.alpha)
        self.halfedge_cotan = compute_halfedge_cotan_weights(mesh, self.alpha)

        # Face bases
        self.N, self.T1, self.T2 = compute_all_face_bases(mesh)

    def total_area(self) -> float:
        return np.sum(self.areas)

    def mean_edge_length(self) -> float:
        return np.mean(self.edge_lengths)
