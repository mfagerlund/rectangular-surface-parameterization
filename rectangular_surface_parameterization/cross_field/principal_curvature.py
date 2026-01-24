"""
Principal curvature computation for triangle meshes.

Computes principal curvatures (k1, k2), directions, and derived quantities
(Gaussian curvature, mean curvature) per face.

Extracted from curvature_field.py for standalone use in visualization.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class PrincipalCurvatures:
    """Principal curvature data per face."""
    k1: np.ndarray          # First principal curvature (nf,)
    k2: np.ndarray          # Second principal curvature (nf,)
    dir1: np.ndarray        # First principal direction as complex (nf,)
    dir2: np.ndarray        # Second principal direction as complex (nf,)
    gaussian: np.ndarray    # Gaussian curvature K = k1 * k2 (nf,)
    mean: np.ndarray        # Mean curvature H = (k1 + k2) / 2 (nf,)
    tensor: np.ndarray      # Curvature tensor [a11, a12, a22] (nf, 3)


def compute_principal_curvatures(mesh, param) -> PrincipalCurvatures:
    """
    Compute principal curvatures and directions for each face.

    This uses the discrete curvature tensor based on dihedral angles,
    projected into the local reference frame of each face.

    Args:
        mesh: Mesh data structure with vertices, triangles, edge_to_vertex,
              edge_to_triangle, T2E, normal, area, num_edges, num_faces
        param: Parameter structure with e1r, e2r (local reference frames)

    Returns:
        PrincipalCurvatures dataclass with k1, k2, directions, and derived quantities
    """

    def comp_angle(u, v, n):
        """Compute signed angle between vectors u and v with normal n."""
        cross_prod = np.cross(u, v)
        sin_angle = np.sum(cross_prod * n, axis=1)
        cos_angle = np.sum(u * v, axis=1)
        return np.arctan2(sin_angle, cos_angle)

    # Edge vectors
    edge = mesh.vertices[mesh.edge_to_vertex[:, 1], :] - mesh.vertices[mesh.edge_to_vertex[:, 0], :]
    edge_length = np.sqrt(np.sum(edge**2, axis=1))
    edge = edge / edge_length[:, np.newaxis]

    # Outer product of edge with itself, stored as 9-column matrix
    Eedge = np.column_stack([
        edge[:, 0] * edge[:, 0], edge[:, 0] * edge[:, 1], edge[:, 0] * edge[:, 2],
        edge[:, 1] * edge[:, 0], edge[:, 1] * edge[:, 1], edge[:, 1] * edge[:, 2],
        edge[:, 2] * edge[:, 0], edge[:, 2] * edge[:, 1], edge[:, 2] * edge[:, 2]
    ])

    # Interior edges: both adjacent triangles exist
    ide_int = np.all(mesh.edge_to_triangle[:, 0:2] >= 0, axis=1)
    dihedral_angle = np.zeros(mesh.num_edges)

    # For interior edges, compute dihedral angle
    int_idx = np.where(ide_int)[0]
    t1 = mesh.edge_to_triangle[int_idx, 0]
    t2 = mesh.edge_to_triangle[int_idx, 1]
    sign_e = mesh.edge_to_triangle[int_idx, 3]

    dihedral_angle[int_idx] = sign_e * comp_angle(
        mesh.normal[t1, :],
        mesh.normal[t2, :],
        edge[int_idx, :]
    )

    # Compute curvature tensor per face
    Curv = np.zeros((mesh.num_faces, 4))
    J = np.array([[0, -1], [1, 0]])
    K = np.zeros(mesh.num_faces)

    for i in range(mesh.num_faces):
        # Find triangles sharing vertices with triangle i
        face_verts = mesh.triangles[i, :]
        idt = np.where(np.any(np.isin(mesh.triangles, face_verts), axis=1))[0]

        # Get unique edge indices from these triangles
        t2e_vals = mesh.T2E[idt, :].ravel()
        ide = np.unique(np.abs(t2e_vals) - 1)

        # Local reference frame
        E = np.column_stack([param.e1r[i, :], param.e2r[i, :]])

        # Sum weighted outer products
        weighted_sum = np.sum(
            dihedral_angle[ide, np.newaxis] * edge_length[ide, np.newaxis] * Eedge[ide, :],
            axis=0
        ) / mesh.area[i]

        # Reshape to 3x3 matrix (column-major)
        A = weighted_sum.reshape((3, 3), order='F')
        A = (A + A.T) / 2

        # Project to local frame and rotate by J
        A = J.T @ E.T @ A @ E @ J
        A = (A + A.T) / 2

        Curv[i, :] = A.ravel(order='F')
        K[i] = np.linalg.det(A)

    # Keep only unique elements of symmetric matrix: [a11, a12, a22]
    Curv = Curv[:, [0, 1, 3]]

    # Compute principal curvatures and directions
    dir_min = np.zeros(mesh.num_faces, dtype=complex)
    kappa = np.zeros((mesh.num_faces, 2))

    for i in range(mesh.num_faces):
        # Reconstruct symmetric 2x2 matrix
        A = np.array([[Curv[i, 0], Curv[i, 1]],
                      [Curv[i, 1], Curv[i, 2]]])

        # Compute eigenvalues and eigenvectors
        D, V = np.linalg.eig(A)

        # Sort by eigenvalue
        idx_sort = np.argsort(D)
        D = D[idx_sort]
        V = V[:, idx_sort]

        # Principal direction as complex number
        dir_min[i] = complex(V[0, 0], V[1, 0])
        kappa[i, :] = D

    # Second principal direction is perpendicular
    dir_max = 1j * dir_min

    # Compute derived quantities
    k1 = kappa[:, 0]
    k2 = kappa[:, 1]
    gaussian = k1 * k2
    mean = (k1 + k2) / 2

    return PrincipalCurvatures(
        k1=k1,
        k2=k2,
        dir1=dir_min,
        dir2=dir_max,
        gaussian=gaussian,
        mean=mean,
        tensor=Curv
    )
