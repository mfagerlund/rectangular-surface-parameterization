# function [A] = angles_of_triangles(V, T)
#     % Computes for each triangle the 3 angles among its edges.
#     % Input:
#     %   option 1:   [A] = angles_of_triangles(V, T)
#     %
#     %               V   - (num_of_vertices x 3) 3D coordinates of
#     %                     the mesh vertices.
#     %               T   - (num_of_triangles x 3) T[i] are the 3 indices
#     %                     corresponding to the 3 vertices of the i-th
#     %                     triangle. The indexing is based on -V-.
#     %
#     %   option 2:   [A] = angles_of_triangles(L)
#     %
#     %               L - (num_of_triangles x 3) L[i] is a triple
#     %                   containing the lengths of the 3 edges
#     %                   corresponding to the i-th triange.
#     %
#     % Output:
#     %

import numpy as np


def angles_of_triangles(V, T):
    """
    Computes for each triangle the 3 corner angles.

    Parameters
    ----------
    V : ndarray (nv, 3)
        3D coordinates of mesh vertices.
    T : ndarray (nf, 3)
        Triangle vertex indices (0-indexed).

    Returns
    -------
    A : ndarray (nf, 3)
        Corner angles in radians. A[i, j] is the angle at vertex T[i, j].
    """

    # 	E1 = V(T(:,2),:) - V(T(:,1),:);
    #     E2 = V(T(:,3),:) - V(T(:,2),:);
    #     E3 = V(T(:,1),:) - V(T(:,3),:);

    # Edge vectors (MATLAB 1-indexed, Python 0-indexed)
    E1 = V[T[:, 1], :] - V[T[:, 0], :]  # edge from vertex 0 to vertex 1
    E2 = V[T[:, 2], :] - V[T[:, 1], :]  # edge from vertex 1 to vertex 2
    E3 = V[T[:, 0], :] - V[T[:, 2], :]  # edge from vertex 2 to vertex 0

    #     E1 = E1./sqrt(sum(E1.^2,2));
    #     E2 = E2./sqrt(sum(E2.^2,2));
    #     E3 = E3./sqrt(sum(E3.^2,2));

    # Normalize edge vectors
    E1 = E1 / np.linalg.norm(E1, axis=1, keepdims=True)
    E2 = E2 / np.linalg.norm(E2, axis=1, keepdims=True)
    E3 = E3 / np.linalg.norm(E3, axis=1, keepdims=True)

    #     A = pi - acos([dot(E3, E1, 2), dot(E1, E2, 2), dot(E2, E3, 2)]);

    # Compute angles using dot products
    # dot(E3, E1, 2) is row-wise dot product in MATLAB
    # Angle at vertex 0: between edges E3 (incoming) and E1 (outgoing)
    # Angle at vertex 1: between edges E1 (incoming) and E2 (outgoing)
    # Angle at vertex 2: between edges E2 (incoming) and E3 (outgoing)
    dot_E3_E1 = np.sum(E3 * E1, axis=1)
    dot_E1_E2 = np.sum(E1 * E2, axis=1)
    dot_E2_E3 = np.sum(E2 * E3, axis=1)

    # Clamp to [-1, 1] to avoid numerical issues with acos
    dot_E3_E1 = np.clip(dot_E3_E1, -1.0, 1.0)
    dot_E1_E2 = np.clip(dot_E1_E2, -1.0, 1.0)
    dot_E2_E3 = np.clip(dot_E2_E3, -1.0, 1.0)

    # A = pi - acos(...) because dot products are between consecutive edges
    # (angle between them is the exterior angle, we want interior)
    A = np.column_stack([
        np.pi - np.arccos(dot_E3_E1),
        np.pi - np.arccos(dot_E1_E2),
        np.pi - np.arccos(dot_E2_E3)
    ])

    #     assert(all(~isnan(A(:))) && all(isreal(A(:))), 'Triangle of size zero.');
    # end

    assert not np.any(np.isnan(A)), "Triangle of size zero."

    return A
