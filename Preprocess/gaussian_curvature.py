# === ISSUES ===
# - accumarray: use np.add.at (np.bincount doesn't work with floating point values)
# - unique(..., 'rows'): use np.unique with axis=0 and return_inverse=True
# === END ISSUES ===

import numpy as np
from .angles_of_triangles import angles_of_triangles


# function [K,idx_bound,ide_bound] = gaussian_curvature(X, T)
# % Output gaussian curvature inside and normal curvature at the boundary

def gaussian_curvature(X, T):
    """
    Compute Gaussian curvature at vertices using angle defect.

    Parameters
    ----------
    X : ndarray (nv, 3)
        Vertex positions
    T : ndarray (nf, 3)
        Triangle indices (0-indexed)

    Returns
    -------
    K : ndarray (nv,)
        Gaussian curvature at each vertex
    idx_bound : ndarray
        Indices of boundary vertices
    ide_bound : ndarray
        Indices of boundary edges in the unique edge list
    """
    nv = X.shape[0]

    # % Compute interior curvature
    # theta = angles_of_triangles(X, T);
    # K = 2*pi - accumarray(T(:), theta(:));

    # Compute interior curvature
    theta = angles_of_triangles(X, T)  # (nf, 3) angles at each corner
    K = np.full(nv, 2 * np.pi)
    # accumarray: sum angles at each vertex
    np.add.at(K, T.ravel(), -theta.ravel())

    # % K(idx_bound) = K(idx_bound) - pi;
    # (This is done below after finding boundary vertices)

    # % Compute boundary normal curvature
    # E2V = sort([T(:,1), T(:,2); T(:,2), T(:,3); T(:,3), T(:,1)], 2);

    # Compute boundary normal curvature
    # Build edge list: each triangle contributes 3 edges
    E2V = np.vstack([
        np.column_stack([T[:, 0], T[:, 1]]),
        np.column_stack([T[:, 1], T[:, 2]]),
        np.column_stack([T[:, 2], T[:, 0]])
    ])
    E2V = np.sort(E2V, axis=1)  # Sort each edge so smaller index comes first

    # [E2V_u,~,ic] = unique(E2V, 'rows');
    # n_adj = accumarray(ic, 1, [size(E2V_u,1),1]);

    E2V_u, ic = np.unique(E2V, axis=0, return_inverse=True)
    n_adj = np.bincount(ic, minlength=E2V_u.shape[0])

    # ide_bound = find(n_adj == 1);
    # E2V_bound = E2V_u(ide_bound,:);
    # idx_bound = unique(E2V_bound);

    ide_bound = np.where(n_adj == 1)[0]
    E2V_bound = E2V_u[ide_bound, :]
    idx_bound = np.unique(E2V_bound)

    # K(idx_bound) = K(idx_bound) - pi;

    K[idx_bound] = K[idx_bound] - np.pi

    return K, idx_bound, ide_bound
