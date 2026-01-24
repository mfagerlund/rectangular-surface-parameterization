

# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union


# function fig = plot_frame_field(fig, Src, param, ang, col)

def plot_frame_field(
    fig: Optional[plt.Figure],
    Src,
    param,
    ang: np.ndarray,
    col: Union[np.ndarray, str] = 'gray'
) -> plt.Figure:
    """
    Plot the mesh with frame field directions.

    Visualizes the mesh surface with per-face colors and overlays the cross-field
    directions as red/green arrows at face barycenters.

    Args:
        fig: Existing figure to plot on, or None to create new figure
        Src: Mesh data structure with T (triangles), X (vertices), nv (vertex count)
        param: Parameter structure with e1r, e2r (per-face reference frames in 3D)
        ang: Frame field angles per face (radians)
        col: Face colors - array of shape (nf,) or (nv,) or color string

    Returns:
        fig: The matplotlib figure
    """
    # if isempty(fig)
    #     fig = figure;
    # end

    if fig is None:
        fig = plt.figure()

    # figure(fig);

    ax = fig.add_subplot(111, projection='3d')

    # trisurf(Src.T, Src.X(:,1), Src.X(:,2), Src.X(:,3), col, 'edgecolor', 'none');

    # Extract vertex coordinates
    X = Src.X
    T = Src.T  # Face indices (0-indexed)

    # Determine color handling
    if isinstance(col, str):
        # Simple color string
        ax.plot_trisurf(
            X[:, 0], X[:, 1], X[:, 2],
            triangles=T,
            color=col,
            edgecolor='none',
            alpha=0.8
        )
    elif hasattr(col, '__len__'):
        col_arr = np.asarray(col)
        if col_arr.ndim == 1 and len(col_arr) == T.shape[0]:
            # Per-face colors
            ax.plot_trisurf(
                X[:, 0], X[:, 1], X[:, 2],
                triangles=T,
                cmap='viridis',
                edgecolor='none',
                alpha=0.8
            )
            # Note: matplotlib's plot_trisurf doesn't directly support per-face scalar colors
            # For proper face coloring, use Poly3DCollection
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax.clear()
            verts = X[T]  # Shape: (nf, 3, 3)
            # Normalize colors
            norm = plt.Normalize(col_arr.min(), col_arr.max())
            cmap = plt.cm.viridis
            facecolors = cmap(norm(col_arr))
            poly = Poly3DCollection(verts, facecolors=facecolors, edgecolor='none', alpha=0.8)
            ax.add_collection3d(poly)
            # Set axis limits
            ax.set_xlim(X[:, 0].min(), X[:, 0].max())
            ax.set_ylim(X[:, 1].min(), X[:, 1].max())
            ax.set_zlim(X[:, 2].min(), X[:, 2].max())
        elif col_arr.ndim == 1 and len(col_arr) == Src.nv:
            # Per-vertex colors - use shading interpolation
            ax.plot_trisurf(
                X[:, 0], X[:, 1], X[:, 2],
                triangles=T,
                cmap='viridis',
                edgecolor='none',
                alpha=0.8,
                shade=True
            )
        else:
            # Default gray
            ax.plot_trisurf(
                X[:, 0], X[:, 1], X[:, 2],
                triangles=T,
                color='gray',
                edgecolor='none',
                alpha=0.8
            )

    # if size(ang,1) == size(Src.T,1)
    #     e1 = exp(1i*ang);
    #     e2 = 1i*e1;
    #     E1 = real(e1).*param.e1r + imag(e1).*param.e2r;
    #     E2 = real(e2).*param.e1r + imag(e2).*param.e2r;
    #
    #     bar = (Src.X(Src.T(:,1),:) + Src.X(Src.T(:,2),:) + Src.X(Src.T(:,3),:))/3;
    #
    #     hold on;
    #     quiver3(bar(:,1), bar(:,2), bar(:,3), E1(:,1), E1(:,2), E1(:,3), 0.5, 'r', 'LineWidth',1);
    #     quiver3(bar(:,1), bar(:,2), bar(:,3), E2(:,1), E2(:,2), E2(:,3), 0.5, 'g', 'LineWidth',1);
    #     hold off;
    # end

    if len(ang) == T.shape[0]:
        # Compute frame directions from angles
        # e1 = exp(1i*ang) = cos(ang) + i*sin(ang)
        e1 = np.exp(1j * ang)
        # e2 = 1i * e1 = i*cos(ang) - sin(ang) = -sin(ang) + i*cos(ang)
        e2 = 1j * e1

        # Map from local 2D to 3D using reference frame
        # E1 = real(e1) * e1r + imag(e1) * e2r
        E1 = np.real(e1)[:, np.newaxis] * param.e1r + np.imag(e1)[:, np.newaxis] * param.e2r
        E2 = np.real(e2)[:, np.newaxis] * param.e1r + np.imag(e2)[:, np.newaxis] * param.e2r

        # Compute face barycenters
        # bar = (X[T[:,0]] + X[T[:,1]] + X[T[:,2]]) / 3
        bar = (X[T[:, 0], :] + X[T[:, 1], :] + X[T[:, 2], :]) / 3

        # Plot arrows for first frame direction (red)
        ax.quiver(
            bar[:, 0], bar[:, 1], bar[:, 2],
            E1[:, 0], E1[:, 1], E1[:, 2],
            length=0.5, normalize=True, color='r', linewidth=1
        )

        # Plot arrows for second frame direction (green)
        ax.quiver(
            bar[:, 0], bar[:, 1], bar[:, 2],
            E2[:, 0], E2[:, 1], E2[:, 2],
            length=0.5, normalize=True, color='g', linewidth=1
        )

    # axis equal; view(0,90); % caxis([-0.5 0.5])

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.view_init(elev=90, azim=0)  # Top-down view

    # if size(col,1) == Src.nv
    #     shading interp;
    # end

    # Note: matplotlib shading is handled by shade=True or shading='gouraud'
    # This is already set above for per-vertex colors

    return fig
