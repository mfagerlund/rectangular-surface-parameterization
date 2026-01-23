"""
UV Visualization utilities for MATLAB-ported Corman-Crane implementation.

Provides functions to visualize UV parameterizations, highlight flipped triangles,
and save results to PNG files.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Tuple
from pathlib import Path


def plot_uv_with_flips(
    Xp: np.ndarray,
    T: np.ndarray,
    detJ: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "UV Layout",
    flip_color: str = 'red',
    normal_color: str = 'lightblue',
    edge_color: str = 'black',
    alpha: float = 0.7
) -> Tuple[plt.Axes, int]:
    """
    Plot UV layout with flipped triangles highlighted in red.

    Args:
        Xp: UV coordinates (nv, 2)
        T: Face indices (nf, 3), 0-indexed
        detJ: Jacobian determinant per face (nf,). Computed if None.
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        flip_color: Color for flipped triangles
        normal_color: Color for normal triangles
        edge_color: Edge color
        alpha: Face transparency

    Returns:
        ax: Matplotlib axes
        n_flipped: Number of flipped triangles
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    nf = T.shape[0]

    # Compute detJ if not provided
    if detJ is None:
        detJ = np.zeros(nf)
        for f in range(nf):
            uv0, uv1, uv2 = Xp[T[f, 0]], Xp[T[f, 1]], Xp[T[f, 2]]
            e1 = uv1 - uv0
            e2 = uv2 - uv0
            detJ[f] = e1[0] * e2[1] - e1[1] * e2[0]  # Cross product z-component

    # Build triangles
    triangles = []
    colors = []
    for f in range(nf):
        tri = [Xp[T[f, 0]], Xp[T[f, 1]], Xp[T[f, 2]]]
        triangles.append(tri)
        colors.append(flip_color if detJ[f] <= 0 else normal_color)

    collection = PolyCollection(
        triangles,
        facecolors=colors,
        edgecolors=edge_color,
        linewidth=0.3,
        alpha=alpha
    )
    ax.add_collection(collection)

    ax.autoscale()
    ax.set_aspect('equal')

    n_flipped = np.sum(detJ <= 0)
    ax.set_title(f"{title}\nFlipped: {n_flipped}/{nf} ({100*n_flipped/nf:.1f}%)")

    return ax, n_flipped


def plot_uv_checkerboard(
    Xp: np.ndarray,
    T: np.ndarray,
    detJ: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    checker_scale: float = 0.1,
    title: str = "UV Checkerboard",
    flip_color: str = 'red'
) -> plt.Axes:
    """
    Plot UV with checkerboard pattern, flipped triangles in red.

    Args:
        Xp: UV coordinates (nv, 2)
        T: Face indices (nf, 3), 0-indexed
        detJ: Jacobian determinant per face (nf,). Computed if None.
        ax: Matplotlib axes
        checker_scale: Checkerboard cell size
        title: Plot title
        flip_color: Color for flipped triangles

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    nf = T.shape[0]

    # Compute detJ if not provided
    if detJ is None:
        detJ = np.zeros(nf)
        for f in range(nf):
            uv0, uv1, uv2 = Xp[T[f, 0]], Xp[T[f, 1]], Xp[T[f, 2]]
            e1 = uv1 - uv0
            e2 = uv2 - uv0
            detJ[f] = e1[0] * e2[1] - e1[1] * e2[0]

    triangles = []
    colors = []

    for f in range(nf):
        tri = [Xp[T[f, 0]], Xp[T[f, 1]], Xp[T[f, 2]]]
        triangles.append(tri)

        if detJ[f] <= 0:
            colors.append(flip_color)
        else:
            # Checkerboard based on centroid
            centroid = (Xp[T[f, 0]] + Xp[T[f, 1]] + Xp[T[f, 2]]) / 3
            checker = (int(centroid[0] / checker_scale) + int(centroid[1] / checker_scale)) % 2
            colors.append('white' if checker == 0 else 'lightgray')

    collection = PolyCollection(
        triangles,
        facecolors=colors,
        edgecolors='black',
        linewidth=0.2
    )
    ax.add_collection(collection)

    ax.autoscale()
    ax.set_aspect('equal')

    n_flipped = np.sum(detJ <= 0)
    ax.set_title(f"{title}\nFlipped: {n_flipped}/{nf}")

    return ax


def plot_mesh_with_flips(
    X: np.ndarray,
    T: np.ndarray,
    detJ: np.ndarray,
    ax=None,
    title: str = "Mesh with Flipped Faces",
    flip_color: str = 'red',
    normal_color: str = 'lightblue'
):
    """
    Plot 3D mesh with flipped triangles highlighted.

    Args:
        X: Vertex positions (nv, 3)
        T: Face indices (nf, 3)
        detJ: Jacobian determinant per face (nf,)
        ax: Matplotlib 3D axes
        title: Plot title
        flip_color: Color for flipped triangles
        normal_color: Color for normal triangles

    Returns:
        3D matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    nf = T.shape[0]

    # Build face colors
    facecolors = [flip_color if detJ[f] <= 0 else normal_color for f in range(nf)]

    # Build vertex array for faces
    verts = X[T]  # Shape: (nf, 3, 3)

    poly = Poly3DCollection(
        verts,
        facecolors=facecolors,
        edgecolor='black',
        linewidth=0.2,
        alpha=0.8
    )
    ax.add_collection3d(poly)

    # Set axis limits
    ax.set_xlim(X[:, 0].min(), X[:, 0].max())
    ax.set_ylim(X[:, 1].min(), X[:, 1].max())
    ax.set_zlim(X[:, 2].min(), X[:, 2].max())

    n_flipped = np.sum(detJ <= 0)
    ax.set_title(f"{title}\nFlipped: {n_flipped}/{nf}")

    return ax


def save_uv_visualization(
    Xp: np.ndarray,
    T: np.ndarray,
    detJ: Optional[np.ndarray],
    output_path: str,
    dpi: int = 150
) -> int:
    """
    Save UV visualization with flipped faces to PNG.

    Creates a 2-panel figure: plain UV layout and checkerboard.
    Returns number of flipped triangles.

    Args:
        Xp: UV coordinates (nv, 2)
        T: Face indices (nf, 3)
        detJ: Jacobian determinant per face (nf,). Computed if None.
        output_path: Output PNG path
        dpi: Resolution

    Returns:
        n_flipped: Number of flipped triangles
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1, n_flipped = plot_uv_with_flips(Xp, T, detJ, ax=axes[0], title="UV Layout")
    plot_uv_checkerboard(Xp, T, detJ, ax=axes[1])

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  Flipped triangles: {n_flipped}/{T.shape[0]} ({100*n_flipped/T.shape[0]:.1f}%)")

    return n_flipped


def compute_uv_quality(
    Xp: np.ndarray,
    T: np.ndarray,
    X: np.ndarray = None,
    T_orig: np.ndarray = None
) -> dict:
    """
    Compute quality metrics for UV parameterization.

    Args:
        Xp: UV coordinates (nv, 2)
        T: Face indices (nf, 3)
        X: Original 3D vertex positions (optional, for angle error)
        T_orig: Original face indices (optional, same as T if not cut)

    Returns:
        dict with:
            flipped_count: Number of flipped triangles
            flipped_fraction: Fraction of flipped triangles
            total_area_uv: Total UV area
            angle_error_mean: Mean angle error in radians (if X provided)
            angle_error_max: Max angle error in radians (if X provided)
    """
    nf = T.shape[0]

    # Compute detJ and areas
    detJ = np.zeros(nf)
    areas_uv = np.zeros(nf)

    for f in range(nf):
        uv0, uv1, uv2 = Xp[T[f, 0]], Xp[T[f, 1]], Xp[T[f, 2]]
        e1 = uv1 - uv0
        e2 = uv2 - uv0
        detJ[f] = e1[0] * e2[1] - e1[1] * e2[0]
        areas_uv[f] = abs(detJ[f]) / 2

    flipped = np.sum(detJ <= 0)

    result = {
        'flipped_count': int(flipped),
        'flipped_fraction': flipped / nf if nf > 0 else 0,
        'total_area_uv': np.sum(areas_uv),
        'detJ': detJ
    }

    # Compute angle error if 3D mesh provided
    if X is not None:
        if T_orig is None:
            T_orig = T

        angle_errors = []
        for f in range(nf):
            for local in range(3):
                # UV angle
                ca, cb, cc = T[f, local], T[f, (local+1)%3], T[f, (local+2)%3]
                uv_a, uv_b, uv_c = Xp[ca], Xp[cb], Xp[cc]
                e1 = uv_b - uv_a
                e2 = uv_c - uv_a
                len1, len2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if len1 > 1e-10 and len2 > 1e-10:
                    cos_uv = np.clip(np.dot(e1, e2) / (len1 * len2), -1, 1)
                    angle_uv = np.arccos(cos_uv)
                else:
                    angle_uv = 0

                # 3D angle
                va, vb, vc = T_orig[f, local], T_orig[f, (local+1)%3], T_orig[f, (local+2)%3]
                e1_3d = X[vb] - X[va]
                e2_3d = X[vc] - X[va]
                len1_3d, len2_3d = np.linalg.norm(e1_3d), np.linalg.norm(e2_3d)
                if len1_3d > 1e-10 and len2_3d > 1e-10:
                    cos_3d = np.clip(np.dot(e1_3d, e2_3d) / (len1_3d * len2_3d), -1, 1)
                    angle_3d = np.arccos(cos_3d)
                else:
                    angle_3d = 0

                angle_errors.append(abs(angle_uv - angle_3d))

        result['angle_error_mean'] = np.mean(angle_errors) if angle_errors else 0
        result['angle_error_max'] = np.max(angle_errors) if angle_errors else 0

    return result


def visualize_run_RSP_result(
    Src,
    SrcCut,
    Xp: np.ndarray,
    disto,
    output_dir: str = "output"
) -> int:
    """
    Comprehensive visualization of run_RSP.py output.

    Creates:
    - uv_layout.png: UV with flips highlighted
    - mesh_flips.png: 3D mesh with flipped faces
    - distortion.png: Distortion metrics

    Args:
        Src: Original MeshInfo
        SrcCut: Cut MeshInfo
        Xp: UV coordinates (nv_cut, 2)
        disto: DistortionMetrics from extract_scale_from_param
        output_dir: Output directory

    Returns:
        n_flipped: Number of flipped triangles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. UV layout with flips
    n_flipped = save_uv_visualization(
        Xp, SrcCut.T, disto.detJ,
        str(output_dir / "uv_layout.png")
    )

    # 2. 3D mesh with flipped faces highlighted
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh_with_flips(Src.X, Src.T, disto.detJ, ax=ax)
    plt.savefig(output_dir / "mesh_flips.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'mesh_flips.png'}")

    # 3. Distortion metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Area distortion
    ax = axes[0, 0]
    im = ax.tripcolor(Xp[:, 0], Xp[:, 1], SrcCut.T,
                      np.log10(disto.area + 1e-16), shading='flat', cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('log10(Area Distortion)')
    plt.colorbar(im, ax=ax)

    # Conformal distortion
    ax = axes[0, 1]
    im = ax.tripcolor(Xp[:, 0], Xp[:, 1], SrcCut.T,
                      np.abs(np.log10(disto.conf + 1e-16)), shading='flat', cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('|log10(Conformal)|')
    plt.colorbar(im, ax=ax)

    # Jacobian determinant
    ax = axes[1, 0]
    im = ax.tripcolor(Xp[:, 0], Xp[:, 1], SrcCut.T,
                      disto.detJ, shading='flat', cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax.set_aspect('equal')
    ax.set_title('Jacobian Determinant (negative = flipped)')
    plt.colorbar(im, ax=ax)

    # Orthogonality
    ax = axes[1, 1]
    im = ax.tripcolor(Xp[:, 0], Xp[:, 1], SrcCut.T,
                      np.degrees(disto.orth - np.pi/2), shading='flat', cmap='RdBu')
    ax.set_aspect('equal')
    ax.set_title('Orthogonality Error (degrees from 90)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "distortion.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'distortion.png'}")

    return n_flipped
