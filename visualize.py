"""
Visualization utilities for Corman-Crane rectangular parameterization.

Uses matplotlib for 2D plots and optional polyscope for 3D visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import Optional, Tuple

from mesh import TriangleMesh


def plot_mesh_2d(
    mesh: TriangleMesh,
    corner_uvs: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    show_edges: bool = True,
    show_faces: bool = True,
    edge_color: str = 'black',
    face_color: str = 'lightblue',
    alpha: float = 0.7,
    title: str = "Mesh"
) -> plt.Axes:
    """
    Plot mesh in 2D (using UV coordinates or XY projection).

    Args:
        mesh: Triangle mesh
        corner_uvs: |C| x 2 UV coordinates per corner (optional)
        ax: Matplotlib axes (creates new if None)
        show_edges: Draw triangle edges
        show_faces: Fill triangles
        edge_color: Color for edges
        face_color: Color for face fill
        alpha: Face transparency
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if corner_uvs is not None:
        # Use UV coordinates
        triangles = []
        for f in range(mesh.n_faces):
            c0 = 3 * f + 0
            c1 = 3 * f + 1
            c2 = 3 * f + 2
            tri = [corner_uvs[c0], corner_uvs[c1], corner_uvs[c2]]
            triangles.append(tri)
    else:
        # Use XY projection of 3D coordinates
        triangles = []
        for f in range(mesh.n_faces):
            v0, v1, v2 = mesh.faces[f]
            tri = [mesh.positions[v0, :2], mesh.positions[v1, :2], mesh.positions[v2, :2]]
            triangles.append(tri)

    if show_faces:
        collection = PolyCollection(triangles, facecolor=face_color,
                                     edgecolor=edge_color if show_edges else 'none',
                                     alpha=alpha, linewidth=0.5)
        ax.add_collection(collection)
    elif show_edges:
        collection = PolyCollection(triangles, facecolor='none',
                                     edgecolor=edge_color, linewidth=0.5)
        ax.add_collection(collection)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title)

    return ax


def plot_mesh_3d(
    mesh: TriangleMesh,
    ax: Optional[Axes3D] = None,
    show_edges: bool = True,
    show_faces: bool = True,
    edge_color: str = 'black',
    face_color: str = 'lightblue',
    alpha: float = 0.7,
    title: str = "Mesh"
) -> Axes3D:
    """
    Plot mesh in 3D.

    Args:
        mesh: Triangle mesh
        ax: 3D matplotlib axes (creates new if None)
        ... (same as plot_mesh_2d)

    Returns:
        3D matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Create list of triangle vertices
    triangles = []
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        tri = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]
        triangles.append(tri)

    if show_faces:
        collection = Poly3DCollection(triangles, facecolor=face_color,
                                       edgecolor=edge_color if show_edges else 'none',
                                       alpha=alpha, linewidth=0.3)
        ax.add_collection3d(collection)
    elif show_edges:
        collection = Poly3DCollection(triangles, facecolor='none',
                                       edgecolor=edge_color, linewidth=0.3)
        ax.add_collection3d(collection)

    # Set axis limits
    pos = mesh.positions
    max_range = np.max(pos.max(axis=0) - pos.min(axis=0)) / 2
    mid = (pos.max(axis=0) + pos.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_title(title)

    return ax


def plot_cross_field(
    mesh: TriangleMesh,
    W: np.ndarray,
    ax: Optional[Axes3D] = None,
    scale: float = 0.3,
    color: str = 'red',
    title: str = "Cross Field"
) -> Axes3D:
    """
    Plot cross field vectors on mesh.

    Args:
        mesh: Triangle mesh
        W: |F| x 3 cross field directions
        ax: 3D matplotlib axes
        scale: Length scale for vectors
        color: Vector color

    Returns:
        3D matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Draw mesh
    plot_mesh_3d(mesh, ax=ax, show_edges=True, show_faces=True, alpha=0.3)

    # Compute face centroids
    centroids = np.zeros((mesh.n_faces, 3))
    for f in range(mesh.n_faces):
        v0, v1, v2 = mesh.faces[f]
        centroids[f] = (mesh.positions[v0] + mesh.positions[v1] + mesh.positions[v2]) / 3

    # Get mean edge length for scaling
    from geometry import compute_edge_lengths
    mean_len = np.mean(compute_edge_lengths(mesh))
    vec_len = scale * mean_len

    # Draw cross field (all 4 directions)
    for f in range(mesh.n_faces):
        c = centroids[f]
        w = W[f]

        # 4-fold symmetry: draw all 4 directions
        from geometry import compute_face_normals
        N = compute_face_normals(mesh)
        n = N[f]

        # Perpendicular direction in tangent plane
        w_perp = np.cross(n, w)

        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            direction = np.cos(angle) * w + np.sin(angle) * w_perp
            end = c + vec_len * direction
            ax.plot([c[0], end[0]], [c[1], end[1]], [c[2], end[2]],
                   color=color, linewidth=0.5, alpha=0.7)

    ax.set_title(title)
    return ax


def plot_cut_graph(
    mesh: TriangleMesh,
    Gamma: np.ndarray,
    ax: Optional[Axes3D] = None,
    cut_color: str = 'red',
    cut_width: float = 3.0,
    title: str = "Cut Graph"
) -> Axes3D:
    """
    Plot cut graph on mesh.

    Args:
        mesh: Triangle mesh
        Gamma: |E| cut edge indicator
        ax: 3D matplotlib axes
        cut_color: Color for cut edges
        cut_width: Line width for cut edges

    Returns:
        3D matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Draw mesh
    plot_mesh_3d(mesh, ax=ax, show_edges=True, show_faces=True, alpha=0.3)

    # Draw cut edges
    cut_lines = []
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            i, j = mesh.edge_vertices[e]
            p1 = mesh.positions[i]
            p2 = mesh.positions[j]
            cut_lines.append([p1, p2])

    if cut_lines:
        collection = Line3DCollection(cut_lines, colors=cut_color, linewidths=cut_width)
        ax.add_collection3d(collection)

    ax.set_title(title)
    return ax


def plot_cones(
    mesh: TriangleMesh,
    cone_vertices: np.ndarray,
    ax: Optional[Axes3D] = None,
    cone_color: str = 'green',
    cone_size: float = 50,
    title: str = "Cone Singularities"
) -> Axes3D:
    """
    Plot cone singularities on mesh.

    Args:
        mesh: Triangle mesh
        cone_vertices: Array of vertex indices that are cones
        ax: 3D matplotlib axes
        cone_color: Color for cone markers
        cone_size: Marker size

    Returns:
        3D matplotlib axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Draw mesh
    plot_mesh_3d(mesh, ax=ax, show_edges=True, show_faces=True, alpha=0.3)

    # Draw cone vertices
    if len(cone_vertices) > 0:
        cone_pos = mesh.positions[cone_vertices]
        ax.scatter(cone_pos[:, 0], cone_pos[:, 1], cone_pos[:, 2],
                  c=cone_color, s=cone_size, marker='o', zorder=10)

    ax.set_title(f"{title} ({len(cone_vertices)} cones)")
    return ax


def plot_uv_checkerboard(
    mesh: TriangleMesh,
    corner_uvs: np.ndarray,
    ax: Optional[plt.Axes] = None,
    checker_scale: float = 1.0,
    title: str = "UV Checkerboard"
) -> plt.Axes:
    """
    Plot UV parameterization with checkerboard pattern.

    Args:
        mesh: Triangle mesh
        corner_uvs: |C| x 2 UV coordinates per corner
        ax: Matplotlib axes
        checker_scale: Scale of checkerboard pattern

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    triangles = []
    colors = []

    for f in range(mesh.n_faces):
        c0 = 3 * f + 0
        c1 = 3 * f + 1
        c2 = 3 * f + 2
        tri = [corner_uvs[c0], corner_uvs[c1], corner_uvs[c2]]
        triangles.append(tri)

        # Checkerboard color based on centroid
        centroid = (corner_uvs[c0] + corner_uvs[c1] + corner_uvs[c2]) / 3
        checker = (int(centroid[0] / checker_scale) + int(centroid[1] / checker_scale)) % 2
        colors.append('white' if checker == 0 else 'gray')

    collection = PolyCollection(triangles, facecolors=colors,
                                 edgecolors='black', linewidth=0.3)
    ax.add_collection(collection)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title)

    return ax


def visualize_phase1(mesh: TriangleMesh, save_path: Optional[str] = None):
    """Visualize Phase 1: mesh structure."""
    from mesh import euler_characteristic, genus

    fig = plt.figure(figsize=(12, 5))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    plot_mesh_3d(mesh, ax=ax1, title=f"Mesh (chi={euler_characteristic(mesh)}, g={genus(mesh)})")

    # 2D projection
    ax2 = fig.add_subplot(122)
    plot_mesh_2d(mesh, ax=ax2, title="XY Projection")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def visualize_phase2(mesh: TriangleMesh, W: np.ndarray, save_path: Optional[str] = None):
    """Visualize Phase 2: cross field."""
    from cross_field import compute_smoothness_energy

    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(111, projection='3d')
    plot_cross_field(mesh, W, ax=ax,
                     title=f"Cross Field (energy={compute_smoothness_energy(mesh, W):.2f})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def visualize_phase3(
    mesh: TriangleMesh,
    Gamma: np.ndarray,
    alpha: np.ndarray,
    omega0: np.ndarray,
    save_path: Optional[str] = None
):
    """Visualize Phase 3: cut graph and cones."""
    from cut_graph import get_cone_vertices, count_cut_edges

    cone_vertices = get_cone_vertices(mesh, alpha, omega0)

    fig = plt.figure(figsize=(12, 5))

    # Cut graph
    ax1 = fig.add_subplot(121, projection='3d')
    plot_cut_graph(mesh, Gamma, ax=ax1, title=f"Cut Graph ({count_cut_edges(Gamma)} edges)")

    # Cones
    ax2 = fig.add_subplot(122, projection='3d')
    plot_cones(mesh, cone_vertices, ax=ax2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def visualize_uv(
    mesh: TriangleMesh,
    corner_uvs: np.ndarray,
    save_path: Optional[str] = None
):
    """Visualize UV parameterization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plain UV layout
    plot_mesh_2d(mesh, corner_uvs, ax=axes[0], title="UV Layout")

    # Checkerboard
    plot_uv_checkerboard(mesh, corner_uvs, ax=axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()
