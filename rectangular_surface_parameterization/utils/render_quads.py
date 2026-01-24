"""
Quad mesh renderer that properly displays quads (not triangulated).

Usage:
    python -m rectangular_surface_parameterization.utils.render_quads mesh_quads.obj
    python -m rectangular_surface_parameterization.utils.render_quads mesh_quads.obj -o output.png
    python -m rectangular_surface_parameterization.utils.render_quads mesh_quads.obj --wireframe
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from pathlib import Path


def read_quad_obj(filepath):
    """Read OBJ file, returning vertices and faces (quads and tris separate)."""
    vertices = []
    quads = []
    tris = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()[1:4]
                vertices.append([float(p) for p in parts])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # Handle v/vt/vn format
                indices = [int(p.split('/')[0]) - 1 for p in parts]  # OBJ is 1-indexed
                if len(indices) == 4:
                    quads.append(indices)
                elif len(indices) == 3:
                    tris.append(indices)

    return np.array(vertices), quads, tris


def render_quad_mesh(vertices, quads, tris,
                     title=None,
                     wireframe=False,
                     quad_color='steelblue',
                     tri_color='coral',
                     edge_color='black',
                     figsize=(12, 10),
                     elev=30, azim=45,
                     output=None):
    """
    Render a quad/tri mesh with proper quad display.

    Parameters
    ----------
    vertices : ndarray (N, 3)
    quads : list of [i,j,k,l] indices
    tris : list of [i,j,k] indices
    wireframe : bool - show only edges
    quad_color : color for quad faces
    tri_color : color for tri faces (hole fills)
    output : str - save to file instead of display
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if wireframe:
        # Draw edges only
        quad_edges = set()
        for q in quads:
            for i in range(4):
                e = tuple(sorted([q[i], q[(i+1) % 4]]))
                quad_edges.add(e)

        tri_edges = set()
        for t in tris:
            for i in range(3):
                e = tuple(sorted([t[i], t[(i+1) % 3]]))
                if e not in quad_edges:
                    tri_edges.add(e)

        # Draw quad edges
        for e in quad_edges:
            pts = vertices[list(e)]
            ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=quad_color, linewidth=0.5)

        # Draw tri edges (hole fills) in different color
        for e in tri_edges:
            pts = vertices[list(e)]
            ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=tri_color, linewidth=0.5)
    else:
        # Draw filled faces
        # Quads
        if quads:
            quad_verts = [[vertices[i] for i in q] for q in quads]
            quad_collection = Poly3DCollection(quad_verts,
                                               facecolors=quad_color,
                                               edgecolors=edge_color,
                                               linewidths=0.3,
                                               alpha=0.9)
            ax.add_collection3d(quad_collection)

        # Tris (hole fills)
        if tris:
            tri_verts = [[vertices[i] for i in t] for t in tris]
            tri_collection = Poly3DCollection(tri_verts,
                                              facecolors=tri_color,
                                              edgecolors=edge_color,
                                              linewidths=0.3,
                                              alpha=0.9)
            ax.add_collection3d(tri_collection)

    # Set axis properties
    all_verts = vertices
    center = all_verts.mean(axis=0)
    max_range = (all_verts.max(axis=0) - all_verts.min(axis=0)).max() / 2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{len(quads)} quads (blue) + {len(tris)} tris (coral)')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=quad_color, edgecolor=edge_color, label=f'Quads ({len(quads)})'),
        Patch(facecolor=tri_color, edgecolor=edge_color, label=f'Hole-fill tris ({len(tris)})')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Saved to {output}')
    else:
        plt.show()

    plt.close()


def render_quad_mesh_2d(vertices, quads, tris,
                        title=None,
                        quad_color='steelblue',
                        tri_color='coral',
                        edge_color='black',
                        figsize=(12, 10),
                        output=None):
    """
    Render mesh as 2D flat view (useful for UV layouts).
    Uses only X,Y coordinates.
    """
    fig, ax = plt.subplots(figsize=figsize)

    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Draw quads
    quad_patches = []
    for q in quads:
        poly = Polygon(vertices[q, :2], closed=True)
        quad_patches.append(poly)

    if quad_patches:
        pc = PatchCollection(quad_patches, facecolor=quad_color,
                            edgecolor=edge_color, linewidth=0.5, alpha=0.8)
        ax.add_collection(pc)

    # Draw tris
    tri_patches = []
    for t in tris:
        poly = Polygon(vertices[t, :2], closed=True)
        tri_patches.append(poly)

    if tri_patches:
        pc = PatchCollection(tri_patches, facecolor=tri_color,
                            edgecolor=edge_color, linewidth=0.5, alpha=0.8)
        ax.add_collection(pc)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{len(quads)} quads + {len(tris)} tris')

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f'Saved to {output}')
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Render quad mesh with proper quad display')
    parser.add_argument('mesh', help='Path to quad mesh OBJ file')
    parser.add_argument('-o', '--output', help='Save to file instead of display')
    parser.add_argument('--wireframe', action='store_true', help='Show wireframe only')
    parser.add_argument('--flat', action='store_true', help='2D view (X,Y only)')
    parser.add_argument('--elev', type=float, default=30, help='Elevation angle (default: 30)')
    parser.add_argument('--azim', type=float, default=45, help='Azimuth angle (default: 45)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 10],
                        help='Figure size (default: 12 10)')

    args = parser.parse_args()

    # Read mesh
    vertices, quads, tris = read_quad_obj(args.mesh)
    print(f'Loaded: {len(vertices)} vertices, {len(quads)} quads, {len(tris)} tris')

    title = Path(args.mesh).stem

    if args.flat:
        render_quad_mesh_2d(vertices, quads, tris,
                           title=title,
                           figsize=tuple(args.figsize),
                           output=args.output)
    else:
        render_quad_mesh(vertices, quads, tris,
                        title=title,
                        wireframe=args.wireframe,
                        elev=args.elev,
                        azim=args.azim,
                        figsize=tuple(args.figsize),
                        output=args.output)


if __name__ == '__main__':
    main()
