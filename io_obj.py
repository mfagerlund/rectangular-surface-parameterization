"""
OBJ file I/O for triangle meshes.

Supports:
- Loading: vertices (v), faces (f), texture coordinates (vt)
- Saving: vertices with optional per-corner UVs
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from mesh import TriangleMesh, build_connectivity, validate_manifold


def load_obj(filepath: str) -> TriangleMesh:
    """
    Load an OBJ file into a TriangleMesh.

    Handles:
    - Vertex positions (v x y z)
    - Faces (f v1 v2 v3 or f v1/vt1 v2/vt2 v3/vt3 or f v1/vt1/vn1 ...)
    - Quads are triangulated into 2 triangles

    Returns:
        TriangleMesh with connectivity built
    """
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                # Vertex position
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])

            elif parts[0] == 'f':
                # Face - extract vertex indices (OBJ uses 1-based indexing)
                face_verts = []
                for p in parts[1:]:
                    # Handle v, v/vt, v/vt/vn, v//vn formats
                    v_idx = int(p.split('/')[0])
                    # Convert to 0-based, handle negative indices
                    if v_idx < 0:
                        v_idx = len(vertices) + v_idx
                    else:
                        v_idx = v_idx - 1
                    face_verts.append(v_idx)

                # Triangulate if needed (fan triangulation)
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])

    if not vertices:
        raise ValueError(f"No vertices found in {filepath}")
    if not faces:
        raise ValueError(f"No faces found in {filepath}")

    positions = np.array(vertices, dtype=np.float64)
    face_array = np.array(faces, dtype=np.int32)

    mesh = TriangleMesh(positions=positions, faces=face_array)
    mesh = build_connectivity(mesh)

    # Validate manifoldness
    is_valid, msg = validate_manifold(mesh)
    if not is_valid:
        print(f"Warning: {msg}")

    return mesh


def load_obj_with_uvs(filepath: str) -> Tuple[TriangleMesh, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load an OBJ file with texture coordinates.

    Returns:
        mesh: TriangleMesh
        uvs: |UV| x 2 array of UV coordinates (or None)
        face_uvs: |F| x 3 array of UV indices per face corner (or None)
    """
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])

            elif parts[0] == 'vt':
                u, v = float(parts[1]), float(parts[2])
                uvs.append([u, v])

            elif parts[0] == 'f':
                face_verts = []
                face_uv_indices = []
                has_uvs = True

                for p in parts[1:]:
                    components = p.split('/')
                    v_idx = int(components[0])
                    if v_idx < 0:
                        v_idx = len(vertices) + v_idx
                    else:
                        v_idx = v_idx - 1
                    face_verts.append(v_idx)

                    if len(components) > 1 and components[1]:
                        vt_idx = int(components[1])
                        if vt_idx < 0:
                            vt_idx = len(uvs) + vt_idx
                        else:
                            vt_idx = vt_idx - 1
                        face_uv_indices.append(vt_idx)
                    else:
                        has_uvs = False

                # Triangulate
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                    if has_uvs and len(face_uv_indices) == len(face_verts):
                        face_uvs.append([face_uv_indices[0], face_uv_indices[i], face_uv_indices[i + 1]])

    positions = np.array(vertices, dtype=np.float64)
    face_array = np.array(faces, dtype=np.int32)

    mesh = TriangleMesh(positions=positions, faces=face_array)
    mesh = build_connectivity(mesh)

    uv_array = np.array(uvs, dtype=np.float64) if uvs else None
    face_uv_array = np.array(face_uvs, dtype=np.int32) if face_uvs else None

    return mesh, uv_array, face_uv_array


def save_obj(filepath: str, mesh: TriangleMesh, corner_uvs: Optional[np.ndarray] = None):
    """
    Save mesh to OBJ file.

    Args:
        filepath: Output path
        mesh: Triangle mesh
        corner_uvs: |C| x 2 array of UV coordinates per corner (optional)
    """
    with open(filepath, 'w') as f:
        f.write("# Corman-Crane rectangular parameterization output\n")
        f.write(f"# Vertices: {mesh.n_vertices}, Faces: {mesh.n_faces}\n\n")

        # Write vertices
        for i in range(mesh.n_vertices):
            v = mesh.positions[i]
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        # Write UVs if provided (per-corner)
        if corner_uvs is not None:
            f.write("\n")
            for c in range(mesh.n_corners):
                uv = corner_uvs[c]
                f.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")

        # Write faces
        f.write("\n")
        if corner_uvs is not None:
            # With UVs: f v/vt v/vt v/vt
            for face_idx in range(mesh.n_faces):
                v0, v1, v2 = mesh.faces[face_idx]
                c0 = 3 * face_idx + 0
                c1 = 3 * face_idx + 1
                c2 = 3 * face_idx + 2
                # OBJ uses 1-based indexing
                f.write(f"f {v0+1}/{c0+1} {v1+1}/{c1+1} {v2+1}/{c2+1}\n")
        else:
            # Without UVs
            for face_idx in range(mesh.n_faces):
                v0, v1, v2 = mesh.faces[face_idx]
                f.write(f"f {v0+1} {v1+1} {v2+1}\n")


def mesh_info(mesh: TriangleMesh) -> str:
    """Return a summary string of mesh properties."""
    from mesh import euler_characteristic, count_boundary_loops, genus

    chi = euler_characteristic(mesh)
    b = count_boundary_loops(mesh)
    g = genus(mesh)

    lines = [
        f"Vertices: {mesh.n_vertices}",
        f"Faces: {mesh.n_faces}",
        f"Edges: {mesh.n_edges}",
        f"Halfedges: {mesh.n_halfedges}",
        f"Corners: {mesh.n_corners}",
        f"Euler characteristic (V-E+F): {chi}",
        f"Boundary loops: {b}",
        f"Genus: {g}",
    ]

    # Validate
    is_valid, msg = validate_manifold(mesh)
    lines.append(f"Manifold: {is_valid} ({msg})")

    return "\n".join(lines)
