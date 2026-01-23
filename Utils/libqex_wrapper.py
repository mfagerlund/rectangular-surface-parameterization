"""
Python wrapper for libQEx quad extraction.

This module provides a Python interface to the libQEx library for extracting
quad meshes from triangle meshes with UV parameterization.

Usage:
    from Utils.libqex_wrapper import extract_quads

    quad_verts, quad_faces = extract_quads(vertices, triangles, uvs_per_triangle)
"""

import numpy as np
import subprocess
import tempfile
from pathlib import Path
import os


def _find_qex_exe():
    """Find the qex_extract.exe executable."""
    this_dir = Path(__file__).parent
    bin_dir = this_dir.parent / "bin"

    exe_name = "qex_extract.exe"
    exe_path = bin_dir / exe_name

    if exe_path.exists():
        return str(exe_path)

    # Check current directory
    if Path(exe_name).exists():
        return exe_name

    raise FileNotFoundError(
        f"Could not find {exe_name}. Expected at {exe_path}\n"
        "See bin/BINARIES.txt for build instructions."
    )


def extract_quads(vertices, triangles, uv_per_triangle, vertex_valences=None):
    """
    Extract a quad mesh from a triangle mesh with UV parameterization.

    Parameters
    ----------
    vertices : ndarray, shape (n_verts, 3)
        3D vertex positions of the input triangle mesh.
    triangles : ndarray, shape (n_tris, 3)
        Triangle indices (0-based).
    uv_per_triangle : ndarray, shape (n_tris, 3, 2)
        UV coordinates for each corner of each triangle.
        uv_per_triangle[i, j, :] is the UV for the j-th corner of triangle i.
    vertex_valences : ndarray, shape (n_verts,), optional
        Expected valence at each vertex. Currently ignored (not passed to exe).

    Returns
    -------
    quad_vertices : ndarray, shape (n_quad_verts, 3)
        3D vertex positions of the output quad mesh.
    quad_faces : ndarray, shape (n_quads, 4)
        Quad indices (0-based).
    """
    exe_path = _find_qex_exe()

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.uint32)
    uv_per_triangle = np.asarray(uv_per_triangle, dtype=np.float64)

    n_verts = vertices.shape[0]
    n_tris = triangles.shape[0]

    assert vertices.shape == (n_verts, 3), f"vertices shape {vertices.shape}"
    assert triangles.shape == (n_tris, 3), f"triangles shape {triangles.shape}"
    assert uv_per_triangle.shape == (n_tris, 3, 2), f"uv shape {uv_per_triangle.shape}"

    # Build input string
    lines = [f"{n_verts} {n_tris}"]

    # Vertices
    for i in range(n_verts):
        lines.append(f"{vertices[i, 0]:.17g} {vertices[i, 1]:.17g} {vertices[i, 2]:.17g}")

    # Triangles with UVs
    for i in range(n_tris):
        t = triangles[i]
        uv = uv_per_triangle[i]
        lines.append(
            f"{t[0]} {t[1]} {t[2]} "
            f"{uv[0, 0]:.17g} {uv[0, 1]:.17g} "
            f"{uv[1, 0]:.17g} {uv[1, 1]:.17g} "
            f"{uv[2, 0]:.17g} {uv[2, 1]:.17g}"
        )

    input_data = "\n".join(lines) + "\n"

    # Run the executable
    result = subprocess.run(
        [exe_path],
        input=input_data,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"qex_extract failed: {result.stderr}")

    # Parse output
    output_lines = result.stdout.strip().split("\n")
    if len(output_lines) < 1:
        raise RuntimeError("qex_extract produced empty output")

    header = output_lines[0].split()
    n_quad_verts = int(header[0])
    n_quads = int(header[1])

    quad_vertices = np.zeros((n_quad_verts, 3), dtype=np.float64)
    quad_faces = np.zeros((n_quads, 4), dtype=np.int32)

    # Read vertices
    for i in range(n_quad_verts):
        parts = output_lines[1 + i].split()
        quad_vertices[i, 0] = float(parts[0])
        quad_vertices[i, 1] = float(parts[1])
        quad_vertices[i, 2] = float(parts[2])

    # Read quads (filter out degenerate quads like 0 0 0 0)
    valid_quads = []
    for i in range(n_quads):
        parts = output_lines[1 + n_quad_verts + i].split()
        q = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]

        # Skip degenerate quads (all zeros or any duplicates)
        if q[0] == q[1] == q[2] == q[3] == 0:
            continue
        if len(set(q)) < 4:
            continue  # Skip quads with duplicate vertices

        valid_quads.append(q)

    quad_faces = np.array(valid_quads, dtype=np.int32)

    return quad_vertices, quad_faces


def save_quad_obj(filepath, vertices, quads):
    """
    Save a quad mesh to OBJ format.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    vertices : ndarray, shape (n_verts, 3)
        Vertex positions.
    quads : ndarray, shape (n_quads, 4)
        Quad face indices (0-based, will be converted to 1-based for OBJ).
    """
    with open(filepath, 'w') as f:
        f.write("# Quad mesh extracted by libQEx\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for q in quads:
            # OBJ uses 1-based indexing
            f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")


if __name__ == "__main__":
    # Simple test with two triangles forming a quad
    print("Testing libQEx wrapper...")

    try:
        exe_path = _find_qex_exe()
        print(f"Found qex_extract at: {exe_path}")

        # Create a simple test case: 4 vertices, 2 triangles
        vertices = np.array([
            [-1, 0, -1],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0]
        ], dtype=np.float64)

        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.uint32)

        # UVs that form a [0,1] x [0,1] quad
        uv_per_triangle = np.array([
            [[-0.1, -0.1], [1.1, -0.1], [1, 1]],
            [[-0.1, -0.1], [1, 1], [-0.1, 1.1]]
        ], dtype=np.float64)

        quad_verts, quad_faces = extract_quads(vertices, triangles, uv_per_triangle)

        print(f"Input: {len(vertices)} vertices, {len(triangles)} triangles")
        print(f"Output: {len(quad_verts)} vertices, {len(quad_faces)} quads")
        print(f"Quad vertices:\n{quad_verts}")
        print(f"Quad faces:\n{quad_faces}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
