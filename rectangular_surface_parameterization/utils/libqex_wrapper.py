"""
Python wrapper for libQEx quad extraction.

This module provides a Python interface to the libQEx library for extracting
quad meshes from triangle meshes with UV parameterization.

Usage:
    from rectangular_surface_parameterization.utils.libqex_wrapper import extract_quads

    quad_verts, quad_faces = extract_quads(vertices, triangles, uvs_per_triangle)
"""

import numpy as np
from collections import defaultdict

# Try to import pyqex, provide helpful error if not installed
_pyqex = None
_pyqex_error = None

try:
    import pyqex as _pyqex
except ImportError as e:
    _pyqex_error = str(e)


def _ensure_pyqex():
    """Ensure pyqex is available, raise helpful error if not."""
    if _pyqex is None:
        raise ImportError(
            f"pyqex is not installed ({_pyqex_error}).\n\n"
            "Install it with:\n"
            "    python scripts/install_pyqex.py\n\n"
            "Or manually from: https://github.com/mfagerlund/pyqex/releases"
        )
    return _pyqex


def _fill_holes_with_triangles(vertices, quads, verbose=True):
    """
    Fill holes in a quad mesh with triangles.

    Traces boundary loops using quad face orientation to correctly handle
    cases where multiple holes share vertices. Each loop is fan-triangulated
    from its own centroid.

    Parameters
    ----------
    vertices : ndarray, shape (n_verts, 3)
        Vertex positions.
    quads : ndarray, shape (n_quads, 4)
        Quad face indices (0-based).
    verbose : bool
        Print information about holes filled.

    Returns
    -------
    triangles : list of [i, j, k]
        Triangle indices to fill holes.
    new_vertices : ndarray
        New centroid vertices added for hole filling.
    """
    # Build undirected edge counts
    edge_count = defaultdict(int)
    for quad in quads:
        for i in range(4):
            v0, v1 = quad[i], quad[(i + 1) % 4]
            edge = (min(v0, v1), max(v0, v1))
            edge_count[edge] += 1

    # Boundary edges appear only once
    boundary_edges_set = {e for e, c in edge_count.items() if c == 1}

    if not boundary_edges_set:
        if verbose:
            print("  No holes to fill (mesh is watertight)")
        return [], np.zeros((0, 3))

    # Build directed boundary edges from quads
    # For quad [a,b,c,d], edges are a->b, b->c, c->d, d->a
    directed_edges_set = set()
    for quad in quads:
        for i in range(4):
            v0, v1 = quad[i], quad[(i + 1) % 4]
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key in boundary_edges_set:
                directed_edges_set.add((v0, v1))

    directed_edges = list(directed_edges_set)

    # Build vertex-to-incident-faces map for edge ordering around vertices
    vertex_faces = defaultdict(list)  # vertex -> list of (quad_idx, position_in_quad)
    for qi, quad in enumerate(quads):
        for i in range(4):
            vertex_faces[quad[i]].append((qi, i))

    # Build next_edge map using face topology
    # For each boundary edge, find its successor by looking at the face containing it
    next_edge = {}

    # For each directed boundary edge (a, b), find the next edge (b, c)
    # If edges (a,b) and (b,c) are both boundary and in the same face, they're consecutive
    # If (a,b) is boundary but (b,c) is interior, walk through connected faces to find
    # the next boundary edge starting at b

    for e in directed_edges:
        a, b = e

        # Find the quad containing directed edge (a, b)
        containing_quad = None
        edge_pos = None
        for qi, quad in enumerate(quads):
            for i in range(4):
                if quad[i] == a and quad[(i + 1) % 4] == b:
                    containing_quad = qi
                    edge_pos = i
                    break
            if containing_quad is not None:
                break

        if containing_quad is None:
            continue

        # Get the next vertex in this quad after b
        quad = quads[containing_quad]
        c = quad[(edge_pos + 2) % 4]
        next_candidate = (b, c)

        if next_candidate in directed_edges_set:
            # Both edges are boundary - they're consecutive
            next_edge[e] = next_candidate
        else:
            # Edge (b, c) is interior - need to walk through faces to find next boundary edge
            # Walk around vertex b through connected faces until we find a boundary outgoing edge
            current_qi = containing_quad
            current_next = c  # The vertex after b in current face

            # Keep walking to adjacent faces until we find a boundary edge
            for _ in range(len(quads)):  # Safety limit
                # Find the face adjacent to current face across edge (b, current_next)
                # That face has edge (current_next, b) as one of its edges
                found_adjacent = False
                for qi, quad in enumerate(quads):
                    if qi == current_qi:
                        continue
                    for i in range(4):
                        if quad[i] == current_next and quad[(i + 1) % 4] == b:
                            # Found adjacent face
                            # The next vertex after b in this face
                            next_v = quad[(i + 2) % 4]
                            out_candidate = (b, next_v)
                            if out_candidate in directed_edges_set:
                                # Found the next boundary edge
                                next_edge[e] = out_candidate
                                found_adjacent = True
                                break
                            else:
                                # Keep walking
                                current_qi = qi
                                current_next = next_v
                                found_adjacent = True
                                break
                    if found_adjacent:
                        break

                if e in next_edge:
                    break
                if not found_adjacent:
                    # No adjacent face found - we've reached a gap (but shouldn't happen)
                    break

    # Walk boundary loops
    visited = set()
    loops = []

    for start_edge in directed_edges:
        if start_edge in visited:
            continue

        loop_vertices = []
        current = start_edge

        while current not in visited:
            visited.add(current)
            loop_vertices.append(current[0])

            if current not in next_edge:
                break

            next_e = next_edge[current]
            if next_e == start_edge:
                # Completed loop
                break
            current = next_e

        if len(loop_vertices) >= 3:
            loops.append(loop_vertices)

    # Split loops at repeated vertices (pinch points) to avoid non-manifold edges
    # A figure-8 loop that visits vertex v twice should be split into 2 sub-loops
    final_loops = []
    for loop in loops:
        # Check for repeated vertices
        seen = {}
        repeated = []
        for i, v in enumerate(loop):
            if v in seen:
                repeated.append((seen[v], i))
            seen[v] = i

        if not repeated:
            # No repeats - keep loop as is
            final_loops.append(loop)
        else:
            # Split at repeated vertices
            # For each pair (first_idx, second_idx), split into sub-loops
            # Sort by first occurrence index
            repeated.sort()

            # Extract sub-loops
            remaining = loop[:]
            for first_idx, second_idx in repeated:
                if second_idx <= first_idx:
                    continue
                # Find current positions in remaining
                try:
                    v = loop[first_idx]
                    curr_first = remaining.index(v)
                    # Find second occurrence
                    curr_second = None
                    for i in range(curr_first + 1, len(remaining)):
                        if remaining[i] == v:
                            curr_second = i
                            break
                    if curr_second is not None:
                        # Extract sub-loop from first to second
                        sub_loop = remaining[curr_first:curr_second]
                        if len(sub_loop) >= 3:
                            final_loops.append(sub_loop)
                        # Remove the sub-loop from remaining, keep the shared vertex
                        remaining = remaining[:curr_first+1] + remaining[curr_second+1:]
                except ValueError:
                    continue

            # Add what remains
            if len(remaining) >= 3:
                final_loops.append(remaining)

    if verbose:
        print(f"  Found {len(final_loops)} holes with {len(boundary_edges_set)} boundary edges")

    # Triangulate each loop using fan from centroid
    triangles = []
    n_verts = len(vertices)
    new_vertices = []

    for loop in final_loops:
        if len(loop) < 3:
            continue

        # Compute centroid
        centroid = vertices[loop].mean(axis=0)
        centroid_idx = n_verts + len(new_vertices)
        new_vertices.append(centroid)

        # Create triangles around the loop
        # Reverse winding to match quad orientation (boundary traced from inside)
        for i in range(len(loop)):
            v0 = loop[i]
            v1 = loop[(i + 1) % len(loop)]
            triangles.append([v1, v0, centroid_idx])

    if verbose and triangles:
        print(f"  Added {len(triangles)} triangles to fill {len(final_loops)} holes ({len(new_vertices)} new vertices)")

    return triangles, np.array(new_vertices) if new_vertices else np.zeros((0, 3))


def extract_quads(vertices, triangles, uv_per_triangle, vertex_valences=None, fill_holes=True, verbose=True):
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
        Expected valence at each vertex (passed to libQEx).
    fill_holes : bool
        If True (default), fill holes at irregular vertices with triangles.
    verbose : bool
        Print information about hole filling.

    Returns
    -------
    quad_vertices : ndarray, shape (n_quad_verts, 3)
        3D vertex positions of the output quad mesh.
    quad_faces : ndarray, shape (n_quads, 4)
        Quad indices (0-based).
    tri_faces : ndarray, shape (n_tris, 3) or None
        Triangle indices for hole fills (0-based), or None if fill_holes=False.
    """
    pyqex = _ensure_pyqex()

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.uint32)
    uv_per_triangle = np.asarray(uv_per_triangle, dtype=np.float64)

    n_verts = vertices.shape[0]
    n_tris = triangles.shape[0]

    assert vertices.shape == (n_verts, 3), f"vertices shape {vertices.shape}"
    assert triangles.shape == (n_tris, 3), f"triangles shape {triangles.shape}"
    assert uv_per_triangle.shape == (n_tris, 3, 2), f"uv shape {uv_per_triangle.shape}"

    # Call pyqex
    if vertex_valences is not None:
        vertex_valences = np.asarray(vertex_valences, dtype=np.uint32)
        quad_vertices, quad_faces = pyqex.extract_quads(
            vertices, triangles, uv_per_triangle, vertex_valences
        )
    else:
        quad_vertices, quad_faces = pyqex.extract_quads(
            vertices, triangles, uv_per_triangle
        )

    # Filter out invalid quads
    valid_quads = []
    n_quad_verts = len(quad_vertices)
    for q in quad_faces:
        # Skip degenerate quads (all zeros or any duplicates)
        if q[0] == q[1] == q[2] == q[3] == 0:
            continue
        if len(set(q)) < 4:
            continue  # Skip quads with duplicate vertices

        # Skip quads with out-of-range vertex indices
        if any(idx < 0 or idx >= n_quad_verts for idx in q):
            continue

        valid_quads.append(q)

    quad_faces = np.array(valid_quads, dtype=np.int32) if valid_quads else np.zeros((0, 4), dtype=np.int32)

    # Fill holes with triangles if requested
    tri_faces = None
    if fill_holes and len(quad_faces) > 0:
        hole_tris, new_verts = _fill_holes_with_triangles(quad_vertices, quad_faces, verbose=verbose)
        if hole_tris:
            tri_faces = np.array(hole_tris, dtype=np.int32)
            if len(new_verts) > 0:
                quad_vertices = np.vstack([quad_vertices, new_verts])

    return quad_vertices, quad_faces, tri_faces


def save_quad_obj(filepath, vertices, quads, tris=None):
    """
    Save a quad/tri mesh to OBJ format.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    vertices : ndarray, shape (n_verts, 3)
        Vertex positions.
    quads : ndarray, shape (n_quads, 4)
        Quad face indices (0-based, will be converted to 1-based for OBJ).
    tris : ndarray, shape (n_tris, 3), optional
        Triangle face indices for hole fills (0-based).
    """
    with open(filepath, 'w') as f:
        f.write("# Quad mesh extracted by libQEx\n")
        if tris is not None and len(tris) > 0:
            f.write(f"# {len(quads)} quads + {len(tris)} triangles (hole fills)\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for q in quads:
            # OBJ uses 1-based indexing
            f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        if tris is not None:
            for t in tris:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


if __name__ == "__main__":
    # Simple test
    print("Testing libQEx wrapper...")

    try:
        pyqex = _ensure_pyqex()
        print(f"pyqex version: {pyqex.__version__ if hasattr(pyqex, '__version__') else 'unknown'}")

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

        quad_verts, quad_faces, tri_faces = extract_quads(vertices, triangles, uv_per_triangle)

        print(f"Input: {len(vertices)} vertices, {len(triangles)} triangles")
        print(f"Output: {len(quad_verts)} vertices, {len(quad_faces)} quads", end="")
        if tri_faces is not None and len(tri_faces) > 0:
            print(f", {len(tri_faces)} hole-fill triangles")
        else:
            print()
        print(f"Quad vertices:\n{quad_verts}")
        print(f"Quad faces:\n{quad_faces}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
