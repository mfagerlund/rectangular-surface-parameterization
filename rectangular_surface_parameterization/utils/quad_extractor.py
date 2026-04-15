"""
Quad mesh extraction from parameterized triangle meshes.

Replaces libQEx with a simpler, more robust algorithm:
1. Find all integer grid points (i,j) inside the UV triangulation
2. Compute their 3D positions via barycentric interpolation
3. Merge duplicate vertices across UV cuts by 3D proximity
4. Form quads from UV-adjacent grid points using merged indices

At singularities, some grid points will be missing, leaving small holes.
This is expected and preferable to the pockets/inverted faces that libQEx produces.

Same API as libqex_wrapper.extract_quads — drop-in replacement.
"""

import numpy as np
from collections import defaultdict


def _barycentric_2d(p, a, b, c):
    """Compute barycentric coordinates of point p in triangle (a, b, c) in 2D.

    Returns (lambda1, lambda2, lambda3) where p = l1*a + l2*b + l3*c.
    """
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-30:
        return (-1.0, -1.0, -1.0)

    inv_denom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (1.0 - u - v, v, u)


def _find_grid_points(vertices_3d, triangles, uv_per_triangle, tolerance=1e-8):
    """Find all integer grid points inside the UV triangulation.

    Returns dict: (iu, iv) -> 3D position
    """
    grid_points = {}

    for ti in range(len(triangles)):
        uv = uv_per_triangle[ti]

        uv_min = uv.min(axis=0)
        uv_max = uv.max(axis=0)

        iu_min = int(np.ceil(uv_min[0] - tolerance))
        iu_max = int(np.floor(uv_max[0] + tolerance))
        iv_min = int(np.ceil(uv_min[1] - tolerance))
        iv_max = int(np.floor(uv_max[1] + tolerance))

        for iu in range(iu_min, iu_max + 1):
            for iv in range(iv_min, iv_max + 1):
                if (iu, iv) in grid_points:
                    continue

                p = np.array([float(iu), float(iv)])
                l1, l2, l3 = _barycentric_2d(p, uv[0], uv[1], uv[2])

                if l1 >= -tolerance and l2 >= -tolerance and l3 >= -tolerance:
                    v0 = vertices_3d[triangles[ti, 0]]
                    v1 = vertices_3d[triangles[ti, 1]]
                    v2 = vertices_3d[triangles[ti, 2]]
                    pos_3d = l1 * v0 + l2 * v1 + l3 * v2
                    grid_points[(iu, iv)] = pos_3d

    return grid_points


def _merge_grid_points(grid_points, tolerance=1e-6):
    """Merge grid points that map to the same 3D position (across UV cuts).

    Returns:
        merged_vertices: list of 3D positions (deduplicated)
        uv_to_merged: dict (iu, iv) -> merged vertex index
    """
    all_keys = sorted(grid_points.keys())
    all_positions = np.array([grid_points[k] for k in all_keys])

    if len(all_positions) == 0:
        return [], {}

    inv_tol = 1.0 / tolerance
    bucket = {}
    merged_vertices = []
    uv_to_merged = {}

    for i, key in enumerate(all_keys):
        v = all_positions[i]
        qk = (round(v[0] * inv_tol), round(v[1] * inv_tol), round(v[2] * inv_tol))

        matched_idx = None
        for dk in range(-1, 2):
            for dj in range(-1, 2):
                for di in range(-1, 2):
                    nk = (qk[0] + di, qk[1] + dj, qk[2] + dk)
                    if nk in bucket:
                        for existing_i, existing_merged in bucket[nk]:
                            if np.linalg.norm(all_positions[existing_i] - v) < tolerance:
                                matched_idx = existing_merged
                                break
                    if matched_idx is not None:
                        break
                if matched_idx is not None:
                    break
            if matched_idx is not None:
                break

        if matched_idx is not None:
            uv_to_merged[key] = matched_idx
        else:
            new_idx = len(merged_vertices)
            merged_vertices.append(v)
            uv_to_merged[key] = new_idx
            bucket.setdefault(qk, []).append((i, new_idx))

    return merged_vertices, uv_to_merged


def _form_quads(grid_points, uv_to_merged):
    """Form quads from UV-adjacent grid points (within single chart only).

    A quad exists at (i,j) if all four corners exist and map to 4 distinct
    merged vertices.
    """
    quads = []
    seen = set()

    for (iu, iv) in grid_points:
        corners = [(iu, iv), (iu + 1, iv), (iu + 1, iv + 1), (iu, iv + 1)]
        if all(c in uv_to_merged for c in corners):
            indices = tuple(uv_to_merged[c] for c in corners)
            if len(set(indices)) == 4 and indices not in seen:
                seen.add(indices)
                quads.append(list(indices))

    return quads


# =========================================================================
# Strategy A: Transition function recovery
# =========================================================================

def _recover_transitions_from_mesh(vertices_3d, triangles, uv_per_triangle,
                                    tolerance=1e-6):
    """Recover UV transition functions across cuts using UV seam data.

    Finds vertices that have different UV coordinates in different triangles
    (UV seams). Clusters the UVs per vertex into "sides" of the cut, then
    uses the correspondences to fit transition functions
    R ∈ {I, R90, R180, R270} and t ∈ Z².

    Returns list of (R, t) transitions.
    """
    # Collect all UVs per vertex index across all triangles
    vert_uvs = defaultdict(list)
    for ti in range(len(triangles)):
        for ci in range(3):
            vi = int(triangles[ti, ci])
            vert_uvs[vi].append(uv_per_triangle[ti, ci].copy())

    # For each vertex, cluster UVs into "sides" of the cut
    uv_pairs = []
    for vi, uvs in vert_uvs.items():
        uvs_arr = np.array(uvs)
        # Cluster by proximity (points within 0.5 are on the same side)
        clusters = []
        for uv_pt in uvs_arr:
            placed = False
            for cl in clusters:
                if np.linalg.norm(cl[0] - uv_pt) < 0.5:
                    cl.append(uv_pt)
                    placed = True
                    break
            if not placed:
                clusters.append([uv_pt])

        if len(clusters) >= 2:
            means = [np.mean(cl, axis=0) for cl in clusters]
            for i in range(len(means)):
                for j in range(i + 1, len(means)):
                    uv_pairs.append((means[i], means[j]))

    if len(uv_pairs) < 2:
        return []

    rotations = [
        np.array([[1, 0], [0, 1]], dtype=float),     # 0°
        np.array([[0, -1], [1, 0]], dtype=float),     # 90°
        np.array([[-1, 0], [0, -1]], dtype=float),    # 180°
        np.array([[0, 1], [-1, 0]], dtype=float),     # 270°
    ]

    results = []
    for R in rotations:
        t_votes = defaultdict(int)
        for uv_a, uv_b in uv_pairs:
            for src, dst in [(uv_a, uv_b), (uv_b, uv_a)]:
                t = dst - R @ src
                t_key = (round(t[0]), round(t[1]))
                t_votes[t_key] += 1

        for t_key, count in t_votes.items():
            if count >= 2:
                t = np.array(t_key, dtype=float)
                results.append((R, t, count))

    results.sort(key=lambda x: -x[2])

    seen = set()
    unique = []
    for R, t, count in results:
        key = (tuple(R.ravel()), tuple(t))
        if key not in seen:
            seen.add(key)
            unique.append((R, t))

    return unique


def _extend_grid_with_transitions(grid_points, transitions):
    """Add virtual grid points by applying transition functions.

    For each existing grid point, apply each transition to create a virtual
    point on the other side of the cut. Only adds points that don't already
    exist in the grid.

    Returns extended grid_points dict.
    """
    extended = dict(grid_points)
    added = 0

    for R, t in transitions:
        for (iu, iv), pos_3d in list(grid_points.items()):
            uv = np.array([iu, iv], dtype=float)
            new_uv = R @ uv + t
            new_key = (round(new_uv[0]), round(new_uv[1]))
            if new_key not in extended:
                extended[new_key] = pos_3d.copy()
                added += 1

    return extended, added


def form_quads_strategy_a(grid_points, uv_to_merged,
                          vertices_3d=None, triangles=None,
                          uv_per_triangle=None, verbose=False):
    """Strategy A: Recover transition functions from mesh, extend grid, form quads.

    Requires original mesh data to find cut correspondences.
    Returns (quads, extended_merged_verts, extended_uv_to_merged, stats_dict).
    """
    if vertices_3d is None or triangles is None or uv_per_triangle is None:
        quads = _form_quads(grid_points, uv_to_merged)
        return quads, None, uv_to_merged, {'transitions': 0, 'added_points': 0}

    transitions = _recover_transitions_from_mesh(
        vertices_3d, triangles, uv_per_triangle)
    if verbose:
        print(f"  Strategy A: found {len(transitions)} transition functions")
        for i, (R, t) in enumerate(transitions[:5]):
            angle = {(1,0,0,1): 0, (0,-1,1,0): 90,
                     (-1,0,0,-1): 180, (0,1,-1,0): 270}.get(
                tuple(R.ravel().astype(int)), '?')
            print(f"    T{i}: R={angle} deg, t=({t[0]:.0f}, {t[1]:.0f})")

    if not transitions:
        quads = _form_quads(grid_points, uv_to_merged)
        return quads, None, uv_to_merged, {'transitions': 0, 'added_points': 0}

    extended, added = _extend_grid_with_transitions(grid_points, transitions)
    if verbose:
        print(f"  Strategy A: extended grid by {added} virtual points "
              f"({len(grid_points)} -> {len(extended)})")

    merged_ext, uv_to_merged_ext = _merge_grid_points(extended)
    if verbose:
        print(f"  Strategy A: merged to {len(merged_ext)} vertices")

    quads = _form_quads(extended, uv_to_merged_ext)

    return quads, merged_ext, uv_to_merged_ext, {
        'transitions': len(transitions), 'added_points': added,
        'merged_verts': len(merged_ext)}


# =========================================================================
# Strategy B: Adjacency graph
# =========================================================================

def form_quads_strategy_b(grid_points, uv_to_merged, verbose=False):
    """Strategy B: Build adjacency graph from all UV charts, form quads.

    For each merged vertex, collect directional neighbors (u+1 and v+1) from
    all UV charts. Then find quads as M-R-D-U where M->R (u+1), M->U (v+1),
    R->D (v+1), U->D (u+1).

    Returns (quads, stats_dict).
    """
    # Build directional neighbor sets
    right = defaultdict(set)  # merged_idx -> {merged_idx reachable by u+1}
    up = defaultdict(set)     # merged_idx -> {merged_idx reachable by v+1}

    for (iu, iv), merged_idx in uv_to_merged.items():
        r_key = (iu + 1, iv)
        if r_key in uv_to_merged:
            r_m = uv_to_merged[r_key]
            if r_m != merged_idx:
                right[merged_idx].add(r_m)

        u_key = (iu, iv + 1)
        if u_key in uv_to_merged:
            u_m = uv_to_merged[u_key]
            if u_m != merged_idx:
                up[merged_idx].add(u_m)

    # Find quads: M -right-> R -up-> D <-right- U <-up- M
    quads = set()
    for M in right:
        for R in right[M]:
            for D in up.get(R, set()):
                if D == M:
                    continue
                for U in up.get(M, set()):
                    if U == R or U == D:
                        continue
                    if D in right.get(U, set()):
                        # Valid quad: M, R, D, U
                        q = (M, R, D, U)
                        # Canonical: start with min, prefer forward winding
                        min_pos = min(range(4), key=lambda i: q[i])
                        norm = q[min_pos:] + q[:min_pos]
                        quads.add(norm)

    if verbose:
        n_right = sum(len(v) for v in right.values())
        n_up = sum(len(v) for v in up.values())
        print(f"  Strategy B: {len(right)} vertices with right-neighbors "
              f"({n_right} edges), {len(up)} with up-neighbors ({n_up} edges)")
        print(f"  Strategy B: found {len(quads)} quads")

    return [list(q) for q in quads], {
        'right_edges': sum(len(v) for v in right.values()),
        'up_edges': sum(len(v) for v in up.values()),
    }


def _validate_mesh(vertices, quads, verbose=True):
    """Check output mesh for basic validity."""
    edge_count = defaultdict(int)
    for q in quads:
        for i in range(4):
            e = (min(q[i], q[(i + 1) % 4]), max(q[i], q[(i + 1) % 4]))
            edge_count[e] += 1

    non_manifold = sum(1 for c in edge_count.values() if c > 2)
    boundary = sum(1 for c in edge_count.values() if c == 1)

    if verbose:
        print(f"  Mesh validation: {len(vertices)} verts, {len(quads)} quads, "
              f"{len(edge_count)} edges, {boundary} boundary, {non_manifold} non-manifold")

    return {
        'n_verts': len(vertices),
        'n_quads': len(quads),
        'n_edges': len(edge_count),
        'boundary_edges': boundary,
        'non_manifold_edges': non_manifold,
    }


def extract_quads(vertices, triangles, uv_per_triangle,
                  vertex_valences=None, fill_holes=True,
                  max_hole_size=None, verbose=True, merge_tolerance=1e-6):
    """Extract a quad mesh from a parameterized triangle mesh.

    Drop-in replacement for libqex_wrapper.extract_quads.

    Parameters
    ----------
    vertices : ndarray, shape (n_verts, 3)
        3D vertex positions of the input triangle mesh.
    triangles : ndarray, shape (n_tris, 3)
        Triangle indices (0-based).
    uv_per_triangle : ndarray, shape (n_tris, 3, 2)
        UV coordinates for each corner of each triangle.
    vertex_valences : ignored (API compatibility with libQEx wrapper)
    fill_holes : bool
        If True, fill small boundary holes with fan triangulation.
    max_hole_size : int or None
        Maximum boundary loop size to fill. Larger holes stay open.
    verbose : bool
        Print progress information.
    merge_tolerance : float
        Distance threshold for merging duplicate vertices across cuts.

    Returns
    -------
    quad_vertices : ndarray, shape (n_quad_verts, 3)
    quad_faces : ndarray, shape (n_quads, 4)
    tri_faces : ndarray or None
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    uv_per_triangle = np.asarray(uv_per_triangle, dtype=np.float64)

    if verbose:
        uv_flat = uv_per_triangle.reshape(-1, 2)
        uv_range = (uv_flat.min(axis=0), uv_flat.max(axis=0))
        print(f"  UV range: [{uv_range[0][0]:.2f}, {uv_range[1][0]:.2f}] x "
              f"[{uv_range[0][1]:.2f}, {uv_range[1][1]:.2f}]")

    # Step 1: Find integer grid points in UV space
    grid_points = _find_grid_points(vertices, triangles, uv_per_triangle)

    if verbose:
        print(f"  Found {len(grid_points)} integer grid points in UV domain")

    if len(grid_points) == 0:
        if verbose:
            print("  Warning: No grid points found. UV scale may be too small.")
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32), None

    # Step 2: Merge grid points across UV cuts by 3D proximity
    merged_vertices, uv_to_merged = _merge_grid_points(
        grid_points, tolerance=merge_tolerance)

    if verbose:
        n_merged = len(grid_points) - len(merged_vertices)
        if n_merged > 0:
            print(f"  Merged {n_merged} duplicate vertices across cuts "
                  f"({len(grid_points)} -> {len(merged_vertices)})")

    # Step 3: Form quads using merged vertex indices
    out_quads = _form_quads(grid_points, uv_to_merged)

    if verbose:
        print(f"  Formed {len(out_quads)} quads")

    # Step 4: Remove unused vertices and reindex
    used = set()
    for q in out_quads:
        used.update(q)

    if len(used) < len(merged_vertices):
        old_to_new = {}
        final_vertices = []
        for old_idx in sorted(used):
            old_to_new[old_idx] = len(final_vertices)
            final_vertices.append(merged_vertices[old_idx])
        out_quads = [[old_to_new[v] for v in q] for q in out_quads]
        out_vertices = np.array(final_vertices)
    else:
        out_vertices = np.array(merged_vertices)

    quad_faces = np.array(out_quads, dtype=np.int32) if out_quads else np.zeros((0, 4), dtype=np.int32)

    # Step 5: Fill holes if requested
    tri_faces = None
    if fill_holes and len(quad_faces) > 0:
        from rectangular_surface_parameterization.utils.libqex_wrapper import (
            _fill_holes_with_triangles)
        hole_tris, new_verts = _fill_holes_with_triangles(
            out_vertices, quad_faces, verbose=verbose,
            max_hole_size=max_hole_size)
        if hole_tris:
            tri_faces = np.array(hole_tris, dtype=np.int32)
            if len(new_verts) > 0:
                out_vertices = np.vstack([out_vertices, new_verts])

    if verbose:
        _validate_mesh(out_vertices, quad_faces, verbose=True)

    return out_vertices, quad_faces, tri_faces
