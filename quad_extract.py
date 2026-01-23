"""
Quad mesh extraction from rectangular parameterization.

Extracts a pure quad mesh by tracing integer iso-lines (u=const, v=const)
through the parameterized triangle mesh.

Reference: Bommes et al., "Integer-Grid Maps for Reliable Quad Meshing", SIGGRAPH 2013
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from mesh import TriangleMesh


@dataclass
class QuadMesh:
    """Quad mesh data structure."""
    vertices: np.ndarray  # |V| x 3: vertex positions
    faces: np.ndarray     # |F| x 4: vertex indices per quad face

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)


@dataclass
class IsolineSegment:
    """A segment of an iso-line within a single triangle."""
    face_idx: int
    # Start and end points in 3D
    p0: np.ndarray
    p1: np.ndarray
    # Start and end points in UV
    uv0: np.ndarray
    uv1: np.ndarray
    # Which iso-line this belongs to (u=value or v=value)
    is_u_isoline: bool
    iso_value: int
    # Edge indices where this segment crosses (for connectivity)
    edge0: int  # local edge index (0, 1, or 2) or -1 if at corner
    edge1: int


def compute_uv_scale(
    mesh: TriangleMesh,
    corner_uvs: np.ndarray,
    target_edge_length: float = 1.0
) -> float:
    """
    Compute scale factor to make average quad edge approximately target_edge_length.

    Args:
        mesh: Triangle mesh
        corner_uvs: |C| x 2 UV coordinates per corner
        target_edge_length: Desired average edge length in scaled UV space

    Returns:
        scale: Factor to multiply UVs by
    """
    # Compute average edge length in UV space
    total_uv_length = 0.0
    n_edges = 0

    for f in range(mesh.n_faces):
        for local in range(3):
            c0 = 3 * f + local
            c1 = 3 * f + (local + 1) % 3
            uv0, uv1 = corner_uvs[c0], corner_uvs[c1]
            total_uv_length += np.linalg.norm(uv1 - uv0)
            n_edges += 1

    avg_uv_length = total_uv_length / n_edges if n_edges > 0 else 1.0

    # Scale so average edge is target_edge_length
    return target_edge_length / avg_uv_length if avg_uv_length > 1e-10 else 1.0


def find_edge_crossing(
    uv0: np.ndarray,
    uv1: np.ndarray,
    iso_value: float,
    is_u: bool
) -> Optional[float]:
    """
    Find where an iso-line crosses an edge.

    Args:
        uv0, uv1: UV coordinates at edge endpoints
        iso_value: The integer value of the iso-line
        is_u: True for u=const line, False for v=const line

    Returns:
        t: Parameter along edge [0, 1] where crossing occurs, or None if no crossing
    """
    coord_idx = 0 if is_u else 1
    val0, val1 = uv0[coord_idx], uv1[coord_idx]

    # Check if iso-line passes between val0 and val1
    if (val0 - iso_value) * (val1 - iso_value) > 0:
        # Both on same side
        return None

    # Compute parameter
    denom = val1 - val0
    if abs(denom) < 1e-10:
        return None

    t = (iso_value - val0) / denom

    # Clamp to [0, 1] with small tolerance
    if t < -1e-10 or t > 1 + 1e-10:
        return None

    return np.clip(t, 0.0, 1.0)


def trace_isolines_in_triangle(
    mesh: TriangleMesh,
    face_idx: int,
    corner_uvs: np.ndarray,
    scale: float
) -> List[IsolineSegment]:
    """
    Find all iso-line segments passing through a triangle.

    Args:
        mesh: Triangle mesh
        face_idx: Face index
        corner_uvs: UV coordinates per corner (already scaled)
        scale: UV scale factor (for reference)

    Returns:
        List of IsolineSegment objects
    """
    segments = []

    c0 = 3 * face_idx
    c1 = 3 * face_idx + 1
    c2 = 3 * face_idx + 2

    uvs = [corner_uvs[c0], corner_uvs[c1], corner_uvs[c2]]
    v0, v1, v2 = mesh.faces[face_idx]
    pos = [mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]]

    # Find range of integer u and v values in this triangle
    u_min = min(uv[0] for uv in uvs)
    u_max = max(uv[0] for uv in uvs)
    v_min = min(uv[1] for uv in uvs)
    v_max = max(uv[1] for uv in uvs)

    # Integer values that pass through the triangle
    u_ints = list(range(int(np.floor(u_min)) + 1, int(np.floor(u_max)) + 1))
    v_ints = list(range(int(np.floor(v_min)) + 1, int(np.floor(v_max)) + 1))

    # Edge definitions: edge i goes from corner i to corner (i+1)%3
    edges = [(0, 1), (1, 2), (2, 0)]

    # Trace u iso-lines
    for u_val in u_ints:
        crossings = []
        for edge_idx, (i, j) in enumerate(edges):
            t = find_edge_crossing(uvs[i], uvs[j], u_val, is_u=True)
            if t is not None:
                # Interpolate position and UV
                p = pos[i] * (1 - t) + pos[j] * t
                uv = uvs[i] * (1 - t) + uvs[j] * t
                crossings.append((edge_idx, p, uv))

        if len(crossings) >= 2:
            # Create segment between first two crossings
            seg = IsolineSegment(
                face_idx=face_idx,
                p0=crossings[0][1],
                p1=crossings[1][1],
                uv0=crossings[0][2],
                uv1=crossings[1][2],
                is_u_isoline=True,
                iso_value=u_val,
                edge0=crossings[0][0],
                edge1=crossings[1][0]
            )
            segments.append(seg)

    # Trace v iso-lines
    for v_val in v_ints:
        crossings = []
        for edge_idx, (i, j) in enumerate(edges):
            t = find_edge_crossing(uvs[i], uvs[j], v_val, is_u=False)
            if t is not None:
                p = pos[i] * (1 - t) + pos[j] * t
                uv = uvs[i] * (1 - t) + uvs[j] * t
                crossings.append((edge_idx, p, uv))

        if len(crossings) >= 2:
            seg = IsolineSegment(
                face_idx=face_idx,
                p0=crossings[0][1],
                p1=crossings[1][1],
                uv0=crossings[0][2],
                uv1=crossings[1][2],
                is_u_isoline=False,
                iso_value=v_val,
                edge0=crossings[0][0],
                edge1=crossings[1][0]
            )
            segments.append(seg)

    return segments


def find_isoline_intersections(
    u_segments: List[IsolineSegment],
    v_segments: List[IsolineSegment],
    tol: float = 1e-6
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Find intersection points between u and v iso-line segments.

    Args:
        u_segments: Segments from u=const lines
        v_segments: Segments from v=const lines
        tol: Tolerance for intersection detection

    Returns:
        List of (3D position, u_value, v_value) tuples
    """
    intersections = []

    for u_seg in u_segments:
        for v_seg in v_segments:
            if u_seg.face_idx != v_seg.face_idx:
                continue

            # Check if u_value is within v_segment's v-range
            # and v_value is within u_segment's u-range
            u_val = u_seg.iso_value
            v_val = v_seg.iso_value

            # For u_seg: u is constant at u_val, v varies
            u_seg_v_min = min(u_seg.uv0[1], u_seg.uv1[1])
            u_seg_v_max = max(u_seg.uv0[1], u_seg.uv1[1])

            # For v_seg: v is constant at v_val, u varies
            v_seg_u_min = min(v_seg.uv0[0], v_seg.uv1[0])
            v_seg_u_max = max(v_seg.uv0[0], v_seg.uv1[0])

            # Check if intersection point (u_val, v_val) is within both segments
            if (v_seg_u_min - tol <= u_val <= v_seg_u_max + tol and
                u_seg_v_min - tol <= v_val <= u_seg_v_max + tol):

                # Interpolate 3D position along u_seg where v = v_val
                if abs(u_seg.uv1[1] - u_seg.uv0[1]) > tol:
                    t = (v_val - u_seg.uv0[1]) / (u_seg.uv1[1] - u_seg.uv0[1])
                    t = np.clip(t, 0.0, 1.0)
                    p = u_seg.p0 * (1 - t) + u_seg.p1 * t
                else:
                    p = 0.5 * (u_seg.p0 + u_seg.p1)

                intersections.append((p, u_val, v_val))

    return intersections


def point_in_triangle_uv(
    uv: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
    tol: float = 0.05  # Increased tolerance to capture edge cases
) -> Optional[Tuple[float, float, float]]:
    """
    Check if a point is inside a triangle in UV space.

    Returns barycentric coordinates (b0, b1, b2) if inside, None otherwise.
    """
    # Compute barycentric coordinates
    v0 = uv2 - uv0
    v1 = uv1 - uv0
    v2 = uv - uv0

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return None

    inv_denom = 1.0 / denom
    b2 = (dot11 * dot02 - dot01 * dot12) * inv_denom
    b1 = (dot00 * dot12 - dot01 * dot02) * inv_denom
    b0 = 1.0 - b1 - b2

    # Check if point is in triangle (with tolerance)
    if b0 >= -tol and b1 >= -tol and b2 >= -tol:
        return (b0, b1, b2)
    return None


def find_triangle_containing_uv(
    mesh: TriangleMesh,
    corner_uvs: np.ndarray,
    uv_point: np.ndarray,
    face_hint: int = -1
) -> Optional[Tuple[int, Tuple[float, float, float]]]:
    """
    Find which triangle contains a UV point.

    Args:
        mesh: Triangle mesh
        corner_uvs: UV coordinates per corner
        uv_point: The UV point to locate
        face_hint: Optional face to check first

    Returns:
        (face_idx, barycentric_coords) or None if not found
    """
    # Check hint first
    if face_hint >= 0:
        c0, c1, c2 = 3 * face_hint, 3 * face_hint + 1, 3 * face_hint + 2
        bary = point_in_triangle_uv(uv_point, corner_uvs[c0], corner_uvs[c1], corner_uvs[c2])
        if bary is not None:
            return (face_hint, bary)

    # Search all triangles
    for f in range(mesh.n_faces):
        c0, c1, c2 = 3 * f, 3 * f + 1, 3 * f + 2
        bary = point_in_triangle_uv(uv_point, corner_uvs[c0], corner_uvs[c1], corner_uvs[c2])
        if bary is not None:
            return (f, bary)

    return None


def uv_to_3d(
    mesh: TriangleMesh,
    face_idx: int,
    bary: Tuple[float, float, float]
) -> np.ndarray:
    """Convert barycentric coordinates in a face to 3D position."""
    v0, v1, v2 = mesh.faces[face_idx]
    p0, p1, p2 = mesh.positions[v0], mesh.positions[v1], mesh.positions[v2]
    return bary[0] * p0 + bary[1] * p1 + bary[2] * p2


def build_quad_mesh(
    mesh: TriangleMesh,
    corner_uvs: np.ndarray,
    target_quads: int = 100,
    periodic: bool = False,
    verbose: bool = True
) -> QuadMesh:
    """
    Extract quad mesh from parameterized triangle mesh.

    Uses the correct triangle-first approach from Bommes et al.:
    For each triangle, find which integer (u, v) points lie inside it.
    This guarantees finding ALL valid quad vertices.

    Args:
        mesh: Input triangle mesh
        corner_uvs: |C| x 2 UV coordinates per corner
        target_quads: Approximate number of quads desired
        periodic: If True, treat UV domain as periodic (for genus > 0 surfaces)
        verbose: Print progress info

    Returns:
        QuadMesh object
    """
    # Compute total UV area to determine scale
    total_uv_area = 0.0
    for f in range(mesh.n_faces):
        c0, c1, c2 = 3*f, 3*f+1, 3*f+2
        uv0, uv1, uv2 = corner_uvs[c0], corner_uvs[c1], corner_uvs[c2]
        area = 0.5 * abs((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) -
                         (uv2[0] - uv0[0]) * (uv1[1] - uv0[1]))
        total_uv_area += area

    # Scale so that total_uv_area * scale^2 = target_quads
    scale = np.sqrt(target_quads / max(total_uv_area, 1e-10))

    if verbose:
        print(f"  UV scale factor: {scale:.4f}")
        print(f"  Target quads: {target_quads}")

    # Find optimal UV offset to maximize integer grid coverage
    def count_quads_for_params(uvs_base, test_scale, offset_u, offset_v):
        """Count quads for a given scale and UV offset."""
        uvs = (uvs_base + np.array([offset_u, offset_v])) * test_scale
        vertex_set = set()
        for f in range(mesh.n_faces):
            c0, c1, c2 = 3*f, 3*f+1, 3*f+2
            uv0, uv1, uv2 = uvs[c0], uvs[c1], uvs[c2]
            u_min = min(uv0[0], uv1[0], uv2[0])
            u_max = max(uv0[0], uv1[0], uv2[0])
            v_min = min(uv0[1], uv1[1], uv2[1])
            v_max = max(uv0[1], uv1[1], uv2[1])
            u_start = int(np.ceil(u_min))
            u_end = int(np.floor(u_max))
            v_start = int(np.ceil(v_min))
            v_end = int(np.floor(v_max))
            for u_int in range(u_start, u_end + 1):
                for v_int in range(v_start, v_end + 1):
                    uv_point = np.array([float(u_int), float(v_int)])
                    bary = point_in_triangle_uv(uv_point, uv0, uv1, uv2)
                    if bary is not None:
                        vertex_set.add((u_int, v_int))
        quad_count = sum(1 for (u, v) in vertex_set
                        if (u+1, v) in vertex_set and (u+1, v+1) in vertex_set and (u, v+1) in vertex_set)
        return len(vertex_set), quad_count

    # Search for best scale and offset combination
    best_scale = scale
    best_offset = (0.0, 0.0)
    best_quads = 0

    # Try different scale multipliers
    for scale_mult in [0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
        test_scale = scale * scale_mult
        for du in np.linspace(-0.5, 0.5, 21):
            for dv in np.linspace(-0.5, 0.5, 21):
                _, quads = count_quads_for_params(corner_uvs, test_scale, du / test_scale, dv / test_scale)
                if quads > best_quads:
                    best_quads = quads
                    best_scale = test_scale
                    best_offset = (du / test_scale, dv / test_scale)

    if verbose:
        if best_scale != scale:
            print(f"  Adjusted scale: {best_scale:.4f} ({best_scale/scale:.2f}x)")
        if best_offset[0] != 0 or best_offset[1] != 0:
            print(f"  UV offset for grid alignment: ({best_offset[0]*best_scale:.2f}, {best_offset[1]*best_scale:.2f})")

    # Use optimized scale
    scale = best_scale

    # Scale UVs with optimal offset
    scaled_uvs = (corner_uvs + np.array(best_offset)) * scale

    # Find UV bounds
    u_min_global = np.min(scaled_uvs[:, 0])
    u_max_global = np.max(scaled_uvs[:, 0])
    v_min_global = np.min(scaled_uvs[:, 1])
    v_max_global = np.max(scaled_uvs[:, 1])

    # Period vectors for periodic domains
    period_u = u_max_global - u_min_global
    period_v = v_max_global - v_min_global

    if verbose:
        print(f"  UV range: u=[{u_min_global:.2f}, {u_max_global:.2f}], v=[{v_min_global:.2f}, {v_max_global:.2f}]")
        if periodic:
            print(f"  Periodic mode: period=({period_u:.2f}, {period_v:.2f})")

    # For periodic domains, compute the integer period for wrapping
    if periodic:
        int_period_u = int(np.round(period_u))
        int_period_v = int(np.round(period_v))
    else:
        int_period_u = 0
        int_period_v = 0

    if verbose and periodic:
        print(f"  Integer period: ({int_period_u}, {int_period_v})")

    # CORRECT APPROACH: Iterate through each triangle and find integer points inside
    vertex_map = {}  # (u_int, v_int) -> 3D position

    for f in range(mesh.n_faces):
        c0, c1, c2 = 3*f, 3*f+1, 3*f+2
        uv0, uv1, uv2 = scaled_uvs[c0], scaled_uvs[c1], scaled_uvs[c2]

        # Find bounding box of this triangle in UV space
        u_min = min(uv0[0], uv1[0], uv2[0])
        u_max = max(uv0[0], uv1[0], uv2[0])
        v_min = min(uv0[1], uv1[1], uv2[1])
        v_max = max(uv0[1], uv1[1], uv2[1])

        # Integer values that COULD be in this triangle
        u_start = int(np.ceil(u_min))
        u_end = int(np.floor(u_max))
        v_start = int(np.ceil(v_min))
        v_end = int(np.floor(v_max))

        # Test each candidate integer point
        for u_int in range(u_start, u_end + 1):
            for v_int in range(v_start, v_end + 1):
                uv_point = np.array([float(u_int), float(v_int)])
                bary = point_in_triangle_uv(uv_point, uv0, uv1, uv2)

                if bary is not None:
                    # Compute 3D position via barycentric interpolation
                    p = uv_to_3d(mesh, f, bary)
                    # Store the vertex (may overwrite if on boundary - that's fine)
                    vertex_map[(u_int, v_int)] = p

    if verbose:
        print(f"  Grid vertices found: {len(vertex_map)}")

    if len(vertex_map) == 0:
        if verbose:
            print("  Warning: No quad vertices found")
        return QuadMesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 4), dtype=np.int32)
        )

    # Convert vertex_map to indexed arrays
    # First, create index mapping: (u, v) -> vertex_index
    vertex_index = {}  # (u, v) -> index
    vertices = []
    for (u, v), pos in vertex_map.items():
        vertex_index[(u, v)] = len(vertices)
        vertices.append(pos)
    vertices = np.array(vertices)

    # Helper to wrap grid coordinates for periodic domains
    def wrap_coord(u, v):
        """Wrap (u, v) to the primary domain for periodic surfaces."""
        if not periodic or int_period_u == 0 or int_period_v == 0:
            return (u, v)
        # Find the minimum u, v in our vertex set
        all_u = [k[0] for k in vertex_index.keys()]
        all_v = [k[1] for k in vertex_index.keys()]
        u_min_int = min(all_u)
        v_min_int = min(all_v)
        u_wrapped = u_min_int + ((u - u_min_int) % int_period_u)
        v_wrapped = v_min_int + ((v - v_min_int) % int_period_v)
        return (u_wrapped, v_wrapped)

    # Build quad faces by checking each (u, v) vertex
    faces = []
    checked = set()  # avoid duplicate quads

    for (u, v) in vertex_index.keys():
        # Check if we can form a quad with (u, v) as bottom-left corner
        c00 = (u, v)
        c10 = wrap_coord(u + 1, v) if periodic else (u + 1, v)
        c11 = wrap_coord(u + 1, v + 1) if periodic else (u + 1, v + 1)
        c01 = wrap_coord(u, v + 1) if periodic else (u, v + 1)

        # Create canonical key to avoid duplicates (use min u, v as key)
        quad_key = (min(u, c10[0], c11[0], c01[0]), min(v, c10[1], c11[1], c01[1]))
        if quad_key in checked:
            continue

        if all(c in vertex_index for c in [c00, c10, c11, c01]):
            face = [
                vertex_index[c00],
                vertex_index[c10],
                vertex_index[c11],
                vertex_index[c01]
            ]
            faces.append(face)
            checked.add(quad_key)

    if len(faces) == 0:
        if verbose:
            print("  Warning: No complete quads found")
        faces = np.zeros((0, 4), dtype=np.int32)
    else:
        faces = np.array(faces, dtype=np.int32)

    if verbose:
        print(f"  Quad faces: {len(faces)}")

    return QuadMesh(vertices=vertices, faces=faces)


def save_quad_obj(filepath: str, quad_mesh: QuadMesh):
    """Save quad mesh to OBJ file."""
    with open(filepath, 'w') as f:
        f.write("# Quad mesh from Corman-Crane parameterization\n")
        f.write(f"# Vertices: {quad_mesh.n_vertices}, Faces: {quad_mesh.n_faces}\n\n")

        for v in quad_mesh.vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")

        f.write("\n")
        for face in quad_mesh.faces:
            # OBJ uses 1-based indexing
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3]+1}\n")


def extract_quads(
    mesh: TriangleMesh,
    corner_uvs: np.ndarray,
    target_quads: int = 100,
    output_path: Optional[str] = None,
    periodic: bool = False,
    verbose: bool = True
) -> QuadMesh:
    """
    Main entry point for quad mesh extraction.

    Args:
        mesh: Input triangle mesh with parameterization
        corner_uvs: |C| x 2 UV coordinates per corner
        target_quads: Approximate number of quads desired
        output_path: Optional path to save OBJ file
        periodic: If True, treat UV domain as periodic (for genus > 0 surfaces)
        verbose: Print progress info

    Returns:
        QuadMesh object
    """
    if verbose:
        print("\n[Quad Extraction]")

    quad_mesh = build_quad_mesh(mesh, corner_uvs, target_quads, periodic, verbose)

    if output_path and quad_mesh.n_faces > 0:
        save_quad_obj(output_path, quad_mesh)
        if verbose:
            print(f"  Saved: {output_path}")

    return quad_mesh


# Command-line interface
if __name__ == "__main__":
    import argparse
    from io_obj import load_obj, load_obj_with_uvs

    parser = argparse.ArgumentParser(description="Extract quad mesh from parameterized mesh")
    parser.add_argument("input", help="Input OBJ file with UVs")
    parser.add_argument("-o", "--output", help="Output quad mesh OBJ file")
    parser.add_argument("-n", "--num-quads", type=int, default=100,
                        help="Target number of quads")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    # Load mesh with UVs
    mesh, uvs, face_uvs = load_obj_with_uvs(args.input)

    if uvs is None or face_uvs is None:
        print("Error: Input mesh has no UV coordinates")
        exit(1)

    # Convert to corner UVs
    corner_uvs = np.zeros((mesh.n_corners, 2))
    for f in range(mesh.n_faces):
        for local in range(3):
            corner = 3 * f + local
            uv_idx = face_uvs[f, local]
            corner_uvs[corner] = uvs[uv_idx]

    # Extract quads
    output_path = args.output or args.input.replace('.obj', '_quads.obj')
    quad_mesh = extract_quads(
        mesh, corner_uvs,
        target_quads=args.num_quads,
        output_path=output_path,
        verbose=not args.quiet
    )

    print(f"\nQuad mesh: {quad_mesh.n_vertices} vertices, {quad_mesh.n_faces} faces")
