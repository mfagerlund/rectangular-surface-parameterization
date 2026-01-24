"""Quad mesh builder for QEx quad extraction.

This module walks around connected grid vertices to form quad faces. It is the
final step in the QEx algorithm after finding grid vertices and tracing edges.

The algorithm is based on libQEx MeshExtractorT.cc generate_faces() method.
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from .grid_vertex import (
    GridVertex,
    opposite_direction,
    DIRECTION_PLUS_U,
    DIRECTION_PLUS_V,
    DIRECTION_MINUS_U,
    DIRECTION_MINUS_V,
)


def next_ccw_direction(direction: int) -> int:
    """Get the next direction in counter-clockwise order.

    In UV space, CCW order is: +u -> +v -> -u -> -v -> +u
    Direction encoding: 0=+u, 1=+v, 2=-u, 3=-v

    Args:
        direction: Current direction (0-3)

    Returns:
        Next direction in CCW order
    """
    return (direction + 1) % 4


def walk_face(
    start_vertex_idx: int,
    start_direction: int,
    grid_vertices: List[GridVertex],
    visited_edges: Set[Tuple[int, int]],
    max_face_size: int = 100,
) -> List[int]:
    """Walk around a face starting from an edge, collecting vertices.

    Starting from a vertex and direction, follows edges in CCW order until
    returning to the starting point. At each vertex, we turn CCW from the
    incoming direction to find the next outgoing edge.

    The CCW turn works as follows:
    - We arrive at a vertex from direction D (from the perspective of where we came from)
    - The incoming direction at the current vertex is opposite(D)
    - To turn CCW, we try directions starting from opposite(D) + 1

    IMPORTANT: Each directed edge is visited exactly once. The reverse direction
    of an edge (going the opposite way) is a separate edge that can be used
    by another face. This allows interior edges to be shared by two faces.

    Args:
        start_vertex_idx: Index of the starting vertex
        start_direction: Direction of the first edge to follow (0-3)
        grid_vertices: List of GridVertex objects with populated outgoing_edges
        visited_edges: Set of (vertex_idx, direction) tuples already visited.
            Will be updated in place as edges are traversed.
        max_face_size: Maximum number of vertices in a face (safety limit)

    Returns:
        List of vertex indices forming the face, in order.
        Empty list if the face cannot be completed.
    """
    face_vertices = []
    current_vertex_idx = start_vertex_idx
    current_direction = start_direction

    for _ in range(max_face_size):
        # Mark this directed edge as visited (only this direction, not the reverse)
        edge_key = (current_vertex_idx, current_direction)
        if edge_key in visited_edges:
            # Already visited this edge - we've completed a loop or hit a dead end
            if current_vertex_idx == start_vertex_idx and len(face_vertices) > 0:
                # Successfully returned to start
                return face_vertices
            else:
                # Hit an already-visited edge without completing the face
                return []

        visited_edges.add(edge_key)

        # Add current vertex to face
        face_vertices.append(current_vertex_idx)

        # Follow edge to next vertex
        current_vertex = grid_vertices[current_vertex_idx]
        connection = current_vertex.get_target(current_direction)

        if connection is None:
            # No connection in this direction - face cannot be completed
            return []

        next_vertex_idx, reverse_direction = connection

        # NOTE: Do NOT mark the reverse edge as visited here!
        # The reverse edge can be used by an adjacent face.

        # Move to next vertex
        current_vertex_idx = next_vertex_idx

        # At the new vertex, find the next outgoing edge by turning CCW
        # We arrived from reverse_direction, so we start looking CCW from there
        # CCW from incoming direction means: opposite of how we arrived + 1 in CCW order
        # But actually, the incoming direction at the new vertex is reverse_direction
        # We want to turn CCW from the incoming direction to find the next edge
        # CCW from reverse_direction is: (reverse_direction + 1) % 4

        # Try directions in CCW order starting from the direction after the incoming one
        found_next = False
        for i in range(4):
            # Try CCW directions starting after the incoming direction
            try_direction = (reverse_direction + 1 + i) % 4
            if grid_vertices[current_vertex_idx].has_connection(try_direction):
                current_direction = try_direction
                found_next = True
                break

        if not found_next:
            # No outgoing edge found
            return []

        # Check if we've returned to the start
        if current_vertex_idx == start_vertex_idx:
            return face_vertices

    # Exceeded max face size
    return []


def deduplicate_vertices(
    grid_vertices: List[GridVertex],
    tolerance: float = 1e-10,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Deduplicate 3D vertex positions and create a mapping.

    Grid vertices may have the same 3D position (e.g., at UV seams).
    This function creates unique 3D vertices and maps grid vertex indices
    to unique vertex indices.

    Args:
        grid_vertices: List of GridVertex objects
        tolerance: Distance tolerance for considering vertices identical

    Returns:
        Tuple of:
        - vertices: np.ndarray shape (N, 3) of unique 3D positions
        - index_map: Dict mapping grid vertex index to unique vertex index
    """
    if not grid_vertices:
        return np.zeros((0, 3)), {}

    # Collect all 3D positions
    positions = np.array([gv.pos_3d for gv in grid_vertices])

    # Find unique vertices using a simple approach
    unique_positions = []
    index_map = {}

    for gv_idx, pos in enumerate(positions):
        # Check if this position matches any existing unique position
        found = False
        for unique_idx, unique_pos in enumerate(unique_positions):
            if np.linalg.norm(pos - unique_pos) < tolerance:
                index_map[gv_idx] = unique_idx
                found = True
                break

        if not found:
            index_map[gv_idx] = len(unique_positions)
            unique_positions.append(pos)

    vertices = np.array(unique_positions) if unique_positions else np.zeros((0, 3))
    return vertices, index_map


def _face_key(face: List[int]) -> Tuple[int, ...]:
    """Create a canonical key for a face (independent of starting vertex or direction).

    The key is the lexicographically smallest rotation of the face vertices.
    This allows detecting duplicate faces that start at different vertices.

    Args:
        face: List of vertex indices forming the face

    Returns:
        Tuple representing the canonical form of the face
    """
    if not face:
        return ()

    # Find all rotations and take the lexicographically smallest
    rotations = []
    n = len(face)
    for i in range(n):
        rotations.append(tuple(face[i:] + face[:i]))

    return min(rotations)


def _is_reversed_face(face1: List[int], face2: List[int]) -> bool:
    """Check if face2 is the reverse (opposite winding) of face1.

    Args:
        face1: First face vertex list
        face2: Second face vertex list

    Returns:
        True if face2 is face1 traversed in opposite direction
    """
    if len(face1) != len(face2):
        return False

    # Reverse face2 and check if it matches face1 (considering rotations)
    reversed_face2 = list(reversed(face2))
    return _face_key(face1) == _face_key(reversed_face2)


def build_quads(
    grid_vertices: List[GridVertex],
    verbose: bool = False,
    filter_boundary: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build quad mesh from connected grid vertices.

    Walks around connected grid vertices to form quad faces. This is the final
    step of the QEx algorithm.

    Args:
        grid_vertices: List of GridVertex objects with populated outgoing_edges
            (after running trace_edges)
        verbose: If True, print progress information
        filter_boundary: If True, filter out the outer boundary face (largest n-gon
            that represents the mesh boundary, not an actual face)

    Returns:
        Tuple of:
        - vertices: np.ndarray shape (N, 3) of unique 3D vertex positions
        - faces: np.ndarray shape (M, 4) of quad face indices.
            For non-quad faces (triangles, n-gons), uses -1 padding.
            For example, a triangle [0, 1, 2] becomes [0, 1, 2, -1].

    Algorithm:
        1. Track visited edges (direction from vertex)
        2. For each unvisited edge:
           - Walk around the face in CCW order
           - Collect vertices until returning to start
           - Record as a face
        3. Remove duplicate faces (same face traversed from different start points)
        4. Optionally filter the outer boundary
        5. Deduplicate 3D vertex positions
        6. Build face connectivity with remapped indices
    """
    if not grid_vertices:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32)

    visited_edges: Set[Tuple[int, int]] = set()
    raw_faces: List[List[int]] = []

    # Find all faces by walking unvisited edges
    for vertex_idx, vertex in enumerate(grid_vertices):
        for direction in range(4):
            edge_key = (vertex_idx, direction)

            # Skip if no connection or already visited
            if not vertex.has_connection(direction):
                continue
            if edge_key in visited_edges:
                continue

            # Walk the face starting from this edge
            face = walk_face(vertex_idx, direction, grid_vertices, visited_edges)

            if face and len(face) >= 3:
                raw_faces.append(face)

    if verbose:
        print(f"  Quad builder: found {len(raw_faces)} raw faces from {len(grid_vertices)} grid vertices")
        face_sizes = {}
        for f in raw_faces:
            size = len(f)
            face_sizes[size] = face_sizes.get(size, 0) + 1
        print(f"  Raw face size distribution: {face_sizes}")

    # Remove duplicate faces (keep only one version of each face)
    # Each face may be found twice (once CW, once CCW traversal)
    unique_faces: List[List[int]] = []
    seen_keys: Set[Tuple[int, ...]] = set()

    for face in raw_faces:
        key = _face_key(face)
        reversed_key = _face_key(list(reversed(face)))

        # Only add if we haven't seen this face or its reverse
        if key not in seen_keys and reversed_key not in seen_keys:
            unique_faces.append(face)
            seen_keys.add(key)

    if verbose and len(unique_faces) != len(raw_faces):
        print(f"  After deduplication: {len(unique_faces)} unique faces")

    # Filter out the boundary face if requested
    # The boundary is typically the largest face (n-gon around the outside)
    if filter_boundary and unique_faces:
        max_size = max(len(f) for f in unique_faces)
        if max_size > 4:
            # Filter out faces larger than quads (likely boundaries)
            filtered_faces = [f for f in unique_faces if len(f) <= 4]
            if verbose:
                removed = len(unique_faces) - len(filtered_faces)
                print(f"  Filtered {removed} boundary faces (size > 4)")
            unique_faces = filtered_faces

    raw_faces = unique_faces

    # Deduplicate vertices
    vertices, index_map = deduplicate_vertices(grid_vertices)

    if verbose:
        print(f"  Quad builder: {len(grid_vertices)} grid vertices -> {len(vertices)} unique 3D vertices")

    # Convert faces to use deduplicated vertex indices
    # Find max face size to determine padding
    max_face_size = max((len(f) for f in raw_faces), default=4)
    max_face_size = max(max_face_size, 4)  # At least 4 for quads

    faces = np.full((len(raw_faces), max_face_size), -1, dtype=np.int32)

    for face_idx, face in enumerate(raw_faces):
        for i, gv_idx in enumerate(face):
            faces[face_idx, i] = index_map[gv_idx]

    return vertices, faces


def extract_quads_only(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract only quad faces from the result.

    Filters the faces array to only include 4-sided faces (quads).
    Also filters out degenerate quads with repeated vertices.

    Args:
        vertices: np.ndarray shape (N, 3) of vertex positions
        faces: np.ndarray shape (M, 4+) of face indices with -1 padding

    Returns:
        Tuple of (vertices, quad_faces) where quad_faces has shape (Q, 4)
    """
    if faces.size == 0:
        return vertices, np.zeros((0, 4), dtype=np.int32)

    quad_faces = []
    for face in faces:
        # Count valid vertices (not -1)
        valid = face[face >= 0]
        if len(valid) == 4:
            # Check for degenerate quads (repeated vertices)
            if len(set(valid)) == 4:
                quad_faces.append(valid)

    if not quad_faces:
        return vertices, np.zeros((0, 4), dtype=np.int32)

    return vertices, np.array(quad_faces, dtype=np.int32)


def faces_to_trimesh_format(
    faces: np.ndarray,
) -> np.ndarray:
    """Convert faces to trimesh-compatible format by triangulating.

    Trimesh expects triangles, so this function triangulates n-gon faces.
    Uses simple fan triangulation from the first vertex.

    Args:
        faces: np.ndarray shape (M, N) of face indices with -1 padding

    Returns:
        np.ndarray shape (T, 3) of triangle indices
    """
    if faces.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    triangles = []
    for face in faces:
        valid = face[face >= 0]
        if len(valid) < 3:
            continue

        # Fan triangulation from first vertex
        for i in range(1, len(valid) - 1):
            triangles.append([valid[0], valid[i], valid[i + 1]])

    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)

    return np.array(triangles, dtype=np.int32)


if __name__ == "__main__":
    # Test with a simple 2x2 grid
    print("Testing quad_builder with a simple 2x2 grid")
    print("=" * 50)

    # Create a 2x2 grid of vertices at integer UV coordinates
    # Grid layout:
    #   (0,1) -- (1,1)
    #     |        |
    #   (0,0) -- (1,0)

    grid_vertices = [
        GridVertex(
            type='face',
            uv=(0, 0),
            pos_3d=np.array([0.0, 0.0, 0.0]),
            face_idx=0,
        ),
        GridVertex(
            type='face',
            uv=(1, 0),
            pos_3d=np.array([1.0, 0.0, 0.0]),
            face_idx=0,
        ),
        GridVertex(
            type='face',
            uv=(1, 1),
            pos_3d=np.array([1.0, 1.0, 0.0]),
            face_idx=0,
        ),
        GridVertex(
            type='face',
            uv=(0, 1),
            pos_3d=np.array([0.0, 1.0, 0.0]),
            face_idx=0,
        ),
    ]

    print(f"Created {len(grid_vertices)} grid vertices:")
    for i, gv in enumerate(grid_vertices):
        print(f"  {i}: UV={gv.uv}, pos={gv.pos_3d}")

    # Trace edges to connect them
    from .edge_tracer import trace_edges

    print("\nTracing edges...")
    edges = trace_edges(grid_vertices, verbose=True)

    print(f"\nConnections after tracing:")
    for i, gv in enumerate(grid_vertices):
        connections = []
        for d in range(4):
            if gv.has_connection(d):
                target, _ = gv.get_target(d)
                dir_names = ['+u', '+v', '-u', '-v']
                connections.append(f"{dir_names[d]}->{target}")
        print(f"  {i}: {connections}")

    # Build quads
    print("\nBuilding quads...")
    vertices, faces = build_quads(grid_vertices, verbose=True)

    print(f"\nResult:")
    print(f"  Vertices ({len(vertices)}):")
    for i, v in enumerate(vertices):
        print(f"    {i}: {v}")

    print(f"  Faces ({len(faces)}):")
    for i, f in enumerate(faces):
        print(f"    {i}: {f}")

    # Test with a 3x3 grid (4 quads)
    print("\n" + "=" * 50)
    print("Testing with a 3x3 grid (should produce 4 quads)")
    print("=" * 50)

    # Create a 3x3 grid
    # Grid layout:
    #   (0,2) -- (1,2) -- (2,2)
    #     |        |        |
    #   (0,1) -- (1,1) -- (2,1)
    #     |        |        |
    #   (0,0) -- (1,0) -- (2,0)

    grid_vertices_3x3 = []
    for v in range(3):
        for u in range(3):
            gv = GridVertex(
                type='face',
                uv=(u, v),
                pos_3d=np.array([float(u), float(v), 0.0]),
                face_idx=0,
            )
            grid_vertices_3x3.append(gv)

    print(f"Created {len(grid_vertices_3x3)} grid vertices")

    # Trace edges
    edges = trace_edges(grid_vertices_3x3, verbose=True)

    # Build quads
    vertices, faces = build_quads(grid_vertices_3x3, verbose=True)

    print(f"\nResult:")
    print(f"  Vertices ({len(vertices)}):")
    for i, v in enumerate(vertices):
        print(f"    {i}: {v}")

    print(f"  Faces ({len(faces)}):")
    for i, f in enumerate(faces):
        print(f"    {i}: {f}")

    # Extract only quads
    vertices_q, faces_q = extract_quads_only(vertices, faces)
    print(f"\n  Quads only: {len(faces_q)} faces")
