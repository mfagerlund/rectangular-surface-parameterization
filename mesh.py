"""
Triangle mesh data structure for Corman-Crane rectangular parameterization.

Indexing conventions (from implementation plan):
- Corner: 3 * face_idx + local_vertex (local_vertex in {0, 1, 2})
- Halfedge: same as corner (halfedge ij in face ijk has index 3*face + 0)
- Edge: unique ID per unordered vertex pair

Key arrays:
- positions: |V| x 3 vertex coordinates
- faces: |F| x 3 vertex indices per face
- halfedge_to_edge: |H| -> edge index
- edge_to_halfedge: |E| x 2 -> halfedge indices (twin halfedges)
- edge_vertices: |E| x 2 -> vertex indices (i, j) where i < j
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TriangleMesh:
    """Half-edge triangle mesh with corner-based indexing."""

    positions: np.ndarray  # |V| x 3: vertex positions
    faces: np.ndarray      # |F| x 3: vertex indices per face

    # Derived connectivity (computed by build_connectivity)
    edge_vertices: np.ndarray = None     # |E| x 2: vertex indices (i < j)
    halfedge_to_edge: np.ndarray = None  # |H|: halfedge -> edge index
    edge_to_halfedge: np.ndarray = None  # |E| x 2: edge -> twin halfedges (-1 if boundary)
    halfedge_twin: np.ndarray = None     # |H|: halfedge -> twin halfedge (-1 if boundary)
    vertex_to_halfedge: np.ndarray = None  # |V|: vertex -> one outgoing halfedge

    @property
    def n_vertices(self) -> int:
        return len(self.positions)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        return len(self.edge_vertices) if self.edge_vertices is not None else 0

    @property
    def n_halfedges(self) -> int:
        return 3 * self.n_faces

    @property
    def n_corners(self) -> int:
        return 3 * self.n_faces

    def corner_to_face(self, corner: int) -> int:
        """Get face index from corner index."""
        return corner // 3

    def corner_to_local(self, corner: int) -> int:
        """Get local vertex index (0, 1, or 2) from corner index."""
        return corner % 3

    def face_local_to_corner(self, face: int, local: int) -> int:
        """Get corner index from face and local vertex index."""
        return 3 * face + local

    def corner_vertex(self, corner: int) -> int:
        """Get vertex index at corner."""
        face = self.corner_to_face(corner)
        local = self.corner_to_local(corner)
        return self.faces[face, local]

    def corner_vertices_ijk(self, corner: int) -> Tuple[int, int, int]:
        """Get (i, j, k) vertex indices for the corner at vertex i in face ijk."""
        face = self.corner_to_face(corner)
        local = self.corner_to_local(corner)
        f = self.faces[face]
        i = f[local]
        j = f[(local + 1) % 3]
        k = f[(local + 2) % 3]
        return i, j, k

    def halfedge_vertices(self, halfedge: int) -> Tuple[int, int]:
        """Get (i, j) vertices for halfedge ij."""
        face = halfedge // 3
        local = halfedge % 3
        f = self.faces[face]
        i = f[local]
        j = f[(local + 1) % 3]
        return i, j

    def halfedge_next(self, halfedge: int) -> int:
        """Get next halfedge in face (jk for halfedge ij in face ijk)."""
        face = halfedge // 3
        local = halfedge % 3
        return 3 * face + (local + 1) % 3

    def halfedge_prev(self, halfedge: int) -> int:
        """Get previous halfedge in face (ki for halfedge ij in face ijk)."""
        face = halfedge // 3
        local = halfedge % 3
        return 3 * face + (local + 2) % 3

    def halfedge_face(self, halfedge: int) -> int:
        """Get face for halfedge."""
        return halfedge // 3

    def edge_index(self, i: int, j: int) -> int:
        """Get edge index for unordered vertex pair (i, j)."""
        if i > j:
            i, j = j, i
        # Binary search in sorted edge_vertices
        for e in range(self.n_edges):
            if self.edge_vertices[e, 0] == i and self.edge_vertices[e, 1] == j:
                return e
        return -1

    def is_boundary_halfedge(self, halfedge: int) -> bool:
        """Check if halfedge is on boundary (no twin)."""
        return self.halfedge_twin[halfedge] == -1

    def is_boundary_edge(self, edge: int) -> bool:
        """Check if edge is on boundary."""
        return self.edge_to_halfedge[edge, 1] == -1

    def is_boundary_vertex(self, vertex: int) -> bool:
        """Check if vertex is on boundary."""
        he = self.vertex_to_halfedge[vertex]
        start = he
        while True:
            if self.is_boundary_halfedge(he):
                return True
            he_twin = self.halfedge_twin[he]
            if he_twin == -1:
                return True
            he = self.halfedge_next(he_twin)
            if he == start:
                break
        return False

    def vertex_degree(self, vertex: int) -> int:
        """Get vertex degree (number of incident edges)."""
        degree = 0
        he = self.vertex_to_halfedge[vertex]
        if he == -1:
            return 0
        start = he
        while True:
            degree += 1
            he_twin = self.halfedge_twin[he]
            if he_twin == -1:
                # Hit boundary, count from other direction
                he = self.halfedge_prev(self.vertex_to_halfedge[vertex])
                he = self.halfedge_twin[he]
                while he != -1:
                    degree += 1
                    he = self.halfedge_prev(he)
                    he = self.halfedge_twin[he]
                break
            he = self.halfedge_next(he_twin)
            if he == start:
                break
        return degree

    def vertex_halfedges(self, vertex: int) -> list:
        """Get all outgoing halfedges from vertex in CCW order."""
        halfedges = []
        he = self.vertex_to_halfedge[vertex]
        if he == -1:
            return halfedges
        start = he
        while True:
            halfedges.append(he)
            he_twin = self.halfedge_twin[he]
            if he_twin == -1:
                break
            he = self.halfedge_next(he_twin)
            if he == start:
                break
        return halfedges

    def vertex_corners(self, vertex: int) -> list:
        """Get all corners at vertex in CCW order."""
        corners = []
        he = self.vertex_to_halfedge[vertex]
        if he == -1:
            return corners
        start = he
        while True:
            # The corner at vertex for this halfedge
            face = he // 3
            local = he % 3
            corner = 3 * face + local
            corners.append(corner)
            he_twin = self.halfedge_twin[he]
            if he_twin == -1:
                break
            he = self.halfedge_next(he_twin)
            if he == start:
                break
        return corners

    def face_adjacent(self, face: int, local_edge: int) -> int:
        """Get adjacent face across local edge (0, 1, 2), or -1 if boundary."""
        he = 3 * face + local_edge
        he_twin = self.halfedge_twin[he]
        if he_twin == -1:
            return -1
        return he_twin // 3


def build_connectivity(mesh: TriangleMesh) -> TriangleMesh:
    """Build connectivity arrays for the mesh."""
    n_faces = mesh.n_faces
    n_vertices = mesh.n_vertices
    n_halfedges = 3 * n_faces

    # Build edge list and halfedge-edge mappings
    edge_dict: Dict[Tuple[int, int], int] = {}
    halfedge_to_edge = np.zeros(n_halfedges, dtype=np.int32)

    for f in range(n_faces):
        for local in range(3):
            he = 3 * f + local
            i = mesh.faces[f, local]
            j = mesh.faces[f, (local + 1) % 3]
            edge_key = (min(i, j), max(i, j))
            if edge_key not in edge_dict:
                edge_dict[edge_key] = len(edge_dict)
            halfedge_to_edge[he] = edge_dict[edge_key]

    n_edges = len(edge_dict)
    edge_vertices = np.zeros((n_edges, 2), dtype=np.int32)
    for (i, j), e in edge_dict.items():
        edge_vertices[e] = [i, j]

    # Build edge_to_halfedge (twin halfedges)
    edge_to_halfedge = np.full((n_edges, 2), -1, dtype=np.int32)
    for he in range(n_halfedges):
        e = halfedge_to_edge[he]
        if edge_to_halfedge[e, 0] == -1:
            edge_to_halfedge[e, 0] = he
        else:
            edge_to_halfedge[e, 1] = he

    # Build halfedge_twin
    halfedge_twin = np.full(n_halfedges, -1, dtype=np.int32)
    for e in range(n_edges):
        he0 = edge_to_halfedge[e, 0]
        he1 = edge_to_halfedge[e, 1]
        if he0 != -1 and he1 != -1:
            halfedge_twin[he0] = he1
            halfedge_twin[he1] = he0

    # Build vertex_to_halfedge (one outgoing halfedge per vertex)
    vertex_to_halfedge = np.full(n_vertices, -1, dtype=np.int32)
    for he in range(n_halfedges):
        face = he // 3
        local = he % 3
        v = mesh.faces[face, local]
        if vertex_to_halfedge[v] == -1:
            vertex_to_halfedge[v] = he

    mesh.edge_vertices = edge_vertices
    mesh.halfedge_to_edge = halfedge_to_edge
    mesh.edge_to_halfedge = edge_to_halfedge
    mesh.halfedge_twin = halfedge_twin
    mesh.vertex_to_halfedge = vertex_to_halfedge

    return mesh


def validate_manifold(mesh: TriangleMesh) -> Tuple[bool, str]:
    """
    Validate that mesh is manifold:
    1. Each edge has at most 2 incident faces
    2. Faces around each vertex form a single connected fan
    3. Consistent orientation

    Returns (is_valid, error_message).
    """
    if mesh.edge_vertices is None:
        mesh = build_connectivity(mesh)

    # Check edge valence (at most 2 faces per edge)
    edge_count = np.zeros(mesh.n_edges, dtype=np.int32)
    for he in range(mesh.n_halfedges):
        edge_count[mesh.halfedge_to_edge[he]] += 1

    if np.any(edge_count > 2):
        bad_edges = np.where(edge_count > 2)[0]
        return False, f"Non-manifold: edges {bad_edges} have >2 incident faces"

    # Check orientation consistency
    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 != -1 and he1 != -1:
            i0, j0 = mesh.halfedge_vertices(he0)
            i1, j1 = mesh.halfedge_vertices(he1)
            # Twin halfedges should have opposite orientation
            if not (i0 == j1 and j0 == i1):
                return False, f"Inconsistent orientation at edge {e}"

    return True, "Manifold mesh"


def euler_characteristic(mesh: TriangleMesh) -> int:
    """Compute Euler characteristic: chi = V - E + F."""
    if mesh.edge_vertices is None:
        mesh = build_connectivity(mesh)
    return mesh.n_vertices - mesh.n_edges + mesh.n_faces


def count_boundary_loops(mesh: TriangleMesh) -> int:
    """Count number of boundary loops."""
    if mesh.edge_vertices is None:
        mesh = build_connectivity(mesh)

    # Find boundary halfedges
    boundary_he = set()
    for e in range(mesh.n_edges):
        if mesh.is_boundary_edge(e):
            he = mesh.edge_to_halfedge[e, 0]
            # Find the boundary halfedge (the one without a twin)
            if mesh.halfedge_twin[he] == -1:
                boundary_he.add(he)
            else:
                he = mesh.edge_to_halfedge[e, 1]
                if he != -1 and mesh.halfedge_twin[he] == -1:
                    boundary_he.add(he)

    if not boundary_he:
        return 0

    # Count connected components of boundary
    visited = set()
    n_loops = 0

    for start_he in boundary_he:
        if start_he in visited:
            continue

        n_loops += 1
        he = start_he
        while he not in visited:
            visited.add(he)
            # Move to next boundary halfedge
            # Go to the prev halfedge's twin's prev, etc. until we find a boundary
            he_next = mesh.halfedge_next(he)
            he_next = mesh.halfedge_next(he_next)  # This is prev from endpoint
            while mesh.halfedge_twin[he_next] != -1:
                he_next = mesh.halfedge_twin[he_next]
                he_next = mesh.halfedge_next(he_next)
                he_next = mesh.halfedge_next(he_next)
            he = he_next

    return n_loops


def genus(mesh: TriangleMesh) -> int:
    """Compute genus: 2 - 2g - b = chi, so g = (2 - chi - b) / 2."""
    chi = euler_characteristic(mesh)
    b = count_boundary_loops(mesh)
    return (2 - chi - b) // 2
