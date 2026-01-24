"""
Quad extraction module for extracting quad meshes from UV-parameterized triangle meshes.

This is a **pure Python/NumPy port** of the QEx algorithm for extracting quad meshes
from triangle meshes with integer-grid UV parameterization.

The algorithm finds where integer iso-lines cross the mesh and connects them into quads:
1. Find grid vertices (integer UV points on faces, edges, and at mesh vertices)
2. Trace edges between adjacent grid vertices (ray tracing through triangles)
3. Build quad faces from connected grid vertices

Acknowledgements
----------------
This implementation is a port of the libQEx library. Full credit goes to the original
authors for the algorithm and C++ implementation:

    Hans-Christian Ebke, David Bommes, Marcel Campen, and Leif Kobbelt. 2013.
    "QEx: Robust Quad Mesh Extraction."
    ACM Trans. Graph. 32, 6, Article 168 (November 2013).
    DOI: https://doi.org/10.1145/2508363.2508372

Source: https://github.com/hcebke/libQEx (GPL-3.0)
"""

# Core data structures
from .grid_vertex import (
    GridVertex,
    GridEdge,
    DIRECTION_PLUS_U,
    DIRECTION_PLUS_V,
    DIRECTION_MINUS_U,
    DIRECTION_MINUS_V,
    opposite_direction,
    direction_to_uv_delta,
    uv_delta_to_direction,
)

# Grid vertex finders
from .find_face_vertices import (
    find_face_vertices,
    point_in_triangle_strict,
    barycentric_coords,
    interpolate_3d,
)

from .find_edge_vertices import (
    find_edge_vertices,
)

from .find_mesh_vertices import (
    find_mesh_vertices,
    find_mesh_vertices_with_seams,
    is_integer,
    is_integer_uv,
)

# Edge tracing
from .edge_tracer import (
    build_uv_index,
    trace_edges,
    trace_edges_with_validation,
    get_connected_component,
    find_all_components,
    validate_edge_symmetry,
    MeshTopology,
    LocalEdgeInfo,
    HalfEdge,
    find_path,
)

# Quad building
from .quad_builder import (
    build_quads,
    extract_quads_only,
    faces_to_trimesh_format,
    next_ccw_direction,
    walk_face,
    deduplicate_vertices,
)

# Main API
from .extract_quads import (
    extract_quads,
    extract_quads_from_parameterization,
)

__all__ = [
    # Data structures
    "GridVertex",
    "GridEdge",
    # Direction constants
    "DIRECTION_PLUS_U",
    "DIRECTION_PLUS_V",
    "DIRECTION_MINUS_U",
    "DIRECTION_MINUS_V",
    # Direction utilities
    "opposite_direction",
    "direction_to_uv_delta",
    "uv_delta_to_direction",
    # Face vertex finding
    "find_face_vertices",
    "point_in_triangle_strict",
    "barycentric_coords",
    "interpolate_3d",
    # Edge vertex finding
    "find_edge_vertices",
    # Mesh vertex finding
    "find_mesh_vertices",
    "find_mesh_vertices_with_seams",
    "is_integer",
    "is_integer_uv",
    # Edge tracing
    "build_uv_index",
    "trace_edges",
    "trace_edges_with_validation",
    "get_connected_component",
    "find_all_components",
    "validate_edge_symmetry",
    "MeshTopology",
    "LocalEdgeInfo",
    "HalfEdge",
    "find_path",
    # Quad building
    "build_quads",
    "extract_quads_only",
    "faces_to_trimesh_format",
    "next_ccw_direction",
    "walk_face",
    "deduplicate_vertices",
    # Main API
    "extract_quads",
    "extract_quads_from_parameterization",
]
