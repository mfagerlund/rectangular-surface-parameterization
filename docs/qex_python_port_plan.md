# QEx Python Port Plan

**Status: IMPLEMENTED** ✓

Port libQEx quad extraction to pure Python/NumPy. Reference: `C:\Slask\libQEx\src\`

## Implementation Status

The Python QEx port is complete with proper ray tracing. Key results:
- Simple test cases (2x2, 3x3, 4x4, 5x5 grids): **100% match** with C++ libQEx
- Proper ray tracing through mesh triangles (following libQEx find_path algorithm)
- Half-edge mesh topology for triangle traversal
- Segment-segment intersection for edge crossing detection

### Test Results (January 2026)
```
Simple Square 3x3: MATCH (9 quads)
Simple Square 5x5: MATCH (25 quads)
Four Triangles 4x4: MATCH (16 quads)
```

Note: Tests with invalid UV parameterizations (e.g., simple planar projection on
closed surfaces like spheres) will show differences because such projections create
inverted triangles in UV space.

## Overview

libQEx extracts quad meshes from triangle meshes with integer-grid UV parameterization.
The algorithm finds where integer iso-lines cross the mesh and connects them into quads.

## Algorithm Steps (from MeshExtractorT.cc)

### 1. Extract Transition Functions
- For seamless UVs, there are "transition functions" across cut edges
- These are rotations (0°, 90°, 180°, 270°) that relate UV coords on either side of a seam
- **For our case**: We already have seamless UVs from RSP, transitions should be identity
- **Simplification**: Skip this initially, assume no seams (closed surfaces work directly)

### 2. Consistent Truncation
- Snap UV coordinates to consistent precision to avoid numerical issues
- Ensures integer crossings are found reliably
- **Python**: Round to reasonable precision (e.g., 1e-10)

### 3. Generate Grid Vertices
Find all points where integer UV coordinates fall on the mesh:

#### 3a. Face vertices (OnFace)
For each triangle:
- Get UV bounding box
- For each integer point (x,y) in bbox:
  - Check if point is strictly inside triangle
  - If yes: compute 3D position via barycentric interpolation
  - Store as GridVertex with 4 outgoing edge directions (+u, +v, -u, -v)

#### 3b. Edge vertices (OnEdge)
For each edge:
- Get UV coords of both endpoints
- Find integer crossings along the edge (where floor(u) or floor(v) changes)
- Compute 3D position via linear interpolation
- Store as GridVertex

#### 3c. Vertex vertices (OnVertex)
For each mesh vertex:
- Check if UV coords are integers
- If yes: store as GridVertex with special handling for valence

### 4. Construct Local Edge Information
For each GridVertex, determine outgoing edges in 4 directions (+u, +v, -u, -v):
- Face vertices: all 4 directions available
- Edge vertices: 2 directions along edge, 2 perpendicular (may leave triangle)
- Vertex vertices: complex - depends on valence and surrounding triangles

### 5. Generate Connections (Edge Tracing) - **RAY TRACING IMPLEMENTED**
For each GridVertex and each outgoing direction:
- Build half-edge mesh topology for triangle adjacency
- Trace ray from current UV position to target integer UV
- Walk through triangles using segment-segment intersection:
  - Find which edge the ray crosses
  - Cross to adjacent triangle via opposite half-edge
  - Repeat until target is found or boundary hit
- Record which GridVertex we connect to

This follows the libQEx `find_path()` algorithm (MeshExtractorT.cc lines 1685-2156).

### 6. Generate Faces
Walk around connected GridVertices to form quads:
- Start at an unvisited edge
- Follow edges CCW until returning to start
- Should form 4-sided faces (quads)
- Some faces may be triangles or n-gons near singularities

## Python Implementation Plan

### Phase 1: Core Data Structures
```python
@dataclass
class GridVertex:
    type: str  # 'face', 'edge', 'vertex'
    uv: np.ndarray  # (2,) integer UV coords
    pos_3d: np.ndarray  # (3,) 3D position
    face_idx: int  # which triangle contains this point
    edges: List[GridEdge]  # outgoing edges (up to 4)

@dataclass
class GridEdge:
    direction: int  # 0=+u, 1=+v, 2=-u, 3=-v
    target_vertex: int  # index of connected GridVertex, or -1
    target_direction: int  # which direction at target
```

### Phase 2: Vertex Generation
1. `find_face_vertices(triangles, uvs)` - integer points inside triangles
2. `find_edge_vertices(triangles, uvs)` - integer crossings on edges
3. `find_vertex_vertices(triangles, uvs)` - mesh vertices with integer UVs

### Phase 3: Edge Tracing
1. `trace_edge(start_vertex, direction, triangles, uvs)` - follow direction until hitting another GridVertex
2. Handle edge crossings between triangles

### Phase 4: Face Generation
1. `generate_quads(grid_vertices)` - walk edges to form quad faces
2. Handle holes/boundaries

### Phase 5: Output
1. Collect unique 3D positions
2. Build quad face connectivity
3. Return vertices and faces arrays

## Key Differences from C++ Version

1. **No OpenMesh dependency** - use simple numpy arrays and custom HalfEdge dataclass
2. **No templates** - single implementation for our mesh format
3. **No transition functions** - assumes seamless UVs (identity transitions)
4. **Pure Python ray tracing** - follows same algorithm as C++ but may be slower
5. **MeshTopology class** - builds half-edge structure from triangle arrays

## Files Created

All files are in `rectangular_surface_parameterization/quad_extraction/`:

1. `grid_vertex.py` ✓ - GridVertex, GridEdge dataclasses, direction utilities
2. `geometry.py` ✓ - Geometric utilities (barycentric coords, orientation tests)
3. `find_face_vertices.py` ✓ - Find integer UV points inside triangles
4. `find_edge_vertices.py` ✓ - Find iso-line crossings on edges
5. `find_mesh_vertices.py` ✓ - Find mesh vertices with integer UVs
6. `edge_tracer.py` ✓ - Trace connections between grid vertices
7. `quad_builder.py` ✓ - Build quad faces from connected vertices
8. `extract_quads.py` ✓ - Main API matching libqex_wrapper interface
9. `__init__.py` ✓ - Public exports

## Usage

```python
from rectangular_surface_parameterization.quad_extraction import extract_quads

# Same interface as libqex_wrapper.extract_quads
quad_verts, quad_faces, tri_faces = extract_quads(
    vertices,           # (N, 3) mesh vertices
    triangles,          # (M, 3) triangle indices
    uv_per_triangle,    # (M, 3, 2) UV coords per triangle corner
    fill_holes=True,
    verbose=True,
)
```

## Testing Strategy

1. Compare output with libQEx on same inputs
2. Start with simple cases (single quad, 2x2 grid)
3. Progress to sphere, torus, then complex meshes
4. Keep libQEx wrapper as fallback until Python version is verified

## Actual Implementation Size

- `grid_vertex.py`: ~175 lines (GridVertex, GridEdge dataclasses)
- `geometry.py`: ~200 lines (ported utilities)
- `find_face_vertices.py`: ~330 lines
- `find_edge_vertices.py`: ~420 lines (iso-line crossings)
- `find_mesh_vertices.py`: ~290 lines
- `edge_tracer.py`: ~670 lines (HalfEdge, MeshTopology, find_path ray tracing)
- `quad_builder.py`: ~525 lines
- `extract_quads.py`: ~250 lines
- `__init__.py`: ~120 lines
- **Total: ~3,000 lines** (includes proper ray tracing implementation)

### Key Classes Added for Ray Tracing
- `MeshTopology`: Builds half-edge structure from triangle arrays
- `HalfEdge`: Half-edge data structure for mesh traversal
- `LocalEdgeInfo`: Connection information (mirrors libQEx structure)
- `find_path()`: Ray tracing function following libQEx algorithm

## References & Acknowledgements

This Python implementation is a port of the **libQEx** library. Full credit goes to the
original authors for the algorithm and C++ implementation.

### QEx Paper

> Hans-Christian Ebke, David Bommes, Marcel Campen, and Leif Kobbelt. 2013.
> **QEx: Robust Quad Mesh Extraction.**
> *ACM Trans. Graph.* 32, 6, Article 168 (November 2013).
> DOI: [10.1145/2508363.2508372](https://doi.org/10.1145/2508363.2508372)

```bibtex
@article{Ebke:2013:QEx,
  author = {Ebke, Hans-Christian and Bommes, David and Campen, Marcel and Kobbelt, Leif},
  title = {QEx: Robust Quad Mesh Extraction},
  journal = {ACM Trans. Graph.},
  volume = {32},
  number = {6},
  year = {2013},
  pages = {168:1--168:10},
  doi = {10.1145/2508363.2508372},
  publisher = {ACM},
}
```

### Source Code

- **libQEx GitHub Repository**: [https://github.com/hcebke/libQEx](https://github.com/hcebke/libQEx)
- **License**: GPL-3.0
- **Key source file**: `src/MeshExtractorT.cc` (main algorithm, ~2500 lines)

### Related Work

- **Integer-Grid Maps**: Bommes et al. 2013, "Integer-Grid Maps for Reliable Quad Meshing", SIGGRAPH
- **OpenMesh**: Half-edge mesh library used by libQEx - [https://www.graphics.rwth-aachen.de/software/openmesh/](https://www.graphics.rwth-aachen.de/software/openmesh/)
