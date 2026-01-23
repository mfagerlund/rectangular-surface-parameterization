# Phase 3: Proper Quad Extraction (Bommes et al.)

## Goal

Replace naive grid-sampling with proper iso-line tracing algorithm from Bommes et al. papers.

## Phase 3A: Extract Algorithms from Papers

### Step 1: Read and Extract from Integer-Grid Maps (2013)

**Source**: `D:\Data\GDrive\FlatrPDFs\2461912.2462014_integer-grid-maps-reliable-quad-meshing.pdf`

**Output**: `docs/algo_integer_grid_maps.md`

Extract:
- [ ] Problem definition and inputs/outputs
- [ ] Iso-line tracing algorithm (how to trace u=const and v=const through triangles)
- [ ] Handling transitions across edges (rotation jumps)
- [ ] Finding iso-line intersections (quad vertices)
- [ ] Building quad connectivity from intersections
- [ ] Handling singularities and boundary cases
- [ ] Pseudocode for key algorithms

### Step 2: Read and Extract from Mixed-Integer Quadrangulation (2009)

**Source**: `D:\Data\GDrive\FlatrPDFs\1531326.1531383_mixed-integer-quadrangulation.pdf`

**Output**: `docs/algo_mixed_integer_quad.md`

Extract:
- [ ] How singularities affect quad mesh validity
- [ ] Integer constraints for singularity positions
- [ ] The optimization formulation
- [ ] Any relevant iso-line concepts that complement the 2013 paper

---

## Phase 3B: Implementation Plan

### Key Insight from Paper

**Current (wrong) approach**: Sample integer grid points, find containing triangles
**Correct approach**: Trace iso-lines through triangles, find their intersections

The iso-line approach works because:
- Finds ALL quad vertices regardless of UV domain shape
- Works with any topology (handles torus, etc.)
- Each triangle is processed locally, then connected

### Algorithm Overview

```
For each triangle:
    1. Find where u=k lines cross edges (for all integer k in range)
    2. Find where v=l lines cross edges (for all integer l in range)
    3. Find intersections of u-lines and v-lines WITHIN the triangle
    4. These intersections are quad vertices at (k, l)

Connect vertices:
    5. For each cell (k, l), if all 4 corners exist, create a quad
```

### Implementation Steps

1. **Iso-line Edge Crossing** (`isoline.py`)
   ```python
   def find_edge_crossings(uv0, uv1, iso_value, is_u_isoline):
       """Find where iso-line crosses edge from uv0 to uv1."""
       # Returns t parameter where crossing occurs
   ```

2. **Iso-line Segments in Triangle**
   ```python
   def trace_isolines_in_triangle(face_idx, corner_uvs):
       """Find all iso-line segments passing through triangle."""
       # For each integer u in [u_min, u_max]: find crossings
       # For each integer v in [v_min, v_max]: find crossings
       # Returns list of (iso_type, iso_value, entry_point, exit_point)
   ```

3. **Find Quad Vertices (Intersections)**
   ```python
   def find_quad_vertices_in_triangle(face_idx, corner_uvs, mesh):
       """Find where u and v iso-lines intersect within triangle."""
       # For each (u_int, v_int) that could be in triangle:
       #   Check if point is inside triangle
       #   Compute 3D position via barycentric interpolation
       # Returns list of (u_int, v_int, 3d_position)
   ```

4. **Build Quad Mesh**
   ```python
   def build_quad_mesh_isoline(mesh, corner_uvs):
       """Build quad mesh by finding all iso-line intersections."""
       vertices = {}  # (u, v) -> 3D position

       for each triangle:
           for each (u, v, pos) in find_quad_vertices_in_triangle(...):
               vertices[(u, v)] = pos

       # Build quads where all 4 corners exist
       quads = []
       for (u, v) in vertices:
           if all corners (u,v), (u+1,v), (u+1,v+1), (u,v+1) exist:
               quads.append(...)

       return QuadMesh(vertices, quads)
   ```

### Why This Works Better

| Current Approach | Iso-line Approach |
|------------------|-------------------|
| Sample (u,v) points, find containing triangle | Find (u,v) points that exist in each triangle |
| Misses points near boundaries | Finds ALL valid points |
| Periodic wrapping is complex | No special handling needed |
| ~30% coverage | ~100% coverage |

### Key Difference

**Current**: "Is point (3, 5) inside any triangle?" - May fail near boundaries
**Iso-line**: "Which integer points are inside triangle T?" - Always finds them

---

## Current Status

- [x] Phase 3A Step 1: Extract Integer-Grid Maps algorithm → `docs/algo_integer_grid_maps.md`
- [ ] Phase 3A Step 2: Extract Mixed-Integer algorithm (optional - mainly about optimization, not extraction)
- [ ] Phase 3B: Implementation

## Files to Create

```
docs/
  algo_integer_grid_maps.md    # Extracted algorithm from 2013 paper
  algo_mixed_integer_quad.md   # Extracted algorithm from 2009 paper
```

## Commands

```bash
# After algorithm extraction, run:
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/torus.obj" -o output/torus_uv.obj --quads 100

# Expected: ~100 quads (vs current 66) with full surface coverage
```
