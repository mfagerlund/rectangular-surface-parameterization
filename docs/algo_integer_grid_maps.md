# Integer-Grid Maps for Quad Meshing

> **Note:** This document describes the algorithm for quad extraction from integer-grid maps.
> This is a **downstream step beyond the Corman-Crane paper**, which produces the seamless
> UV parameterization that serves as input to quad extraction.
>
> **Implementation:** We use [libQEx](https://github.com/hcebke/libQEx) for robust quad extraction.
> Pre-built binaries are automatically downloaded from GitHub Releases on first use.
> See `bin/BINARIES.txt` for details or `docs/libqex_setup.md` to build from source.

**Source**: Bommes et al., "Integer-Grid Maps for Reliable Quad Meshing", SIGGRAPH 2013

## Core Concept

The main principle (Figure 2 in paper):

```
Surface M  --f-->  UV Domain Ω  <--intersect-->  Integer Grid G
                        |
                   Ω ∩ G = quad vertices
                        |
                   f^(-1) maps back to surface
                        |
                   Quad mesh Q on surface
```

**Key insight**: The quad mesh is formed by the **image of integer iso-lines** (u=k and v=k curves) mapped onto the surface.

## Integer-Grid Map (IGM) Definition

An IGM is a piecewise linear map f: M_IGM → M that satisfies:

### 1. Transition Functions
When crossing from triangle t_i to neighboring triangle t_j, the transition function must be an **integer-grid automorphism**:

```
g_{i→j}(a) = R^r_{90} * a + t
```

Where:
- R^r_{90} = rotation by r × 90° (r ∈ {0, 1, 2, 3})
- t ∈ Z² (integer translation)

### 2. Singular Points at Integers
All singular vertices (cone singularities) must map to integer coordinates:

```
f^(-1)(s_i) ∈ Z²  for all singularities s_i
```

### 3. Consistent Orientation
All triangles must have positive orientation (no flips):

```
det[v - u, w - u] > 0  for all triangles (u, v, w)
```

## Quad Extraction Algorithm

Once you have a valid IGM, quad extraction is straightforward **contouring**:

### Step 1: Trace Integer Iso-lines

For each triangle with UV corners (uv0, uv1, uv2):

```python
def trace_isolines_in_triangle(triangle, corner_uvs):
    """Find where integer iso-lines cross the triangle edges."""
    segments = []

    # Find range of integers in this triangle
    u_min, u_max = min/max of u coordinates
    v_min, v_max = min/max of v coordinates

    # For each integer u value that passes through triangle
    for u_int in range(ceil(u_min), floor(u_max) + 1):
        # Find where u = u_int crosses the triangle edges
        crossings = []
        for each edge (p0, p1) with UVs (uv0, uv1):
            if u_int is between uv0.u and uv1.u:
                t = (u_int - uv0.u) / (uv1.u - uv0.u)
                crossing_point = p0 + t * (p1 - p0)
                crossing_uv = uv0 + t * (uv1 - uv0)
                crossings.append((edge_id, crossing_point, crossing_uv))

        if len(crossings) == 2:
            # Iso-line segment from crossing[0] to crossing[1]
            segments.append(IsolineSegment(u_int, crossings))

    # Same for v iso-lines
    for v_int in range(ceil(v_min), floor(v_max) + 1):
        # ... similar logic for v = v_int

    return segments
```

### Step 2: Connect Segments Across Triangles

When an iso-line exits one triangle through an edge, it enters the adjacent triangle through the same edge. **Important**: Apply the transition function when crossing cut edges!

```python
def trace_isoline_chain(mesh, corner_uvs, Gamma, zeta, iso_type, iso_value):
    """Trace a complete iso-line across the mesh."""
    chain = []

    # Start from a boundary or singularity
    current_tri, entry_edge = find_start_point(iso_type, iso_value)

    while not done:
        # Find where iso-line exits this triangle
        segment = trace_in_triangle(current_tri, iso_value)
        chain.append(segment)

        exit_edge = segment.exit_edge

        # Cross to adjacent triangle
        adjacent_tri = get_adjacent_triangle(current_tri, exit_edge)

        if Gamma[exit_edge] == 1:  # Cut edge
            # Apply transition function rotation
            # The iso-line type may change! (u becomes v after 90° rotation)
            iso_value, iso_type = apply_transition(iso_value, iso_type, zeta[exit_edge])

        current_tri = adjacent_tri

    return chain
```

### Step 3: Find Iso-line Intersections (Quad Vertices)

Within each triangle, find where u-isolines and v-isolines cross:

```python
def find_quad_vertices_in_triangle(triangle, corner_uvs):
    """Find intersection of u=k and v=l lines within triangle."""
    vertices = []

    u_segments = trace_u_isolines(triangle, corner_uvs)
    v_segments = trace_v_isolines(triangle, corner_uvs)

    for u_seg in u_segments:
        for v_seg in v_segments:
            # u_seg has constant u = u_int, v varies
            # v_seg has constant v = v_int, u varies

            # Check if (u_int, v_int) is within both segments
            if point_in_segment(u_int, v_int, u_seg) and \
               point_in_segment(u_int, v_int, v_seg):
                # Interpolate 3D position
                p = interpolate_3d_position(triangle, u_int, v_int)
                vertices.append((u_int, v_int, p))

    return vertices
```

### Step 4: Build Quad Connectivity

Connect adjacent integer points to form quads:

```python
def build_quads(all_vertices):
    """Build quad faces from integer grid vertices."""
    vertex_map = {}  # (u, v) -> vertex_index

    for u, v, pos in all_vertices:
        vertex_map[(u, v)] = len(vertex_map)

    quads = []
    for (u, v) in vertex_map:
        # Check if all 4 corners of this cell exist
        corners = [(u, v), (u+1, v), (u+1, v+1), (u, v+1)]
        if all(c in vertex_map for c in corners):
            quad = [vertex_map[c] for c in corners]
            quads.append(quad)

    return quads
```

## Handling Transition Functions (Cut Edges)

This is the tricky part. When tracing across a cut edge with rotation r and translation t:

```python
def apply_transition(uv, r, t):
    """Apply transition function: g(uv) = R^r * uv + t"""
    # Rotation matrices for r ∈ {0, 1, 2, 3}
    # r=0: identity
    # r=1: 90° CCW  -> (u,v) -> (-v, u)
    # r=2: 180°    -> (u,v) -> (-u, -v)
    # r=3: 270° CCW -> (u,v) -> (v, -u)

    if r == 0:
        rotated = uv
    elif r == 1:
        rotated = (-uv[1], uv[0])
    elif r == 2:
        rotated = (-uv[0], -uv[1])
    elif r == 3:
        rotated = (uv[1], -uv[0])

    return (rotated[0] + t[0], rotated[1] + t[1])
```

**Important**: When r is odd (1 or 3), u-isolines become v-isolines and vice versa!

## Key Observations

1. **The quad mesh is implicit in the IGM** - once you have a valid parameterization where singularities are at integer positions, the quad mesh exists.

2. **Iso-line tracing is local** - each triangle can be processed independently, then segments are connected.

3. **Transition functions preserve the integer grid** - that's why they must be rotations by 90° multiples plus integer translations.

4. **Quad vertices = integer coordinate points** - every point (i, j) where i, j ∈ Z that lies in the UV domain becomes a quad vertex.

5. **Quad edges = iso-line segments between adjacent integer points** - the segment of u=k between (k, j) and (k, j+1).

## Complexity

- O(F) to trace iso-lines through all faces
- O(F) to find all intersections
- O(V_quad) to build connectivity

Where F = number of triangles, V_quad = number of quad vertices ≈ target_quads.
