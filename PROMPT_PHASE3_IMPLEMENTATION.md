# Task: Implement Correct Quad Extraction Algorithm

## Context

This is a Corman-Crane rectangular parameterization project that converts triangle meshes to quad meshes. The parameterization (UV mapping) is working, but the quad extraction has a fundamental bug causing only ~30% coverage.

## The Problem

Current `quad_extract.py` uses the **wrong approach**:
- Iterates through integer grid points (u, v)
- For each point, searches for a triangle that contains it
- Misses many valid points near triangle boundaries

## The Solution (from Bommes et al. 2013)

Use the **correct approach**:
- Iterate through each triangle
- For each triangle, find which integer (u, v) points lie inside it
- This guarantees finding ALL valid quad vertices

## Algorithm to Implement

```python
def build_quad_mesh_correct(mesh, corner_uvs, target_quads):
    """
    Build quad mesh by finding integer points inside each triangle.

    For each triangle:
        1. Get UV coordinates of the 3 corners
        2. Find bounding box: u_min, u_max, v_min, v_max
        3. For each integer (u, v) in that range:
            - Test if (u, v) is inside triangle (barycentric coords)
            - If inside, compute 3D position via barycentric interpolation
            - Store vertex: (u, v) -> 3D position

    Build quads:
        For each (u, v) in vertex map:
            If (u, v), (u+1, v), (u+1, v+1), (u, v+1) all exist:
                Create quad face
    """
```

## Key Files

Read these first:
1. `docs/algo_integer_grid_maps.md` - Extracted algorithm with full pseudocode
2. `PLAN_PHASE3_QUAD_EXTRACTION.md` - Implementation plan
3. `quad_extract.py` - Current (wrong) implementation to replace

Supporting files:
- `mesh.py` - TriangleMesh data structure
- `corman_crane.py` - Main entry point
- `test_meshes.py` - Test suite

## What to Change

Replace `build_quad_mesh()` in `quad_extract.py` with the correct algorithm. The key change:

**Before (wrong):**
```python
for u_int in u_ints:
    for v_int in v_ints:
        # Search ALL triangles for one containing this point
        for f in range(mesh.n_faces):
            if point_in_triangle(u_int, v_int, triangle_f):
                ...
```

**After (correct):**
```python
for f in range(mesh.n_faces):
    # Get integer points that COULD be in this triangle
    u_min, u_max = get_u_range(triangle_f)
    v_min, v_max = get_v_range(triangle_f)

    for u_int in range(ceil(u_min), floor(u_max) + 1):
        for v_int in range(ceil(v_min), floor(v_max) + 1):
            if point_in_triangle(u_int, v_int, triangle_f):
                ...
```

## Expected Results

| Mesh | Current | Expected |
|------|---------|----------|
| Sphere | 15 quads (34% coverage) | ~50 quads (~100% coverage) |
| Torus | 66 quads (31% coverage) | ~100 quads (~100% coverage) |

## Test Command

```bash
cd C:/Dev/Corman-Crane
python test_meshes.py
```

## Notes

- The `point_in_triangle_uv()` function already exists and works correctly
- The `uv_to_3d()` function already exists for barycentric interpolation
- Scale factor calculation is correct, don't change it
- No need for periodic wrapping with the correct approach - it handles all topologies naturally

## Don't Change

- `uv_recovery.py` - UV parameterization is working
- `corman_crane.py` - Main pipeline is fine
- Test pass criteria - keep the same thresholds
