# Mesh Quality Investigation

This document summarizes the mesh quality issues discovered when testing the RSP pipeline on various meshes.

## Summary

| Mesh | Vertices | Faces | Status | Issue |
|------|----------|-------|--------|-------|
| sphere320.obj | 162 | 320 | ✓ Works | Clean manifold mesh |
| torus (generated) | 512 | 1024 | ✓ Works | Clean manifold mesh |
| stanford-bunny.obj | 35,947 | 69,451 | ✗ Fails | Negative Voronoi area |
| cow.obj | ~2,900 | ~5,800 | ✗ Fails | Singular matrix in cross field |
| spot.obj | ~4,000 | ~8,000 | ✗ Fails | Singular matrix in cross field |
| teapot.obj | ~3,300 | ~6,600 | ✗ Fails | Gaussian curvature mismatch |
| suzanne.obj | ~500 | ~1,000 | ✗ Fails | Non-manifold edges |

## Detailed Issues

### stanford-bunny.obj

**Error:**
```
AssertionError: Negative vertex area.
```

**Location:** `preprocessing/dec.py:207`

**Cause:** The Voronoi area calculation produces negative values. This typically happens when:
- The mesh has non-Delaunay triangulation (obtuse triangles)
- There are nearly degenerate triangles
- The mesh has very poor triangle quality

**Note:** The bunny mesh has no degenerate triangles (all areas > 1e-8), but the Voronoi dual area computation is sensitive to triangle shape.

### cow.obj and spot.obj

**Error:**
```
MatrixRankWarning: Matrix is exactly singular
AssertionError: NaN vector field.
```

**Location:** `cross_field/face_field.py:294`

**Warnings before failure:**
- `Non Delaunay mesh: risk of convergence issues!`
- `overflow encountered in scalar multiply`

**Cause:** The cross field computation solves a linear system that becomes singular. This happens when:
- The mesh has poor triangle quality (non-Delaunay)
- There are nearly degenerate configurations
- The Laplacian matrix has numerical issues

### teapot.obj

**Error:**
```
AssertionError: Gaussian curvature does not match topology.
```

**Location:** `preprocessing/preprocess.py:635`

**Cause:** The sum of discrete Gaussian curvature should equal 2πχ (where χ is the Euler characteristic). This fails when:
- The mesh has boundaries (holes)
- The mesh is not closed
- There are topological inconsistencies

**Note:** The Utah teapot traditionally has a hole at the bottom and the lid is often separate geometry.

### suzanne.obj

**Error:**
```
ValueError: cannot reshape array of size 2946 into shape (1472,2)
```

**Location:** `preprocessing/connectivity.py:132`

**Warning before failure:**
- `Trivially triangulating high degree facets`

**Cause:** The mesh has non-manifold edges (edges shared by more than 2 triangles). The connectivity computation expects each edge to have exactly 2 adjacent triangles.

**Note:** Suzanne (Blender monkey) often has non-manifold geometry at the eyes and ears.

## Requirements for RSP Pipeline

For the RSP pipeline to work correctly, the input mesh must be:

1. **Manifold** - Each edge shared by exactly 2 triangles
2. **Closed** - No boundary edges (or boundaries must be explicitly handled)
3. **Well-shaped triangles** - Avoid very obtuse or skinny triangles
4. **Consistent orientation** - All face normals pointing outward
5. **No degenerate triangles** - All triangles must have positive area

## Recommendations

### Preprocessing Steps

Before running the RSP pipeline, consider:

1. **Remeshing** - Use MeshLab, Blender, or similar to improve triangle quality
2. **Hole filling** - Close any boundaries
3. **Manifold repair** - Fix non-manifold edges/vertices
4. **Intrinsic Delaunay** - Flip edges to achieve Delaunay triangulation

### MeshLab Workflow

```
Filters > Remeshing > Isotropic Explicit Remeshing
  - Target edge length: ~average edge length
  - Iterations: 5-10
  - Check "Refine" and "Collapse"

Filters > Cleaning > Remove Unreferenced Vertices
Filters > Cleaning > Remove Duplicate Vertices
Filters > Cleaning > Remove Duplicate Faces
```

### Programmatic Options

- **PyMeshLab** - Python bindings for MeshLab
- **trimesh** - `trimesh.repair.fix_normals()`, `trimesh.repair.fill_holes()`
- **libigl** - Intrinsic Delaunay triangulation

## Working Test Meshes

The following meshes are known to work:

1. **sphere320.obj** - Icosphere, 162 vertices, genus 0
2. **Generated torus** - 512 vertices, genus 1 (created by `extract_quads.py` test)

To generate a clean torus:
```python
# See extract_quads.py or use:
python -c "
import numpy as np
R, r = 1.0, 0.4  # Major/minor radius
n_major, n_minor = 32, 16
# ... (see Results/torus.obj for output)
"
```
