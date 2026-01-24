# Robustness Improvements

This document describes improvements made to handle meshes that previously failed.

## Changes Made

### 1. Mesh Preprocessing Utility (`rectangular_surface_parameterization/utils/preprocess_mesh.py`)

**New file** that uses PyMeshLab to clean meshes before running RSP.

**Features:**
- `preprocess_mesh()` - Full preprocessing pipeline:
  - Remove duplicate vertices/faces
  - Remove unreferenced vertices
  - Remove zero-area faces
  - Repair non-manifold edges/vertices
  - Close small holes
  - Re-orient faces consistently
  - Isotropic remeshing for better triangle quality

- `check_mesh_quality()` - Diagnostic tool:
  - Reports vertex/face counts
  - Checks for non-manifold geometry
  - Checks for boundary edges (holes)
  - Reports topological measures (genus, Euler characteristic)

- `make_delaunay()` - Apply Delaunay triangulation:
  - Flips edges to improve triangle quality
  - Helps avoid negative Voronoi areas

**Usage:**
```python
from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh, check_mesh_quality

# Check mesh quality
check_mesh_quality("bunny.obj")

# Preprocess mesh
clean_path = preprocess_mesh("bunny.obj", "bunny_clean.obj")
```

**Command line:**
```bash
python -m rectangular_surface_parameterization.utils.preprocess_mesh bunny.obj bunny_clean.obj
```

**Requirements:**
```bash
pip install pymeshlab
```

### 2. DEC Operator Fixes (`rectangular_surface_parameterization/preprocessing/dec.py`)

**Problem:** Meshes with unreferenced vertices caused "Negative vertex area" assertion.

**Root cause:** The Voronoi area computation uses `np.bincount` which returns 0 for vertices that don't appear in any triangle. The assertion `vor_area > 0` then fails.

**Fix:** Added handling for unreferenced vertices (lines 207-220):

```python
# Check for unreferenced vertices (vor_area = 0) and handle them
unreferenced = (vor_area == 0)
n_unreferenced = np.sum(unreferenced)
if n_unreferenced > 0:
    warnings.warn(f"Mesh has {n_unreferenced} unreferenced vertices. Assigning small area.")
    # Assign a small positive area to unreferenced vertices
    min_positive_area = vor_area[vor_area > 0].min() if np.any(vor_area > 0) else 1e-10
    vor_area[unreferenced] = min_positive_area * 1e-3

# Check for truly negative areas (shouldn't happen with barycentric subdivision)
if np.any(vor_area < 0):
    warnings.warn("Negative vertex areas detected. Using absolute values.")
    vor_area = np.abs(vor_area)
```

**Impact:** Stanford bunny (35,947 vertices with 1,113 unreferenced) now passes DEC computation.

### 3. libQEx Wrapper Fixes (`rectangular_surface_parameterization/utils/libqex_wrapper.py`)

**Problem:** Some extracted quads had invalid vertex indices or were degenerate.

**Fix:** Added filtering for:
- All-zero quads (`0 0 0 0`)
- Quads with duplicate vertices
- Quads with out-of-range vertex indices

```python
# Skip quads with out-of-range vertex indices
if any(idx < 0 or idx >= n_quad_verts for idx in q):
    continue
```

## Test Results

### Before fixes:
| Mesh | Status | Error |
|------|--------|-------|
| stanford-bunny.obj | ✗ FAIL | Negative vertex area |
| cow.obj | ✗ FAIL | Singular matrix |
| spot.obj | ✗ FAIL | Singular matrix |
| teapot.obj | ✗ FAIL | Gaussian curvature mismatch |
| suzanne.obj | ✗ FAIL | Non-manifold edges |

### After fixes:
| Mesh | Status | Notes |
|------|--------|-------|
| stanford-bunny.obj | ⏳ Testing | DEC passes, preprocessing running |
| cow.obj | TBD | May need preprocessing |
| spot.obj | TBD | May need preprocessing |
| teapot.obj | TBD | Topology issue (holes) |
| suzanne.obj | TBD | Needs preprocessing for non-manifold |

## Recommended Workflow

For meshes that fail:

1. **Check quality first:**
   ```python
   from rectangular_surface_parameterization.utils.preprocess_mesh import check_mesh_quality
   check_mesh_quality("problematic_mesh.obj")
   ```

2. **Preprocess if needed:**
   ```python
   from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh
   preprocess_mesh("problematic_mesh.obj", "mesh_clean.obj")
   ```

3. **Run pipeline:**
   ```bash
   python extract_quads.py mesh_clean.obj -o Results/ --scale 10
   ```

## Future Improvements

1. **Auto-preprocessing in extract_quads.py** - Detect mesh issues and auto-clean
2. **Better cross-field solver** - Handle singular matrices gracefully
3. **Boundary handling** - Support meshes with holes
4. **Mixed Voronoi/barycentric areas** - True mixed area computation for obtuse triangles
