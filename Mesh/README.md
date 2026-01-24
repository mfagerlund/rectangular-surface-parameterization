# Test Meshes

This folder contains test meshes for the RSP pipeline. The meshes from the original
MATLAB implementation (B36, pig, SquareMyles) are included for compatibility testing.

## Mesh Summary

| Mesh | Vertices | Faces | Genus | Source | Notes |
|------|----------|-------|-------|--------|-------|
| sphere320.obj | 162 | 320 | 0 | Standard test | Icosphere, simplest test case |
| torus.obj | 576 | 576 | 1 | Standard test | Clean torus, tests genus-1 handling |
| pig.obj | 1,843 | 3,560 | 0 | MATLAB repo | Used in curvature-aligned example |
| B36.obj | 2,200 | 4,396 | 0 | MATLAB repo | Used in smooth cross field example |
| SquareMyles.obj | 706 | 1,328 | 0 | MATLAB repo | Used in Chebyshev net example |

## Recommended Usage

From the MATLAB README, these are the canonical examples:

### 1. Smooth cross field with hard edges (B36)
```bash
python run_RSP.py Mesh/B36.obj -o Results/ --frame-field smooth -v
```

### 2. Curvature-aligned parameterization (pig)
```bash
python run_RSP.py Mesh/pig.obj -o Results/ --frame-field curvature --energy alignment -v
```

### 3. Chebyshev net with boundary alignment (SquareMyles)
```bash
python run_RSP.py Mesh/SquareMyles.obj -o Results/ --frame-field trivial --energy chebyshev -v
```

### 4. Simple verification (sphere, torus)
```bash
python run_RSP.py Mesh/sphere320.obj -o Results/ -v
python run_RSP.py Mesh/torus.obj -o Results/ -v
```

## Expected Results

### sphere320.obj
- **Singularities:** 8 positive (index +1/4 each), sum = 2 = Euler characteristic
- **Cross field:** Smooth field with singularities at roughly octahedral positions
- **UV layout:** Should produce 0 flipped triangles
- **Genus:** 0 (sphere topology)

### torus.obj
- **Singularities:** 0 (flat Euler characteristic for genus-1)
- **Cross field:** Smooth wrapping field
- **UV layout:** Should produce 0 flipped triangles
- **Genus:** 1 (torus topology)

### pig.obj
- **Singularities:** Varies by cross field type
- **Best with:** `--frame-field curvature --energy alignment`
- **Notes:** Organic shape, good for demonstrating curvature alignment

### B36.obj
- **Singularities:** Concentrated at high-curvature regions
- **Best with:** `--frame-field smooth` with hard edge detection
- **Notes:** Mechanical part with sharp features

### SquareMyles.obj
- **Best with:** `--frame-field trivial --energy chebyshev`
- **Notes:** Designed for Chebyshev net demonstration

## Known Issues / Failure Modes

### Meshes that may require preprocessing

Some meshes have quality issues that can cause pipeline failures:

| Issue | Symptom | Solution |
|-------|---------|----------|
| Non-manifold edges | `ValueError` during connectivity | Use `--preprocess` |
| Unreferenced vertices | Warning, may cause `d1d has zero rows` | Use `--preprocess` |
| Disconnected components | `Mesh has disconnected components` | Use `--preprocess` |
| Holes/boundaries | `Gaussian curvature incompatible` warning | May need boundary support |
| Very obtuse triangles | `Non Delaunay mesh` warning | Usually OK, but may affect convergence |

### Cross field solver failures

The cross field computation can fail on certain meshes:

| Mesh Type | Issue | Workaround |
|-----------|-------|------------|
| High-genus meshes | Singular matrix in eigensolver | Try different `--frame-field` |
| Very coarse meshes | Insufficient resolution for smooth field | Subdivide mesh first |
| Meshes with spikes | Extreme curvature causes instability | Smooth/remesh first |

## Licensing

- **sphere320.obj, torus.obj:** Public domain / freely usable
- **B36.obj, pig.obj, SquareMyles.obj:** Included in the original MATLAB implementation by
  Etienne Corman and Keenan Crane. The MATLAB code is AGPL-3.0-or-later, but the meshes
  themselves have no explicit license. For commercial use, contact the original authors.

## Adding New Test Meshes

When adding meshes for testing, check:

1. **Manifold:** Each edge shared by exactly 2 faces
2. **Closed:** No boundary edges (unless testing boundary handling)
3. **Connected:** Single connected component
4. **Reasonable resolution:** 500-5000 faces is good for testing
5. **License:** Verify redistribution is permitted
