# Phase 2: Testing & Quad Extraction

## Current Status

Rectangular parameterization is working on sphere320.obj:
- 0 flipped triangles
- 14.04° mean angle error
- 5 iteration convergence

## Phase 2A: Test on More Meshes

### Test Meshes

| Mesh | V | F | Topology | Flips | Angle Error | Status |
|------|---|---|----------|-------|-------------|--------|
| sphere320.obj | 162 | 320 | Genus 0 | 0 | 14.04° | ✓ Pass |
| torus.obj | 576 | 1152 | Genus 1 | 0 | 10.74° | ✓ Pass |
| stanford-bunny.obj | ~35k | ~70k | Genus 0 | - | - | Too slow |

### Performance Note

Current implementation is too slow for large meshes (>5k faces). The bunny (~70k faces) times out. Optimization opportunities:
- Sparse Cholesky instead of dense solve
- Vectorized constraint evaluation
- Numba/Cython for inner loops

### Success Criteria

- [x] Sphere: 0 flipped triangles ✓
- [x] Torus: 0 flipped triangles, genus 1 handled ✓
- [ ] Bunny: Requires performance optimization

### Commands

```bash
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/stanford-bunny.obj" -o output/bunny_uv.obj -v
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/torus.obj" -o output/torus_uv.obj -v
```

---

## Phase 2B: Quad Mesh Extraction

The rectangular parameterization produces UV coordinates aligned with a cross field. To get actual quad meshes, we need:

### Step 1: Integer-Grid Mapping

Snap UV coordinates to an integer grid while preserving:
- Orthogonality of iso-lines
- Alignment with cross field singularities
- Minimal distortion

**Approach**: Scale UVs so average quad has unit size, then round to integers at singularities.

### Step 2: Iso-Line Tracing

Extract curves where u=const and v=const on the surface:
- Start from singularities and boundaries
- Trace along the surface following the parameterization gradient
- Handle crossings at integer coordinates

### Step 3: Quad Extraction

Build quad mesh from iso-line intersections:
- Vertices: intersection points of u=const and v=const lines
- Edges: segments of iso-lines between intersections
- Faces: regions bounded by four edges

### References

Available locally:
- `D:\Data\GDrive\FlatrPDFs\2461912.2462014_integer-grid-maps-reliable-quad-meshing.pdf` - Bommes et al., SIGGRAPH 2013
- `D:\Data\GDrive\FlatrPDFs\1531326.1531383_mixed-integer-quadrangulation.pdf` - Bommes et al., SIGGRAPH 2009

Also relevant:
- Campen et al., "Quantized Global Parametrization", SIGGRAPH 2015

### Output

New file: `quad_extract.py`
- Input: triangle mesh + corner UVs
- Output: quad mesh (V, F where F is Nx4)

---

## Files to Create

```
quad_extract.py     # Quad mesh extraction from parameterization
test_meshes.py      # Automated tests on bunny, torus, etc.
```

## Milestones

| Task | Status |
|------|--------|
| Test bunny | Pending |
| Test torus | Pending |
| Integer-grid mapping | Pending |
| Iso-line tracing | Pending |
| Quad extraction | Pending |
| End-to-end quad mesh | Pending |
