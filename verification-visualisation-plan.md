# Verification Visualisation Plan

Implementation plan for `Utils/verify_pipeline.py` - a unified visual verification tool for each pipeline stage.

## Goal

Provide step-by-step visual confirmation that each pipeline stage produces correct output. Each visualization should be:
1. Automatically generated (no manual plotting)
2. Saved to predictable filenames
3. Self-documenting (what to look for is clear from the image)

## Architecture

```
Utils/verify_pipeline.py
├── verify_geometry(Src, output_dir)      -> stage1_*.png
├── verify_cross_field(Src, ang, sing)    -> stage2_*.png
├── verify_cut_graph(Src, cut_edges, k21) -> stage3_*.png
├── verify_optimization(Src, u, v, theta) -> stage4_*.png
├── verify_uv_recovery(Xp, T, detJ)       -> stage5_*.png
└── verify_all(mesh_path, output_dir)     -> runs full pipeline
```

## CLI Interface

```bash
# Full pipeline verification
python Utils/verify_pipeline.py <mesh.obj> -o output/

# Single stage (requires intermediate data)
python Utils/verify_pipeline.py <mesh.obj> -o output/ --stage 2
```

## Stage 1: Geometry

**Inputs:** `Src` (MeshInfo)

**Outputs:**
- `stage1_mesh.png` - 3D mesh wireframe from two angles
- `stage1_curvature.png` - Vertex colors = discrete Gaussian curvature (angle defect)

**Implementation:**
```python
def verify_geometry(Src, output_dir):
    # Mesh wireframe - use matplotlib 3D with edges
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    # Plot mesh from two views

    # Curvature - vertex colors based on angle defect
    # angle_defect = 2*pi - sum of angles at vertex
    # For sphere: should be uniform positive
```

**What to check:**
- Mesh topology intact (no missing faces)
- Sphere: uniform curvature everywhere
- Cube: high curvature at corners only

---

## Stage 2: Cross Field

**Inputs:** `Src`, `ang` (per-face angle), `sing` (per-vertex singularity index)

**Outputs:**
- `stage2_cross_field.png` - Cross glyphs on face centroids
- `stage2_singularities.png` - Mesh with singularity markers

**Implementation:**
```python
def verify_cross_field(Src, param, ang, sing, output_dir):
    # Cross field - draw 4-way cross at each face centroid
    # Cross orientation from ang[face]
    # Use quiver or custom line segments

    # Singularities - mark vertices where |sing| > 0.1
    # Red = positive index (+1/4), Blue = negative index (-1/4)
    # Size proportional to |sing|
```

**What to check:**
- Crosses rotate smoothly across faces
- Singularities at expected locations (8 for sphere, 0 for torus)
- Sum of indices = Euler characteristic

---

## Stage 3: Cut Graph

**Inputs:** `Src`, `cut_edges` (list of edge indices), `cones` (cone vertex indices)

**Outputs:**
- `stage3_cut_graph.png` - Mesh with cut edges highlighted in red, cones marked

**Implementation:**
```python
def verify_cut_graph(Src, cut_edges, cones, output_dir):
    # Draw mesh wireframe (light gray)
    # Overlay cut edges in red (thicker)
    # Mark cone vertices with large dots
    # Verify: cut graph is connected and includes all cones
```

**What to check:**
- Cut edges form a connected tree
- All cone singularities lie on the cut
- No unnecessary loops (minimal cut)

---

## Stage 4: Optimization

**Inputs:** `Src`, `u` (scale), `v` (anisotropy), `theta` (angle adjustment)

**Outputs:**
- `stage4_scale_u.png` - Face colors = u values
- `stage4_scale_v.png` - Face colors = v values
- `stage4_integrability.png` - Integrability error per face (should be ~0)

**Implementation:**
```python
def verify_optimization(Src, u, v, theta, output_dir):
    # Color faces by u value (colormap: coolwarm centered at 0)
    # Color faces by v value
    # Compute integrability error, color by magnitude
```

**What to check:**
- u, v fields are smooth (no discontinuities except at cuts)
- Integrability error near zero everywhere
- No NaN or extreme values

---

## Stage 5: UV Recovery

**Inputs:** `Xp` (UV coords), `T` (faces), `detJ` (Jacobian determinants)

**Outputs:**
- `stage5_uv_layout.png` - 2D triangle layout, flipped = red
- `stage5_checkerboard.png` - UV checkerboard texture

**Implementation:**
```python
def verify_uv_recovery(Xp, T, detJ, output_dir):
    # Use existing Utils/visualize_uv.py functions
    # plot_uv_with_flips for layout
    # plot_uv_checkerboard for texture
```

**What to check:**
- 0 red triangles (no flips)
- Triangles fill UV space compactly
- Checkerboard is regular (no severe distortion)

---

## Expected Results: sphere320.obj

| Stage | Key Metric | Expected Value |
|-------|------------|----------------|
| 1 | Euler characteristic | 2 (genus 0) |
| 2 | Singularity count | 8 |
| 2 | Sum of indices | 2.0 |
| 3 | Cut edge count | ~41 |
| 4 | Max integrability error | < 1e-10 |
| 5 | Flipped triangles | 0 |

---

## Implementation Tasks

1. [ ] Create `Utils/verify_pipeline.py` with CLI argument parsing
2. [ ] Implement `verify_geometry()` - wireframe + curvature
3. [ ] Implement `verify_cross_field()` - cross glyphs + singularities
4. [ ] Implement `verify_cut_graph()` - cut edges + cones
5. [ ] Implement `verify_optimization()` - u, v, integrability heatmaps
6. [ ] Implement `verify_uv_recovery()` - wrap existing visualize_uv functions
7. [ ] Implement `verify_all()` - run full pipeline and call all verify functions
8. [ ] Add test: `test_verify_pipeline.py` - runs on sphere320, checks files exist
9. [ ] Generate reference images for sphere320 and commit to `docs/reference_images/`

---

## Dependencies

- matplotlib (3D plotting, colormaps)
- numpy (data manipulation)
- Existing: `Utils/visualize_uv.py` (reuse for stage 5)

## Notes

- All plots should have titles indicating stage and what's shown
- Colorbars where applicable
- Save at 150 DPI for reasonable file size
- Non-interactive (plt.savefig, not plt.show)
