# Project Instructions

## Overview / Goal
- Corman & Crane rectangular parameterization (SIGGRAPH 2025) for quad meshing.
- Orthogonal (not necessarily isotropic) UVs aligned to a cross field.
- Not origami unfolding: need **compact** UVs (high fill) so integer iso-lines become quad edges.
- Goal: 0 flipped triangles + compact UV layout.

## Pipeline
Stages: `Geometry -> Cross Field -> Cut Graph -> Optimization -> UV Recovery -> [Quad Extraction]`
Phases: Load mesh -> geometry -> cross field -> cut graph -> sparse ops -> optimization -> UV recovery -> (optional) quad mesh.

**Note:** The Corman-Crane paper covers stages 1-5 (producing seamless UV parameterization). Quad extraction (stage 6) is a separate downstream step, included here for completeness.

## Implementation - `run_RSP.py`
Entry point. Line-by-line translation from official MATLAB code.
- `rectangular_surface_parameterization/preprocessing/` - MeshInfo, angles, curvature, connectivity, DEC operators
- `rectangular_surface_parameterization/cross_field/` - trivial connection, cross field computation
- `rectangular_surface_parameterization/optimization/` - reduce_corner_var_2d, optimize_RSP
- `rectangular_surface_parameterization/parameterization/` - cut_mesh, mesh_to_disk_seamless, parametrization_from_scales
- `rectangular_surface_parameterization/io/` - I/O
- `rectangular_surface_parameterization/utils/` - visualization, preprocessing

## Requirements
Python 3.8+, NumPy, SciPy, Matplotlib, trimesh. Install: `pip install numpy scipy matplotlib trimesh`

Optional for mesh preprocessing: `pip install pymeshlab`

## Commands

See **[USAGE.md](USAGE.md)** for complete CLI reference.

Quick examples:
```bash
python run_RSP.py mesh.obj -o Results/ -v          # Parameterization
python extract_quads.py mesh.obj -o Results/ --scale 10  # Full pipeline
pytest tests/ -v                                    # Run tests
```

Test meshes included in `Mesh/` folder - see [Mesh/README.md](Mesh/README.md) for details.

## Example Output

### Sphere (genus 0) - UV Layout
![Sphere UV Layout](docs/images/sphere320_uv_layout.jpg)
*Left: UV layout with triangle mesh. Right: Checkerboard pattern for distortion visualization. **0 flipped triangles.***

### Sphere - Distortion Analysis
![Sphere Distortion](docs/images/sphere320_distortion.jpg)
*Four quality metrics: Area distortion, conformal distortion, Jacobian determinant (negative = flipped), orthogonality error.*

### Sphere - Quad Mesh
![Sphere Quads](docs/examples/sphere320_smooth/sphere320_quads.jpg)
*Extracted quad mesh: 908 quads*

### Torus (genus 1) - UV Layout
![Torus UV Layout](docs/images/torus_uv_layout.jpg)
*Torus parameterization showing characteristic cut structure for genus-1 surface. **0 flipped triangles.***

## References
- MATLAB implementation: https://github.com/etcorman/RectangularSurfaceParameterization
- Paper: https://www.cs.cmu.edu/~kmcrane/Projects/RectangularSurfaceParameterization/
- Quad extraction (libQEx): https://github.com/hcebke/libQEx

## Visualization Utilities
`rectangular_surface_parameterization/io/visualize.py`:
- `plot_uv_with_flips(Xp, T, detJ)` - UV layout with flipped triangles in red
- `plot_uv_checkerboard(Xp, T, detJ)` - checkerboard pattern, flips in red
- `plot_mesh_with_flips(X, T, detJ)` - 3D mesh with flipped faces highlighted
- `save_uv_visualization(Xp, T, detJ, path)` - save 2-panel PNG
- `visualize_run_RSP_result(Src, SrcCut, Xp, disto, output_dir)` - full visualization suite
- `compute_uv_quality(Xp, T, X, T_orig)` - quality metrics (flip count, angle error)

## Current Status
| Stage | Status |
|-------|--------|
| 1. Geometry | VERIFIED (54 pytest tests pass) |
| 2. Cross Field | VERIFIED (8 singularities, sum=chi, matches MATLAB) |
| 3. Cut Graph | VERIFIED (41 cut edges, 7 tests pass) |
| 4. Optimization | VERIFIED (normalization bug fixed, 2 tests pass) |
| 5. UV Recovery | VERIFIED (**0 flips** - rotation matrix bug fixed) |

**ALL STAGES VERIFIED.** Pipeline produces 0 flipped triangles. See [Validation Approach](#validation-approach) below.

## Visual Verification

Stage visualizations are generated automatically by `run_RSP.py`:

```bash
python run_RSP.py mesh.obj -o output/ -v                    # All stages (default)
python run_RSP.py mesh.obj -o output/ -v --visualize 1,5    # Only geometry + UV
python run_RSP.py mesh.obj -o output/ -v --visualize none   # No visualizations
```

| Stage | Output Files | What to Check |
|-------|--------------|---------------|
| 1. Geometry | `stage1_mesh.jpg`, `stage1_curvature.jpg` | Mesh intact, curvature at vertices |
| 2. Cross Field | `stage2_cross_field.jpg`, `stage2_singularities.jpg` | Crosses aligned, 8 singularities for sphere |
| 3. Cut Graph | `stage3_cut_graph.jpg` | Cut edges connect all cones |
| 4. Optimization | `stage4_scales.jpg`, `stage4_distributions.jpg` | Smooth scale fields |
| 5. UV Recovery | `stage5_uv_layout.jpg`, `stage5_quality.jpg` | 0 flipped triangles (no red) |

## Validation Approach

This implementation has been **validated against the original MATLAB code** via GNU Octave 10.3.0. All three benchmark meshes (pig, B36, SquareMyles) produce structurally identical results. See [docs/octave-validation-report.md](docs/octave-validation-report.md).

See [README.md](README.md) for example outputs and the `tests/` directory for the full test suite.

## Quad Meshing

The Corman-Crane paper produces a **seamless UV parameterization** — the input for quad meshing, not the quad mesh itself.

### Recommended: QuadriFlow (via Blender)

**Use QuadriFlow for quad mesh generation.** It produces pure quad meshes with curvature-aligned edges in a single step, with no intermediate parameterization/quantization/extraction pipeline.

QuadriFlow is built into Blender (5.0+). Usage:

```bash
blender --background --python-expr "
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath='input.obj')
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
bpy.ops.object.quadriflow_remesh(
    target_faces=2000,
    use_preserve_boundary=True,
    seed=0
)
bpy.ops.wm.obj_export(filepath='output_quads.obj', export_selected_objects=True)
"
```

Results (pure quads, 0 triangles, no holes):
- Pig (1843 verts) → 2146 quads
- Bunny (2503 verts) → 1764 quads

QuadriFlow (Huang et al., SGP 2018) is MIT-licensed. Source: https://github.com/hjwdzh/QuadriFlow

### Why not RSP + libQEx?

The RSP parameterization → quantization → libQEx extraction pipeline is broken in practice:
1. RSP produces near-degenerate UV faces at cut seams (det ~1e-6)
2. The quantizer (pyquantization) rejects these faces or runs out of memory
3. Without quantization, the raw UVs have floating-point errors at every cut edge
4. libQEx (and our Python port) hit hundreds of path-tracing failures on non-trivial meshes
5. Result: 5-20% missing quads, holes everywhere

The RSP parameterization itself is valid (0 flipped triangles, verified against MATLAB/Octave). The downstream toolchain for converting UVs to quads is the broken part.

### Legacy code (kept for reference)

- `utils/quad_extractor.py` — Python port of libQEx's MeshExtractorT.cc (~1740 lines, 52 tests)
- `utils/libqex_wrapper.py` — Hole-filling utilities
- `quadmesh.py` — Full pipeline (RSP + quantization + extraction)
- `extract_quads.py` — RSP + extraction without quantization

## Mesh Preprocessing

Many real-world meshes fail the RSP pipeline due to quality issues. Use the preprocessing utilities:

```python
from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh, check_mesh_quality

# Diagnose issues
check_mesh_quality("mesh.obj")

# Clean mesh (remesh, fill holes, fix non-manifold)
preprocess_mesh("mesh.obj", "mesh_clean.obj")
```

Or use `--preprocess` flag with `extract_quads.py`.

### Robustness Fixes
The pipeline includes fixes for common mesh issues:
- **Unreferenced vertices**: Handled in `preprocessing/dec.py` (assigns small Voronoi area)
- **Curvature mismatches**: Relaxed to warnings in `preprocessing/preprocess.py`
- **Invalid quad indices**: Filtered in `utils/libqex_wrapper.py`

See `docs/robustness-improvements.md` for details.

## Docs
`docs/algo_integer_grid_maps.md`, `docs/libqex_setup.md`, `docs/robustness-improvements.md`, `docs/mesh-quality-investigation.md`

## Future Work

### Medium Priority
- **Auto-detect preprocessing needs**: Check mesh quality and auto-preprocess if needed

### Lower Priority
- **Boundary support**: Handle meshes with holes (teapot fails on Gaussian curvature check)
- **Mixed Voronoi/barycentric areas**: True mixed area computation for obtuse triangles
- **Mesh decimation**: Auto-simplify very large meshes before processing

### Known Limitations
| Mesh | Issue | Workaround |
|------|-------|------------|
| teapot | Meshes with holes | Needs boundary support |
| suzanne | Non-manifold edges | Use --preprocess |

## License

AGPL-3.0-or-later (GNU Affero General Public License v3.0 or later)

This is a derivative work of the original MATLAB implementation by Etienne Corman
and Keenan Crane. See LICENSE file for full attribution and terms.

## Project card

`project-card/` holds this project's one-liner, tags and image for the cross-project index.
When the purpose or the look of the project changes materially, regenerate it with
`/project-index Corman-Crane`.

