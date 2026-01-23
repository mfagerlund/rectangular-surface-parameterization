# CLAUDE.md

## DO NOT RUSH
- Read `verification-plan.md` before anything.
- Verify each pipeline stage with tests + visualizations; get explicit signoff before next stage.
- If a stage fails, fix it before touching downstream.

## Overview / Goal
- Corman & Crane rectangular parameterization (SIGGRAPH 2025) for quad meshing.
- Orthogonal (not necessarily isotropic) UVs aligned to a cross field.
- Not origami unfolding: need **compact** UVs (high fill) so integer iso-lines become quad edges.
- Goal: 0 flipped triangles + compact UV layout.

## Pipeline
Stages: `Geometry -> Cross Field -> Cut Graph -> Optimization -> UV Recovery`
Phases: Load mesh -> geometry -> cross field -> cut graph -> sparse ops -> optimization -> UV recovery.

## Two Implementations

### MATLAB-ported (PRIMARY) - `run_RSP.py`
Entry point for production use. Line-by-line translation from official MATLAB code.
- `Preprocess/` - MeshInfo, angles, curvature, connectivity, DEC operators
- `FrameField/` - trivial connection, cross field computation
- `Orthotropic/` - optimization (reduce_corner_var_2d, optimize_RSP)
- `ComputeParam/` - cut_mesh, mesh_to_disk_seamless, parametrization_from_scales
- `Utils/` - I/O, visualization

### Old implementation (DEPRECATED) - `corman_crane.py`
Done without MATLAB reference. Contains hacks/cheats to reduce flips - DO NOT USE as reference.
- `mesh.py`, `geometry.py`, `cross_field.py`, `cut_graph.py`, `optimization.py`, `sparse_ops.py`, `uv_recovery.py`
- Lower flip count achieved through algorithm deviations, not correct fixes

## Requirements
Python 3.8+, NumPy, SciPy, Matplotlib. Install: `pip install numpy scipy matplotlib`

## Commands

### MATLAB-ported pipeline (PRIMARY)
```bash
# Basic run
python run_RSP.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o Results/ -v

# With UV visualization PNGs (shows flipped faces in red)
python run_RSP.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o Results/ -v --save-viz

# Interactive plots
python run_RSP.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o Results/ -v --plot
```
Output files: `Results/<mesh>_param.obj`, `Results/uv_layout.png`, `Results/mesh_flips.png`, `Results/distortion.png`

### Verification
```bash
python verify_pipeline.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" --stage geometry
pytest tests/ -v
```

Test meshes: `C:/Dev/Colonel/Data/Meshes/sphere320.obj` (genus 0), `C:/Dev/Colonel/Data/Meshes/torus.obj` (genus 1).

## References (READ)
MATLAB impl: https://github.com/etcorman/RectangularSurfaceParameterization (local: `C:\Slask\RectangularSurfaceParameterization`)
Paper: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025.pdf`
Supplement: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025-supplement.pdf`
Quad extraction: `D:\Data\GDrive\FlatrPDFs\2461912.2462014_integer-grid-maps-reliable-quad-meshing.pdf`,
`D:\Data\GDrive\FlatrPDFs\1531326.1531383_mixed-integer-quadrangulation.pdf`

## Visualization Utilities
`Utils/visualize_uv.py` - works with MATLAB-ported implementation:
- `plot_uv_with_flips(Xp, T, detJ)` - UV layout with flipped triangles in red
- `plot_uv_checkerboard(Xp, T, detJ)` - checkerboard pattern, flips in red
- `plot_mesh_with_flips(X, T, detJ)` - 3D mesh with flipped faces highlighted
- `save_uv_visualization(Xp, T, detJ, path)` - save 2-panel PNG
- `visualize_run_RSP_result(Src, SrcCut, Xp, disto, output_dir)` - full visualization suite
- `compute_uv_quality(Xp, T, X, T_orig)` - quality metrics (flip count, angle error)

## Current Status
See `verification-plan.md`. Summary:
| Stage | Status |
|-------|--------|
| 1. Geometry | VERIFIED (54 pytest tests pass) |
| 2. Cross Field | VERIFIED (8 singularities, sum=chi, matches MATLAB) |
| 3. Cut Graph | VERIFIED (41 cut edges, 7 tests pass) |
| 4. Optimization | VERIFIED (normalization bug fixed, 2 tests pass) |
| 5. UV Recovery | IN PROGRESS |

**Flip count:** 46 flips (14.4%) on sphere320 - this is the real bug to fix.

The old implementation shows fewer flips but through algorithm deviations (cheating), not correct fixes.
The MATLAB code is proven to work, so the bug is in the Python translation.

## Docs
`verification-plan.md`, `PLAN_UV_FIXES.md`, `bug-report.md`, `docs/algo_integer_grid_maps.md`

## License
MIT

## Note
`README.md` is a symlink to this file.
