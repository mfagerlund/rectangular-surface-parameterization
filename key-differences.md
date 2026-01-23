# Key Differences: RectangularSurfaceParameterization -> Corman-Crane

This compares the MATLAB/C++ reference repo at `C:\Slask\RectangularSurfaceParameterization` with the Python port in `C:\Dev\Corman-Crane`, based on a read-through of the main entry points and core modules.

## High-level structure and entry points
- MATLAB repo: `run_RSP.m` is the single driver with options and plotting; core algorithm spread across `Preprocess/`, `FrameField/`, `Orthotropic/`, and `ComputeParam/`. Quantization lives in `QuantizationYoann/` (C++).
- Python repo: `corman_crane.py` is the CLI entry point; core algorithm is consolidated into Python modules (`mesh.py`, `geometry.py`, `cross_field.py`, `cut_graph.py`, `optimization.py`, `uv_recovery.py`), with tests and debug scripts.

## Pipeline alignment (algorithm phases)
- MATLAB pipeline in `run_RSP.m`:
  1) Preprocess geometry/constraints (`MeshInfo`, `dec_tri`, `preprocess_ortho_param`)
  2) Cross-field selection (`compute_face_cross_field`, `compute_curvature_cross_field`, or `trivial_connection`)
  3) Corner reduction (`reduce_corner_var_2d`, `reduction_from_ff2d`)
  4) Optimization (`optimize_RSP`)
  5) Cut-to-disk + seamless assembly (`mesh_to_disk_seamless`)
  6) Param reconstruction (`parametrization_from_scales`)
  7) Optional quantization + save (`save_param`)
- Python pipeline in `corman_crane.py`:
  1) Geometry (`compute_corner_angles`, `compute_edge_lengths`)
  2) Cross field (connection Laplacian in `compute_smooth_cross_field`)
  3) Cut graph + jump data (`compute_cut_jump_data`)
  4) Constraint solve only (`solve_constraints_only`)
  5) UV recovery via Poisson solve (`recover_parameterization`)
  6) Normalize UVs + quality metrics

## Cross-field options and constraints
- MATLAB supports 3 field modes (`trivial`, `curvature`, `smooth`) and integrates hard-edge and boundary alignment constraints during preprocessing.
- Python currently uses the smooth connection-Laplacian field only; there is no CLI or module support yet for curvature-aligned or trivial fields, nor explicit hard-edge or boundary alignment constraints.

## Optimization and energy choices
- MATLAB exposes energy types (`distortion`, `chebyshev`, `alignment`) with multiple weights and regularizers, and uses `optimize_RSP` (Newton) as the main solver.
- Python defaults to a feasibility-only solve (`solve_constraints_only`) and does not wire the full Newton optimization (`solve_optimization`) into the CLI. Energy-type selection and corresponding weights are not implemented at the user level.

## Cut graph and seam handling
- MATLAB: seamlessness is explicitly controlled via `ifseamless_const`, and cutting is handled in `mesh_to_disk_seamless` along with alignment constraints.
- Python: `compute_cut_jump_data` builds a cut graph and includes a pruning step that intentionally allows cycles to reduce boundary complexity (trade-off for UV quality). This is a behavioral divergence from the strict topology assumptions in the paper and MATLAB implementation.

## UV recovery and outputs
- MATLAB reconstructs UVs via `parametrization_from_scales` after cutting to a disk and exports through `save_param` (with singularities and hard-edge data).
- Python reconstructs UVs via a Poisson solve in `recover_parameterization`, normalizes to `[0,1]`, and saves with `save_obj`. It also reports flip counts and angle error metrics.

## Quantization and quad extraction
- MATLAB provides a quantization step through external C++ (`QuantizationYoann/`) for integer-seamless maps.
- Python includes a quad extraction utility (`quad_extract.py`) but no integer-seamless quantization step yet.

## Tooling, diagnostics, and tests
- MATLAB repo includes plotting in `run_RSP.m` and some helper utilities in `Utils/`.
- Python repo includes extensive debug scripts, visualization helpers, and unit tests (`tests/`), plus diagnostic markdown notes (e.g., `fix-flips.md`, `verification-plan.md`).

## Licensing
- MATLAB repo is AGPL-3.0-or-later.
- Python repo is MIT (per `README.md`).
