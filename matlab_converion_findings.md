# MATLAB Conversion Review Findings

Scope: Review of `MATLAB_CONVERSION.md` guidance and all Python files that still carry MATLAB comment blocks (identified via `# %`). This document focuses on conversion completeness, known pitfalls (column-major flattening, signed edge encoding, boundary sentinels), and dependencies called out in the per-file `ISSUES` blocks.

## File Inventory (MATLAB-commented sources)

- `ComputeParam/cut_mesh.py` (was cut_mesh_with_python.py)
- `ComputeParam/parametrization_from_scales.py`
- `ComputeParam/matrix_vector_multiplication.py`
- `ComputeParam/mesh_to_disk_seamless.py`
- `FrameField/brush_frame_field.py`
- `FrameField/compute_face_cross_field.py`
- `FrameField/trivial_connection.py`
- `FrameField/compute_curvature_cross_field.py`
- `Orthotropic/objective_ortho_param.py`
- `Orthotropic/omega_from_scale.py`
- `Orthotropic/optimize_RSP.py`
- `Orthotropic/reduction_from_ff2d.py`
- `Orthotropic/oracle_integrability_condition.py`
- `Orthotropic/reduce_corner_var_2d_cut.py`
- `Preprocess/connectivity.py`
- `Preprocess/dec_tri.py`
- `Preprocess/find_graph_generator.py`
- `Preprocess/gaussian_curvature.py`
- `Preprocess/MeshInfo.py`
- `Preprocess/preprocess_ortho_param.py`
- `Preprocess/sort_triangles_comp.py`
- `Utils/extract_scale_from_param.py`
- `Utils/readOBJ.py`
- `Utils/writeObj.py`
- `run_RSP.py`

## Global Checklist Against `MATLAB_CONVERSION.md`

### 1) Issues Block at Top of Each .py
- Present: most MATLAB-commented files include an `# === ISSUES ===` block.
- Missing: `ComputeParam/cut_mesh.py` and `ComputeParam/cut_mesh_with_python.py` do not include the required `ISSUES` block.

### 2) Column-Major Flattening (vec/x(:))
Positive signals:
- Multiple files explicitly use `flatten('F')` / `ravel('F')` when MATLAB column-major semantics are required:
  - `ComputeParam/parametrization_from_scales.py`
  - `ComputeParam/mesh_to_disk_seamless.py`
  - `Preprocess/dec_tri.py`
  - `Preprocess/preprocess_ortho_param.py`
  - `Orthotropic/objective_ortho_param.py`
  - `Orthotropic/reduction_from_ff2d.py`
  - `Orthotropic/oracle_integrability_condition.py`
  - `FrameField/compute_face_cross_field.py`
  - `Utils/extract_scale_from_param.py`

Potentially ambiguous / verify:
- Some `.ravel()` or `.flatten()` calls are row-major but likely used where order is not semantically important (set-like operations or scatter-add), e.g.:
  - `ComputeParam/cut_mesh_with_python.py` (degree counts, index remapping)
  - `Preprocess/connectivity.py` (edge counts)
  - `Preprocess/gaussian_curvature.py` (vertex accumulation)
  - `Preprocess/sort_triangles_comp.py` (edge set extraction)
  - `Orthotropic/omega_from_scale.py` (row-wise flattening of 3x3 blocks)
  - `Orthotropic/optimize_RSP.py` (vectorization of u/v/omega/ang)

Recommendation: confirm each row-major flatten is either order-independent or intentionally row-major. If any are meant to replicate MATLAB vec(), switch to `ravel('F')` or use `Utils/vec.py`.

### 3) Signed Edge Encoding (T2E)
Positive signals:
- `Preprocess/connectivity.py` explicitly encodes signed edge indices as `(edge_idx + 1) * sign` and documents the decode rule.
- `Preprocess/dec_tri.py` and `FrameField/compute_curvature_cross_field.py` decode with `abs(T2E) - 1` and `sign(T2E)`.
- `ComputeParam/cut_mesh_with_python.py` returns signed **1-based** `ide_cut_inv` to preserve sign information for edge 0.

Recommendation: ensure all callers agree on the signed-1-based contract. For any input expected to be 0-based signed, validate that it was encoded with the +1 rule to avoid the edge-0 sign loss.

### 4) Boundary Sentinel (-1)
Positive signals:
- `Preprocess/connectivity.py` explicitly uses `-1` for missing neighbors in `E2T` and notes the MATLAB vs Python difference.
- `ComputeParam/cut_mesh_with_python.py` shifts 1-based 0-sentinels to `-1` and uses `-1` in boundary checks.

Recommendation: verify downstream logic does not still treat `0` as sentinel. Search for any remaining boundary checks that compare to `0` in translated Python and update to `-1` where applicable.

## File-by-File Findings

### `ComputeParam/cut_mesh.py`
- Status: Full Python implementation with interleaved MATLAB comments. **(FIXED: consolidated from cut_mesh_with_python.py)**
- Issues block: present. **(FIXED)**
- Indexing handling:
  - Detects 1-based inputs and shifts indices to 0-based for internal use.
  - Converts 0-sentinel to `-1` for boundary adjacency (`E2T`, `T2T`).
  - Returns signed **1-based** `ide_cut_inv` to avoid edge-0 sign loss.
- Column-major concerns: uses `.ravel()` for degree accumulation and index mapping. These are order-insensitive operations (multiset counts and first-write), so row-major flattening should not alter results.
- Risk: `T2E` input expectations are not fully stated. If callers pass 0-based signed edges without +1 encoding, sign for edge 0 may already be lost before this function sees it.

### `ComputeParam/parametrization_from_scales.py`
- Issues block: present; notes QP, blkdiag, accumarray, complex usage.
- Column-major: uses `flatten('F')` in the key places that correspond to MATLAB vec.
- Risk: relies on QP solving and matrix assembly details from MATLAB. The pipeline will need a clear choice of solver and KKT formulation for parity.

### `ComputeParam/mesh_to_disk_seamless.py`
- Issues block: present; notes `wrapToPi`, `cut_mesh`, `dec_tri`, `ismember`, `blkdiag`.
- Column-major: uses `flatten('F')` for triangle indices and states intent.
- Uses row-major `ravel()` for rotation matrices; this appears intentional (treat each 2x2 as row-major flattened). Ensure all consumers interpret these flattened rotations consistently.

### `ComputeParam/matrix_vector_multiplication.py`
- Issues block: present; function is mostly straightforward sparse construction.
- Column-major: documentation says matrices are flattened row-major. This is acceptable as long as callers match that convention.

### `FrameField/brush_frame_field.py`
- Issues block: present (none).
- BFS over graph edges; indexing comments show awareness of 0-based conversion.
- Risk: verify that `E2T` passed in is already 0-based with `-1` sentinel (or that the code never sees boundary edges).

### `FrameField/compute_face_cross_field.py`
- Issues block: present; relies on QP, eigensolver, wrap function.
- Column-major: `E2T` flatten uses `ravel('F')`, which matches MATLAB vec.
- Risk: uses `ide_int` indexing; any mismatch between `-1` sentinel and expected interior edges could change the set of edges considered.

### `FrameField/compute_curvature_cross_field.py`
- Issues block: present.
- Uses `T2E` signed-1-based decoding, explicitly noted in comments.
- `t2e_vals = Src.T2E[idt, :].ravel()` uses row-major. Here, order is used for unique edge extraction; order should not matter but verify no ordering assumptions exist downstream.

### `FrameField/trivial_connection.py`
- Issues block: present; depends on QP and `brush_frame_field`.
- Risk: unimplemented solver or `wrapToPi` parity will affect alignment/transport behavior.

### `Orthotropic/objective_ortho_param.py`
- Issues block: present.
- Uses `ravel('F')` for `ut` and `vt` to match MATLAB vec.
- Several `.ravel()` calls are on constructed arrays (`tile`/`reshape`) where order is not semantically tied to MATLAB vec; likely OK.

### `Orthotropic/omega_from_scale.py`
- Issues block: present (none identified).
- Uses `.ravel()` to flatten 3x3 blocks for sparse assembly. This is likely correct if the intended convention is row-major flattening; confirm with the consuming code.

### `Orthotropic/optimize_RSP.py`
- Issues block: present; notes QP, wrap, plotting, blkdiag.
- Vectorization uses `np.ravel` without order. If `u`, `v`, `omega`, `ang` are column vectors or 1D arrays, order is irrelevant. If any are 2D with MATLAB-style layout, consider forcing `ravel('F')`.

### `Orthotropic/reduction_from_ff2d.py`
- Issues block: present.
- Uses `ravel('F')` for consistent column-major assembly.

### `Orthotropic/oracle_integrability_condition.py`
- Issues block: present; depends on `omega_from_scale` and uses several MATLAB sparse ops.
- Uses `ravel('F')` and column-major assembly consistently.

### `Orthotropic/reduce_corner_var_2d_cut.py`
- Issues block: present; depends on `sort_triangles` and MATLAB-style helpers.
- No obvious indexing pitfalls seen in header scan; verify use of `sign_edge` with signed edge encoding.

### `Preprocess/connectivity.py`
- Issues block: present; has explicit signed-encoding and boundary rules.
- Signed T2E is encoded as `(edge_idx + 1) * sign` and documented.
- Boundary sentinel: `-1` is used internally and returned.

### `Preprocess/dec_tri.py`
- Issues block: present; notes vec/accumarray/sparse.
- Decodes signed T2E using `abs(T2E) - 1` and `sign(T2E)`.
- Uses `flatten('F')` for column-major matching in critical matrix assembly.
- Note: some `.flatten()` calls are explicitly row-major with comments; those appear intentional. Confirm that they correspond to MATLAB row-major flattening (not vec).

### `Preprocess/find_graph_generator.py`
- Issues block: present; MST fallback to Kruskal is unimplemented.
- Risk: depends on SciPy MST and graph traversal; behavior should be checked for parity with MATLAB `graph/minspantree` and `dfsearch`.

### `Preprocess/gaussian_curvature.py`
- Issues block: present.
- Uses `np.add.at(K, T.ravel(), -theta.ravel())` to accumulate on vertices; order is irrelevant.

### `Preprocess/MeshInfo.py`
- Issues block: present; depends on `connectivity`.
- Uses `T.flatten()` for building per-vertex areas; this is order-insensitive since it maps triangle areas to vertex indices.

### `Preprocess/preprocess_ortho_param.py`
- Issues block: present; heavy use of MATLAB-style graph utilities.
- Column-major conversion is explicitly handled in places where vec is used.
- There are several `ravel()` uses for `tri_hard` and `tri_fix` concatenation; verify whether the order matters (MATLAB `tri_hard(:)` would be column-major).
- Boundary handling: identifies boundary edges via `E2T[:, :2] < 0`, consistent with the `-1` sentinel.

### `Preprocess/sort_triangles_comp.py`
- Issues block: present; highlights vec and row-wise operations.
- Uses `.ravel()` to gather edges for set operations; ordering should not be significant.

### `Utils/extract_scale_from_param.py`
- Issues block: present.
- Uses `ravel('F')` for `T(:)` semantics where needed.

### `Utils/readOBJ.py`
- Issues block: present; straightforward parser.
- Minor `pass` for unsupported or ignored lines; likely OK.

### `Utils/writeObj.py`
- Issues block: present; straightforward writer.

### `run_RSP.py`
- Issues block: present; notes missing external quantization tool.
- Uses `.flatten()` for `E2V` index stamping; order is irrelevant.

## Cross-Cutting Risks / Gaps

1) ~~Missing `ISSUES` headers in `cut_mesh.py` and `cut_mesh_with_python.py`.~~ **(FIXED)**
2) ~~One MATLAB source (`cut_mesh.py`) has no Python translation.~~ **(FIXED: consolidated into single cut_mesh.py)**
3) ~~Some row-major `.ravel()` calls standing in for MATLAB `vec()`.~~ **(FIXED in preprocess_ortho_param.py)**
4) Signed edge encoding appears consistent in `connectivity.py` and downstream decoders, but any code path that constructs signed indices without the +1 offset would silently lose sign on edge 0.

## Suggested Next Actions

- ~~Add `ISSUES` blocks to `ComputeParam/cut_mesh.py` and `ComputeParam/cut_mesh_with_python.py`.~~ **(DONE)**
- ~~Decide whether `ComputeParam/cut_mesh.py` should be removed, renamed, or fully implemented.~~ **(DONE: consolidated)**
- ~~Audit all `.ravel()` / `.flatten()` calls in MATLAB-derived files.~~ **(DONE: fixed preprocess_ortho_param.py)**
- Confirm every creation of signed edge indices uses `(edge_idx + 1) * sign` and that all consumers decode with `abs(x) - 1`.
- Add a small set of parity tests for edge cases: boundary edges, edge 0 with negative sign, and vec() ordering on 2D arrays.

## Fixes Applied (2025-01-23)

1. **Consolidated `cut_mesh` files**: Deleted redundant MATLAB-only file and renamed `cut_mesh_with_python.py` → `cut_mesh.py` to match original MATLAB filename. The Python file contains full MATLAB source as comments per convention.

2. **Added ISSUES block to `cut_mesh.py`**: Documents key conversion notes including:
   - Local MeshInfo implementation
   - Union-find translation with path compression
   - Auto-detection of 1-based/0-based indexing
   - Signed 1-based ide_cut_inv output for edge 0 sign preservation
   - Boundary sentinel convention (-1)

3. **Fixed `.ravel()` calls in `preprocess_ortho_param.py`**: Changed 4 instances of `tri_hard.ravel()` to `tri_hard.ravel('F')` to match MATLAB `tri_hard(:)` column-major semantics (lines 221, 276, 1096, 1102).

**Remaining items from original list:**
- Confirm every signed edge encoding uses `(edge_idx + 1) * sign` pattern
- Add parity tests for edge cases (boundary edges, edge 0 sign, vec() ordering)
