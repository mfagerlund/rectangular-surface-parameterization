# Your Half Code Review (Python vs MATLAB Comments)

Scope
- Reviewed: `FrameField/brush_frame_field.py`, `FrameField/compute_face_cross_field.py`, `FrameField/compute_curvature_cross_field.py`, `FrameField/plot_frame_field.py`, `FrameField/trivial_connection.py`, `FrameField/__init__.py`, `Orthotropic/reduce_corner_var_2d.py`, `Orthotropic/reduce_corner_var_2d_cut.py`, `Orthotropic/reduction_from_ff2d.py`, `Orthotropic/omega_from_scale.py`, `Orthotropic/oracle_integrability_condition.py`, `Orthotropic/objective_ortho_param.py`, `Orthotropic/optimize_RSP.py`, `run_RSP.py`, and `tests/*.py`.
- Basis: comparison against the embedded MATLAB comment blocks and `MATLAB_CONVERSION.md` rules (no `.m` sources in this repo).

Findings (ordered by severity)
1) Trivial-frame singularity seed differs from MATLAB
   - `run_RSP.py:150-166`: The MATLAB block initializes boundary singularities with `param.K`, but the Python code uses `param.Kt` for `sing[param.idx_bound]`. If `param.K` includes extended vertices (hard-edge corners) this change may be intentional, but it is a parity deviation and can alter the trivial-field singularity assignment.

2) Link-constraint check uses a different quantity than the MATLAB block
   - `FrameField/trivial_connection.py:213-224`: The MATLAB comment uses `om_cycle` when validating `sing_link`, while the Python code uses `om_link` (with a note about the MATLAB line being a typo). This is a direct divergence from the embedded MATLAB block; if MATLAB is authoritative, the Python check is validating a different constraint.

3) Duplicate BFS/brush implementations can bypass MATLAB boundary filtering
   - `FrameField/trivial_connection.py:118-296` and `FrameField/compute_face_cross_field.py:285-406` define local `brush_frame_field`/`breadth_first_search` implementations instead of using `FrameField/brush_frame_field.py`.
   - The shared version filters out boundary sentinels (`adj >= 0`), while the local versions rely on `param.ide_int` containing no boundary edges. If boundary edges leak in (or `E2T` carries `-1`), the local implementations diverge from MATLAB’s `adj == 0` filtering and can propagate through invalid faces.

4) Signed edge decoding assumes no zero entries, with no guard
   - `FrameField/compute_curvature_cross_field.py:140-149` and `Orthotropic/omega_from_scale.py:63-79`: Both decode signed edges via `abs(T2E) - 1`. The conversion guide requires `(edge_idx + 1) * sign` encoding; if any upstream `T2E` still uses `0` or unsigned indices, this will silently map to `-1` (last edge) instead of failing as MATLAB would. There is no assertion or sanitization here.

Notes
- Tests are extensive and reflect the conversion expectations, but most are Python-native (not MATLAB comment translations), so parity validation is limited to the algorithmic files with embedded MATLAB blocks.
