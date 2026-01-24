# My Half Code Review (Python vs MATLAB Comments)

Scope
- Reviewed: `ComputeParam/cut_mesh.py`, `ComputeParam/matrix_vector_multiplication.py`, `ComputeParam/mesh_to_disk_seamless.py`, `ComputeParam/parametrization_from_scales.py`, `Preprocess/angles_of_triangles.py`, `Preprocess/connectivity.py`, `Preprocess/dec_tri.py`, `Preprocess/find_graph_generator.py`, `Preprocess/gaussian_curvature.py`, `Preprocess/MeshInfo.py`, `Preprocess/preprocess_ortho_param.py`, `Preprocess/sort_triangles.py`, `Preprocess/sort_triangles_comp.py`, `Utils/vec.py`, `Utils/extract_scale_from_param.py`, `Utils/readOBJ.py`, `Utils/save_param.py`, `Utils/visualize_uv.py`, `Utils/writeObj.py`.
- Basis: comparison against the embedded MATLAB comment blocks (no `.m` sources present in this repo).

Findings (ordered by severity)
1) Potentially silent cut-edge pairing mismatch
   - `ComputeParam/mesh_to_disk_seamless.py:222-230`: The MATLAB code asserts that cut edges are paired (`abs(ide_cut(1:2:end-1)) == abs(ide_cut(2:2:end))`). The Python code slices into pairs but does not validate even length; if `ide_cut` is odd-length, the last unpaired edge is silently ignored. This can misalign `ide_cut_cor`/`Rot` constraints and drift from MATLAB behavior.

2) `ismember` mismatch is silently dropped
   - `ComputeParam/mesh_to_disk_seamless.py:374-385`: MATLAB `ismember` would return `0` for missing entries, which would fail when used for indexing (forcing the issue to be fixed). The Python translation drops not-found entries via `found_mask`, which can mask a data mismatch and silently shrink `tri_fix_param`/`ide_fix_cut_final`, altering alignment constraints relative to MATLAB.

3) Graph generator construction may diverge from MATLAB
   - `Preprocess/find_graph_generator.py:101-105` and `Preprocess/find_graph_generator.py:188-206`: MATLAB?s `minspantree` with a root/forest can yield different predecessor arrays versus SciPy?s `minimum_spanning_tree` + BFS. The Python version also hard-codes the dual BFS root to `0` and does not build a predecessor for all components in the dual ?forest? case. This can change the cycle/cocycle generators compared to MATLAB, especially with disconnected dual graphs or weight ties.

4) UV scale extraction masks index mismatches
   - `Utils/extract_scale_from_param.py:243-247`: The MATLAB code indexes `v(T)` after building `v` from `T_cut`. The Python port uses `v[T]` but silently falls back to zeros when `T` exceeds `len(v)`. MATLAB would error in that case; the Python fallback can hide a bad T/T_cut mismatch and produce incorrect `ut` without warning.

Notes
- No `.m` sources were found in this repo, so verification is limited to the embedded MATLAB comments and documented conversion rules.
