# Code Review 1

## Findings

### High
- Default CLI path enables quantization even when `QuantizationYoann` is missing, and `save_param` raises a hard error. This makes the default run path fail on clean machines; auto-detect or default to off, and degrade to warning if the binary is missing. (`run_RSP.py:146`, `Utils/save_param.py:91`, `Utils/save_param.py:111`)
- `extract_scale_from_param` silently zero-fills `v` when `T` references vertices not present in `T_cut`, masking a T/T_cut mismatch and producing incorrect `ut`/distortion outputs instead of failing fast. Add an explicit index range check and raise. (`Utils/extract_scale_from_param.py:243`)

### Medium
- Cut-edge pairing in `mesh_to_disk_seamless` drops the last edge when the boundary cut count is odd (`min_len`), so a malformed cut can pass without error and silently remove a constraint. Validate even pairing or raise with a clearer message. (`ComputeParam/mesh_to_disk_seamless.py:227`)
- Quantization executable path is hard-coded to `./QuantizationYoann/build/Quantization` without OS-specific suffix handling or repo-root resolution, so it fails on Windows and when run from a different CWD. Resolve via `Path(__file__)` and append `.exe` on Windows. (`Utils/save_param.py:91`)

### Low
- Integration test references a local mesh path outside the repo, so it is skipped in most environments and leaves the full pipeline unverified. Add a small fixture mesh under `tests/fixtures` or generate one in-test. (`tests/test_review_issues.py:329`)
- `optimize_RSP` prints every iteration unconditionally; add a `verbose` flag or reuse `if_plot` to prevent noisy output in batch runs. (`Orthotropic/optimize_RSP.py:187`)
- `run_RSP.py` imports unused modules (`scipy.sparse`, `Axes3D`, `writeObj`), which are obsolete in the current entry point. (`run_RSP.py:30`, `run_RSP.py:32`, `run_RSP.py:36`)

## Questions / Assumptions
- Is quantization supposed to be a default part of the CLI workflow in 2026, or should it remain opt-in until a cross-platform build is in-repo?
- Should `extract_scale_from_param` hard-fail on any `T`/`T_cut` mismatch, or is there a documented case where the silent zero-fill is desired?