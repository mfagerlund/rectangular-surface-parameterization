# Code Review Findings

## High
- ComputeParam/mesh_to_disk_seamless.py:219-228 silently accepts an odd number of cut boundary edges by truncating the last unpaired edge (min_len logic). This can mispair cut constraints and produce invalid seamless constraints; the MATLAB intent was to fail on odd pairing. Consider explicit odd-count detection and a hard error.
- Utils/extract_scale_from_param.py:243-247 silently zero-fills v(T) when T references more vertices than the cut mesh provides. This hides T/T_cut mismatches and yields incorrect per-corner scales; tests in tests/test_review_issues.py:272-281 expect an IndexError instead. Make this a hard validation error instead of returning zeros.

## Medium
- Utils/verify_pipeline.py:56-67 computes angle defects as 2*pi minus corner sums for all vertices, then infers Euler characteristic by rounding total curvature. This is wrong for meshes with boundaries (boundary vertices should use pi - sum), and it can diverge from the topological V-E+F that is already available. The stage-1 “total curvature” metrics and plots are therefore misleading for open meshes.
- Utils/save_param.py:90-111 invokes QuantizationYoann via a relative path (./QuantizationYoann/...). Running run_RSP.py from any directory other than repo root will fail to find the executable even if it exists. Resolve the path relative to the repo (e.g., Path(__file__).parent.parent) or expose it as a CLI option.
- run_RSP.py:146 default behavior keeps quantization enabled, but QuantizationYoann is not included. On a clean checkout, this can hard-fail save_param unless users pass --no-quantization. Either default to disabled or degrade to a warning when the external tool is missing.

## Low / Docs / Cleanup
- Utils/verify_pipeline.py:409-432 prints “expected χ/4” but uses index_sum itself, so the expected comparison is a no-op and misleading. Compute χ from V-E+F and display a real expected value (or note boundary corrections).
- CLAUDE.md:56 shows `python verify_pipeline.py ...` but the tool lives at Utils/verify_pipeline.py; the command is inaccurate.
- .gitignore:1-23 does not include `.pytest_cache/`, while a `.pytest_cache` directory exists in the repo. Add it to avoid accidental check-ins, and remove the directory if it’s tracked.
