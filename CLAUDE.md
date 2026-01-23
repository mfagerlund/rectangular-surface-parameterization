# CLAUDE.md

Instructions for Claude Code when working on this project.

## CRITICAL: DO NOT RUSH

**Read `verification-plan.md` before doing ANYTHING.**

Previous work rushed ahead as soon as one thing appeared to work. We have NOTHING that's a proven foundation. The rule:

1. Verify each pipeline stage with tests AND visualizations
2. Get explicit signoff before moving to the next stage
3. NO rushing ahead when something "looks like it works"
4. If a stage fails, fix it BEFORE touching anything downstream

## Project Overview

Python implementation of Corman & Crane's rectangular parameterization (SIGGRAPH 2025) for **quad meshing**.

**This is for quad meshing, NOT origami unfolding.**

- Origami: any valid UV layout works (just flatten without flips)
- Quad meshing: UV domain must be **compact** (high fill ratio ~100%)
  - Integer iso-lines (u=k, v=k) become quad edges
  - Fragmented "octopus" UV = sparse quad coverage

**Goal**: 0 flipped triangles AND compact UV layout (high fill ratio)

## Pipeline Stages

```
1. Geometry → 2. Cross Field → 3. Cut Graph → 4. Optimization → 5. UV Recovery
```

Each stage must be verified independently. A bug in stage N corrupts all stages N+1 onward.

## Key Files

```
corman_crane.py     # Main CLI entry point
mesh.py             # Half-edge mesh data structure
geometry.py         # Geometric computations (angles, areas, cotans)
cross_field.py      # Cross field propagation
cut_graph.py        # Cut graph and jump data (Algorithm 2)
sparse_ops.py       # Sparse matrix operators (Algorithms 5-10)
optimization.py     # Constraint solver (Algorithms 3-4)
uv_recovery.py      # UV coordinate recovery (Algorithm 11)
verify_pipeline.py  # Stage-by-stage verification
visualize.py        # Visualization utilities
```

## Commands

Run full pipeline:
```bash
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o output/sphere_uv.obj -v
```

Verify a stage:
```bash
python verify_pipeline.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" --stage geometry
```

Test meshes:
- `C:/Dev/Colonel/Data/Meshes/sphere320.obj` - genus 0, should have ~4 cones
- `C:/Dev/Colonel/Data/Meshes/torus.obj` - genus 1

## References (READ THE PAPERS)

**Official MATLAB implementation:**
- GitHub: https://github.com/etcorman/RectangularSurfaceParameterization
- **Local copy: `C:\Slask\RectangularSurfaceParameterization`** - USE THIS FOR LINE-BY-LINE TRANSLATION

**Main paper:**
- Corman & Crane, "Rectangular Parameterization", SIGGRAPH 2025
- Local: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025.pdf`
- Supplement: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025-supplement.pdf`

**Quad extraction:**
- Bommes et al., "Integer-Grid Maps for Reliable Quad Meshing", SIGGRAPH 2013
- Local: `D:\Data\GDrive\FlatrPDFs\2461912.2462014_integer-grid-maps-reliable-quad-meshing.pdf`

- Bommes et al., "Mixed-Integer Quadrangulation", SIGGRAPH 2009
- Local: `D:\Data\GDrive\FlatrPDFs\1531326.1531383_mixed-integer-quadrangulation.pdf`

## Current Status

See `verification-plan.md` for detailed status. Summary:

| Stage | Status |
|-------|--------|
| 1. Geometry | ✓ VERIFIED (54 pytest tests pass) |
| 2. Cross Field | ~ PARTIAL (math correct, 14 singularities vs optimal 4) |
| 3. Cut Graph | ? INVESTIGATE (cone/singularity mismatch suspected) |
| 4. Optimization | ? UNKNOWN (depends on 2-3) |
| 5. UV Recovery | ✗ BROKEN (28 flips, overlapping/inverted) |

**Key insight:** MATLAB passes cross-field singularities directly to cut_mesh.
Python recomputes cones differently, which may cause the mismatch.

## Documentation

- `verification-plan.md` - Current verification status and action items
- `PLAN_UV_FIXES.md` - Planned fixes for UV issues
- `bug-report.md` - Analysis of fragmentation issue
- `docs/algo_integer_grid_maps.md` - Extracted algorithm from Bommes 2013
