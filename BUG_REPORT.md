# Bug Report: 8 Remaining Flipped Triangles

## Current State

UV recovery now works with minor remaining issues:
- **Flipped triangles**: 8 / 320 (2.5%)
- **Angle error**: 15.05°

## Recent Fixes Applied

1. **b_rhs formula**: Changed to subtraction `b = 0.5 * (R_zeta * mu[he] - mu[he_twin])`
2. **Y-coordinate flip**: Added `f[:, 1] = -f[:, 1]` after solve

## Remaining Issue

8 triangles are still flipped. Potential causes:

1. **Singularities**: Triangles near cone vertices where the frame field has discontinuities
2. **Cut edges**: Triangles adjacent to cut edges may have inconsistent corner identification
3. **Zeta rotation**: The zeta sign convention may be inconsistent for certain edge orientations

## Diagnostic Questions

1. Are the 8 flipped triangles clustered near singularities or scattered?
2. Does the zeta rotation need halfedge-orientation-dependent sign flip?
3. Is the corner identification constraint (U matrix) correct for cut edges?

## Files

- `uv_recovery.py` - UV recovery implementation (lines 108-131 for b_rhs, line 275-277 for y-flip)
- `cut_graph.py` - Zeta and phi computation

## Reproduce

```bash
cd c:\Dev\Corman-Crane
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o sphere_uv.obj -v
```
