# Issue: 10 Flipped Triangles in UV Parameterization

## Problem

The pipeline produces 10 flipped triangles (3.1%) on `sphere320.obj`. Goal is 0 flips.

## Root Cause

**MATLAB pre-marks jump edges as cut edges BEFORE building the spanning tree.**

In `cut_mesh.m` line 12:
```matlab
visited_edge(edge_jump_tag) = true;  % Jump edges MUST be cut
```

Then the BFS spanning tree (lines 13-31) only traverses non-jump edges. This guarantees:
- Spanning tree uses ONLY identity edges (k21 = 1)
- Jump edges (k21 != 1) are always in the cut

Our code builds the spanning tree first, THEN computes jumps. This is backwards.

## Source Code

- **Python**: `c:\Dev\Corman-Crane\cut_graph.py` - `compute_cut_jump_data()`
- **MATLAB**: `C:\Slask\RectangularSurfaceParameterization\ComputeParam\cut_mesh.m`
- **MATLAB k21**: `C:\Slask\RectangularSurfaceParameterization\Orthotropic\reduction_from_ff2d.m`

## How to Test

```bash
python -m pytest tests/test_cut_graph_topology.py -v
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o output/sphere320_uv.obj -v
```

Check output:
- `Flipped triangles: 0`
- `UV fill ratio: > 90%`

## Definition of Done

1. 0 flipped triangles on `sphere320.obj`
2. All 91+ tests still pass
3. Cut graph uses identity edges for spanning tree (like MATLAB)
