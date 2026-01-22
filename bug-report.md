# Bug Report: Fragmented UV Domain (Low Fill Ratio)

## Summary

The UV parameterization produces a fragmented domain with only 34-53% coverage of the bounding box instead of the expected ~100%. This causes:
- "Spiky" UV visualizations
- ~50% holes in the output mesh
- Poor quad extraction coverage

## Reproduction

```bash
python test_meshes.py
```

Output shows:
```
Mesh              V     F  G Flips    Err  Fill Quads Status
sphere320       162   320  0     4   14.7   53%    52   FAIL
torus           576  1152  1     0    9.5   34%   105   PASS

WARNING: 2 mesh(es) have UV fill < 80% (fragmented UV domain)
```

## Expected Behavior

Per the Corman-Crane paper (Section D, Supplement):

> We make just one simple change to our basic setup: rather than store values at vertices, we will store values at all triangle corners, using linear constraints to identify equivalent values as needed.

The UV domain should form a connected, roughly rectangular region with:
- **Fill ratio ~100%** (all triangles pack tightly)
- **1 connected component** after cutting to disk topology
- **Corners identified** across non-cut edges (same UV position)

## Actual Behavior

The UV domain is fragmented into many disconnected pieces:
- **223 corner components** (sphere) instead of 1
- **34-53% fill ratio** instead of ~100%
- Triangles scattered with gaps between them

## Root Cause Analysis

The issue is in **cut_graph.py** - the cut graph has too many edges:

| Mesh | Expected Cut Edges | Actual Cut Edges |
|------|-------------------|------------------|
| Sphere (g=0) | ~4 (connecting 4 singularities) | 62 |
| Torus (g=1) | ~48 (two loops of ~24) | 75 |

### Why Too Many Cuts?

1. **Cone detection** (`CONE_THRESHOLD = 0.5`) detects ~29 false-positive cones on sphere
2. Pruning can't remove degree-1 vertices that are "cones"
3. Result: 62 edges remain instead of ~4

### Why 223 Components?

Each cut edge splits corner identifications around its endpoints:
- Vertex with 6 incident faces has 6 corners
- If 1 cut edge: corners split into 2 groups
- If 2 cut edges: corners split into 2-3 groups
- With 62 cuts distributed across 162 vertices: massive fragmentation

## Algorithm Reference

From Algorithm 11 (RecoverParameterization), lines 23-28:
```
if Γ_ij == 0 then                        ▷ if ij NOT in the cut, add corner constraints
    L_U ← Append(L_U, (p, i^{jk}, +1))
    L_U ← Append(L_U, (p, i^{lj}, −1))
    ...
```

Corner constraints are only added for **non-cut edges**. With too many cut edges, too few constraints are added, and the UV domain fragments.

## Potential Fixes

1. **Improve cone detection** - Current threshold 0.5 rad is a tradeoff:
   - Lower (0.3): More cones detected, less pruning, more fragmentation
   - Higher (0.7): Fewer cones, more pruning, but causes flipped triangles

2. **Fix cross field** - The cross field holonomy errors cause spurious cone indices:
   - BFS propagation accumulates errors around handles
   - `initialize_smooth_cross_field()` helps but doesn't fully solve it

3. **Improve pruning algorithm** - Current pruning can disconnect the cut graph:
   - Should preserve connectivity while removing leaves
   - Need to ensure resulting cut forms a tree connecting true singularities

## Files Involved

- `cut_graph.py:CONE_THRESHOLD` - Cone detection sensitivity
- `cut_graph.py:compute_cut_jump_data()` - BFS traversal and pruning
- `uv_recovery.py:recover_parameterization()` - Corner constraint generation
- `cross_field.py:propagate_cross_field()` - Cross field that determines cones

## Related Documentation

- `fix-flips.md` - Documents Algorithm 11 and RHS averaging fix
- `PLAN_PHASE2.md` - Phase 2 implementation notes
- Paper Supplement, Section D: "Singularities and Cuts"
