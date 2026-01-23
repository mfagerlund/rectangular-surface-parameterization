# Plan: Fix Fragmented UV Domain

## Problem Summary

The UV parameterization produces fragmented domains (34-53% fill) instead of connected regions (~100%). Root causes:

1. **False cone detection** blocks pruning → oversized cut graph
2. **BFS cross-field** accumulates holonomy errors → noisy omega0
3. **Naive pruning** doesn't enforce minimal tree → excess cut edges
4. **RHS sign bug** distorts UV solution → exacerbates spiky output

---

## Phase 1: RHS Sign Fix (Low Risk)

**File**: `uv_recovery.py`
**Lines**: 145-156

### 1.1 Problem

Algorithm 11 line 13: `b = (1/2)(R_ζ μ^k_ij + μ^l_ji)` uses **addition**.

Current code (line 156):
```python
b_rhs[he] = 0.5 * (mu_rot - mu[he_twin])  # WRONG: subtraction
```

### 1.2 Fix

Change line 156 to:
```python
b_rhs[he] = 0.5 * (mu_rot + mu[he_twin])  # CORRECT: addition per Algorithm 11
```

### 1.3 Test

```bash
python test_meshes.py
```

**Expected**: May reduce angle error slightly. Won't fix fragmentation (that's caused by cut graph issues).

**Metrics to check**:
- Angle error (should decrease)
- Flipped count (may change)
- Fill ratio (likely unchanged - dominated by cut graph issues)

---

## Phase 2: Cone Index Quantization (Medium Risk)

**File**: `cut_graph.py`
**Lines**: 199-216

### 2.1 Problem

Cone detection uses raw `c_vertex` (angle defect + omega0):
```python
c_vertex = K.copy()
for e in range(n_edges):
    i, j = mesh.edge_vertices[e]
    c_vertex[i] += omega0[e]
    c_vertex[j] -= omega0[e]

is_cone = np.abs(np.mod(c_vertex + np.pi/4, np.pi/2) - np.pi/4) > CONE_THRESHOLD
```

When omega0 is noisy, this creates ~29 false cones (sphere should have 4).

### 2.2 Fix

Quantize `c_vertex` to multiples of π/2 before threshold test:
```python
# Quantize cone index to nearest multiple of π/2
c_vertex_quantized = np.round(2 * c_vertex / np.pi) * (np.pi / 2)

# Cone if quantized value is non-zero (±π/2, ±π, ±3π/2, etc.)
# with tolerance for numerical noise
is_cone = np.abs(c_vertex_quantized) > 0.1
```

### 2.3 Test

```bash
python test_meshes.py
```

**Expected**:
- Fewer cones detected (4 for sphere, ~4 for torus)
- More edges pruned
- Higher fill ratio

**Metrics**:
- Count cones: should be ~4 for sphere
- Cut edges: should drop from ~62 to ~4-10
- Fill ratio: should increase significantly

### 2.4 Fallback

If quantization is too aggressive (creates flips), add hysteresis:
```python
is_cone = np.abs(c_vertex_quantized) > 0.1 or np.abs(c_vertex - c_vertex_quantized) > 0.3
```

---

## Phase 3: Cycle-Breaking in Cut Graph (Medium-High Risk)

**File**: `cut_graph.py`
**Lines**: 218-238

### 3.1 Problem

Current pruning only removes degree-1 non-cones. It doesn't:
- Remove cycles (cotree edges that don't connect singularities)
- Ensure the cut graph is a tree/forest

Section D of the supplement says:
> We then iteratively remove any degree-1 vertex from the primal graph, except for singular vertices, until no more vertices can be removed.

This is necessary but not sufficient. The cut graph should be a minimal tree connecting cones.

### 3.2 Fix

After current pruning, add cycle-breaking:

```python
def break_cycles_in_cut_graph(mesh, Gamma, is_cone):
    """
    Remove edges from cut graph until it forms a tree/forest.
    Preserve connectivity between cones.
    """
    # Build adjacency list for cut graph
    adj = {v: [] for v in range(mesh.n_vertices)}
    for e in range(mesh.n_edges):
        if Gamma[e] == 1:
            i, j = mesh.edge_vertices[e]
            adj[i].append((j, e))
            adj[j].append((i, e))

    # Find cone vertices
    cones = set(np.where(is_cone)[0])

    # BFS to find spanning tree connecting cones
    # Start from arbitrary cone (or vertex if no cones)
    start = next(iter(cones)) if cones else 0
    visited = {start}
    tree_edges = set()
    queue = [start]

    while queue:
        v = queue.pop(0)
        for neighbor, edge_idx in adj[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                tree_edges.add(edge_idx)
                queue.append(neighbor)

    # Remove non-tree edges (these are the cycles)
    for e in range(mesh.n_edges):
        if Gamma[e] == 1 and e not in tree_edges:
            Gamma[e] = 0

    return Gamma
```

### 3.3 Test

```bash
python test_meshes.py
```

**Expected**:
- Cut edges = |V_cones| - 1 + 2*genus (for tree connecting cones on genus-g surface)
- For sphere (g=0, 4 cones): ~3-4 cut edges
- For torus (g=1, ~4 cones): ~5-6 cut edges + genus loops
- Fill ratio should increase dramatically

### 3.4 Caution

This may break connectivity on genus > 0 surfaces. The cut must include homology generators (loops around handles). May need to preserve specific edges.

---

## Phase 4: Cross-field Holonomy Correction (High Risk)

**File**: `cross_field.py`
**Function**: `propagate_cross_field()` and/or `initialize_smooth_cross_field()`

### 4.1 Problem

BFS propagation accumulates holonomy errors:
- Transporting around a handle adds/subtracts rotation
- This corrupts omega0, creating spurious cone indices
- The smoothing in `initialize_smooth_cross_field()` is local, not holonomy-aware

### 4.2 Options

**Option A: Post-hoc holonomy correction**

After BFS propagation, measure holonomy around each handle and redistribute the error:

```python
def correct_holonomy(mesh, W, xi):
    """
    Measure and correct cross-field holonomy around handles.
    """
    # Find basis loops (homology generators)
    loops = find_homology_basis(mesh)  # TODO: implement

    for loop in loops:
        # Measure total rotation around loop
        holonomy = measure_holonomy(mesh, xi, loop)

        # Should be multiple of π/2 for cross field
        target = round(2 * holonomy / np.pi) * (np.pi / 2)
        error = holonomy - target

        # Distribute error along loop edges
        n_edges = len(loop)
        for edge in loop:
            xi[edge] -= error / n_edges

    return xi
```

**Option B: Use optimization-based cross field**

Replace BFS + smoothing with proper energy minimization:

```python
def optimize_cross_field(mesh, n_iters=1000):
    """
    Minimize cross-field smoothness energy with holonomy constraints.
    """
    # This is a more invasive change requiring:
    # - Smoothness energy: sum of squared angle differences (mod π/2)
    # - Holonomy constraints: total rotation around handles = k*π/2
    # - Solver: gradient descent or sparse linear solve
    pass
```

### 4.3 Recommendation

Defer Phase 4. Phases 1-3 may be sufficient. If fragmentation persists after Phase 3, revisit cross-field optimization.

### 4.4 Test

If implemented:
```bash
python test_meshes.py
```

**Expected**:
- omega0 values should be near multiples of π/2
- Fewer false cones
- Smoother cross field visualization

---

## Test Matrix

| Phase | Change | Test Command | Key Metrics |
|-------|--------|--------------|-------------|
| 1 | RHS sign | `python test_meshes.py` | Angle error ↓ |
| 2 | Cone quantization | `python test_meshes.py` | Cones ≈ 4, Cut edges ↓ |
| 3 | Cycle breaking | `python test_meshes.py` | Fill ratio ↑↑ |
| 4 | Holonomy fix | `python test_meshes.py` | omega0 quantized |

---

## Implementation Order

1. **Phase 1** (RHS sign): Safe, isolated change. Do first.
2. **Phase 2** (Cone quantization): Moderate risk. Test thoroughly.
3. **Phase 3** (Cycle breaking): Higher risk. May need iteration.
4. **Phase 4** (Holonomy): Only if still fragmented after Phase 3.

---

## Success Criteria

| Mesh | Current | Target |
|------|---------|--------|
| Sphere | 53% fill, 223 components | >90% fill, 1 component |
| Torus | 34% fill, many components | >80% fill, 1-2 components |

---

## Files Modified

| Phase | File | Lines |
|-------|------|-------|
| 1 | uv_recovery.py | 156 |
| 2 | cut_graph.py | 199-216 |
| 3 | cut_graph.py | 218-238 (add function) |
| 4 | cross_field.py | New function |

---

## References

- Paper Supplement, Section D: "Singularities and Cuts"
- Algorithm 11: RecoverParameterization (full pseudocode)
- bug-report.md: Current symptom analysis
