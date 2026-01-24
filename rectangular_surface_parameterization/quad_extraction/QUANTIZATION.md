# Quantization: The Missing Piece

## The Problem

After RSP produces a **seamless UV parameterization**, singularities (where the cross-field has ±90° rotation) sit at arbitrary real coordinates like `(2.347, 5.891)`. But quad extraction traces iso-lines at **integer** UV coordinates. If singularities don't land on integers, the extracted quad mesh has:

- Suboptimal quad density near singularities
- Triangular holes where singularities should become irregular vertices
- Distortion that could have been avoided

**Quantization** solves this by snapping singularities to integer coordinates `(2, 6)` while minimizing map distortion.

## Current Workaround (The Cheat)

```bash
python extract_quads.py mesh.obj --scale 10  # Scale UVs, hope for the best
```

This just multiplies all UVs by a constant. Singularities still land at arbitrary positions—we're just making the grid finer so the error is less noticeable. It's not a real solution.

## The Real Solution: Quantization

### Traditional Approach: T-Mesh (QGP)

The reference method [Campen et al. 2015] extracts a **T-mesh** (motorcycle graph) from the seamless map:

1. Trace iso-curves from all singularities simultaneously
2. These traces partition the surface into axis-aligned rectangular regions
3. Optimize integer edge lengths for each T-mesh arc
4. Propagate constraints back to original mesh

**Problems:**
- T-mesh is a complex data structure with T-junctions
- Requires careful handling of degenerate cases (zero-length edges)
- Re-embedding T-mesh into original mesh is tedious

### Better Approach: Decimated Mesh (QaWiTM)

**"Quad Mesh Quantization Without a T-Mesh"** [Coudert-Osmont et al. 2023] replaces the T-mesh with a simple decimated triangular mesh:

1. **Decimate** the original mesh (edge collapse + flips)
2. **Optimize** integer edge geometry on the coarse mesh
3. **Propagate** constraints via a linear matrix `D`

## Why QaWiTM is Awesome

### No Commercial Solver Required

| Approach | Solver Dependency |
|----------|-------------------|
| QuantizationYoann (MATLAB reference) | **Gurobi** (commercial, expensive) |
| QGP | Mixed-integer solver |
| **QaWiTM** | **None** - just Dijkstra's algorithm |

### Simpler Data Structures

```
T-Mesh (QGP):
- Polygonal faces with T-junctions
- Complex boundary decomposition
- Zero-length edges for aligned singularities
- Tedious re-embedding

Decimated Mesh (QaWiTM):
- Standard triangle mesh
- Half-edge data structure (we already have this!)
- No degenerate elements needed
- Matrix multiplication for re-embedding
```

### Same Quality

From the paper (Section 5.2):
> "quad mesh quality obtained from QGP and our method are very similar"

And Figure 13 shows side-by-side comparisons confirming equivalent output quality.

### Additional Flexibility

The decimated mesh approach naturally supports:

- **Free boundaries** - edges aren't forced to be axis-aligned
- **Feature curve alignment** - can be enforced at quantization step, not before
- **Direct quad extraction** on coarse mesh - fewer numerical precision issues

## Algorithm Overview

### Input
- Triangular mesh `M` with half-edges `{h}`
- Frame field `F` (from cross-field computation)
- Cut graph `C` (edges where the map is discontinuous)
- Seamless map `U^seamless` (from RSP)

### Output
- Grid-preserving map `U^grid` (singularities at integer coordinates)

### Steps

```
Algorithm 1: Grid Preserving Map Generation
───────────────────────────────────────────
1. U^seamless ← minimize f(U) subject to seamless constraints
   (This is RSP - we already have it!)

2. [A, ω] ← quantize(M, F, U^seamless)
   (This is what we need to implement)

3. U^grid ← minimize f(U) subject to AU = ω and seamless constraints
   (Re-optimize with integer constraints)
```

### Quantization Detail (Algorithm 3)

```
Algorithm 3: Quantize — Decimated Mesh Version
──────────────────────────────────────────────
Input: Triangular mesh M, seamless map U^seamless
Output: Linear system A, ω such that AU = ω implies integer constraints

// Step 1: Compute simpler proxy problem (§4.1)
1. M^proxy, D ← decimate(M)
2. U^proxy ← D · U^seamless

// Step 2: Optimize integer degrees of freedom (§4.2)
3. {ω_h} ← solve_edge_geometry(U^proxy)

// Step 3: Move solution back to original problem (§4.3)
4. A ← D
```

### Key Insight: Edge Geometry vs Edge Length

QGP optimizes **edge lengths** `l_i` (scalar, axis-aligned).

QaWiTM optimizes **edge geometry** `ω_h = U^-_{next(h)} - U^-_h` (complex/2D vector).

This allows edges that aren't axis-aligned, enabling:
- More expressive quad mesh structures
- No degenerate triangles for aligned singularities
- Direct coarse-to-fine map transfer

## Implementation Roadmap

### Phase 1: Mesh Decimation (§4.1)

```python
def decimate_mesh(vertices, triangles, uv_seamless, singularities):
    """
    Decimate mesh while preserving:
    - Singular vertices
    - Manifold topology
    - Positive Jacobian in UV space

    Returns:
        coarse_mesh: Decimated mesh
        D: Matrix such that U^proxy = D @ U^seamless
    """
```

Operations needed:
- Edge collapse (with UV coordinate linear combination)
- Edge flip (Delaunay criterion in UV space)
- Cut graph handling (move cuts before operations)

### Phase 2: Integer Optimization (§4.2)

```python
def optimize_integer_geometry(mesh_proxy, uv_proxy):
    """
    Find integer edge geometry ω that:
    - Is closed (boundary of each triangle sums to zero)
    - Minimizes distortion from seamless map
    - Keeps map valid (positive Jacobian)

    Returns:
        omega: Dict of half-edge → Gaussian integer (a + bi)
    """
```

Key components:
- Dual edge graph construction
- Dijkstra shortest path with custom weights
- Atomic operations (quad loop insertion/deletion)
- Scale-and-round initialization

### Phase 3: Constraint Propagation (§4.3)

```python
def propagate_constraints(D, omega):
    """
    Convert coarse mesh constraints to original mesh.

    Constraint: (D_{next(h)} - D_h) @ U^grid = ω_h

    Returns:
        A, omega: Linear system for final optimization
    """
```

### Phase 4: Final Optimization

Re-run seamless map optimization with added integer constraints. This uses the same solver as RSP but with additional linear equality constraints.

## Complexity Analysis

| Step | Time Complexity | Notes |
|------|-----------------|-------|
| Decimation | O(n log n) | n = original triangles |
| Integer optimization | O(m² log m) | m = coarse triangles, Dijkstra per edge |
| Constraint propagation | O(1) | Just matrix assignment |
| Final optimization | O(n) | Same as RSP |

From the paper (Figure 14):
- Decimation: ~10-1000ms depending on mesh size
- Quantization: ~0.1-10000ms depending on coarse mesh size
- Total: Negligible compared to full pipeline

## References

### Primary Paper

**Quad Mesh Quantization Without a T-Mesh**
Yoann Coudert-Osmont, David Desobry, Martin Heistermann, David Bommes, Nicolas Ray, Dmitry Sokolov
Computer Graphics Forum, 2023

- PDF: https://inria.hal.science/hal-04395861/file/main%20(1).pdf
- HAL: https://inria.hal.science/hal-04395861
- AlgoHex: https://www.algohex.eu/publications/quad-mesh-quantization-without-a-t-mesh/

### Background Papers

**Quantized Global Parametrization (QGP)** - The T-mesh approach this paper improves upon
Campen, Bommes, Kobbelt. ACM TOG 2015.
https://doi.org/10.1145/2816795.2818140

**Integer-Grid Maps for Reliable Quad Meshing** - Foundational decimation approach
Bommes, Campen, Ebke, Alliez, Kobbelt. ACM TOG 2013.
https://doi.org/10.1145/2461912.2462014

**Mixed-Integer Quadrangulation** - Original integer-grid map formulation
Bommes, Zimmer, Kobbelt. ACM TOG 2009.
https://doi.org/10.1145/1531326.1531383

## File Organization (Proposed)

```
quad_extraction/
├── __init__.py
├── extract_quads.py      # Main API (existing)
├── edge_tracer.py        # Ray tracing (existing)
├── quad_builder.py       # Quad construction (existing)
├── ...
│
├── quantization/         # NEW: QaWiTM implementation
│   ├── __init__.py
│   ├── decimate.py       # Mesh decimation with UV tracking
│   ├── dual_graph.py     # Dual edge graph for Dijkstra
│   ├── optimize.py       # Integer geometry optimization
│   ├── propagate.py      # Constraint propagation
│   └── quantize.py       # Main quantization API
│
└── QUANTIZATION.md       # This document
```

## Integration with Existing Pipeline

```python
# Current pipeline (with scaling cheat)
uv = run_rsp(mesh)
uv_scaled = uv * scale  # Cheat!
quads = extract_quads(mesh, uv_scaled)

# Proper pipeline (with quantization)
uv_seamless = run_rsp(mesh)
uv_grid = quantize(mesh, uv_seamless)  # NEW!
quads = extract_quads(mesh, uv_grid)
```

The `--scale` parameter becomes unnecessary (or optional for density control) once quantization is implemented.

## Why This Matters

1. **Correctness**: Singularities become proper irregular vertices (valence 3 or 5)
2. **Quality**: Optimal quad density everywhere, not just away from singularities
3. **Independence**: No Gurobi, no external C++ tools, pure Python
4. **Completeness**: The full pipeline, as the papers intended

This is the missing piece that turns a "working demo" into a "production implementation."
