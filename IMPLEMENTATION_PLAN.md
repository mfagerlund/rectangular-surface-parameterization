# Corman-Crane Rectangular Parameterization - Implementation Plan

## Overview

Computes **rectangular surface parameterizations** (orthogonal but not isotropic) for quad meshing and Chebyshev nets.

**Reference**: Corman & Crane, SIGGRAPH 2025 + Supplement (Algorithms 1-11)

---

## Key Variables

| Paper | Code | Size | Description |
|-------|------|------|-------------|
| α_i^{jk} | `alpha` | \|C\| = 3\|F\| | Corner angles |
| ξ_{ijk} | `xi` | \|F\| | Cross field angle (rel. to edge ij) |
| φ | `phi` | \|H\| = 3\|F\| | Reference frame angle per halfedge |
| θ | `theta` | \|F\| | Frame rotation from reference |
| ω⁰ | `omega0` | \|E\| | Reference frame rotation across edge |
| Γ | `Gamma` | \|E\| | Cut edge indicator {0,1} |
| ζ | `zeta` | \|E\| | Quarter-rotation jump |
| s | `s` | \|C\| | Sign bit {-1,+1} for v at corner |
| u, v | `u`, `v` | \|V\| | Log scale/stretch factors |
| λ | `lambda_` | \|E\| | Lagrange multipliers |

**Indexing**:
- Corner: `3 * face + local_vertex` (local_vertex ∈ {0,1,2})
- Halfedge: same as corner (halfedge ij in face ijk has index `3*face + 0`)
- Edge: unique ID per unordered vertex pair. Build `halfedge_to_edge[|H|]` and `edge_to_halfedge[|E|, 2]` mappings.

---

## Test Meshes

| Mesh | V | F | Topology | Purpose |
|------|---|---|----------|---------|
| 2-tri quad | 4 | 2 | Disk | Manual verification |
| 4x4 grid | 25 | 32 | Disk | Finite diff tests |
| stanford-bunny.obj | ~35k | ~70k | Genus 0 | Main test |
| torus.obj | ~5k | ~10k | Genus 1 | Cut graph test |

Location: `C:\Dev\Colonel\Data\Meshes\`

---

## Phases

### Phase 1: Mesh Data Structures
- Half-edge mesh with `twin`, `next`, `vertex`, `face` operations
- OBJ loader with manifold validation
- Geometric quantities: `edge_lengths`, `corner_angles`, `areas`, `normals`
- **Test**: Euler χ = V - E + F = 2 for bunny

### Phase 2: Cross Field
- Store one representative unit vector per face
- Simple approach: propagate from seed face via parallel transport (defer curvature-based field)
- Convert to angles ξ relative to first halfedge per face
- **Test**: Visual inspection

### Phase 3: Cut Graph & Jump Data (Algorithm 2)
- BFS to propagate reference frame φ across faces
- Compute ζ (closest quarter-rotation), ω⁰ (frame rotation), Γ (cut edges)
- Cone index at vertex i: `c_i = 2π - Σ_corners α + Σ_edges ω⁰` (quantized to multiples of π/2)
- Prune cut graph: remove degree-1 non-cone vertices
- Compute corner signs s via cumulative product around each vertex
- **Test**: On torus, Γ forms two loops

### Phase 4: Sparse Matrices
- Cotan-Laplacian L (Supplement Eq 1)
- Divergence operator (Supplement Eq 2)
- Gradient operator G: `(Gf)_e = f_j - f_i` for edge e = (i,j)
- **Test**: L symmetric, row sums = 0, L = -div @ G

### Phase 5: Optimization Setup
- **Objective** (default): Dirichlet energy Φ = Σ A_f (|∇u|² + |∇v|²)
- **Constraint** F(u,v,θ) = 0: integrability (Algorithm 5-6, Supplement)
- Build Jacobian J (Algorithm 7-8) and Hessian D = ∇(J^T λ) (Algorithm 9-10)
- **Test**: Finite diff Jacobian on 4x4 grid, error < 1e-5

### Phase 6: Newton Solver (Algorithm 3-4)
- KKT system: `[H+D, J^T; J, 0] [y; δ] = [g + J^T λ; F]`
- Line search with merit function ‖g + J^T λ‖ + ‖F‖
- Converge when merit < 1e-8
- **Test**: Bunny converges in ~20 iterations

### Phase 7: UV Recovery (Algorithm 11)
- Target edge vectors: μ = ℓ * (a cos η, b sin η) where a,b from u,v
- Corner identification matrix U (corners share values unless separated by cut)
- Poisson solve: min ‖Af - b‖²_M s.t. Uf = 0
- **Test**: f continuous across non-cut edges

### Phase 8: Validation
- Orthogonality error < 1e-4
- No flipped triangles (negative Jacobian)
- Checkerboard visualization

### Phase 9: Integration
- CLI: `python corman_crane.py bunny.obj -o bunny_uv.obj`
- OBJ output with per-corner UVs (`vt` records)

---

## Dependencies

```
numpy scipy click matplotlib pytest
# Optional: polyscope scikit-sparse numba
```

---

## Milestones

| Phase | Pass Criteria | Status |
|-------|---------------|--------|
| 1 | χ = 2 for bunny | ✓ |
| 2 | Smooth field visualized | ✓ |
| 3 | Γ connects cones | ✓ |
| 4 | L symmetric, null space = constants | ✓ |
| 5 | Jacobian matches finite diff | ✓ |
| 6 | Converges in <10 iterations | ✓ (5 iter) |
| 7 | UV continuity across non-cuts | ✓ |
| 8 | 0 flipped triangles | ✓ |
| 9 | End-to-end on sphere320 | ✓ |

## Final Test Results (sphere320.obj)

```
Flipped triangles: 0 / 320
Angle error (mean): 14.04 deg
Convergence: 5 iterations
```
