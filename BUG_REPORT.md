# Bug Report: Constraint System Not Converging

## Summary

The optimization constraint solver in `solve_constraints_only` does not converge. The constraint residual `|F|` stays at ~27-96 instead of converging to ~0. This causes the UV recovery to produce degenerate (collapsed) triangles.

## Symptoms

1. Running `python corman_crane.py` on sphere or torus meshes produces broken UV visualizations
2. Constraint residual `|F|` stuck at high values (~27 for torus, ~96 for sphere)
3. UV triangles collapse to thin slivers or points

## What Works

- **Mesh data structures** - Euler characteristic verified (χ=2 for sphere, χ=0 for torus)
- **Visualization pipeline** - Simple XY projection produces correct images
- **Jacobian computation** - Verified against finite differences (error < 1e-8)
- **Laplacian** - Verified: symmetric, row sums = 0, L = -div @ G

## Root Cause Investigation

The problem appears to be in **`cut_graph.py:compute_cut_jump_data`**, specifically in how `omega0` (reference frame rotation across edges) is computed.

### Evidence

From `debug_constraints.py` output on sphere320.obj:

```
Omega0 analysis:
  Range: [-6.9392, 6.9392]    # WAY TOO LARGE - more than 2π radians!
  Mean: -0.6715
  Std: 4.3364

  Vertex angle defect range: [0.0693, 0.0823]  # This is correct for a sphere
  Vertex omega0 sum range: [-25.2069, 24.4084] # Should be close to angle defect!
```

**Key insight**: For a smooth cross field on a sphere, `omega0` should be small (related to curvature, ~0.07 rad per vertex). But we're seeing values up to ±6.9 radians per edge and ±25 radians summed per vertex.

### The Constraint Equation

From `sparse_ops.py:build_constraint_system`:

```
rho[e] = (theta[f0] - theta[f1]) - omega0[e]  +  RHS_contributions
```

At initialization (u=v=theta=0), the RHS contributions are 0, so:
```
rho[e] = -omega0[e]
```

With `|omega0|` values up to 6.9, we get `|F| ≈ 96`, which matches observations.

### Suspect Code

In `cut_graph.py`, line 112:
```python
omega0[e] = phi[he_ij] - xi_star + np.pi
```

Where:
- `phi[he_ij]` = reference frame angle at halfedge
- `xi_star = zeta[e] + xi_he[he_twin]` = rotated cross field at twin
- `zeta[e] = (π/2) * n_star` = quarter-rotation jump

The formula should yield small values for a smoothly varying cross field, but it's producing large values.

## Files Involved

1. **`cut_graph.py`** - `compute_cut_jump_data()` computes omega0, zeta, phi, s, Gamma
2. **`sparse_ops.py`** - `build_constraint_system()` uses omega0 in constraint equations
3. **`optimization.py`** - `solve_constraints_only()` tries to solve F(u,v,theta) = 0
4. **`cross_field.py`** - `propagate_cross_field()` and `cross_field_to_angles()` compute xi

## Questions to Investigate

1. Is `omega0` being computed with the correct sign convention?
2. Is `phi` (reference frame) being propagated correctly across edges?
3. Should `omega0` be normalized to [-π, π] range?
4. Is there an indexing bug in how edges/halfedges/faces are related?
5. Does Algorithm 2 from the supplement have different conventions than our implementation?

## How to Reproduce

```bash
cd c:\Dev\Corman-Crane
python debug_constraints.py
```

This will print diagnostic information showing the omega0 distribution and residual analysis.

## Reference

The algorithm is from: **Corman & Crane, "Rectangular Parameterization", SIGGRAPH 2025**

Supplement PDF with Algorithm 2 (ComputeCutJumpData) is at:
`D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025-supplement.pdf`
