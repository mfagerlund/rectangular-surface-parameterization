# Fix Flipped Triangles in UV Recovery

## Problem Statement

The UV recovery phase produces flipped triangles (~0.78%) on genus > 0 surfaces (e.g., torus).

**Root cause (identified in Stage 1):** Two bugs in `uv_recovery.py`:

1. **Bug 1 (line 140):** Cut edges skip RHS averaging - should average with rotation like non-cut edges
2. **Bug 2 (lines 275-284):** Adds edge-vector constraints for cut edges - paper adds NO constraints

**Original hypothesis (WRONG):**
> The paper requires `e_A = R_zeta * e_B`, which couples u/v and requires a coupled solve.

**Actual fix (from Algorithm 11):**
- Apply rotated averaging `b = (1/2)(R_ζ μ_A + μ_B)` to ALL interior edges (cut and non-cut)
- For cut edges: add NO constraints (corners are independent)
- No coupled solve needed - the existing separate u/v solves work fine

---

## Stage 1: Research (Read Papers)

### 1.1 Main Paper: Corman & Crane, "Rectangular Parameterization", SIGGRAPH 2025

**Questions to answer:**
- [x] How does Algorithm 11 (RecoverParameterization) handle cut edges?
- [x] What is the exact definition of zeta? Direction of rotation (A->B or B->A)?
- [x] Are there any special cases for genus > 0 surfaces?
- [x] What is the relationship between zeta, phi, and theta at cut edges?

**Findings:**

The main paper defers cut handling details to the supplement (Section D).

### 1.2 Supplemental Material

**Questions to answer:**
- [x] Full pseudocode for Algorithm 11
- [x] Definition of variables at cut edges (Gamma, zeta)
- [x] How are edge vectors transformed across cuts?
- [x] Any notes on handling higher-genus surfaces?

**Findings:**

---

#### Equation 3: Discrete Poisson (Page 1, Section B.3)

> Finally, we discretize the Poisson equation in Equation 13 as a scalar Poisson equation for
> each coordinate f¹, f² of the final rectangular parameterization f : V → R², namely
>
>     L f^p = div μ^p,    p = 1, 2.                                 (Equation 3)
>
> Here L, div, and μ are the discrete Laplacian, divergence, and differential given in
> Equations 1, 2, and 18, resp.

This is the basic Poisson solve that Section D modifies for cut edges.

---

#### FULL TEXT: Section D (Singularities and Cuts) - Page 2

> In the case where either (i) the field has singularities and/or (ii) the domain M is not a
> topological disk, we must effectively cut along the curve Γ, plus additional topological cuts,
> before flattening the mesh. In practice, we compute topological cuts by considering the
> traversal tree used to construct Γ, and taking the complementary graph of primal edges not
> crossed by any edge in this dual tree. We then iteratively remove any degree-1 vertex from
> the primal graph, except for singular vertices, until no more vertices can be removed. The
> remaining primal edges define the cut. (This approach is similar in spirit to the tree-cotree
> strategy of Eppstein [2002].)
>
> This problem of solving for a parameterization on a cut mesh comes up frequently in geometry
> processing, and is a perennial nuisance. Hence, we will first introduce a perspective that
> simplifies the situation both conceptually and in terms of implementation, before giving the
> details of our particular algorithm. **We make just one simple change to our basic setup: rather
> than store values at vertices, we will store values at all triangle corners, using linear
> constraints to identify equivalent values as needed.** In the absence of any cuts, our formulation
> ultimately yields the same solution as before, but with the additional flexibility to easily
> incorporate cuts.
>
> In more detail, for any function f_i at vertices, we now store values f_i^{jk} at all triangle
> corners. A sparse matrix U ∈ R^{m×|C|} encodes equivalence of values at different corners,
> where m is the number of identifications, and |C| is the number of corners. **Two corners around
> a common vertex are equivalent if and only if they are not separated by a cut edge.** For the
> nth equivalence f_i^{jk} = f_i^{kl} of consecutive corners, U will have nonzero entries
> U_{n,i}^{jk} = 1 and U_{n,i}^{kl} = −1, i.e., the equation number determines the row index,
> and the corners are specified via the column indices.
>
> Next, suppose we want to solve any linear equation Af = b, which in general may be under- or
> over-determined. We can express this system in terms of values on corners, plus the
> identification matrix U, as a least-squares problem subject to linear constraints:
>
>     min_{f∈R^|C|} ||Af − b||²_M   s.t. Uf = 0.                    (Equation 6)
>
> Here |A|²_M := A^T M A denotes the ℓ-2 norm with a mass matrix M ∈ R^{|C|×|C|}. We use in
> particular a diagonal matrix with nonzero entries
>
>     M_{i,i}^{jk,jk} = (1/2) cot α_i^{jk}
>
> i.e., we use the cotan weight associated with each corner. In the case where there is no cut,
> our problem amounts to solving the exact same cotan-Poisson problem in Equation 3.
>
> Equation 6 can be solved using any available method—for instance, the method of Lagrange
> multipliers, we find the solution to this problem is given by the block linear system
>
>     [ A^T M A    U^T ]  [ f ]   [ A^T M b ]
>     [    U       0   ]  [ λ ] = [    0    ]
>
> where λ ∈ R^m are the Lagrange multipliers. By formulating the problem this way, rather than
> directly applying least-squares to the combined system [AU]^T f = [b 0], we ensure that corner
> values are identified exactly, and only the original equations are approximated.
>
> **In the case of our particular problem, we also have to modify Equation 3, splitting it into
> two different equations on the two sides of every edge. In particular, for each edge ij in Γ
> we modify the right-hand side so that both vectors μ^k_ij, μ^l_ji are in the same coordinate
> system.** In particular, consider the angles β_ij from Section 5.3.1, which give the rotation
> between frames (after parallel transport) across any cut edge ij. Hence, to evaluate
> Equation 3 for an edge from corner i^{jk} to corner j^{ki}, we use
>
>     f_j^{ki} − f_i^{jk} = (1/2)(μ^k_ij + R_{−ζ_ij} μ^l_ji),      (Equation 7)
>
> where **ζ_ij := (π/2)⌈(β − π/4)/(π/2)⌉ is the closest quarter-rotation taking the frame in
> triangle ijk to the frame in triangle jil.** In other words, we account for the jump in the
> frame across the cut. Likewise, on the other side of the edge, with endpoints at corners
> i^{lj} and j^{il}, we have
>
>     f_j^{il} − f_i^{lj} = (1/2)(R_{ζ_ij} μ^k_ij + μ^l_ji).       (Equation 8)

---

#### FULL TEXT: Algorithm 11 - RecoverParameterization (Page 6)

```
Algorithm 11 RecoverParameterization(M, Γ, ζ, ℓ, α, φ, θ, u, v)

Input: The mesh connectivity M = (V, E, F), the cut curve Γ : E → {0, 1} cutting
       the mesh to disk topology, the closest quarter-rotation ζ : E → R between
       two frames in adjacent triangles, the length ℓ_ij ∈ R≥0 of each edge ij ∈ E,
       angles α_i^{jk} at each triangle corner, the angle φ_ijk ∈ R of the reference
       frame relative to edge ij in each triangle ijk, the angle θ_ijk of the
       optimized frame relative to the reference frame, and the log scale/stretch
       factors u_i, v_i at each vertex i ∈ V.

Output: f : C → R² giving the uv-coordinates per triangle corner.

 1: for each ij ∈ E do                          ▷ edge scale from log-scale
 2:     a_ij ← e^{(u_i + u_j + v_i + v_j)/2}
 3:     b_ij ← e^{(u_i + u_j − v_i − v_j)/2}

 4: for each ijk ∈ F do
 5:     η_ij ← φ_ijk + θ_ijk                    ▷ angle of X_1^θ relative to each edge
 6:     η_jk ← η_ij − (π − α_j^{ki})
 7:     η_ki ← η_jk − (π − α_k^{ij})
 8:                                              ▷ target edge vector
 9:     μ^k_ij ← ℓ_ij(+a_ij cos(η_ijk), +b_ij sin(η_ijk))
10:     μ^i_jk ← ℓ_jk(−a_jk cos(η_ijk + α_j^{ki}), −b_jk sin(η_ijk + α_j^{ki}))
11:     μ^j_ki ← ℓ_ki(−a_ki cos(η_ijk − α_i^{jk}), −b_ki sin(η_ijk − α_i^{jk}))

12:                                              ▷ right hand-side (THIS IS THE KEY LINE)
13: for each k^{ij} ∈ C do b_k^{ij} ← (1/2)(R_{ζ_ij} μ^k_ij + μ^l_ji)

14:
15: L_A ← ()
16: L_U ← ()
17: p ← 0
18: for each k^{ij} ∈ C do
19:                                              ▷ build matrix A in Equation 6
20:     L_A ← Append(L_A, (k^{ij}, i^{jk}, +1))
21:     L_A ← Append(L_A, (k^{ij}, j^{ki}, −1))
22:                                              ▷ build matrix U in Equation 6
23:     if Γ_ij == 0 then                        ▷ if ij NOT in the cut, add corner constraints
24:         L_U ← Append(L_U, (p, i^{jk}, +1))
25:         L_U ← Append(L_U, (p, i^{lj}, −1))
26:         L_U ← Append(L_U, (p + 1, j^{ki}, +1))
27:         L_U ← Append(L_U, (p + 1, j^{il}, −1))
28:         p ← p + 2
                                                 ▷ Build matrices from lists of nonzeros
29: A ← SparseFromTriplets(L_A, |C|, |C|)
30: U ← SparseFromTriplets(L_U, p, |C|)
31: f ← SolveQP(A^T W A, −A^T W b, U, 0)         ▷ corner coordinates
```

---

#### KEY OBSERVATIONS FROM ALGORITHM 11

1. **Line 13 is critical**: The RHS `b` is computed as `(1/2)(R_{ζ} μ^k_ij + μ^l_ji)` for ALL corners.
   - R_ζ is a 2D rotation matrix by angle ζ
   - This averaging with rotation happens for EVERY edge, not just non-cut edges
   - The rotation brings both μ vectors into the same coordinate system

2. **Lines 23-28**: Corner identification constraints (U matrix)
   - Only added when `Γ_ij == 0` (edge is NOT in the cut)
   - When `Γ_ij == 1` (edge IS in the cut): NO constraints added, corners are independent
   - This is the ONLY difference between cut and non-cut edges in corner handling

3. **No edge-vector matching**: The algorithm does NOT add any constraint like `e_A = R_ζ e_B`.
   The periodicity is handled implicitly through the rotated RHS averaging.

---

#### NOTATION MAPPING (Paper → Code)

| Paper Notation | Code Variable | Description |
|----------------|---------------|-------------|
| Γ_ij | `Gamma[e]` | Cut edge indicator (0=not cut, 1=cut) |
| ζ_ij | `zeta[e]` | Quarter-rotation jump (0, π/2, π, 3π/2) |
| μ^k_ij | `mu[he]` | Target edge vector for halfedge |
| α_i^{jk} | `alpha[c]` | Corner angle at corner c |
| φ_ijk | `phi[he]` | Reference frame angle per halfedge |
| θ_ijk | `theta[f]` | Frame rotation from reference |
| R_ζ | 2D rotation | `[[cos(ζ), -sin(ζ)], [sin(ζ), cos(ζ)]]` |

---

**Critical insight**: The paper does NOT add edge-vector matching constraints for cut edges. Instead:
- The RHS already includes proper rotation when averaging μ values
- Cut edges simply have NO corner identification (corners are independent)
- This works because the Poisson solve finds a least-squares solution

### 1.3 Current Implementation Analysis

**Files involved:**
- `cut_graph.py`: Computes Gamma (cut indicator), zeta (rotation jump), phi, omega0
- `uv_recovery.py`: Algorithm 11 implementation (lines 21-329)

**Current zeta values on torus cut edges:**
- 0 (no rotation)
- π/2 (90° rotation)
- 3π/2 (-90° rotation)

**Questions:**
- [x] Is zeta the rotation from A->B or B->A?
- [x] How is zeta computed in `cut_graph.py`? (check lines 52-57, propagation logic)
- [x] What is the relationship between zeta and the cross-field matching?

**Findings:**

1. **Zeta direction**: ζ rotates FROM triangle ijk TO triangle jil (A→B direction).
   - In `cut_graph.py:113`: `zeta[e] = (np.pi / 2) * n_star`
   - n* is computed from `phi_ij_to_ji - xi_he[he_twin]`

2. **Current implementation bugs in `uv_recovery.py`**:

---

#### BUG 1: RHS computation skips rotation for cut edges (lines 140-154)

**Current code:**
```python
# uv_recovery.py lines 140-154
if he_twin == -1 or Gamma[e] == 1:
    # Boundary or cut edge - use local μ (no averaging across the edge)
    b_rhs[he] = mu[he]                          # <-- WRONG for cut edges
else:
    # Interior edge - rotate current mu by oriented zeta, subtract twin
    he0 = mesh.edge_to_halfedge[e, 0]
    z = zeta[e] if he == he0 else -zeta[e]
    cos_z, sin_z = np.cos(z), np.sin(z)

    mu_rot = np.array([
        cos_z * mu[he, 0] - sin_z * mu[he, 1],
        sin_z * mu[he, 0] + cos_z * mu[he, 1]
    ])

    b_rhs[he] = 0.5 * (mu_rot - mu[he_twin])    # <-- Note: using SUBTRACTION
```

**Problems:**
1. Cut edges (`Gamma[e] == 1`) skip the rotation averaging entirely
2. Uses subtraction (`-`) instead of addition (`+`) for averaging
3. Paper's Algorithm 11 line 13: `b = (1/2)(R_ζ μ^k_ij + μ^l_ji)` uses ADDITION

---

#### BUG 2: Adds incorrect edge-vector constraints for cut edges (lines 272-284)

**Current code:**
```python
# uv_recovery.py lines 272-284
else:
    # Cut edge: for genus > 0 surfaces, enforce edge-vector matching
    # This creates periodic boundary conditions
    if use_periodic:
        # BUG: This is a simplification that causes flipped triangles.
        # Currently enforcing e_A = e_B (ignoring rotation by zeta).
        # The paper says e_A = R_zeta * e_B, which requires a coupled u/v solve.

        U_rows.extend([constraint_idx, constraint_idx, constraint_idx, constraint_idx])
        U_cols.extend([c0_end, c0_start, c1_start, c1_end])
        U_data.extend([1.0, -1.0, -1.0, 1.0])
        constraint_idx += 1
    # else: genus 0 - cut edges have no constraints (unfold to disk)
```

**This constraint encodes:** `f[c0_end] - f[c0_start] - f[c1_start] + f[c1_end] = 0`

Which is: `(f[c0_end] - f[c0_start]) = (f[c1_start] - f[c1_end])` i.e., `e_A = e_B`

**Problem:** The paper's Algorithm 11 lines 23-28 show that for cut edges (`Γ_ij == 1`),
NO constraints should be added at all. The periodicity is handled implicitly by the
rotated RHS averaging, not by explicit constraints.

---

3. **Root cause**: The current code tries to enforce periodicity via edge-vector
   constraints, but ignores the rotation ζ. This creates inconsistent constraints
   that cause flipped triangles.

4. **Paper's approach** (from Algorithm 11):
   - Compute RHS with rotation for ALL edges: `b = 1/2(R_ζ * μ_A + μ_B)`
   - For non-cut edges: identify corners (U constraints)
   - For cut edges: NO corner identification, NO edge-vector constraints
   - The least-squares solve naturally handles the discontinuity

---

## Stage 2: Implementation Plan

**Based on Stage 1 findings, the fix is simpler than expected!**

The paper does NOT require a coupled u/v solve. The fix is:
1. Fix the RHS computation to include rotation for cut edges
2. Change subtraction to addition in RHS averaging
3. REMOVE the incorrect edge-vector matching constraints for cut edges

### 2.1 Fix 1: RHS Computation (lines 136-154)

**Current (incorrect):**
```python
# uv_recovery.py lines 140-154
if he_twin == -1 or Gamma[e] == 1:
    b_rhs[he] = mu[he]  # No averaging for cut edges  <-- BUG: should average
else:
    ...
    b_rhs[he] = 0.5 * (mu_rot - mu[he_twin])  # <-- BUG: should be + not -
```

**Fixed (matching Algorithm 11 line 13):**
```python
# Replace lines 140-154 with:
if he_twin == -1:
    # Boundary edge only - no twin to average with
    b_rhs[he] = mu[he]
else:
    # ALL interior edges (cut or not) - average with rotation
    he0 = mesh.edge_to_halfedge[e, 0]
    z = zeta[e] if he == he0 else -zeta[e]
    cos_z, sin_z = np.cos(z), np.sin(z)

    # Rotate mu[he] by zeta
    mu_rot = np.array([
        cos_z * mu[he, 0] - sin_z * mu[he, 1],
        sin_z * mu[he, 0] + cos_z * mu[he, 1]
    ])

    # Paper uses ADDITION: b = (1/2)(R_ζ μ^k_ij + μ^l_ji)
    b_rhs[he] = 0.5 * (mu_rot + mu[he_twin])
```

**Key changes:**
1. Remove `Gamma[e] == 1` condition - cut edges should also be averaged
2. Change `mu_rot - mu[he_twin]` to `mu_rot + mu[he_twin]` (paper uses addition)

### 2.2 Fix 2: Remove Edge-Vector Constraints (lines 272-284)

**Current (incorrect):**
```python
# uv_recovery.py lines 272-284
else:  # Cut edge
    if use_periodic:
        # Edge-vector matching constraint (WRONG - paper adds NO constraints)
        U_rows.extend([constraint_idx, constraint_idx, constraint_idx, constraint_idx])
        U_cols.extend([c0_end, c0_start, c1_start, c1_end])
        U_data.extend([1.0, -1.0, -1.0, 1.0])
        constraint_idx += 1
```

**Fixed (matching Algorithm 11 lines 23-28):**
```python
# Replace lines 272-284 with:
else:  # Cut edge (Gamma[e] == 1)
    # Paper's Algorithm 11: NO constraints for cut edges
    # Corners on opposite sides of cut are independent
    # Periodicity handled implicitly by rotated RHS averaging
    pass
```

### 2.3 Summary of Changes

| Location | Current | Fixed |
|----------|---------|-------|
| Line 140 | `if he_twin == -1 or Gamma[e] == 1:` | `if he_twin == -1:` |
| Line 154 | `0.5 * (mu_rot - mu[he_twin])` | `0.5 * (mu_rot + mu[he_twin])` |
| Lines 275-284 | Add edge-vector constraints | Delete entire block (just `pass`) |

### 2.4 Implementation Steps

1. [x] ~~Verify zeta direction from paper~~ (A→B direction confirmed)
2. [ ] Fix RHS: remove `Gamma[e] == 1` from boundary check (line 140)
3. [ ] Fix RHS: change `-` to `+` in averaging (line 154)
4. [ ] Remove edge-vector constraints for cut edges (lines 275-284)
5. [ ] Test on torus - expect 0 flips
6. [ ] Test on sphere - verify no regression (should still have 0 flips)

---

## References

**Local PDFs:**
- Main paper: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025.pdf`
- Supplemental: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025-supplement.pdf`

**Key sections in supplement:**
- Section D (page 2): Singularities and Cuts - explains corner-based formulation and cut handling
- Equations 7 & 8: RHS formulas for cut edges with rotation
- Algorithm 11 (page 6): Full RecoverParameterization pseudocode

**Key algorithms:**
- Algorithm 2: ComputeCutJumpData (cut_graph.py)
- Algorithm 11: RecoverParameterization (uv_recovery.py)

---

## Test Cases

| Mesh | Genus | Current Flips | Expected After Fix |
|------|-------|---------------|-------------------|
| sphere320.obj | 0 | 0 | 0 |
| torus.obj | 1 | 9 (0.78%) | 0 |

---

## Notes

_(Space for additional findings during research)_
