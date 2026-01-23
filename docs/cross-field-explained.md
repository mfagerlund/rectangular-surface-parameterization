# Cross-Field in Corman-Crane Rectangular Parameterization

## Summary

In the Corman-Crane rectangular parameterization method, a **cross-field is an INPUT** to the algorithm, not computed by it. The paper explicitly states that cross-field generation is handled by "existing algorithms" and is considered pre-processing. The cross-field determines where singularities (cones) appear and ultimately controls the structure of the quad mesh.

The paper's Algorithm 1 (FlattenMesh) has this signature:
```
Input: ... a cross field given by any one of four representative unit vectors W_ijk in each triangle ijk
```

This is critical: the paper does NOT describe how to compute the cross-field. It assumes you have one.

---

## Definition

### What is a Cross-Field?

A **cross-field** (also called a 4-direction field or 4-RoSy field) assigns four orthogonal directions to each point on the surface. Due to the 4-fold rotational symmetry, we only store ONE representative unit vector per face, since the other three directions are obtained by rotating by multiples of pi/2.

### Key Variables

| Symbol | Name | Size | Description |
|--------|------|------|-------------|
| W | Cross-field vectors | \|F\| x 3 | One representative unit tangent vector per face |
| xi (ξ) | Cross-field angles | \|F\| | Angle of W relative to first edge of each face |

### Angle Representation

The paper uses angle representation for efficiency. From Algorithm 1, line 9:
```
ξ_ijk = atan2(<T2, W_ijk>, <T1, W_ijk>)
```

Where:
- `T1` = unit vector along first edge ij of triangle ijk
- `T2` = N x T1 (perpendicular tangent vector)
- `N` = face normal

This converts the 3D unit vector W to a single angle ξ relative to the local coordinate frame (T1, T2).

---

## Input or Computed?

### The Paper's Position

**The cross-field is an INPUT.** From the supplement, Section E (Pseudocode):

> "We omit pre-processing (namely, generation of the reference field) and post-processing (namely, quantization and contouring of the parameterization in the case of meshing), since these steps are **handled by existing algorithms**."

The paper explicitly defers cross-field computation to external methods. This means:
1. They assume you already have a smooth cross-field
2. Cross-field singularities determine cone locations
3. The quality of the cross-field directly affects the parameterization quality

### Our Implementation

In the current codebase (`cross_field.py`), we use a simple BFS propagation approach:
1. Start from a seed face with initial direction (typically T1)
2. Propagate via parallel transport across edges
3. Optionally smooth with local averaging

**This is NOT what the paper authors used.** They likely used an optimization-based cross-field method (see "What the Paper Doesn't Say" below).

---

## Required Properties

Based on Algorithm 1 inputs, a valid cross-field must satisfy:

### 1. Unit Length
```
|W_ijk| = 1 for all faces ijk
```
The vectors must be unit vectors.

### 2. Tangent to Surface
```
<W_ijk, N_ijk> = 0 for all faces ijk
```
Each W must lie in the tangent plane of its face.

### 3. Smoothness (Implicit)
While not explicitly stated, the cross-field should be "smooth" across edges. A noisy cross-field creates many spurious singularities, leading to fragmented UV layouts.

### 4. Four-Fold Symmetry
The cross-field has 4-fold symmetry: if W is a valid direction, so are W rotated by pi/2, pi, and 3*pi/2. When comparing across edges, we match to the "closest" of these four directions.

---

## Testable Properties (for verification)

### 1. Unit Length Check
```python
def test_unit_length(W):
    norms = np.linalg.norm(W, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
```

### 2. Tangent Check
```python
def test_tangent(W, normals):
    dots = np.sum(W * normals, axis=1)
    assert np.allclose(dots, 0.0, atol=1e-6)
```

### 3. Singularity Index Sum (Gauss-Bonnet for Cross-Fields)

For a cross-field on a surface of Euler characteristic chi:
```
Sum of cone indices = 2*pi*chi
```

For genus 0 (sphere): chi = 2, so sum of indices = 4*pi = 8 * (pi/2)

Since cross-field singularities have indices that are multiples of pi/2:
- **Sphere (genus 0):** Typically 4 cones of index +pi/2 each, or 8 of +pi/4 each
- **Torus (genus 1):** chi = 0, so indices must sum to 0 (can have no singularities, or balanced +/- pairs)

### 4. Expected Number of Singularities

| Surface | Genus | Chi | Expected Cones |
|---------|-------|-----|----------------|
| Sphere | 0 | 2 | 4-8 (typically 4 of index +1/4 turn) |
| Torus | 0 | 0 | 0 or balanced pairs |
| Double torus | 2 | -2 | Need negative curvature singularities |

**CRITICAL TEST:** Our sphere320.obj was showing 29 "cones" instead of ~4. This indicates a problem in either:
1. The cross-field (too noisy)
2. The cone detection threshold
3. The omega0 computation

### 5. Smoothness Energy

Cross-field smoothness can be measured as:
```python
def smoothness_energy(W, mesh):
    energy = 0
    for each edge:
        # Transport W from face0 to face1
        w_transported = parallel_transport(W[f0], N[f0], N[f1])
        # Find angle to closest of 4 cross directions
        diff = angle_mod_pi_over_2(W[f1], w_transported)
        energy += diff^2
    return energy
```

A good cross-field should have low smoothness energy. For the paper's results, energy should be close to zero except at singularities.

### 6. No NaN or Inf Values
```python
assert np.all(np.isfinite(W))
assert np.all(np.isfinite(xi))
```

---

## Relationship to Other Pipeline Stages

### Cross-Field -> Cut Graph (Algorithm 2)

The cross-field directly determines:

1. **phi (reference frame angles):** Initialized from the cross-field
   ```
   phi_ijk = xi_ijk  (Algorithm 2, line 7)
   ```

2. **omega0 (frame rotation across edges):** Measures how much the cross-field rotates across each edge
   ```
   omega0_ij = phi_ij - xi* + pi  (Algorithm 2, line 15)
   ```
   Where xi* is the closest of the 4 cross directions in the neighboring face.

3. **zeta (quarter-rotation jump):** The discrete rotation needed to match cross-field directions
   ```
   zeta_ij = (pi/2) * n*  where n* = Round(2*(phi_ij->ji - xi_ji)/pi) mod 4
   ```

4. **Cone vertices:** A vertex is a cone if its "cone index" is non-zero:
   ```
   c_i = (2*pi - sum of angles at i) + sum of omega0 around i
   ```
   Cones happen where the cross-field has singularities.

### Cross-Field -> Optimization (Algorithms 3-8)

The cross-field affects optimization through:

1. **omega0:** Part of the constraint equations (Equation 17 in supplement)
2. **phi:** Used to express the current frame angle theta relative to the cross-field
3. **s (sign bits):** Determined by parity of zeta, which comes from cross-field matching

### Cross-Field -> UV Recovery (Algorithm 11)

The final UV coordinates align with the cross-field:
- The u iso-lines follow one cross direction
- The v iso-lines follow the perpendicular direction
- Singularities in the cross-field become cone singularities in the UV map

---

## Relevant Equations

### From Algorithm 1 (FlattenMesh)

Line 7-9: Compute cross-field angles relative to edge ij
```
T1 = Unit(x_j - x_i)
T2 = N x T1
xi_ijk = atan2(<T2, W_ijk>, <T1, W_ijk>)
```

### From Algorithm 2 (ComputeCutJumpData)

Lines 1-4: Express cross directions relative to halfedges
```
xi_ij = xi_ijk
xi_jk = xi_ij - (pi - alpha_j^ki)
xi_ki = xi_jk - (pi - alpha_k^ij)
```

Line 12: Closest cross index
```
n* = Mod(Round(2 * (phi_ij->ji - xi_ji) / pi), 4)
```

Line 13: Jump angle
```
zeta_ij = (pi/2) * n*
```

Line 15: Frame rotation
```
omega0_ij = phi_ij - xi* + pi
```

Lines 40-45: Cone indices
```
c_i = K_i + sum_j(omega0_ij)  where K_i = 2*pi - sum of angles at vertex i
```

---

## What the Paper Doesn't Say

### 1. Cross-Field Generation Method

The paper completely defers this to "existing algorithms." They likely used one of:
- Knoppel et al. 2013: "Globally Optimal Direction Fields" (eigenvalue problem)
- Ray et al. 2008: "N-Symmetry Direction Field Design" (optimization-based)
- Bommes et al. 2009: "Mixed-Integer Quadrangulation" (MIQ) approach

### 2. Singularity Placement Strategy

The paper doesn't discuss:
- How to choose where singularities should go
- Whether to prescribe cone locations
- How to balance number of cones vs. distortion

### 3. Numerical Tolerances

No explicit tolerances mentioned for:
- How smooth is "smooth enough"
- How close to multiples of pi/2 omega0 should be
- Threshold for cone detection

### 4. Boundary Handling for Open Meshes

The supplement says "For simplicity we assume M is without boundary." No guidance for meshes with boundary.

### 5. Initialization Quality Requirements

The paper doesn't say how good the cross-field needs to be for convergence. A noisy cross-field might:
- Create too many cones
- Make the optimization harder to solve
- Cause UV fragmentation

---

## Implications for Our Implementation

### Current Problem

Our BFS-based cross-field propagation (`propagate_cross_field()`) is causing:
1. Too many detected cones (29 instead of ~4 for sphere)
2. Noisy omega0 values
3. Fragmented UV layout

### Possible Solutions

1. **Better cross-field algorithm:** Use optimization-based methods (Knoppel, Ray, etc.)

2. **Post-processing:** Smooth the cross-field before passing to Algorithm 2

3. **Different cone detection:** Instead of thresholding c_vertex, use topological analysis to find actual singularities

4. **Prescribe cones:** For known surfaces (sphere), manually place 4 cones at optimal locations

### Priority

The cross-field is the ROOT CAUSE of our downstream problems. Before fixing optimization or UV recovery, we need a better cross-field.

---

## References

### Paper Citations

1. **Section in Supplement:** Algorithm 1 (FlattenMesh) - lines 3-9 define cross-field input
2. **Section in Supplement:** Algorithm 2 (ComputeCutJumpData) - uses cross-field for frame propagation
3. **Section in Supplement:** Section E intro - "We omit pre-processing (namely, generation of the reference field)"

### External References for Cross-Field Computation

1. Knoppel, Crane, Pinkall, Schroder. "Globally Optimal Direction Fields." SIGGRAPH 2013.
2. Ray, Vallet, Li, Levy. "N-Symmetry Direction Field Design." TOG 2008.
3. Bommes, Zimmer, Kobbelt. "Mixed-Integer Quadrangulation." SIGGRAPH 2009.
4. Crane, de Goes, Desbrun, Schroder. "Digital Geometry Processing with Discrete Exterior Calculus." SIGGRAPH Course 2013.
