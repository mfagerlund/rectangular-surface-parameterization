# Examples Gallery

This page shows output from all included test meshes with various configuration options.

Each mesh shows:
- **UV Layout** - The 2D parameterization (left) and checkerboard pattern (right)
- **Distortion** - Four quality metrics: area distortion, conformal distortion, Jacobian determinant, orthogonality error
- **3D Mesh** - Original mesh with flipped faces highlighted in red (if any)

---

## Sphere (genus 0)

A 320-face sphere - the simplest test case. Genus 0 surfaces require at least 8 singularities (by Euler characteristic).

### Smooth Cross Field

```bash
python run_RSP.py Mesh/sphere320.obj --frame-field smooth --save-viz
```

![Sphere Smooth UV](docs/examples/sphere320_smooth/sphere320_uv_layout.png)
*UV Layout: **0 flipped triangles** - perfect parameterization*

![Sphere Smooth Distortion](docs/examples/sphere320_smooth/sphere320_distortion.png)
*Distortion analysis showing area, conformal, Jacobian, and orthogonality metrics*

![Sphere Smooth Mesh](docs/examples/sphere320_smooth/sphere320_mesh_flips.png)
*3D mesh view - no flipped faces (all blue)*

### Curvature-Aligned Cross Field

```bash
python run_RSP.py Mesh/sphere320.obj --frame-field curvature --save-viz
```

![Sphere Curvature UV](docs/examples/sphere320_curvature/sphere320_uv_layout.png)
*Curvature-aligned field on a sphere (uniform curvature, so similar to smooth)*

---

## Torus (genus 1)

A 1152-face torus - genus 1 surfaces can have **zero singularities** since their Euler characteristic is 0.

### Smooth Cross Field

```bash
python run_RSP.py Mesh/torus.obj --frame-field smooth --save-viz
```

![Torus Smooth UV](docs/examples/torus_smooth/torus_uv_layout.png)
*UV Layout: **0 flipped triangles** - characteristic elongated shape from torus topology*

![Torus Smooth Distortion](docs/examples/torus_smooth/torus_distortion.png)
*Distortion is relatively uniform across the surface*

![Torus Smooth Mesh](docs/examples/torus_smooth/torus_mesh_flips.png)
*3D torus mesh - no flipped faces*

---

## B36 (complex shape)

A 4556-face mesh with interesting topology - demonstrates handling of complex geometry.

> **Note:** B36's UV layout shows **overlapping regions** - this is expected behavior, not a bug.
> The algorithm produces *seamless* UVs for quad meshing, not *bijective* (overlap-free) UVs for texturing.
> See [Understanding UV Overlaps](#understanding-uv-overlaps) for details.

### Smooth Cross Field

```bash
python run_RSP.py Mesh/B36.obj --frame-field smooth --save-viz
```

![B36 Smooth UV](docs/examples/B36_smooth/B36_uv_layout.png)
*UV Layout: **0 flipped triangles** - note: overlapping regions are expected for complex shapes (see [Understanding UV Overlaps](#understanding-uv-overlaps))*

![B36 Smooth Distortion](docs/examples/B36_smooth/B36_distortion.png)
*Distortion varies across the surface due to geometric complexity*

![B36 Smooth Mesh](docs/examples/B36_smooth/B36_mesh_flips.png)
*3D mesh view*

### Curvature-Aligned Cross Field

```bash
python run_RSP.py Mesh/B36.obj --frame-field curvature --save-viz
```

![B36 Curvature UV](docs/examples/B36_curvature/B36_uv_layout.png)
*Curvature-aligned version - different singularity placement*

---

## Pig (organic shape)

A 3678-face pig mesh - tests handling of organic geometry with varying curvature.

### Smooth Cross Field

```bash
python run_RSP.py Mesh/pig.obj --frame-field smooth --save-viz
```

![Pig Smooth UV](docs/examples/pig_smooth/pig_uv_layout.png)
*UV Layout: **10 flipped triangles (0.3%)** - very low flip rate for complex organic shape*

![Pig Smooth Distortion](docs/examples/pig_smooth/pig_distortion.png)
*Distortion concentrated at high-curvature areas (ears, snout)*

![Pig Smooth Mesh](docs/examples/pig_smooth/pig_mesh_flips.png)
*3D mesh - flipped faces (if any) would appear in red*

### Curvature-Aligned Cross Field

```bash
python run_RSP.py Mesh/pig.obj --frame-field curvature --save-viz
```

![Pig Curvature UV](docs/examples/pig_curvature/pig_uv_layout.png)
*UV Layout: **23 flipped triangles (0.6%)** - curvature alignment creates different fold patterns*

![Pig Curvature Distortion](docs/examples/pig_curvature/pig_distortion.png)
*Curvature-aligned distortion pattern*

---

## SquareMyles (mesh with hole)

A 1328-face mesh with a square hole - tests handling of meshes with boundaries.

### Smooth Cross Field (Distortion Energy)

```bash
python run_RSP.py Mesh/SquareMyles.obj --frame-field smooth --save-viz
```

![SquareMyles Smooth UV](docs/examples/SquareMyles_smooth/SquareMyles_uv_layout.png)
*UV Layout: **0 flipped triangles** - hole preserved in UV space*

![SquareMyles Smooth Distortion](docs/examples/SquareMyles_smooth/SquareMyles_distortion.png)
*Distortion analysis*

![SquareMyles Smooth Mesh](docs/examples/SquareMyles_smooth/SquareMyles_mesh_flips.png)
*3D mesh showing the square hole*

### Chebyshev Energy

```bash
python run_RSP.py Mesh/SquareMyles.obj --frame-field smooth --energy chebyshev --save-viz
```

![SquareMyles Chebyshev UV](docs/examples/SquareMyles_chebyshev/SquareMyles_uv_layout.png)
*UV Layout: **0 flipped triangles** - Chebyshev energy produces more uniform grid spacing*

![SquareMyles Chebyshev Distortion](docs/examples/SquareMyles_chebyshev/SquareMyles_distortion.png)
*Chebyshev energy optimizes for constant grid spacing - note the different orthogonality pattern*

---

## Summary Table

| Mesh | Faces | Cross Field | Energy | Flipped | Rate |
|------|-------|-------------|--------|---------|------|
| sphere320 | 320 | smooth | distortion | 0 | 0.0% |
| sphere320 | 320 | curvature | distortion | 0 | 0.0% |
| torus | 1152 | smooth | distortion | 0 | 0.0% |
| B36 | 4556 | smooth | distortion | 0 | 0.0% |
| B36 | 4556 | curvature | distortion | 0 | 0.0% |
| pig | 3678 | smooth | distortion | 10 | 0.3% |
| pig | 3678 | curvature | distortion | 23 | 0.6% |
| SquareMyles | 1328 | smooth | distortion | 0 | 0.0% |
| SquareMyles | 1328 | smooth | chebyshev | 0 | 0.0% |

## Understanding UV Overlaps

**Important:** This algorithm produces **seamless** parameterizations, not **bijective** (overlap-free) ones.

| Property | Meaning | Required for |
|----------|---------|--------------|
| **Seamless** | UVs match across cut edges | Quad meshing |
| **No flipped triangles** | Local orientation preserved | Both |
| **Bijective (no overlaps)** | No self-intersection in UV space | Texture mapping |

Complex shapes like B36 and pig may have **overlapping regions** in UV space. This is expected and OK for quad meshing - libQEx handles overlaps during extraction. If you need overlap-free UVs for texturing, additional post-processing would be required.

## Visualization Guide

### UV Layout Panel

The UV layout shows two views:
- **Left (blue)**: Triangle mesh in UV space. Flipped triangles appear in red.
- **Right (checkerboard)**: Alternating gray/white pattern to visualize distortion. Irregular checkers indicate stretching.

### Distortion Panel

Four metrics visualized:
1. **Area Distortion** (log scale): How much each triangle is stretched/compressed. Values near 0 are ideal.
2. **Conformal Distortion** (log scale): Angle preservation. Lower is better.
3. **Jacobian Determinant**: Sign indicates orientation. Negative = flipped triangle.
4. **Orthogonality Error**: Degrees deviation from 90. Measures how rectangular the grid is.

### 3D Mesh Panel

Shows the original mesh with:
- **Blue faces**: Valid orientation
- **Red faces**: Flipped (inverted) triangles that may cause rendering issues

## Regenerating Examples

To regenerate all examples:

```bash
# All meshes with smooth cross field
for mesh in sphere320 torus B36 pig SquareMyles; do
    python run_RSP.py Mesh/${mesh}.obj -o docs/examples/${mesh}_smooth/ --frame-field smooth --save-viz
done

# Curvature variants
for mesh in sphere320 B36 pig; do
    python run_RSP.py Mesh/${mesh}.obj -o docs/examples/${mesh}_curvature/ --frame-field curvature --save-viz
done

# Chebyshev energy
python run_RSP.py Mesh/SquareMyles.obj -o docs/examples/SquareMyles_chebyshev/ --energy chebyshev --save-viz
```
