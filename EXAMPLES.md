# Examples Gallery

This page shows output from all included test meshes with various configuration options.

Each mesh shows:
- **UV Layout** - The 2D parameterization (left) and checkerboard pattern (right)
- **Distortion** - Four quality metrics: area distortion, conformal distortion, Jacobian determinant, orthogonality error
- **Quad Mesh** - Extracted quad mesh using libQEx (when available)

---

## Sphere (genus 0)

A 320-face sphere - the simplest test case. Genus 0 surfaces require at least 8 singularities (by Euler characteristic).

### Smooth Cross Field

```bash
python run_RSP.py Mesh/sphere320.obj --frame-field smooth -v
```

![Sphere Smooth UV](docs/examples/sphere320_smooth/sphere320_uv_layout.png)
*UV Layout: **0 flipped triangles** - perfect parameterization*

![Sphere Smooth Distortion](docs/examples/sphere320_smooth/sphere320_distortion.png)
*Distortion analysis showing area, conformal, Jacobian, and orthogonality metrics*

![Sphere Smooth Quads](docs/examples/sphere320_smooth/sphere320_quads.png)
*Extracted quad mesh: 400 quads, 16 triangular hole-fills at singularities*

### Curvature-Aligned Cross Field

```bash
python run_RSP.py Mesh/sphere320.obj --frame-field curvature -v
```

![Sphere Curvature UV](docs/examples/sphere320_curvature/sphere320_uv_layout.png)
*Curvature-aligned field on a sphere (uniform curvature, so similar to smooth)*

![Sphere Curvature Quads](docs/examples/sphere320_curvature/sphere320_quads.png)
*Extracted quad mesh: 102 quads*

---

## Torus (genus 1)

A 1152-face torus - genus 1 surfaces can have **zero singularities** since their Euler characteristic is 0.

### Smooth Cross Field

```bash
python run_RSP.py Mesh/torus.obj --frame-field smooth -v
```

![Torus Smooth UV](docs/examples/torus_smooth/torus_uv_layout.png)
*UV Layout: **0 flipped triangles** - characteristic elongated shape from torus topology*

![Torus Smooth Distortion](docs/examples/torus_smooth/torus_distortion.png)
*Distortion is relatively uniform across the surface*

![Torus Smooth Quads](docs/examples/torus_smooth/torus_quads.png)
*Extracted quad mesh: 887 quads with clean periodic structure*

---

## B36 (complex shape)

A 4556-face mesh with interesting topology - demonstrates handling of complex geometry.

> **Note:** B36's UV layout shows **overlapping regions** - this is expected behavior, not a bug.
> The algorithm produces *seamless* UVs for quad meshing, not *bijective* (overlap-free) UVs for texturing.
> See [Understanding UV Overlaps](#understanding-uv-overlaps) for details.

### Smooth Cross Field

```bash
python run_RSP.py Mesh/B36.obj --frame-field smooth -v
```

![B36 Smooth UV](docs/examples/B36_smooth/B36_uv_layout.png)
*UV Layout: **0 flipped triangles** - note: overlapping regions are expected for complex shapes (see [Understanding UV Overlaps](#understanding-uv-overlaps))*

![B36 Smooth Distortion](docs/examples/B36_smooth/B36_distortion.png)
*Distortion varies across the surface due to geometric complexity*

> **Note:** Quad extraction with libQEx fails for B36 due to overlapping UVs - this is a known limitation. The parameterization itself is valid for applications that handle overlaps.

### Curvature-Aligned Cross Field

```bash
python run_RSP.py Mesh/B36.obj --frame-field curvature -v
```

![B36 Curvature UV](docs/examples/B36_curvature/B36_uv_layout.png)
*Curvature-aligned version - different singularity placement*

---

## Pig (organic shape)

A 3678-face pig mesh - tests handling of organic geometry with varying curvature.

### Smooth Cross Field

```bash
python run_RSP.py Mesh/pig.obj --frame-field smooth -v
```

![Pig Smooth UV](docs/examples/pig_smooth/pig_uv_layout.png)
*UV Layout: **10 flipped triangles (0.3%)** - very low flip rate for complex organic shape*

![Pig Smooth Distortion](docs/examples/pig_smooth/pig_distortion.png)
*Distortion concentrated at high-curvature areas (ears, snout)*

![Pig Smooth Quads](docs/examples/pig_smooth/pig_quads.png)
*Extracted quad mesh: 888 quads*

### Curvature-Aligned Cross Field

```bash
python run_RSP.py Mesh/pig.obj --frame-field curvature -v
```

![Pig Curvature UV](docs/examples/pig_curvature/pig_uv_layout.png)
*UV Layout: **23 flipped triangles (0.6%)** - curvature alignment creates different fold patterns*

![Pig Curvature Distortion](docs/examples/pig_curvature/pig_distortion.png)
*Curvature-aligned distortion pattern*

![Pig Curvature Quads](docs/examples/pig_curvature/pig_quads.png)
*Extracted quad mesh: 938 quads*

---

## SquareMyles (mesh with hole)

A 1328-face mesh with a square hole - tests handling of meshes with boundaries.

### Smooth Cross Field (Distortion Energy)

```bash
python run_RSP.py Mesh/SquareMyles.obj --frame-field smooth -v
```

![SquareMyles Smooth UV](docs/examples/SquareMyles_smooth/SquareMyles_uv_layout.png)
*UV Layout: **0 flipped triangles** - hole preserved in UV space*

![SquareMyles Smooth Distortion](docs/examples/SquareMyles_smooth/SquareMyles_distortion.png)
*Distortion analysis*

![SquareMyles Smooth Quads](docs/examples/SquareMyles_smooth/SquareMyles_quads.png)
*Extracted quad mesh: 917 quads*

### Chebyshev Energy

```bash
python run_RSP.py Mesh/SquareMyles.obj --frame-field smooth --energy chebyshev -v
```

![SquareMyles Chebyshev UV](docs/examples/SquareMyles_chebyshev/SquareMyles_uv_layout.png)
*UV Layout: **0 flipped triangles** - Chebyshev energy produces more uniform grid spacing*

![SquareMyles Chebyshev Distortion](docs/examples/SquareMyles_chebyshev/SquareMyles_distortion.png)
*Chebyshev energy optimizes for constant grid spacing - note the different orthogonality pattern*

![SquareMyles Chebyshev Quads](docs/examples/SquareMyles_chebyshev/SquareMyles_quads.png)
*Extracted quad mesh: 948 quads*

---

## Summary Table

| Mesh | Faces | Cross Field | Energy | Flipped | Rate | Quads |
|------|-------|-------------|--------|---------|------|-------|
| sphere320 | 320 | smooth | distortion | 0 | 0.0% | 400 |
| sphere320 | 320 | curvature | distortion | 0 | 0.0% | 102 |
| torus | 1152 | smooth | distortion | 0 | 0.0% | 887 |
| B36 | 4556 | smooth | distortion | 0 | 0.0% | N/A* |
| B36 | 4556 | curvature | distortion | 0 | 0.0% | N/A* |
| pig | 3678 | smooth | distortion | 10 | 0.3% | 888 |
| pig | 3678 | curvature | distortion | 23 | 0.6% | 938 |
| SquareMyles | 1328 | smooth | distortion | 0 | 0.0% | 917 |
| SquareMyles | 1328 | smooth | chebyshev | 0 | 0.0% | 948 |

*B36 quad extraction fails due to UV overlaps (libQEx limitation)

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

### Quad Mesh Panel

Shows the extracted quad mesh:
- **Light blue faces**: Quads extracted by libQEx
- **Yellow faces**: Triangular hole-fills at singularities (where cross field has rotational defect)

## Regenerating Examples

To regenerate all examples:

```bash
# All meshes with smooth cross field
for mesh in sphere320 torus B36 pig SquareMyles; do
    python run_RSP.py Mesh/${mesh}.obj -o docs/examples/${mesh}_smooth/ --frame-field smooth -v
done

# Curvature variants
for mesh in sphere320 B36 pig; do
    python run_RSP.py Mesh/${mesh}.obj -o docs/examples/${mesh}_curvature/ --frame-field curvature -v
done

# Chebyshev energy
python run_RSP.py Mesh/SquareMyles.obj -o docs/examples/SquareMyles_chebyshev/ --energy chebyshev -v

# Quad extraction (for meshes without overlapping UVs)
for mesh in sphere320 torus pig SquareMyles; do
    python extract_quads.py Mesh/${mesh}.obj -o docs/examples/${mesh}_smooth/ --scale 10 --skip-rsp
done
```
