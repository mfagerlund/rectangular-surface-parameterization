# Corman-Crane Rectangular Parameterization

Python implementation of the rectangular surface parameterization algorithm from Corman & Crane, SIGGRAPH 2025.

## Overview

This tool computes orthogonal (but not necessarily isotropic) UV parameterizations for triangle meshes. Unlike conformal maps which preserve angles, rectangular parameterization aligns UV coordinates with a cross field, producing axis-aligned texture patterns suitable for quad meshing.

## Status

**In Progress** - Core algorithm implemented, UV quality issues being resolved:
- Constraint solver converges in 5 iterations
- UV recovery has flipped triangles and fragmentation issues
- See `verification-plan.md` for current status

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for visualization)

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python corman_crane.py input.obj -o output_uv.obj -v
```

Options:
- `-o, --output` - Output OBJ file with UV coordinates
- `-v, --visualize` - Save visualization images
- `-q, --quiet` - Suppress progress output

Example:
```bash
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o output/sphere_uv.obj -v
```

Output files (visualizations) are saved to the `output/` folder.

## Algorithm Phases

1. **Mesh Loading** - Load OBJ, build half-edge connectivity
2. **Geometry** - Compute corner angles, edge lengths, cotan weights
3. **Cross Field** - Propagate smooth cross field via parallel transport
4. **Cut Graph** - Compute cuts to make mesh simply connected (Algorithm 2)
5. **Sparse Operators** - Build Laplacian, divergence, constraint system
6. **Optimization** - Solve integrability constraints (Algorithms 3-8)
7. **UV Recovery** - Recover UV coordinates via Poisson solve (Algorithm 11)

## File Structure

```
corman_crane.py     # Main CLI entry point
mesh.py             # Half-edge mesh data structure
io_obj.py           # OBJ file loading/saving
geometry.py         # Geometric computations (angles, areas, cotans)
cross_field.py      # Cross field propagation
cut_graph.py        # Cut graph and jump data (Algorithm 2)
sparse_ops.py       # Sparse matrix operators (Algorithms 5-10)
optimization.py     # Constraint solver (Algorithms 3-4)
uv_recovery.py      # UV coordinate recovery (Algorithm 11)
lscm.py             # LSCM baseline for comparison
visualize.py        # Visualization utilities
```

## Why Compact UVs Matter

**This is for quad meshing, NOT origami unfolding.**

For origami/texture atlas applications, any valid UV layout works - the mesh just needs to flatten to 2D without flips.

For quad meshing, the UV domain must be **compact** (high fill ratio, ~100%):
- Integer iso-lines (u=k, v=k for k ∈ Z) become quad edges
- A fragmented "octopus" UV layout means integer iso-lines miss large parts of the mesh
- Result: sparse quad coverage instead of full surface coverage

**Goal**: 0 flipped triangles AND compact UV layout (high fill ratio)

## References

**Main paper:**
- Corman & Crane, "Rectangular Parameterization", SIGGRAPH 2025
- Local: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025.pdf`
- Supplement: `D:\Data\GDrive\FlatrPDFs\corman-crane-rectangular-parameterization-siggraph2025-supplement.pdf`

**Quad extraction:**
- Bommes et al., "Integer-Grid Maps for Reliable Quad Meshing", SIGGRAPH 2013
- Local: `D:\Data\GDrive\FlatrPDFs\2461912.2462014_integer-grid-maps-reliable-quad-meshing.pdf`

- Bommes et al., "Mixed-Integer Quadrangulation", SIGGRAPH 2009
- Local: `D:\Data\GDrive\FlatrPDFs\1531326.1531383_mixed-integer-quadrangulation.pdf`

The algorithm produces parameterizations where:
- Iso-lines align with a user-specified cross field
- The map is locally orthogonal (angles between iso-lines are 90°)
- Singularities are confined to cone points

## License

MIT
