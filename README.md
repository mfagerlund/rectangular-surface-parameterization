# Corman-Crane Rectangular Parameterization

Python implementation of the rectangular surface parameterization algorithm from Corman & Crane, SIGGRAPH 2025.

## Overview

This tool computes orthogonal (but not necessarily isotropic) UV parameterizations for triangle meshes. Unlike conformal maps which preserve angles, rectangular parameterization aligns UV coordinates with a cross field, producing axis-aligned texture patterns suitable for quad meshing.

## Status

**Working** - All phases implemented and tested on sphere320.obj:
- Constraint solver converges in 5 iterations
- UV recovery produces 0 flipped triangles
- Mean angle error: 14.04°

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
python corman_crane.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o sphere_uv.obj -v
```

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

## Reference

Based on: **Corman & Crane, "Rectangular Parameterization", SIGGRAPH 2025**

The algorithm produces parameterizations where:
- Iso-lines align with a user-specified cross field
- The map is locally orthogonal (angles between iso-lines are 90°)
- Singularities are confined to cone points

## License

MIT
