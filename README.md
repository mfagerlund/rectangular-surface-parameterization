# Rectangular Surface Parameterization (Python)

> **Python implementation** of the algorithm by Etienne Corman and Keenan Crane.
> Original MATLAB code: [etcorman/RectangularSurfaceParameterization](https://github.com/etcorman/RectangularSurfaceParameterization)

![icon](https://github.com/user-attachments/assets/c5910a61-4265-4186-92f2-e5a7779c25d8)

This is a complete Python port of the MATLAB implementation for the paper:

[_Rectangular Surface Parameterization_](https://www.cs.cmu.edu/~kmcrane/Projects/RectangularSurfaceParameterization/RectangularSurfaceParameterization.pdf)
Etienne Corman and Keenan Crane
_ACM Transactions on Graphics (SIGGRAPH)_, 2025

## What This Does

Computes an orthogonal (rectangular) UV parameterization of a triangle mesh, aligned to a cross field. The resulting UVs can be used for quad meshing, texture mapping, or architectural/fabrication applications where rectangular grid patterns are desired.

**Key features:**
- Cross field computation (smooth, curvature-aligned, or trivial connection)
- Seamless parameterization with integrability constraints
- Hard edge and boundary alignment support
- Integration with [libQEx](https://github.com/hcebke/libQEx) for quad mesh extraction

<!-- TODO: Add hero image showing input mesh -> quad mesh result -->
<!-- ![Example result](images/hero_example.png) -->

## Installation

```bash
# Clone the repository
git clone https://github.com/mfagerlund/rectangular-surface-parameterization.git
cd rectangular-surface-parameterization

# Install dependencies
pip install numpy scipy matplotlib

# Optional: for mesh preprocessing
pip install pymeshlab
```

## Command-Line Interface

Two main commands are provided:

### `run_RSP.py` - Parameterization

Computes UV parameterization for a triangle mesh.

```bash
python run_RSP.py mesh.obj -o Results/ -v
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `Results/` | Output directory |
| `-v, --verbose` | off | Verbose output |
| `--frame-field` | `smooth` | Cross field type: `smooth`, `curvature`, `trivial` |
| `--energy` | `distortion` | Energy type: `distortion`, `chebyshev`, `alignment` |
| `--w-conf-ar` | `0.5` | Conformal/area weight (0=area, 0.5=isometric, 1=conformal) |
| `--no-hardedge` | off | Disable hard edge constraints |
| `--no-boundary` | off | Disable boundary alignment |
| `--save-viz` | off | Save UV visualization PNGs |
| `--plot` | off | Show interactive matplotlib plots |

**Output:** `<mesh>_param.obj` with UV coordinates

### `extract_quads.py` - Full Pipeline

Runs parameterization + quad mesh extraction via libQEx.

```bash
python extract_quads.py mesh.obj -o Results/ --scale 10
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `Results/` | Output directory |
| `--scale` | `1.0` | UV scale factor (higher = more quads) |
| `--preprocess` | off | Clean mesh with PyMeshLab first |
| `--skip-rsp` | off | Use existing `*_param.obj` file |

**Output:** `<mesh>_quads.obj`

> **Note:** Quad extraction requires Windows x64 (pre-built libQEx binaries included).
> Parameterization works on all platforms.

See **[USAGE.md](USAGE.md)** for complete reference including troubleshooting and Python API.

## Python Conversion Notes

This is a **line-by-line port** of the original MATLAB implementation by Mattias Fagerlund.
The Python code preserves the exact structure and algorithms of the MATLAB source.

For the original version with interleaved MATLAB comments alongside each Python section,
see [commit 7d1aab4](https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4).

**Core port:**
- Complete translation of all MATLAB algorithms to Python/NumPy/SciPy
- All pipeline stages verified with test suites (see `tests/`)

**Additional pipeline extensions (beyond the original):**
- Mesh preprocessing utilities for handling real-world meshes
- Integration with libQEx for quad mesh extraction
- Visualization and verification tools

For implementation details, see [CLAUDE.md](CLAUDE.md).

## Citation

If you use this software in academic work, please cite the original paper:

```bibtex
@article{Corman:2025:RSP,
  author = {Corman, Etienne and Crane, Keenan},
  title = {Rectangular Surface Parameterization},
  journal = {ACM Trans. Graph.},
  volume = {44},
  number = {4},
  year = {2025},
  publisher = {ACM},
  address = {New York, NY, USA},
}
```

## License

AGPL-3.0-or-later

Copyright (C) 2025 Etienne Corman and Keenan Crane (original algorithm)
Copyright (C) 2025 Mattias Fagerlund (Python conversion)

See [LICENSE](LICENSE) for full terms. Commercial licensing available from the original authors.

## Acknowledgments

- **Etienne Corman** and **Keenan Crane** for the algorithm and original MATLAB implementation
- **Yoann Coudert-Osmont** for the quantization code
- **libQEx** authors for the quad mesh extraction library
- **PyMeshLab** for mesh preprocessing and repair
