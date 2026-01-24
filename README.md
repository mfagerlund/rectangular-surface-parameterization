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
- Cross field computation (smooth, curvature-aligned, or user-specified)
- Seamless parameterization with integrability constraints
- Hard edge and boundary alignment support
- Integration with [libQEx](https://github.com/hcebke/libQEx) for quad mesh extraction

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

## Quick Start

```bash
# Basic parameterization
python run_RSP.py mesh.obj -o Results/ -v

# Full pipeline: parameterization + quad extraction
python extract_quads.py mesh.obj -o Results/ --scale 10
```

See **[USAGE.md](USAGE.md)** for complete command-line reference, options, and examples.

## Python Conversion Notes

This is a **line-by-line port** of the original MATLAB implementation by Mattias Fagerlund.
The Python code preserves the exact structure and algorithms of the MATLAB source, with
commented MATLAB code alongside each Python section for verification.

**Core port:**
- Complete translation of all MATLAB algorithms to Python/NumPy/SciPy
- Identical numerical results to the original implementation
- All pipeline stages verified against MATLAB output

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
