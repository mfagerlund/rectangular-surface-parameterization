# libQEx Setup Guide

libQEx is used for robust quad mesh extraction from integer-grid maps.

> **Note:** Quad extraction is **beyond the scope of the Corman-Crane paper**.
> The paper produces seamless UV parameterization; libQEx extracts quads from it.

## Pre-built Binaries

**Windows x64**: Binaries are included in the `bin/` directory. No setup required.

**Linux/macOS**: Build from source (see below).

## Building from Source

### Prerequisites

- CMake 3.10+
- C++ compiler with C++11 support
- Git

### Build Steps

1. Clone and build OpenMesh:
   ```bash
   git clone https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
   cd OpenMesh && mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=OFF
   cmake --build . --config Release
   cmake --install . --config Release --prefix ../install
   cd ../..
   ```

2. Clone and build libQEx:
   ```bash
   git clone https://github.com/hcebke/libQEx.git
   cd libQEx && mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=../../OpenMesh/install
   cmake --build . --config Release
   ```

3. Copy `qex_extract` (or `qex_extract.exe`) to your cache directory.

See `.github/workflows/build-libqex.yml` for the CI build process.

## Algorithm Overview

libQEx implements the QEx algorithm (SIGGRAPH Asia 2013):

1. **Input**: Integer-grid map (seamless UV parameterization)
2. **Trace iso-lines**: Find where u=k and v=k lines cross triangle edges
3. **Find intersections**: Where u-isolines and v-isolines cross = quad vertices
4. **Build connectivity**: Connect adjacent integer points to form quads

## Python Port

This project includes a **pure Python/NumPy port** of the QEx algorithm that doesn't
require the C++ binaries. See [qex_python_port_plan.md](qex_python_port_plan.md) for details.

```python
from rectangular_surface_parameterization.quad_extraction import extract_quads
quad_verts, quad_faces, tri_faces = extract_quads(vertices, triangles, uv_per_triangle)
```

## License

libQEx is GPL-3.0. See `bin/BINARIES.txt` for details.

## References & Acknowledgements

### QEx Paper

> Hans-Christian Ebke, David Bommes, Marcel Campen, and Leif Kobbelt. 2013.
> **QEx: Robust Quad Mesh Extraction.**
> *ACM Trans. Graph.* 32, 6, Article 168 (November 2013).
> DOI: [10.1145/2508363.2508372](https://doi.org/10.1145/2508363.2508372)

```bibtex
@article{Ebke:2013:QEx,
  author = {Ebke, Hans-Christian and Bommes, David and Campen, Marcel and Kobbelt, Leif},
  title = {QEx: Robust Quad Mesh Extraction},
  journal = {ACM Trans. Graph.},
  volume = {32},
  number = {6},
  year = {2013},
  pages = {168:1--168:10},
  doi = {10.1145/2508363.2508372},
  publisher = {ACM},
}
```

### Links

- [libQEx GitHub Repository](https://github.com/hcebke/libQEx)
- [QEx Paper (ACM DL)](https://dl.acm.org/doi/10.1145/2508363.2508372)
- [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/)
