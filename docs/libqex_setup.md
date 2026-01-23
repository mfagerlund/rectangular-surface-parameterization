# libQEx Setup Guide

libQEx is used for robust quad mesh extraction from integer-grid maps.

> **Note:** Quad extraction is **beyond the scope of the Corman-Crane paper**.
> The paper produces seamless UV parameterization; libQEx extracts quads from it.

## Pre-built Binaries (Recommended)

Pre-built Windows x64 binaries are included in the `bin/` directory. No build required.

```
bin/
  libQEx.dll        # Main library
  qex_extract.exe   # Command-line tool
  OpenMesh.dll      # Dependency
```

### Usage

```bash
# From Python (via subprocess)
python quad_extract.py input_param.obj -o output_quads.obj

# Direct CLI
bin/qex_extract.exe input_param.obj output_quads.obj
```

## Building from Source (Optional)

If you need to build from source (e.g., for a different platform):

### Prerequisites

1. **CMake** 2.6 or higher
2. **Visual Studio** 2019 or higher (Windows)
3. **OpenMesh** library
   - Download: https://www.graphics.rwth-aachen.de/software/openmesh/
   - Or via vcpkg: `vcpkg install openmesh`

### Build Steps

```bash
# Clone libQEx
git clone https://github.com/hcebke/libQEx.git
cd libQEx

# Create build directory
mkdir build && cd build

# Configure (adjust OpenMesh path as needed)
cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenMesh_DIR=<path-to-openmesh>

# Build
cmake --build . --config Release

# Copy binaries to this repo's bin/ directory
copy Release\*.dll ..\..\bin\
copy Release\*.exe ..\..\bin\
```

### Dependencies

| Library | Version | Notes |
|---------|---------|-------|
| OpenMesh | 8.0+ | Mesh data structures |
| CMake | 2.6+ | Build system |

## Algorithm Overview

libQEx implements the QEx algorithm (SIGGRAPH Asia 2013):

1. **Input**: Integer-grid map (seamless UV parameterization with singularities at integer coords)
2. **Trace iso-lines**: Find where u=k and v=k lines cross triangle edges
3. **Find intersections**: Where u-isolines and v-isolines cross = quad vertices
4. **Build connectivity**: Connect adjacent integer points to form quads
5. **Handle edge cases**: Fold-overs, numerical precision, boundary conditions

See `algo_integer_grid_maps.md` for detailed algorithm description.

## License

libQEx is GPL v3. Commercial licensing available from the authors.

## References

- [libQEx GitHub](https://github.com/hcebke/libQEx)
- [QEx Paper](https://dl.acm.org/doi/10.1145/2508363.2508372) (SIGGRAPH Asia 2013)
- [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/)
