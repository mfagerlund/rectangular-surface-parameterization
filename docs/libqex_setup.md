# libQEx Setup Guide

libQEx is used for robust quad mesh extraction from integer-grid maps.

> **Note:** Quad extraction is **beyond the scope of the Corman-Crane paper**.
> The paper produces seamless UV parameterization; libQEx extracts quads from it.

## Pre-built Binaries

Pre-built Windows x64 binaries will be in the `bin/` directory after building.

**Status:** Binaries not yet built. Run the build script first (see below).

```
bin/
  libQEx.dll        # Main library (after build)
  qex_demo.exe      # Demo/CLI tool (after build)
  OpenMeshCore.dll  # Dependency (after build)
```

## Building (Required First Time)

### Prerequisites

1. **CMake** 3.10 or higher - https://cmake.org/download/ (add to PATH)
2. **Visual Studio 2019 or 2022** with "Desktop development with C++" workload
3. **Git** (for cloning dependencies)

### Build Steps

**Option 1: Automated script (recommended)**
```powershell
# From repo root
.\scripts\build_libqex.ps1
```

Or double-click `scripts\build_libqex.bat`

The script will:
1. Clone OpenMesh source from GitLab
2. Build OpenMesh with CMake
3. Clone libQEx from GitHub
4. Build libQEx
5. Copy binaries to `bin/`

Build directory: `C:\Slask\libqex_build` (sources already downloaded there)

**Option 2: Manual build**
```powershell
cd C:\Slask\libqex_build

# Build OpenMesh
cd OpenMesh-src
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=../install -DBUILD_APPS=OFF
cmake --build . --config Release
cmake --install . --config Release
cd ../..

# Build libQEx
cd libQEx
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DOpenMesh_INCLUDE_DIR=../../OpenMesh-src/install/include -DOpenMesh_LIBRARY=../../OpenMesh-src/install/lib/OpenMeshCore.lib
cmake --build . --config Release

# Copy to repo
copy Release\*.dll C:\Dev\Corman-Crane\bin\
copy Release\*.exe C:\Dev\Corman-Crane\bin\
copy ..\..\OpenMesh-src\install\bin\*.dll C:\Dev\Corman-Crane\bin\
```

### Dependencies

| Library | Version | Notes |
|---------|---------|-------|
| OpenMesh | 11.0 | Mesh data structures (cloned from GitLab) |
| CMake | 3.10+ | Build system |
| Visual Studio | 2019/2022 | C++ compiler with SSE support |

### Downloaded Sources

Sources are already downloaded to `C:\Slask\libqex_build\`:
- `libQEx/` - libQEx source from GitHub
- `OpenMesh-src/` - OpenMesh source from GitLab

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
