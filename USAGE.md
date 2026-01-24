# Usage Guide

Command-line reference for rectangular-surface-parameterization.

## Basic Commands

### Parameterization Only

```bash
# Basic run - outputs parameterized mesh with UVs
python run_RSP.py mesh.obj -o Results/ -v

# With visualization PNGs (UV layout, flipped faces highlighted)
python run_RSP.py mesh.obj -o Results/ -v --save-viz

# Interactive matplotlib plots
python run_RSP.py mesh.obj -o Results/ -v --plot
```

**Output:** `Results/<mesh>_param.obj` with UV coordinates

### Full Pipeline (Parameterization + Quad Extraction)

> **Note:** Quad extraction currently requires **Windows x64**. Pre-built binaries for
> libQEx are included in `bin/`. See [Other Platforms](#other-platforms) below.

```bash
# Generate quad mesh from triangle mesh
python extract_quads.py mesh.obj -o Results/ --scale 10

# With mesh preprocessing (for problematic meshes)
python extract_quads.py mesh.obj -o Results/ --scale 10 --preprocess

# Skip parameterization if already done
python extract_quads.py mesh.obj -o Results/ --scale 10 --skip-rsp
```

**Output:** `Results/<mesh>_quads.obj`

The `--scale` parameter controls quad density (higher = more quads).

## Cross Field Options

Control how the guiding cross field is computed:

```bash
# Curvature-aligned (default) - follows principal curvatures
python run_RSP.py mesh.obj --frame-field curvature

# Smoothest field - minimizes field variation
python run_RSP.py mesh.obj --frame-field smooth

# Trivial connection - user-specified singularities
python run_RSP.py mesh.obj --frame-field trivial
```

## Constraint Options

```bash
# Align to sharp edges (detected automatically)
python run_RSP.py mesh.obj --hard-edges

# Align to mesh boundary
python run_RSP.py mesh.obj --boundary

# Enforce seamless constraints (required for quad extraction)
python run_RSP.py mesh.obj --seamless
```

These can be combined:
```bash
python run_RSP.py mesh.obj --hard-edges --boundary --seamless
```

## Energy Types

Control the optimization objective:

```bash
# Distortion energy (default)
# --w-conf-ar 0.0 = area-preserving
# --w-conf-ar 0.5 = isometric
# --w-conf-ar 1.0 = conformal
python run_RSP.py mesh.obj --energy distortion --w-conf-ar 0.5

# Chebyshev net (for fabrication applications)
python run_RSP.py mesh.obj --energy chebyshev

# Alignment energy (match target directions)
python run_RSP.py mesh.obj --energy alignment
```

## Mesh Preprocessing

For meshes with quality issues (non-manifold edges, holes, etc.):

```bash
# Standalone preprocessing
python Utils/preprocess_mesh.py input.obj output_clean.obj

# Check mesh quality
python -c "from Utils.preprocess_mesh import check_mesh_quality; check_mesh_quality('mesh.obj')"
```

Or use the `--preprocess` flag with extract_quads.py.

## Verification

```bash
# Verify specific pipeline stage
python verify_pipeline.py mesh.obj --stage geometry
python verify_pipeline.py mesh.obj --stage cross_field
python verify_pipeline.py mesh.obj --stage cut_graph
python verify_pipeline.py mesh.obj --stage optimization
python verify_pipeline.py mesh.obj --stage uv_recovery

# Run all tests
pytest tests/ -v
```

## Output Files

| File | Description |
|------|-------------|
| `<mesh>_param.obj` | Parameterized mesh with UV coordinates |
| `<mesh>_quads.obj` | Extracted quad mesh |
| `uv_layout.png` | UV space visualization |
| `mesh_flips.png` | 3D mesh with flipped faces in red |
| `distortion.png` | Distortion heatmap |

## Test Meshes

```bash
# Genus 0 (sphere-like)
python run_RSP.py "C:/Dev/Colonel/Data/Meshes/sphere320.obj" -o Results/

# Genus 1 (torus)
python run_RSP.py "C:/Dev/Colonel/Data/Meshes/torus.obj" -o Results/
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Singular matrix error | Try `--preprocess` or simplify mesh |
| Flipped triangles (red in viz) | Normal for complex geometry; check if count is acceptable |
| Gaussian curvature mismatch | Mesh may have holes; needs boundary support |
| Non-manifold edges | Use `--preprocess` to clean mesh |

## Python API

```python
from run_RSP import run_rsp_pipeline

# Run parameterization
result = run_rsp_pipeline(
    mesh_path="mesh.obj",
    output_dir="Results/",
    frame_field_type="curvature",
    energy_type="distortion",
    hard_edges=True,
    boundary=True,
    seamless=True
)

# Access results
uv_coords = result.Xp        # (n_vertices, 2)
triangles = result.T         # (n_faces, 3)
flip_count = result.n_flips  # Number of flipped triangles
```

## Other Platforms

### Current Status

The **parameterization** (`run_RSP.py`) works on all platforms (Windows, Linux, macOS) -
it's pure Python with NumPy/SciPy.

**Quad extraction** (`extract_quads.py`) currently requires Windows x64 because it uses
pre-built libQEx binaries. On other platforms, you can still generate the parameterized
mesh with UVs, then use external tools for quad extraction.

### Building libQEx for Linux/macOS

libQEx is open source and can be built from source:

**Requirements:**
- CMake 3.10+
- C++ compiler (g++ on Linux, clang on macOS)
- OpenMesh library

**Build steps:**
```bash
# Clone dependencies
git clone https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
git clone https://github.com/hcebke/libQEx.git

# Build OpenMesh
cd OpenMesh && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=OFF
make -j4
sudo make install
cd ../..

# Build libQEx
cd libQEx && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Copy binary to repo
cp qex_extract /path/to/rectangular-surface-parameterization/bin/
```

See `docs/libqex_setup.md` for detailed instructions.

### Contributing Binaries

If you build libQEx for Linux or macOS, consider contributing the binaries via a pull
request. Please include:

1. The compiled `qex_extract` binary (and any required `.so`/`.dylib` files)
2. Platform info (OS version, architecture)
3. Build instructions you used

For security, we prefer contributors who can provide reproducible build instructions or
submit via GitHub Actions CI. Future improvement: automated cross-platform builds via CI.
