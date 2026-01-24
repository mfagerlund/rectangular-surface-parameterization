# Pythonification Plan

## Goal
Transform the codebase from a MATLAB port with interleaved comments and MATLAB conventions into idiomatic, maintainable Python that follows community standards.

## Current State (Updated 2025-01-24)

| Artifact | Status | Notes |
|----------|--------|-------|
| MATLAB comments (`# %`) | ✅ Removed | Commit 5696cf1 |
| Dict-based mesh structs | ✅ Converted | `MeshInfo`, `DEC`, `OrthoParam`, `Weight` are dataclasses |
| Signed 1-based edge encoding | ✅ Abstracted | `SignedEdgeArray` class with `.indices`/`.signs` API |
| MATLAB naming (`T2E`, `nf`, `nv`) | ✅ Complete | All fields renamed except T2E (uses SignedEdgeArray) |
| Solver params hardcoded | ✅ Extracted | `OptimizationParams` in `Orthotropic/optimization_params.py` |
| Custom OBJ reader | ✅ Replaced | Using trimesh for multi-format, encoding-safe I/O |
| Package structure | ✅ Complete | All 39 modules migrated, old directories removed |

**Note:** The 66 uses of `ravel('F')` / `flatten('F')` are intentional and correct - they match the algorithm's mathematical derivation. Do not change.

## Phase 1: Remove MATLAB Comments ✅ COMPLETE

**Effort:** Low | **Risk:** Low | **Impact:** High for readability

### What
Strip all `# %` MATLAB comments and `# === ISSUES ===` blocks from Python files.

### Why
- They clutter the code and confuse contributors
- The MATLAB reference is available in the original repo
- The code is verified working; we no longer need line-by-line parity

### How
```bash
# Preview which files have MATLAB comments
grep -rn "# %" --include="*.py" .

# Automated removal (review each file after)
sed -i '/^[ \t]*# %/d' file.py
sed -i '/^# === ISSUES ===/,/^# === END ISSUES ===/d' file.py
```

### Preserve
Create `docs/matlab-reference.md` with:
- Link to original MATLAB repo
- Key algorithmic differences/adaptations
- The "Critical Pitfalls" section from `MATLAB_CONVERSION.md`

---

## Phase 2: Introduce Dataclasses ✅ COMPLETE

**Effort:** Medium | **Risk:** Low | **Impact:** High for maintainability

**Status:** All dataclasses implemented: `MeshInfo`, `DEC`, `OrthoParam`, `Weight`, `OptimizationParams`.

### What
Replace dict-based mesh structures with typed dataclasses.

### Current Pattern
```python
Src = {
    'X': vertices,      # (nv, 3)
    'T': triangles,     # (nf, 3)
    'T2E': t2e,         # (nf, 3) signed 1-based
    'nv': nv,
    'nf': nf,
    'ne': ne,
}
result = Src['T2E']
```

### Target Pattern
```python
@dataclass
class TriangleMesh:
    """Triangle mesh with connectivity."""
    vertices: np.ndarray          # (num_vertices, 3)
    triangles: np.ndarray         # (num_faces, 3)
    triangle_to_edge: np.ndarray  # (num_faces, 3) - see EdgeIndex

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.triangles.shape[0]

result = mesh.triangle_to_edge
```

### Files to Create
- `core/mesh.py` - `TriangleMesh`, `CutMesh`
- `core/operators.py` - `DECOperators` (replaces `dec` dict)
- `core/parameters.py` - `OptimizationParams` (replaces `param` dict)
- `core/edge_index.py` - `SignedEdgeIndex` wrapper (see Phase 4)

### Migration Strategy
1. Create dataclasses with same field names first
2. Add `@classmethod from_dict()` for compatibility
3. Update one module at a time, running tests after each
4. Remove dict compatibility after all modules converted

---

## Phase 3: Pythonic Naming (Low Risk)

**Effort:** Medium | **Risk:** Low | **Impact:** Medium

### Naming Conventions

| MATLAB | Python | Notes |
|--------|--------|-------|
| `nv`, `nf`, `ne` | `num_vertices`, `num_faces`, `num_edges` | Or use properties |
| `T` | `triangles` | |
| `X` | `vertices` | |
| `T2E` | `triangle_to_edge` | |
| `E2T` | `edge_to_triangle` | |
| `E2V` | `edge_to_vertex` | |
| `Src` | `source_mesh` or `mesh` | |
| `SrcCut` | `cut_mesh` | |
| `ang` | `angles` or `frame_angles` | |
| `om`, `omega` | `edge_rotation` | |
| `ut`, `vt` | `scale_u_corners`, `scale_v_corners` | |
| `dec` | `operators` or `dec_operators` | |
| `ide_fix` | `fixed_edge_indices` | |
| `tri_fix` | `fixed_face_indices` | |

### Tool
Use `agent-refactor` for safe multi-file renames:
```bash
agent-refactor ts rename --file src/mesh.py --line 12 --col 5 --to num_vertices
```

(Note: Would need Python support in agent-refactor, or use rope/jedi refactoring)

---

## Phase 4: Edge Index Abstraction (Medium Risk)

**Effort:** High | **Risk:** Medium | **Impact:** High for correctness

### Problem
The signed 1-based encoding `(edge_idx + 1) * sign` is error-prone and non-obvious:
```python
# Current: easy to forget the +1
T2E_signed = (edge_idx + 1) * sign
edge_idx = np.abs(T2E) - 1
sign = np.sign(T2E)
```

### Solution
Create an abstraction that handles encoding/decoding:

```python
@dataclass
class SignedEdgeArray:
    """Array of signed edge references.

    Internally stores 1-based indices to preserve sign for edge 0.
    External API uses 0-based indices.
    """
    _data: np.ndarray  # Internal 1-based signed storage

    @classmethod
    def from_edges_and_signs(cls, edges: np.ndarray, signs: np.ndarray):
        """Create from 0-based edge indices and signs."""
        return cls((edges + 1) * signs)

    @property
    def indices(self) -> np.ndarray:
        """0-based edge indices."""
        return np.abs(self._data) - 1

    @property
    def signs(self) -> np.ndarray:
        """Edge orientation signs (+1 or -1)."""
        return np.sign(self._data)

    def __getitem__(self, key):
        """Index access returns new SignedEdgeArray (not tuple).

        Slicing/indexing preserves the abstraction:
            arr[0]      -> SignedEdgeArray with single element
            arr[mask]   -> SignedEdgeArray with filtered elements
            arr[:, 0]   -> SignedEdgeArray for column 0

        To get raw values, use .indices and .signs properties.
        """
        return SignedEdgeArray(self._data[key])

    def __array__(self) -> np.ndarray:
        """Allow np.asarray() to get raw internal data if needed."""
        return self._data
```

### Migration
1. Create `SignedEdgeArray` class
2. Update `connectivity.py` to return it
3. Update consumers one at a time
4. Run tests after each change

---

## Phase 5: Module Reorganization (Low Risk)

**Effort:** Low | **Risk:** Low | **Impact:** Medium

### Current Structure
```
Preprocess/     # Mesh loading, connectivity, DEC
FrameField/     # Cross field computation
Orthotropic/    # Optimization
ComputeParam/   # UV recovery
Utils/          # I/O, visualization
```

### Target Structure
```
rectangular_surface_parameterization/
    __init__.py
    core/
        mesh.py           # TriangleMesh, CutMesh dataclasses
        operators.py      # DECOperators dataclass
        parameters.py     # OptimizationParams dataclass
        edge_index.py     # SignedEdgeArray
    preprocessing/
        connectivity.py
        curvature.py
        dec.py
    cross_field/
        trivial_connection.py
        curvature_field.py
        face_field.py
    optimization/
        objective.py
        constraints.py
        solver.py
    parameterization/
        cut_mesh.py
        seamless.py
        integrate.py
    io/
        obj.py
        visualization.py
    cli.py               # Entry point (was run_RSP.py)
```

### pyproject.toml Updates Required
When reorganizing, update `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["rectangular_surface_parameterization*"]

[project.scripts]
rsp = "rectangular_surface_parameterization.cli:main"
```

---

## Cross-Platform CI for libQEx

Currently quad extraction only works on Windows (pre-built binaries). GitHub Actions
can build libQEx for all platforms automatically, eliminating:
- Trust issues with contributed binaries
- Manual build overhead for users
- Platform fragmentation

### Workflow Design

```yaml
# .github/workflows/build-libqex.yml
name: Build libQEx
on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            artifact_path: libQEx/build/qex_extract
          - os: windows-latest
            artifact_path: libQEx/build/Release/qex_extract.exe
          - os: macos-latest
            artifact_path: libQEx/build/qex_extract
          - os: macos-14  # ARM
            artifact_path: libQEx/build/qex_extract
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Build OpenMesh
        run: |
          git clone https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
          cd OpenMesh && mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_APPS=OFF -DCMAKE_INSTALL_PREFIX=../install
          cmake --build . --config Release
          cmake --install . --config Release
      - name: Build libQEx
        run: |
          git clone https://github.com/hcebke/libQEx.git
          cd libQEx && mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          cmake --build . --config Release
      - uses: actions/upload-artifact@v4
        with:
          name: libqex-${{ matrix.os }}
          path: ${{ matrix.artifact_path }}
```

### Benefits
- Every binary built from source in controlled environment
- Triggered on version tags or manually
- Users download from GitHub Releases
- No virus risk from untrusted contributors

---

## GPU Acceleration Assessment

### Short Answer: Not Worth It

The computational core is **sparse linear algebra** (18 files use `scipy.sparse`). The main operations are:

1. **Sparse matrix assembly** - CPU-bound index manipulation
2. **Sparse direct solves** (`spsolve`) - Already uses optimized CHOLMOD
3. **Newton iteration** - Sequential with data dependencies

### Why GPU Won't Help Much

| Factor | Reality |
|--------|---------|
| Matrix size | Typical meshes: 1K-100K vertices. GPU overhead dominates. |
| Sparsity pattern | Irregular mesh connectivity. GPU sparse solvers struggle. |
| Transfer overhead | Data must move CPU↔GPU each iteration. |
| Direct vs iterative | Direct solvers (CHOLMOD) beat GPU iterative for these sizes. |

### When GPU Would Help
- **Batch processing**: Many meshes simultaneously (embarrassingly parallel)
- **Very large meshes**: 1M+ vertices where GPU memory bandwidth wins
- **Complete rewrite**: Use iterative GPU-native solvers (CG, GMRES with GPU preconditioners)

### Incremental Options (5-20% speedup, not transformative)
1. **CuPy** for dense operations (exponentials, element-wise)
2. **Numba JIT** for hot loops in index manipulation
3. **cuSPARSE** for SpMV (but not direct solves)

### Honest Recommendation
Focus on **algorithmic** improvements instead:
- Better preconditioners for iterative solves
- Multigrid methods for the Poisson system
- Warm-starting Newton with previous solution

---

## Already Completed

### Documentation Structure
- [x] Created `LICENSE` with AGPL-3.0 and proper attribution
- [x] Created `README.md` (user-facing: what it is, install, citation)
- [x] Created `USAGE.md` (CLI reference: all commands, options, troubleshooting)
- [x] Updated `CLAUDE.md` (developer-facing: architecture, internals)
- [x] Renamed GitHub repo to `rectangular-surface-parameterization`
- [x] Updated git remote URL

---

## Implementation Order

### Phase 1: Quick Wins (1-2 sessions) ✅ COMPLETE
1. [x] Strip MATLAB comments from Python files (commit 5696cf1)
2. [x] Add header comment to each file referencing commit `7d1aab4` for line-by-line translation
3. [x] Rename local folder from `Corman-Crane` to `rectangular-surface-parameterization`

**Reference commit for MATLAB translation:** `7d1aab4`

**Approach used:** Python script (`scripts/strip_matlab_comments.py`, now deleted) handled
cross-platform comment stripping and header insertion. Verified with:
```bash
# Confirm no MATLAB comments remain
rg "# %" --type py
# Confirm all files have reference header
rg "commit 7d1aab4" --type py
```

### Phase 2: Type Safety (2-3 sessions) ✅ COMPLETE
4. [x] `MeshInfo` dataclass exists in `Preprocess/MeshInfo.py` (14 fields)
5. [x] `DEC` dataclass exists in `Preprocess/dec_tri.py` (15 fields)
6. [x] `OrthoParam` dataclass exists in `Preprocess/preprocess_ortho_param.py`
7. [x] `Weight` dataclass exists in `run_RSP.py`
8. [x] Create `OptimizationParams` dataclass for hardcoded solver parameters in `optimize_RSP.py`
   - Created `Orthotropic/optimization_params.py` with all solver parameters
   - Updated `optimize_RSP()` to accept optional `opt_params` argument

### Phase 3: Naming (1-2 sessions) ✅ COMPLETE
9. [x] Renamed `Src` -> `mesh`, `SrcCut` -> `disk_mesh` (1765 occurrences)
10. [x] Renamed MeshInfo fields: `nv/nf/ne` -> `num_vertices/num_faces/num_edges`
11. [x] Renamed MeshInfo fields: `X` -> `vertices`, `T` -> `triangles`
12. [x] Renamed connectivity fields (409 occurrences, commit 839c67f):
    - `E2V` -> `edge_to_vertex`
    - `E2T` -> `edge_to_triangle`
    - `T2T` -> `triangle_to_triangle`
    - `Nv` -> `vertex_normals`
    - `SqEdgeLength` -> `sq_edge_length`
    - `T2E` retained (encapsulates SignedEdgeArray)

### Phase 4: Edge Encoding (2-3 sessions) ✅ COMPLETE
13. [x] Implement `SignedEdgeArray` class
14. [x] Migrate connectivity.py to return SignedEdgeArray
15. [x] Migrate all consumers to use .indices/.signs API
16. [x] Full test suite verification (728 tests pass)
17. [x] Replaced custom OBJ reader with trimesh (handles encoding, multi-format)

### Phase 5: Module Reorganization (1 session) ✅ COMPLETE
18. [x] Created `rectangular_surface_parameterization/` package structure
19. [x] Core classes importable: `from rectangular_surface_parameterization import MeshInfo, SignedEdgeArray, load_mesh`
20. [x] Updated pyproject.toml for package discovery
21. [x] Added `py.typed` marker
22. [x] Migrated all 39 modules using scripts/migrate_module.py and scripts/migrate_all.py
23. [x] Removed old directories (Preprocess/, FrameField/, Orthotropic/, ComputeParam/, Utils/)
24. [x] Test run (746 tests pass)

### Phase 6: Cross-Platform CI (1-2 sessions) ✅ COMPLETE
21. [x] Create `.github/workflows/build-libqex.yml`
22. [x] Build libQEx for Windows x64
23. [x] Build libQEx for Linux x64 (Ubuntu)
24. [x] Build libQEx for macOS (Intel + Apple Silicon)
25. [x] Auto-upload binaries to GitHub Releases (on version tags)
26. [ ] Update `bin/` with CI-built binaries (after first successful build)
27. [x] Add workflow for running tests on all platforms (`.github/workflows/test.yml`)

---

## Files to Delete After Pythonification

- `MATLAB_CONVERSION.md` (archived to docs)
- `matlab_converion_findings.md` (findings incorporated)
- `key-differences.md` (if outdated)
- `Orthotropic/findings.md` (if outdated)

---

## Success Criteria

1. All 54+ tests still pass
2. `run_RSP.py sphere320.obj` produces identical output (0 flipped triangles)
3. No MATLAB comments remain in Python files
4. All public APIs use typed dataclasses
5. Variable names follow Python conventions
6. Package installable via `pip install -e .`
