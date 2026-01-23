# MATLAB to Python Conversion Guide

Files have commented MATLAB code. Add Python **interleaved** below each section.

## Critical Pitfalls

### 1. vec() / x(:) is COLUMN-MAJOR
MATLAB's `x(:)` flattens in **column-major** (Fortran) order:
```python
# WRONG: np.ravel() or .flatten() - these are row-major (C order)
# RIGHT: np.ravel('F') or .flatten('F')
```

### 2. Signed Edge Indices: Edge 0 Problem
MATLAB uses 1-based indexing, so `T2E * sign` works (edge 1 * -1 = -1).
Python with 0-based: `edge 0 * -1 = 0` **loses sign info!**

**Solution**: Use 1-based encoding for signed indices:
```python
# Encode: T2E_signed = (edge_idx + 1) * sign
# Decode: edge_idx = np.abs(T2E) - 1, sign = np.sign(T2E)
```

### 3. Boundary Markers
- MATLAB uses `0` for "no neighbor"
- Python uses `-1` for "no neighbor" (since 0 is a valid index)

## Rules

1. **Issues first**: Add a comment block at TOP OF EACH .py FILE listing unsupported MATLAB functions that need custom implementation
2. Keep MATLAB comments above corresponding Python code
3. Translate section by section, not line by line
4. MATLAB is 1-indexed, Python is 0-indexed
5. **Cache scope**: If caching per-vertex results, key cache by mesh identity too (or require `clear_cache()` before each mesh) to avoid stale cross-mesh results.
6. **Signed edge encoding**: Always store signed edge indices as `(edge_idx + 1) * sign` and decode with `edge_idx = abs(T2E) - 1`, `sign = sign(T2E)`. Never store signed indices directly with 0-based edge IDs.
7. **Boundary sentinel**: Use `-1` consistently for "no neighbor" (never `0`, which is a valid index).

## Example

```python
# === ISSUES ===
# - accumarray: use np.add.at or np.bincount
# - ismember(..., 'rows'): need custom row matching
# === END ISSUES ===

# % Dual spanning tree
# Q = 1;
# tri_pred = -ones(nf,1);
# tri_pred(Q) = 0;

# Dual spanning tree
queue = deque([0])
tri_pred = -np.ones(nf, dtype=int)
tri_pred[0] = 0

# visited_edge = false(ne,1);
# visited_edge(edge_jump_tag) = true;

visited_edge = np.zeros(ne, dtype=bool)
visited_edge[edge_jump_tag] = True
```
