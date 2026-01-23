# MATLAB Cut Graph Algorithm Analysis

## Purpose
Understand MATLAB's cut graph algorithm to fix Python implementation.

## MATLAB Data Flow

### Input to `mesh_to_disk_seamless.m`
```
Src     - mesh structure
param   - preprocessing parameters (including para_trans, K, E2T, etc.)
ang     - cross-field angles per face
sing    - singularity indices per vertex (from cross-field)
k21     - rotation jump index per edge (1=identity, 2=90°, 3=180°, 4=270°)
```

### Key Call (mesh_to_disk_seamless.m:4-5)
```matlab
idcone = param.idx_int(abs(sing(param.idx_int)) > 0.1);  % Find cones
[SrcCut,...] = cut_mesh(..., idcone, k21 ~= 1);          % Call cut_mesh
```

**Key insight**: `edge_jump_tag = (k21 ~= 1)` - edges where rotation is NOT identity.

---

## `cut_mesh.m` Algorithm

### Inputs
- `X, T, E2V, E2T, T2E, T2T` - mesh connectivity
- `idcone` - vertex indices that are cones (singularities)
- `edge_jump_tag` - boolean array, true for edges with non-identity rotation

### Step 1: Dual Spanning Tree (lines 7-31)
```matlab
Q = 1;                              % Start from face 1
tri_pred = -ones(nf,1);             % Predecessor for each face
tri_pred(Q) = 0;                    % Face 1 has no predecessor
visited_edge = false(ne,1);         % Track visited edges
visited_edge(edge_jump_tag) = true; % PRE-MARK non-identity edges as visited!

while ~isempty(Q)
    idtri = Q(1);  Q(1) = [];

    % Get adjacent faces and their shared edges
    adj = T2T(idtri,:);
    adjedge = abs(T2E(idtri,:));

    for i = 1:length(adj)
        if (tri_pred(adj(i)) == -1) && ~visited_edge(adjedge(i))
            tri_pred(adj(i)) = idtri;
            Q = [Q; adj(i)];
            visited_edge(adjedge(i)) = true;
        end
    end
end
```

**Critical**: `visited_edge(edge_jump_tag) = true` means BFS CANNOT use edges with
non-identity rotation. The spanning tree uses ONLY identity-rotation edges.

### Step 2: Initial Cut (line 35)
```matlab
visited_edge(edge_jump_tag) = false;  % Reset jump edges
edge_cut = ~visited_edge;             % Cut = edges NOT visited by BFS
```

After this: `edge_cut = edges not in spanning tree`.

### Step 3: Prune Leaves (lines 36-45)
```matlab
deg = accumarray(E2V(edge_cut,:), 1, [nv,1]);  % Vertex degrees in cut
deg1 = deg == 1;  % Degree-1 vertices

while sum(deg1(idcone)) ~= sum(deg1)  % While non-cone degree-1 exists
    deg1(idcone(deg1(idcone))) = false;     % Don't prune cones
    edge_cut(any(deg1(E2V),2)) = false;      % Remove edges to deg-1 vertices
    deg = accumarray(E2V(edge_cut,:), 1, [nv,1]);
    deg1 = deg == 1;
end
```

This removes "leaf" edges that don't connect to cones.

### Step 4: Cut Mesh (lines 47-92)
Create new mesh topology with cut edges as boundary. Uses union-find for vertex equivalences.

### Output
- `SrcCut` - cut mesh (should have chi=1 for disk)
- `idx_cut_inv` - mapping from cut vertices to original vertices
- `ide_cut_inv` - mapping from cut edges to original edges
- `edge_cut` - boolean array of cut edges

---

## Comparison: MATLAB vs Python

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Cone source | `sing` from cross-field | Was recomputing, now fixed |
| Cone threshold | `abs(sing) > 0.1` | `abs(singularities) > 0.1` ✓ |
| k21/zeta | Pre-computed, input to cut_mesh | Computed during BFS |
| Spanning tree constraint | Uses ONLY identity edges | Uses ANY edges |
| Cut = | Edges NOT in spanning tree | Same ✓ |
| Prune leaves | Same algorithm | Same ✓ |

**Key difference**: MATLAB's spanning tree is constrained to identity-rotation edges.
Our spanning tree has no such constraint.

---

## Where is k21 computed in MATLAB?

Found in `reduction_from_ff2d.m` (called from `run_RSP.m:97`):

```matlab
% Line 5: Find rotation index that aligns cross-field across edge
[~,k21i] = min(abs(
    exp(1i*ang(param.E2T(param.ide_int,2)) + (0:3)*1i*pi/2 + 1i*(omega - para_trans))
    - exp(1i*ang(param.E2T(param.ide_int,1)))
), [], 2);
k21(param.ide_int) = k21i;
```

This finds k21 ∈ {1,2,3,4} that minimizes cross-field misalignment.

---

## k21 vs zeta Relationship

| MATLAB k21 | Python n_star | Python zeta | Meaning |
|------------|---------------|-------------|---------|
| 1 | 0 | 0 | Identity (no rotation) |
| 2 | 1 | π/2 | 90° rotation |
| 3 | 2 | π | 180° rotation |
| 4 | 3 | 3π/2 | 270° rotation |

**Key equivalence**:
- `edge_jump_tag = (k21 ~= 1)` in MATLAB
- `edge_jump_tag = (zeta ~= 0)` or `(n_star ~= 0)` in Python

---

## Algorithm Structure Difference

**MATLAB (two-phase)**:
1. Phase A: `reduction_from_ff2d` computes k21 for ALL edges
2. Phase B: `cut_mesh` builds spanning tree, AVOIDING edges where k21 ≠ 1

**Python (one-phase)**:
1. BFS computes n_star/zeta AND builds spanning tree simultaneously
2. No constraint on which edges can be in spanning tree

---

## Why This Matters

MATLAB's constraint ensures the spanning tree uses only "smooth" edges (identity rotation).
Non-smooth edges are FORCED into the cut.

Our algorithm lets the spanning tree use non-smooth edges, which might cause issues
downstream because:
1. UV coordinates might not align correctly across cut edges
2. The parameterization constraints might be harder to satisfy

---

## Proposed Fix

Two options:

### Option A: Two-pass algorithm (match MATLAB)
1. Pass 1: BFS to compute zeta for all edges (no spanning tree)
2. Pass 2: Build spanning tree using only zeta=0 edges

### Option B: Modify single-pass algorithm
1. When crossing an edge, if n_star ≠ 0, don't add to spanning tree
2. Keep the edge as Gamma=1 (in cut) and still propagate phi

**Issue with Option B**: If we don't add the edge to spanning tree but still propagate phi,
we might leave some faces unreachable (phi = inf).

### Option C: Two BFS passes
1. BFS 1: Compute zeta for all edges (visit all faces)
2. BFS 2: Build spanning tree using only zeta=0 edges

This is cleanest and matches MATLAB's structure.

---

## Test Plan

1. Implement Option C
2. Compare number of cut edges with current implementation
3. Compare flipped triangles
4. If improved, run full test suite
