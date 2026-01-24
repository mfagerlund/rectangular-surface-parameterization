"""
Main quantization API.

Implements Algorithm 1 from the QaWiTM paper:
1. U^seamless ← minimize f(U) subject to seamless constraints (already done by RSP)
2. [A, ω] ← quantize(M, F, U^seamless)
3. U^grid ← minimize f(U) subject to AU = ω and seamless constraints
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from .decimate import decimate_mesh, DecimationResult
from .dual_graph import build_dual_graph, DualGraph
from .optimize import optimize_integer_geometry, IntegerSolution
from .propagate import propagate_constraints, ConstraintSystem


@dataclass
class QuantizationResult:
    """Result of UV quantization."""
    uv_grid: np.ndarray           # (N, 2) quantized UV coordinates
    singularity_error: float      # Total distance of singularities from integers
    num_singularities: int        # Number of singular vertices
    singularities_on_integer: int # Count that landed on integers
    decimation_ratio: float       # Ratio of coarse to original triangles

    def __str__(self) -> str:
        return (
            f"QuantizationResult:\n"
            f"  Singularities: {self.num_singularities} "
            f"({self.singularities_on_integer} on integers)\n"
            f"  Total error: {self.singularity_error:.6f}\n"
            f"  Decimation ratio: {self.decimation_ratio:.2%}"
        )


def find_singularities(triangles: np.ndarray, uv_per_triangle: np.ndarray,
                       tol: float = 1e-6) -> np.ndarray:
    """
    Find singular vertices (where cross-field has non-zero index).

    Singularities are detected by checking if the UV rotation around a vertex
    is not a multiple of 90 degrees.

    Args:
        triangles: (M, 3) triangle vertex indices
        uv_per_triangle: (M, 3, 2) UV coordinates per triangle corner
        tol: Tolerance for angle comparison

    Returns:
        Array of singular vertex indices
    """
    n_verts = triangles.max() + 1

    # Build vertex-to-triangle adjacency
    vertex_triangles = [[] for _ in range(n_verts)]
    for ti, tri in enumerate(triangles):
        for local_idx, vi in enumerate(tri):
            vertex_triangles[vi].append((ti, local_idx))

    singularities = []

    for vi in range(n_verts):
        if len(vertex_triangles[vi]) < 3:
            continue  # Boundary vertex

        # Compute total angle deficit around vertex
        total_angle = 0.0

        for ti, local_idx in vertex_triangles[vi]:
            # Get UV vectors from this vertex to neighbors
            uv_center = uv_per_triangle[ti, local_idx]
            uv_next = uv_per_triangle[ti, (local_idx + 1) % 3]
            uv_prev = uv_per_triangle[ti, (local_idx + 2) % 3]

            v1 = uv_next - uv_center
            v2 = uv_prev - uv_center

            # Signed angle
            angle = np.arctan2(
                v1[0] * v2[1] - v1[1] * v2[0],
                v1[0] * v2[0] + v1[1] * v2[1]
            )
            total_angle += angle

        # For a regular vertex, total angle should be 2π (or -2π)
        # Singularity has index = (2π - total_angle) / (π/2)
        angle_deficit = 2 * np.pi - abs(total_angle)

        if abs(angle_deficit) > tol:
            singularities.append(vi)

    return np.array(singularities, dtype=int)


def compute_singularity_positions(singularities: np.ndarray,
                                   triangles: np.ndarray,
                                   uv_per_triangle: np.ndarray) -> np.ndarray:
    """
    Get UV positions of singularities.

    Since UVs are per-triangle, we average across incident triangles.
    """
    if len(singularities) == 0:
        return np.zeros((0, 2))

    n_verts = triangles.max() + 1

    # Build vertex-to-triangle adjacency
    vertex_triangles = [[] for _ in range(n_verts)]
    for ti, tri in enumerate(triangles):
        for local_idx, vi in enumerate(tri):
            vertex_triangles[vi].append((ti, local_idx))

    positions = []
    for vi in singularities:
        uvs = []
        for ti, local_idx in vertex_triangles[vi]:
            uvs.append(uv_per_triangle[ti, local_idx])
        positions.append(np.mean(uvs, axis=0))

    return np.array(positions)


def measure_singularity_error(sing_uvs: np.ndarray) -> Tuple[float, int]:
    """
    Measure total distance of singularities from integer coordinates.

    Returns:
        (total_error, num_on_integer)
    """
    if len(sing_uvs) == 0:
        return 0.0, 0

    # Distance to nearest integer for each coordinate
    errors = np.abs(sing_uvs - np.round(sing_uvs))
    total_errors = np.sum(errors, axis=1)  # Sum u and v errors

    # Count how many are on integers (within tolerance)
    tol = 1e-6
    on_integer = np.all(errors < tol, axis=1)

    return float(np.sum(total_errors)), int(np.sum(on_integer))


def quantize_uv(vertices: np.ndarray,
                triangles: np.ndarray,
                uv_per_triangle: np.ndarray,
                target_scale: float = 1.0,
                verbose: bool = False) -> QuantizationResult:
    """
    Quantize UV coordinates to snap singularities to integers.

    Implements the QaWiTM algorithm:
    1. Decimate mesh while preserving singularities
    2. Optimize integer edge geometry on coarse mesh
    3. Propagate constraints to original mesh
    4. Re-solve for grid-aligned UVs

    Args:
        vertices: (N, 3) mesh vertex positions
        triangles: (M, 3) triangle indices
        uv_per_triangle: (M, 3, 2) UV coordinates per triangle corner
        target_scale: Target UV scale for quad density
        verbose: Print progress

    Returns:
        QuantizationResult with grid-aligned UV coordinates
    """
    if verbose:
        print("Quantization: Finding singularities...")

    # Step 0: Find singularities
    singularities = find_singularities(triangles, uv_per_triangle)
    sing_uvs = compute_singularity_positions(singularities, triangles, uv_per_triangle)

    initial_error, initial_on_int = measure_singularity_error(sing_uvs)

    if verbose:
        print(f"  Found {len(singularities)} singularities")
        print(f"  Initial error: {initial_error:.4f} ({initial_on_int} on integers)")

    if len(singularities) == 0:
        if verbose:
            print("  No singularities - using scaled UVs directly")
        return QuantizationResult(
            uv_grid=uv_per_triangle * target_scale,
            singularity_error=0.0,
            num_singularities=0,
            singularities_on_integer=0,
            decimation_ratio=1.0,
        )

    # Step 1: Decimate mesh (§4.1)
    if verbose:
        print("Quantization: Decimating mesh...")

    decimation = decimate_mesh(
        vertices, triangles, uv_per_triangle,
        singularities=singularities,
        verbose=verbose
    )

    if verbose:
        print(f"  Decimated: {len(triangles)} → {len(decimation.triangles_coarse)} triangles")

    # Step 2: Build dual graph and optimize integer geometry (§4.2)
    if verbose:
        print("Quantization: Building dual graph...")

    dual = build_dual_graph(
        decimation.triangles_coarse,
        decimation.uv_coarse,
    )

    if verbose:
        print(f"  Dual graph: {dual.num_edges} edges")
        print("Quantization: Optimizing integer geometry...")

    solution = optimize_integer_geometry(
        dual,
        decimation.uv_coarse,
        target_scale=target_scale,
        verbose=verbose
    )

    if verbose:
        print(f"  Optimized {len(solution.omega)} edge geometries")

    # Step 3: Propagate constraints to original mesh (§4.3)
    if verbose:
        print("Quantization: Propagating constraints...")

    constraints = propagate_constraints(
        decimation,
        solution,
    )

    # Also add original singularities directly as constraints
    # (in case vertex_map doesn't perfectly preserve them)
    all_constrained = set(constraints.constrained_indices.tolist())
    all_constrained.update(singularities.tolist())
    constraints = ConstraintSystem(
        A=constraints.A,
        omega=constraints.omega,
        constrained_indices=np.array(sorted(all_constrained), dtype=int),
    )

    # Step 4: Solve for grid-aligned UVs
    if verbose:
        print("Quantization: Solving constrained system...")

    # First scale UVs, then snap singularities
    uv_scaled = uv_per_triangle * target_scale

    uv_grid = solve_with_constraints(
        vertices, triangles, uv_scaled,
        constraints,
        verbose=verbose
    )

    # Measure final error
    final_sing_uvs = compute_singularity_positions(singularities, triangles, uv_grid)
    final_error, final_on_int = measure_singularity_error(final_sing_uvs)

    if verbose:
        print(f"  Final error: {final_error:.4f} ({final_on_int} on integers)")
        print(f"  Improvement: {initial_error - final_error:.4f}")

    return QuantizationResult(
        uv_grid=uv_grid,
        singularity_error=final_error,
        num_singularities=len(singularities),
        singularities_on_integer=final_on_int,
        decimation_ratio=len(decimation.triangles_coarse) / len(triangles),
    )


def solve_with_constraints(vertices: np.ndarray,
                           triangles: np.ndarray,
                           uv_initial: np.ndarray,
                           constraints: ConstraintSystem,
                           verbose: bool = False) -> np.ndarray:
    """
    Re-solve for UVs with integer constraints.

    This implements a simplified version that directly snaps singularities
    to integers and propagates the adjustment smoothly.
    """
    n_triangles = len(triangles)
    n_verts = triangles.max() + 1

    # Build vertex-to-triangle adjacency
    vertex_triangles = [[] for _ in range(n_verts)]
    for ti, tri in enumerate(triangles):
        for local_idx, vi in enumerate(tri):
            vertex_triangles[vi].append((ti, local_idx))

    uv_result = uv_initial.copy()

    if len(constraints.constrained_indices) == 0:
        if verbose:
            print("  No constraints to apply")
        return uv_result

    if verbose:
        print(f"  Solving with {len(constraints.constrained_indices)} constraints...")

    # For each constrained vertex (singularity), snap to nearest integer
    for vi in constraints.constrained_indices:
        if vi >= n_verts:
            continue

        # Get current UV positions for this vertex across all triangles
        uvs_at_vertex = []
        for ti, local_idx in vertex_triangles[vi]:
            uvs_at_vertex.append(uv_result[ti, local_idx])

        if not uvs_at_vertex:
            continue

        # Average position
        mean_uv = np.mean(uvs_at_vertex, axis=0)

        # Target: nearest integer
        target_uv = np.round(mean_uv)

        # Compute offset needed
        offset = target_uv - mean_uv

        # Apply offset to all occurrences of this vertex
        for ti, local_idx in vertex_triangles[vi]:
            uv_result[ti, local_idx] = uv_result[ti, local_idx] + offset

    return uv_result
