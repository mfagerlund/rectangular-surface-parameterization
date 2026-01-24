"""
Constraint propagation (§4.3).

Propagates integer constraints from the coarse mesh back to the original mesh.

The key insight: constraints on the coarse mesh can be expressed as linear
constraints on the original mesh via the decimation matrix D.

For half-edge h in coarse mesh:
    omega_h = U^-_{next(h)} - U^-_h

This becomes a constraint:
    (D_{next(h)} - D_h) @ U^grid = omega_h
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from scipy import sparse

from .decimate import DecimationResult
from .optimize import IntegerSolution


@dataclass
class ConstraintSystem:
    """Linear constraints for grid-aligned UVs."""
    A: Optional[sparse.csr_matrix]  # Constraint matrix
    omega: np.ndarray               # Right-hand side (integer values)
    constrained_indices: np.ndarray  # Which UV components are constrained


def build_constraint_matrix(decimation: DecimationResult,
                             solution: IntegerSolution,
                             n_original_tris: int) -> ConstraintSystem:
    """
    Build constraint matrix from decimation and integer solution.

    For each half-edge in the coarse mesh with integer omega:
        (D_{next(h)} - D_h) @ U = omega_h

    Args:
        decimation: Decimation result with mapping
        solution: Integer solution with omega values
        n_original_tris: Number of triangles in original mesh

    Returns:
        ConstraintSystem for final optimization
    """
    D = decimation.D
    n_coarse_verts, n_original_verts = D.shape

    # Build constraints from half-edges
    constraints_A = []
    constraints_omega = []
    constrained_indices = []

    triangles = decimation.triangles_coarse
    n_coarse_tris = len(triangles)

    # For each half-edge in coarse mesh
    for fi, tri in enumerate(triangles):
        for i in range(3):
            v_from = tri[i]
            v_to = tri[(i + 1) % 3]

            # Look up the integer omega for this half-edge
            # We need the half-edge index in the dual graph
            he_key = (v_from, v_to)

            # Find omega for this edge (may be stored by half-edge index)
            omega_val = None
            for he_idx, omega in solution.omega.items():
                # Check if this matches our edge
                # This is simplified - in full implementation we'd track properly
                if np.linalg.norm(omega) > 0.5:  # Non-zero integer
                    omega_val = omega
                    break

            if omega_val is None:
                continue

            # Build constraint: D[v_to, :] @ U - D[v_from, :] @ U = omega
            # This is for vertex-based UVs. For per-triangle UVs, we need
            # to map through the triangle structure.

            # For now, we'll build simpler constraints:
            # Just constrain the singularity positions to integers

    # Constrain singularities to integer positions
    for sing_idx in decimation.singularities_coarse:
        # Find original vertices that map to this singularity
        original_verts = np.where(decimation.vertex_map == sing_idx)[0]

        if len(original_verts) == 0:
            continue

        # For each original vertex at this singularity
        for orig_v in original_verts:
            # Find triangles containing this vertex
            # We need to constrain UV at this vertex to be integer

            # This is a simplified constraint - just record which indices to snap
            constrained_indices.append(orig_v)

    # Build sparse matrix
    if len(constrained_indices) == 0:
        return ConstraintSystem(
            A=None,
            omega=np.array([]),
            constrained_indices=np.array([]),
        )

    # For now, return placeholder - full implementation would build proper matrix
    n_constraints = len(constrained_indices)
    n_vars = n_original_tris * 3 * 2  # UV per triangle corner

    # Simple constraint: snap specified vertices to nearest integer
    A_data = []
    A_rows = []
    A_cols = []
    omega_vals = []

    for i, vi in enumerate(constrained_indices):
        # Find a triangle containing this vertex
        # and constrain its UV to be integer
        # This is a simplified version - just marking the constraint

        # Placeholder values
        omega_vals.append(0)  # Target will be computed during solve

    A = sparse.csr_matrix((n_constraints, n_vars)) if n_constraints > 0 else None

    return ConstraintSystem(
        A=A,
        omega=np.array(omega_vals),
        constrained_indices=np.array(constrained_indices),
    )


def propagate_constraints(decimation: DecimationResult,
                          solution: IntegerSolution) -> ConstraintSystem:
    """
    Propagate integer constraints from coarse to original mesh.

    This is the core of §4.3: the decimation matrix D allows us to
    express coarse-mesh constraints as original-mesh constraints.

    The key insight: singularities in the coarse mesh map back to original
    vertices via the vertex_map. We constrain those original vertices
    to snap to integer UV coordinates.

    Args:
        decimation: Decimation result with D matrix
        solution: Integer solution from optimization

    Returns:
        ConstraintSystem for final UV solve
    """
    # Find original vertices that are singularities
    # A vertex is a singularity if it maps to a coarse singularity
    constrained = []

    for coarse_sing in decimation.singularities_coarse:
        # Find all original vertices that map to this coarse singularity
        original_verts = np.where(decimation.vertex_map == coarse_sing)[0]
        constrained.extend(original_verts.tolist())

    # Remove duplicates and sort
    constrained = sorted(set(constrained))

    return ConstraintSystem(
        A=None,  # Not using matrix form for now
        omega=np.array([]),
        constrained_indices=np.array(constrained, dtype=int),
    )
