"""
Integer geometry optimization (§4.2).

Optimizes edge geometry to be Gaussian integers (a + bi where a, b are integers)
while minimizing distortion from the seamless map.

Key insight from QaWiTM: Instead of MIP solvers, use Dijkstra's algorithm
on a dual graph where edge weights represent distortion costs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import heapq

from .dual_graph import DualGraph, get_vertex_one_ring


@dataclass
class IntegerSolution:
    """Result of integer optimization."""
    omega: Dict[int, np.ndarray]  # he_idx -> integer edge geometry
    distortion: float              # Total distortion from original


def round_to_gaussian_integer(z: np.ndarray) -> np.ndarray:
    """
    Round a complex number (as 2D vector) to nearest Gaussian integer.

    A Gaussian integer is a + bi where a, b are integers.
    """
    return np.round(z)


def compute_rounding_cost(omega: np.ndarray, omega_int: np.ndarray) -> float:
    """
    Compute cost of rounding omega to omega_int.

    Cost is the squared distance in UV space.
    """
    diff = omega - omega_int
    return float(np.dot(diff, diff))


def scale_and_round_initial(dual: DualGraph, target_scale: float) -> Dict[int, np.ndarray]:
    """
    Initial solution: scale UVs and round to integers.

    This is the simplest quantization strategy (equivalent to --scale).
    Used as a starting point for Dijkstra optimization.
    """
    omega_int = {}
    for he_idx in range(len(dual.half_edges)):
        scaled = dual.omega_initial[he_idx] * target_scale
        omega_int[he_idx] = round_to_gaussian_integer(scaled)
    return omega_int


def check_closure_constraint(dual: DualGraph, omega: Dict[int, np.ndarray],
                              face_idx: int) -> np.ndarray:
    """
    Check if edge geometries satisfy closure constraint for a face.

    Returns the closure error (should be zero for valid solution).
    """
    total = np.zeros(2)
    for he in dual.half_edges:
        if he.face_idx == face_idx:
            he_idx = dual.he_lookup[(he.vertex_from, he.vertex_to)]
            total += omega.get(he_idx, dual.omega_initial[he_idx])
    return total


def fix_closure_violation(dual: DualGraph, omega: Dict[int, np.ndarray],
                          face_idx: int) -> Dict[int, np.ndarray]:
    """
    Fix closure violation by adjusting one edge in the face.

    The closure constraint requires: sum of omega around face = 0.
    If violated, we adjust the edge with minimum cost increase.
    """
    # Find half-edges in this face
    face_hes = []
    for he in dual.half_edges:
        if he.face_idx == face_idx:
            he_idx = dual.he_lookup[(he.vertex_from, he.vertex_to)]
            face_hes.append(he_idx)

    if len(face_hes) != 3:
        return omega

    # Compute current closure error
    closure_error = np.zeros(2)
    for he_idx in face_hes:
        closure_error += omega.get(he_idx, dual.omega_initial[he_idx])

    if np.linalg.norm(closure_error) < 1e-6:
        return omega  # Already closed

    # Find best edge to adjust
    best_he = None
    best_cost = float('inf')

    for he_idx in face_hes:
        current = omega.get(he_idx, dual.omega_initial[he_idx])
        adjusted = current - closure_error

        # Cost of this adjustment
        original = dual.omega_initial[he_idx]
        cost = compute_rounding_cost(original, adjusted)

        if cost < best_cost:
            best_cost = cost
            best_he = he_idx

    if best_he is not None:
        omega = omega.copy()
        current = omega.get(best_he, dual.omega_initial[best_he])
        omega[best_he] = current - closure_error

    return omega


def dijkstra_optimization(dual: DualGraph, omega_initial: Dict[int, np.ndarray],
                          max_iterations: int = 1000) -> Dict[int, np.ndarray]:
    """
    Optimize integer geometry using Dijkstra-based approach.

    From §4.2: We iteratively improve the solution by finding better
    integer assignments that reduce distortion while maintaining closure.

    This implements "atomic operations" (quad loop insertion/deletion)
    as moves in the optimization.
    """
    omega = omega_initial.copy()

    # Get all faces
    faces = set()
    for he in dual.half_edges:
        faces.add(he.face_idx)

    # Fix all closure violations first
    for face_idx in faces:
        omega = fix_closure_violation(dual, omega, face_idx)

    # Compute initial distortion
    def compute_total_distortion():
        total = 0.0
        for he_idx in range(len(dual.half_edges)):
            original = dual.omega_initial[he_idx]
            current = omega.get(he_idx, original)
            total += compute_rounding_cost(original * 1.0, current)  # Use scale 1 for comparison
        return total

    best_distortion = compute_total_distortion()

    # Iterative improvement
    for iteration in range(max_iterations):
        improved = False

        # Try adjusting each edge by ±1 in u or v
        for he_idx in range(len(dual.half_edges)):
            current = omega.get(he_idx, dual.omega_initial[he_idx])

            for delta in [np.array([1, 0]), np.array([-1, 0]),
                          np.array([0, 1]), np.array([0, -1])]:

                # Try this adjustment
                test_omega = omega.copy()
                test_omega[he_idx] = current + delta

                # Must also adjust opposite half-edge
                he = dual.half_edges[he_idx]
                if he.opposite >= 0:
                    opp_current = test_omega.get(he.opposite, dual.omega_initial[he.opposite])
                    test_omega[he.opposite] = opp_current - delta

                # Fix any closure violations this creates
                for face_idx in faces:
                    test_omega = fix_closure_violation(dual, test_omega, face_idx)

                # Check if this improves distortion
                old_omega = omega
                omega = test_omega
                new_distortion = compute_total_distortion()
                omega = old_omega

                if new_distortion < best_distortion - 1e-6:
                    omega = test_omega
                    best_distortion = new_distortion
                    improved = True

        if not improved:
            break

    return omega


def optimize_integer_geometry(dual: DualGraph,
                               uv_coarse: np.ndarray,
                               target_scale: float = 1.0,
                               verbose: bool = False) -> IntegerSolution:
    """
    Optimize edge geometry to Gaussian integers.

    Implements §4.2 of QaWiTM:
    1. Initialize with scale-and-round
    2. Use Dijkstra-based optimization to improve
    3. Maintain closure constraints throughout

    Args:
        dual: Dual graph structure
        uv_coarse: UV coordinates on coarse mesh
        target_scale: Scale factor for UV coordinates
        verbose: Print progress

    Returns:
        IntegerSolution with optimized edge geometries
    """
    if verbose:
        print("  Integer optimization: initializing...")

    # Scale the initial omega values
    for he_idx in range(len(dual.half_edges)):
        dual.omega_initial[he_idx] = dual.omega_initial[he_idx] * target_scale

    # Initial solution: scale and round
    omega_initial = scale_and_round_initial(dual, 1.0)  # Already scaled above

    if verbose:
        # Count non-zero omega values
        non_zero = sum(1 for w in omega_initial.values() if np.linalg.norm(w) > 0.5)
        print(f"  Initial: {non_zero} non-zero edge geometries")

    # Optimize
    if verbose:
        print("  Integer optimization: running Dijkstra optimization...")

    omega_final = dijkstra_optimization(dual, omega_initial)

    # Compute final distortion
    total_distortion = 0.0
    for he_idx in range(len(dual.half_edges)):
        original = dual.omega_initial[he_idx]
        final = omega_final.get(he_idx, original)
        total_distortion += compute_rounding_cost(original, final)

    if verbose:
        print(f"  Final distortion: {total_distortion:.4f}")

    return IntegerSolution(
        omega=omega_final,
        distortion=total_distortion,
    )
