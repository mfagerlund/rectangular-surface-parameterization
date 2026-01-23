"""
Pipeline Verification Script

Tests each stage of the Corman-Crane pipeline independently.
Run: python verify_pipeline.py <mesh.obj>
"""

import sys
import numpy as np
from pathlib import Path

from mesh import TriangleMesh, euler_characteristic, genus
from io_obj import load_obj
from geometry import compute_corner_angles, compute_edge_lengths, compute_face_areas
from cross_field import propagate_cross_field, cross_field_to_angles, compute_smoothness_energy
from geometry import compute_all_face_bases


def verify_stage1_geometry(mesh: TriangleMesh):
    """Verify geometry computations."""
    print("\n" + "="*60)
    print("STAGE 1: Geometry Verification")
    print("="*60)

    alpha = compute_corner_angles(mesh)
    ell = compute_edge_lengths(mesh)
    areas = compute_face_areas(mesh)

    errors = []

    # Test 1: Sum of angles in each triangle = π
    print("\n[1.1] Triangle angle sums...")
    angle_sum_errors = []
    for f in range(mesh.n_faces):
        angle_sum = alpha[3*f] + alpha[3*f+1] + alpha[3*f+2]
        error = abs(angle_sum - np.pi)
        angle_sum_errors.append(error)
    max_error = max(angle_sum_errors)
    mean_error = np.mean(angle_sum_errors)
    if max_error < 1e-10:
        print(f"  PASS: All triangles sum to π (max error: {max_error:.2e})")
    else:
        print(f"  FAIL: Angle sum errors (max: {max_error:.2e}, mean: {mean_error:.2e})")
        errors.append("angle_sums")

    # Test 2: Total angle defect = 2πχ (Gauss-Bonnet)
    print("\n[1.2] Gauss-Bonnet theorem...")
    chi = euler_characteristic(mesh)
    expected_defect = 2 * np.pi * chi

    # Compute angle defect at each vertex
    vertex_angles = np.zeros(mesh.n_vertices)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        vertex_angles[v] += alpha[c]
    angle_defect = 2 * np.pi - vertex_angles
    total_defect = np.sum(angle_defect)

    gauss_bonnet_error = abs(total_defect - expected_defect)
    if gauss_bonnet_error < 1e-6:
        print(f"  PASS: Total angle defect = {total_defect:.4f}, expected 2πχ = {expected_defect:.4f}")
    else:
        print(f"  FAIL: Gauss-Bonnet error = {gauss_bonnet_error:.4e}")
        errors.append("gauss_bonnet")

    # Test 3: Edge lengths > 0
    print("\n[1.3] Edge lengths...")
    if np.all(ell > 0):
        print(f"  PASS: All {mesh.n_edges} edges have positive length")
        print(f"        Range: [{ell.min():.4f}, {ell.max():.4f}]")
    else:
        n_bad = np.sum(ell <= 0)
        print(f"  FAIL: {n_bad} edges have non-positive length")
        errors.append("edge_lengths")

    # Test 4: Areas > 0
    print("\n[1.4] Triangle areas...")
    if np.all(areas > 0):
        print(f"  PASS: All {mesh.n_faces} triangles have positive area")
        print(f"        Range: [{areas.min():.6f}, {areas.max():.6f}]")
    else:
        n_bad = np.sum(areas <= 0)
        print(f"  FAIL: {n_bad} triangles have non-positive area")
        errors.append("areas")

    # Summary
    print("\n" + "-"*40)
    if errors:
        print(f"STAGE 1 FAILED: {errors}")
        return False, alpha, ell
    else:
        print("STAGE 1 PASSED")
        return True, alpha, ell


def verify_stage2_cross_field(mesh: TriangleMesh):
    """Verify cross field computation."""
    print("\n" + "="*60)
    print("STAGE 2: Cross Field Verification")
    print("="*60)

    W = propagate_cross_field(mesh)
    xi = cross_field_to_angles(mesh, W)
    N, T1, T2 = compute_all_face_bases(mesh)

    errors = []

    # Test 1: W vectors are unit length
    print("\n[2.1] Unit length check...")
    norms = np.linalg.norm(W, axis=1)
    max_norm_error = np.max(np.abs(norms - 1.0))
    if max_norm_error < 1e-10:
        print(f"  PASS: All W vectors are unit length (max error: {max_norm_error:.2e})")
    else:
        print(f"  FAIL: W norm errors (max: {max_norm_error:.2e})")
        errors.append("unit_length")

    # Test 2: W vectors are tangent (orthogonal to normal)
    print("\n[2.2] Tangency check...")
    dots = np.array([np.dot(W[f], N[f]) for f in range(mesh.n_faces)])
    max_dot = np.max(np.abs(dots))
    if max_dot < 1e-10:
        print(f"  PASS: All W vectors are tangent (max |W·N|: {max_dot:.2e})")
    else:
        print(f"  FAIL: W not tangent (max |W·N|: {max_dot:.2e})")
        errors.append("tangency")

    # Test 3: Smoothness energy
    print("\n[2.3] Smoothness energy...")
    energy = compute_smoothness_energy(mesh, W)
    avg_energy_per_edge = energy / mesh.n_edges
    print(f"  INFO: Total smoothness energy = {energy:.4f}")
    print(f"        Average per edge = {avg_energy_per_edge:.6f}")
    print(f"        (Lower is smoother, 0 = perfectly smooth)")

    # For a smooth cross field on sphere, this should be small
    # but not zero due to singularities
    if avg_energy_per_edge < 0.1:
        print(f"  PASS: Cross field is reasonably smooth")
    else:
        print(f"  WARN: Cross field may be noisy (avg energy > 0.1)")

    # Summary
    print("\n" + "-"*40)
    if errors:
        print(f"STAGE 2 FAILED: {errors}")
        return False, W, xi
    else:
        print("STAGE 2 PASSED")
        return True, W, xi


def verify_stage3_cut_graph(mesh: TriangleMesh, alpha: np.ndarray, xi: np.ndarray):
    """Verify cut graph computation."""
    print("\n" + "="*60)
    print("STAGE 3: Cut Graph Verification")
    print("="*60)

    from cut_graph import compute_cut_jump_data, count_cut_edges

    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi)

    errors = []

    # Test 1: phi covers all halfedges
    print("\n[3.1] Frame coverage check...")
    n_inf = np.sum(phi == np.inf)
    if n_inf == 0:
        print(f"  PASS: All {mesh.n_halfedges} halfedges have frame angles")
    else:
        print(f"  FAIL: {n_inf} halfedges have no frame angle")
        errors.append("phi_coverage")

    # Test 2: omega0 distribution
    print("\n[3.2] Frame rotation (omega0) distribution...")
    print(f"  INFO: omega0 range: [{omega0.min():.4f}, {omega0.max():.4f}]")
    print(f"        omega0 mean: {np.mean(omega0):.4f}, std: {np.std(omega0):.4f}")

    # For a good cross field, omega0 should be close to 0 at most edges
    n_small = np.sum(np.abs(omega0) < 0.1)
    pct_small = 100 * n_small / mesh.n_edges
    print(f"        {n_small}/{mesh.n_edges} edges ({pct_small:.1f}%) have |omega0| < 0.1")

    # Test 3: Cone indices
    print("\n[3.3] Cone index computation...")
    n_vertices = mesh.n_vertices

    # Compute K (angle defect)
    K = np.zeros(n_vertices)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    # Compute cone index
    c_vertex = K.copy()
    for e in range(mesh.n_edges):
        i, j = mesh.edge_vertices[e]
        c_vertex[i] += omega0[e]
        c_vertex[j] -= omega0[e]

    print(f"  INFO: Cone index range: [{c_vertex.min():.4f}, {c_vertex.max():.4f}]")
    print(f"        Cone index mean: {np.mean(c_vertex):.4f}, std: {np.std(c_vertex):.4f}")

    # Check if cone indices sum to 2πχ
    chi = euler_characteristic(mesh)
    expected_sum = 2 * np.pi * chi
    actual_sum = np.sum(c_vertex)
    sum_error = abs(actual_sum - expected_sum)
    print(f"  INFO: Sum of cone indices = {actual_sum:.4f}, expected 2πχ = {expected_sum:.4f}")
    if sum_error < 0.1:
        print(f"  PASS: Sum matches Gauss-Bonnet (error: {sum_error:.4e})")
    else:
        print(f"  WARN: Sum differs from expected (error: {sum_error:.4f})")

    # Count vertices with cone index close to multiples of π/2
    deviation = np.abs(np.mod(c_vertex + np.pi/4, np.pi/2) - np.pi/4)
    n_regular = np.sum(deviation < 0.1)  # Close to 0, π/2, π, etc.
    n_irregular = mesh.n_vertices - n_regular
    print(f"  INFO: {n_regular} regular vertices (deviation < 0.1)")
    print(f"        {n_irregular} irregular vertices (possible cones or noise)")

    # Test 4: Cut edge count
    print("\n[3.4] Cut graph structure...")
    n_cut = count_cut_edges(Gamma)
    print(f"  INFO: Cut edges: {n_cut}")

    # For genus 0, expected cut = tree connecting cones
    g = genus(mesh)
    if g == 0:
        # Count actual cones (vertices with significant deviation)
        from cut_graph import CONE_THRESHOLD
        is_cone = deviation > CONE_THRESHOLD
        n_cones = np.sum(is_cone)
        expected_min_cuts = max(0, n_cones - 1)
        print(f"  INFO: Detected {n_cones} cones (threshold={CONE_THRESHOLD})")
        print(f"        Minimum cuts for tree: {expected_min_cuts}")

        if n_cut >= expected_min_cuts:
            print(f"  PASS: Cut count is sufficient")
        else:
            print(f"  WARN: Cut count may be insufficient")

    # Summary
    print("\n" + "-"*40)
    if errors:
        print(f"STAGE 3 FAILED: {errors}")
        return False, Gamma, zeta, s, phi, omega0
    else:
        print("STAGE 3 PASSED (with warnings to investigate)")
        return True, Gamma, zeta, s, phi, omega0


def verify_stage4_optimization(mesh: TriangleMesh, alpha: np.ndarray,
                               phi: np.ndarray, omega0: np.ndarray, s: np.ndarray):
    """Verify optimization."""
    print("\n" + "="*60)
    print("STAGE 4: Optimization Verification")
    print("="*60)

    from optimization import solve_constraints_only

    print("\n[4.1] Running constraint solver...")
    u, v, theta = solve_constraints_only(mesh, alpha, phi, omega0, s,
                                         max_iters=500, tol=1e-6, verbose=False)

    errors = []

    # Test 1: Solutions are finite
    print("\n[4.2] Solution finiteness...")
    if np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(theta)):
        print(f"  PASS: All solutions are finite")
        print(f"        u range: [{u.min():.4f}, {u.max():.4f}]")
        print(f"        v range: [{v.min():.4f}, {v.max():.4f}]")
        print(f"        theta range: [{theta.min():.4f}, {theta.max():.4f}]")
    else:
        print(f"  FAIL: Solutions contain NaN/inf")
        errors.append("finite")

    # Test 2: Constraint residual
    print("\n[4.3] Constraint satisfaction...")
    from sparse_ops import build_constraint_system
    F = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)
    residual = np.linalg.norm(F)
    print(f"  INFO: Constraint residual |F| = {residual:.2e}")
    if residual < 1e-4:
        print(f"  PASS: Constraints satisfied")
    else:
        print(f"  WARN: Constraint residual is large")

    # Summary
    print("\n" + "-"*40)
    if errors:
        print(f"STAGE 4 FAILED: {errors}")
        return False, u, v, theta
    else:
        print("STAGE 4 PASSED")
        return True, u, v, theta


def verify_stage5_uv_recovery(mesh: TriangleMesh, Gamma, zeta, ell, alpha,
                              phi, theta, s, u, v):
    """Verify UV recovery."""
    print("\n" + "="*60)
    print("STAGE 5: UV Recovery Verification")
    print("="*60)

    from uv_recovery import recover_parameterization, compute_uv_quality

    print("\n[5.1] Recovering UV coordinates...")
    corner_uvs = recover_parameterization(mesh, Gamma, zeta, ell, alpha, phi, theta, s, u, v)

    errors = []

    # Test 1: UVs are finite
    print("\n[5.2] UV finiteness...")
    if np.all(np.isfinite(corner_uvs)):
        print(f"  PASS: All UVs are finite")
        print(f"        U range: [{corner_uvs[:,0].min():.4f}, {corner_uvs[:,0].max():.4f}]")
        print(f"        V range: [{corner_uvs[:,1].min():.4f}, {corner_uvs[:,1].max():.4f}]")
    else:
        n_bad = np.sum(~np.isfinite(corner_uvs))
        print(f"  FAIL: {n_bad} UV values are NaN/inf")
        errors.append("finite")

    # Test 2: Quality metrics
    print("\n[5.3] Quality metrics...")
    quality = compute_uv_quality(mesh, corner_uvs)

    n_flips = quality['flipped_count']
    pct_flips = 100 * n_flips / mesh.n_faces
    print(f"  INFO: Flipped triangles: {n_flips}/{mesh.n_faces} ({pct_flips:.1f}%)")

    if n_flips == 0:
        print(f"  PASS: No flipped triangles")
    else:
        print(f"  FAIL: {n_flips} flipped triangles")
        errors.append("flips")

    angle_error = np.degrees(quality['angle_error_mean'])
    print(f"  INFO: Mean angle error: {angle_error:.2f} degrees")

    # Test 3: Fill ratio (connectedness)
    print("\n[5.4] UV domain fill ratio...")
    uv_min = corner_uvs.min(axis=0)
    uv_max = corner_uvs.max(axis=0)
    bbox_area = (uv_max[0] - uv_min[0]) * (uv_max[1] - uv_min[1])

    # Compute actual UV area
    uv_area = 0
    for f in range(mesh.n_faces):
        uv0 = corner_uvs[3*f]
        uv1 = corner_uvs[3*f+1]
        uv2 = corner_uvs[3*f+2]
        # Signed area (negative if flipped)
        area = 0.5 * abs((uv1[0]-uv0[0])*(uv2[1]-uv0[1]) - (uv2[0]-uv0[0])*(uv1[1]-uv0[1]))
        uv_area += area

    fill_ratio = uv_area / bbox_area if bbox_area > 0 else 0
    print(f"  INFO: UV area / bbox area = {fill_ratio:.2%}")

    if fill_ratio > 0.8:
        print(f"  PASS: UV domain is well-filled")
    elif fill_ratio > 0.5:
        print(f"  WARN: UV domain may be fragmented")
    else:
        print(f"  FAIL: UV domain is highly fragmented")
        errors.append("fragmented")

    # Summary
    print("\n" + "-"*40)
    if errors:
        print(f"STAGE 5 FAILED: {errors}")
        return False, corner_uvs
    else:
        print("STAGE 5 PASSED")
        return True, corner_uvs


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_pipeline.py <mesh.obj>")
        sys.exit(1)

    mesh_path = sys.argv[1]
    print(f"\nLoading mesh: {mesh_path}")
    mesh = load_obj(mesh_path)

    print(f"\nMesh info:")
    print(f"  Vertices: {mesh.n_vertices}")
    print(f"  Faces: {mesh.n_faces}")
    print(f"  Edges: {mesh.n_edges}")
    print(f"  Euler characteristic: {euler_characteristic(mesh)}")
    print(f"  Genus: {genus(mesh)}")

    # Stage 1
    ok1, alpha, ell = verify_stage1_geometry(mesh)
    if not ok1:
        print("\n*** STOPPING: Stage 1 failed ***")
        sys.exit(1)

    # Stage 2
    ok2, W, xi = verify_stage2_cross_field(mesh)
    if not ok2:
        print("\n*** STOPPING: Stage 2 failed ***")
        sys.exit(1)

    # Stage 3
    ok3, Gamma, zeta, s, phi, omega0 = verify_stage3_cut_graph(mesh, alpha, xi)
    if not ok3:
        print("\n*** STOPPING: Stage 3 failed ***")
        sys.exit(1)

    # Stage 4
    ok4, u, v, theta = verify_stage4_optimization(mesh, alpha, phi, omega0, s)
    if not ok4:
        print("\n*** STOPPING: Stage 4 failed ***")
        sys.exit(1)

    # Stage 5
    ok5, corner_uvs = verify_stage5_uv_recovery(mesh, Gamma, zeta, ell, alpha,
                                                 phi, theta, s, u, v)

    # Final summary
    print("\n" + "="*60)
    print("PIPELINE VERIFICATION SUMMARY")
    print("="*60)
    stages = [
        ("Stage 1: Geometry", ok1),
        ("Stage 2: Cross Field", ok2),
        ("Stage 3: Cut Graph", ok3),
        ("Stage 4: Optimization", ok4),
        ("Stage 5: UV Recovery", ok5),
    ]

    all_passed = True
    for name, passed in stages:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL STAGES PASSED")
    else:
        print("SOME STAGES FAILED - investigate before proceeding")


if __name__ == "__main__":
    main()
