"""
Debug script to investigate why constraints don't converge.
"""

import numpy as np
from pathlib import Path

from mesh import TriangleMesh, build_connectivity
from io_obj import load_obj
from geometry import compute_corner_angles, compute_edge_lengths
from cross_field import propagate_cross_field, cross_field_to_angles
from cut_graph import compute_cut_jump_data, count_cut_edges
from sparse_ops import build_constraint_system, build_jacobian


def verify_jacobian_numerically(mesh, alpha, u, v, s, phi, theta, omega0, eps=1e-7):
    """Verify Jacobian by finite differences."""
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges

    J = build_jacobian(mesh, alpha, u, v, s, phi, theta)
    F0 = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)

    # Check Jacobian w.r.t. u
    print("\nJacobian verification (finite differences):")
    max_err_u = 0
    for i in range(min(5, n_vertices)):  # test a few
        u_plus = u.copy()
        u_plus[i] += eps
        F_plus = build_constraint_system(mesh, alpha, u_plus, v, s, phi, theta, omega0)
        Jcol_numerical = (F_plus - F0) / eps
        Jcol_analytic = J[:, i].toarray().flatten()
        err = np.max(np.abs(Jcol_numerical - Jcol_analytic))
        max_err_u = max(max_err_u, err)
    print(f"  Max error w.r.t. u: {max_err_u:.2e}")

    # Check Jacobian w.r.t. v
    max_err_v = 0
    for i in range(min(5, n_vertices)):
        v_plus = v.copy()
        v_plus[i] += eps
        F_plus = build_constraint_system(mesh, alpha, u, v_plus, s, phi, theta, omega0)
        Jcol_numerical = (F_plus - F0) / eps
        Jcol_analytic = J[:, n_vertices + i].toarray().flatten()
        err = np.max(np.abs(Jcol_numerical - Jcol_analytic))
        max_err_v = max(max_err_v, err)
    print(f"  Max error w.r.t. v: {max_err_v:.2e}")

    # Check Jacobian w.r.t. theta
    max_err_theta = 0
    for f in range(min(5, n_faces)):
        theta_plus = theta.copy()
        theta_plus[f] += eps
        F_plus = build_constraint_system(mesh, alpha, u, v, s, phi, theta_plus, omega0)
        Jcol_numerical = (F_plus - F0) / eps
        Jcol_analytic = J[:, 2*n_vertices + f].toarray().flatten()
        err = np.max(np.abs(Jcol_numerical - Jcol_analytic))
        max_err_theta = max(max_err_theta, err)
    print(f"  Max error w.r.t. theta: {max_err_theta:.2e}")

    return max(max_err_u, max_err_v, max_err_theta)


def analyze_residual(mesh, rho, Gamma):
    """Analyze the constraint residual."""
    n_edges = mesh.n_edges

    print("\nResidual analysis:")
    print(f"  Total edges: {n_edges}")
    print(f"  Cut edges: {np.sum(Gamma)}")
    print(f"  |F| = {np.linalg.norm(rho):.4f}")
    print(f"  |F|_inf = {np.max(np.abs(rho)):.4f}")
    print(f"  Mean |rho|: {np.mean(np.abs(rho)):.4f}")

    # Count boundary edges
    n_boundary = 0
    for e in range(n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 == -1 or he1 == -1:
            n_boundary += 1
    print(f"  Boundary edges: {n_boundary}")

    # Distribution of residual
    sorted_rho = np.sort(np.abs(rho))[::-1]
    print(f"\n  Top 10 residuals:")
    for i in range(min(10, len(sorted_rho))):
        print(f"    {i+1}: {sorted_rho[i]:.4f}")

    # How many edges have significant residual?
    n_significant = np.sum(np.abs(rho) > 0.1)
    print(f"\n  Edges with |rho| > 0.1: {n_significant}")
    n_significant = np.sum(np.abs(rho) > 0.01)
    print(f"  Edges with |rho| > 0.01: {n_significant}")


def analyze_omega0(mesh, omega0, alpha):
    """Analyze omega0 values."""
    print("\nOmega0 analysis:")
    print(f"  Range: [{omega0.min():.4f}, {omega0.max():.4f}]")
    print(f"  Mean: {np.mean(omega0):.4f}")
    print(f"  Std: {np.std(omega0):.4f}")

    # omega0 should be related to curvature
    # At a vertex, sum of omega0 should give angle defect
    n_vertices = mesh.n_vertices
    vertex_omega_sum = np.zeros(n_vertices)
    for e in range(mesh.n_edges):
        i, j = mesh.edge_vertices[e]
        vertex_omega_sum[i] += omega0[e]
        vertex_omega_sum[j] -= omega0[e]

    # Angle defect
    K = np.zeros(n_vertices)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    print(f"\n  Vertex angle defect range: [{K.min():.4f}, {K.max():.4f}]")
    print(f"  Vertex omega0 sum range: [{vertex_omega_sum.min():.4f}, {vertex_omega_sum.max():.4f}]")


def analyze_sign_bits(s, mesh):
    """Analyze sign bits."""
    print("\nSign bits analysis:")
    n_pos = np.sum(s == 1)
    n_neg = np.sum(s == -1)
    print(f"  +1 corners: {n_pos}")
    print(f"  -1 corners: {n_neg}")

    # Check consistency around each vertex
    n_vertices = mesh.n_vertices
    for v in range(min(5, n_vertices)):
        corners = mesh.vertex_corners(v)
        signs = [s[c] for c in corners]
        print(f"  Vertex {v}: corners={corners}, signs={signs}")


def main():
    # Load a simple mesh
    mesh_path = Path("C:/Dev/Colonel/Data/Meshes/sphere320.obj")
    if not mesh_path.exists():
        print(f"Mesh not found: {mesh_path}")
        return

    print(f"Loading mesh: {mesh_path}")
    mesh = load_obj(str(mesh_path))
    print(f"Vertices: {mesh.n_vertices}, Faces: {mesh.n_faces}, Edges: {mesh.n_edges}")

    # Compute geometry and cross field
    alpha = compute_corner_angles(mesh)
    W = propagate_cross_field(mesh)
    xi = cross_field_to_angles(mesh, W)

    # Compute cut graph and jump data
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi)
    print(f"\nCut edges: {count_cut_edges(Gamma)}")

    # Initialize variables
    n_vertices = mesh.n_vertices
    n_faces = mesh.n_faces

    u = np.zeros(n_vertices)
    v = np.zeros(n_vertices)
    theta = np.zeros(n_faces)

    # Build initial constraint system
    rho = build_constraint_system(mesh, alpha, u, v, s, phi, theta, omega0)

    # Analyze
    analyze_residual(mesh, rho, Gamma)
    analyze_omega0(mesh, omega0, alpha)
    analyze_sign_bits(s, mesh)

    # Verify Jacobian
    verify_jacobian_numerically(mesh, alpha, u, v, s, phi, theta, omega0)

    # Try a different initial guess for theta
    print("\n" + "="*60)
    print("Testing with theta = omega0 sum / 2")

    # If theta_f0 - theta_f1 = omega0, then we need theta to satisfy this
    # This is a tree traversal problem

    # BFS to propagate theta
    theta_new = np.full(n_faces, np.inf)
    from collections import deque

    theta_new[0] = 0  # seed
    queue = deque([0])

    while queue:
        f = queue.popleft()

        # Get adjacent faces
        for local in range(3):
            he = 3 * f + local
            he_twin = mesh.halfedge_twin[he]
            if he_twin == -1:
                continue

            f_neighbor = he_twin // 3
            if theta_new[f_neighbor] != np.inf:
                continue  # already visited

            e = mesh.halfedge_to_edge[he]

            # theta[f] - theta[f_neighbor] = omega0[e] for correct orientation
            # But we need to check which side of the edge we're on
            he0 = mesh.edge_to_halfedge[e, 0]
            f0 = he0 // 3

            if f == f0:
                # theta[f0] - theta[f1] = omega0[e]
                # theta[f_neighbor] = theta[f] - omega0[e]
                theta_new[f_neighbor] = theta_new[f] - omega0[e]
            else:
                # theta[f0] - theta[f1] = omega0[e]
                # theta[f] is f1, so: theta[f0] = theta[f] + omega0[e]
                # But f_neighbor is f0? No wait...
                # If f != f0, then f must be in he1's face
                theta_new[f_neighbor] = theta_new[f] + omega0[e]

            queue.append(f_neighbor)

    # Check residual with new theta
    rho_new = build_constraint_system(mesh, alpha, u, v, s, phi, theta_new, omega0)
    print(f"\nWith BFS-propagated theta:")
    print(f"  |F| = {np.linalg.norm(rho_new):.4f}")

    # The residual should now be just the RHS contributions
    # If theta satisfies the LHS perfectly, residual = RHS only

    # Let's see what the residual looks like at each edge
    analyze_residual(mesh, rho_new, Gamma)


if __name__ == "__main__":
    main()
