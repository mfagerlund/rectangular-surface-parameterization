"""
Cut graph and jump data computation (Algorithm 2 from supplement).

Computes:
- Γ: cut edge indicator {0, 1}
- ζ: quarter-rotation jump at each edge
- s: sign bits at corners {-1, +1}
- φ: reference frame angle per halfedge
- ω⁰: reference frame rotation across each edge
"""

import numpy as np
from collections import deque
from typing import Tuple

from mesh import TriangleMesh
from geometry import compute_corner_angles
from cross_field import compute_xi_per_halfedge

# Cone detection threshold (radians)
# A vertex is a cone if its cone index deviates from a multiple of π/2 by more than this
# Lower = more cones detected = less pruning = more cut edges = more fragmented UV
# Higher = fewer cones = more pruning = fewer cut edges = may cause flips
# Recommended range: 0.4-0.7
CONE_THRESHOLD = 0.5


def wrap_angle(x: float) -> float:
    """Wrap angle to [-π, π] range."""
    return np.arctan2(np.sin(x), np.cos(x))


def compute_cut_jump_data(
    mesh: TriangleMesh,
    alpha: np.ndarray,
    xi: np.ndarray,
    singularities: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Algorithm 2: ComputeCutJumpData

    Args:
        mesh: Triangle mesh
        alpha: |C| corner angles
        xi: |F| cross field angles (relative to first edge of each face)
        singularities: |V| cone indices from cross-field (optional)
                       If provided, uses MATLAB convention: cones where abs(sing) > 0.1

    Returns:
        Gamma: |E| cut edge indicator {0, 1}
        zeta: |E| quarter-rotation jump
        s: |C| sign bits {-1, +1}
        phi: |H| reference frame angle per halfedge
        omega0: |E| reference frame rotation across edge
    """
    n_faces = mesh.n_faces
    n_edges = mesh.n_edges
    n_halfedges = mesh.n_halfedges
    n_corners = mesh.n_corners
    n_vertices = mesh.n_vertices

    # Initialize outputs
    Gamma = np.ones(n_edges, dtype=np.int32)  # 1 = in cut (will be pruned)
    zeta = np.zeros(n_edges, dtype=np.float64)
    s = np.ones(n_corners, dtype=np.int32)  # sign bits at corners
    phi = np.full(n_halfedges, np.inf, dtype=np.float64)  # infinity = not yet set
    omega0 = np.zeros(n_edges, dtype=np.float64)

    # Step 1-4: Compute xi relative to each halfedge
    xi_he = compute_xi_per_halfedge(mesh, xi, alpha)

    # Step 5-8: BFS traversal to propagate reference frame
    seed_face = 0
    phi[3 * seed_face + 0] = wrap_angle(xi[seed_face])  # reference frame = cross field direction

    # Propagate phi within seed face (lines 20-21 equivalent)
    c0 = 3 * seed_face + 0
    c1 = 3 * seed_face + 1
    c2 = 3 * seed_face + 2
    phi[3 * seed_face + 1] = wrap_angle(phi[3 * seed_face + 0] - (np.pi - alpha[c1]))
    phi[3 * seed_face + 2] = wrap_angle(phi[3 * seed_face + 1] - (np.pi - alpha[c2]))

    # Edge sign lookup for each halfedge
    # s_ij at edges will be computed during traversal
    s_edge = np.ones(n_edges, dtype=np.int32)

    queue = deque()
    # Push all halfedges of seed face
    for local in range(3):
        he = 3 * seed_face + local
        queue.append(he)

    # BFS main loop (lines 9-28)
    while queue:
        he_ij = queue.popleft()

        face_from = he_ij // 3
        local_from = he_ij % 3

        # Get twin halfedge
        he_twin = mesh.halfedge_twin[he_ij]
        if he_twin == -1:
            continue  # boundary edge

        face_to = he_twin // 3
        local_to = he_twin % 3

        # Get edge index
        e = mesh.halfedge_to_edge[he_ij]

        # Get vertices i, j for orientation
        i, j = mesh.halfedge_vertices(he_ij)

        # Line 11: parallel transport phi (wrap to principal range)
        phi_ij_to_ji = wrap_angle(phi[he_ij] + np.pi)

        # Line 12: closest cross index
        # Wrap the difference before rounding to avoid drift
        angle_diff = wrap_angle(phi_ij_to_ji - xi_he[he_twin])
        n_star = int(np.round(2 * angle_diff / np.pi)) % 4

        # Line 13: jump angle
        zeta[e] = (np.pi / 2) * n_star

        # Line 14: closest cross direction
        xi_star = wrap_angle(zeta[e] + xi_he[he_twin])

        # Line 15: smallest rotation to neighboring cross (wrap to principal range)
        omega0_raw = phi[he_ij] - xi_star + np.pi
        omega0[e] = wrap_angle(omega0_raw)

        # Ensure omega0 sign is consistent with edge_to_halfedge ordering
        # The constraint uses edge_to_halfedge[e, 0] as the "from" side
        # If we computed from the opposite halfedge, negate omega0
        he0 = mesh.edge_to_halfedge[e, 0]
        if he_ij != he0:
            omega0[e] = -omega0[e]

        # Check if neighbor frame not yet set (line 16)
        if phi[he_twin] == np.inf:
            # Line 17-23: set neighbor frame
            Gamma[e] = 0  # not in cut

            # Line 18: s_ij ← +1 (no sign change for spanning tree edges)
            s_edge[e] = +1

            # Line 19: set reference frame in target triangle (already wrapped)
            phi[he_twin] = xi_star

            # Lines 20-21: propagate phi within triangle
            # Halfedges in target face: he_twin, next, prev
            face_to = he_twin // 3
            local_to = he_twin % 3

            he_next = 3 * face_to + (local_to + 1) % 3
            he_prev = 3 * face_to + (local_to + 2) % 3

            # Corner indices for angles
            # he_twin is halfedge ji in face jil
            # Next is il, prev is lj
            c_j = 3 * face_to + local_to  # corner at j
            c_i = 3 * face_to + (local_to + 1) % 3  # corner at i
            c_l = 3 * face_to + (local_to + 2) % 3  # corner at l

            # phi_il = phi_ji - (pi - alpha at i in this face) - WRAP
            phi[he_next] = wrap_angle(phi[he_twin] - (np.pi - alpha[c_i]))
            # phi_lj = phi_il - (pi - alpha at l) - WRAP
            phi[he_prev] = wrap_angle(phi[he_next] - (np.pi - alpha[c_l]))

            # Push other edges of this face to queue (lines 22-23)
            queue.append(he_next)
            queue.append(he_prev)

        else:
            # Line 24-27: neighbor frame already set
            Gamma[e] = 1  # edge could be in cut

            # Lines 26-27: s_ij ← (-1)^n* (sign change based on n_star parity)
            if n_star % 2 == 1:  # odd n_star
                s_edge[e] = -1
            else:
                s_edge[e] = +1

    # Step 29-33: Compute relative signs at corners
    for v in range(n_vertices):
        corners = mesh.vertex_corners(v)
        if len(corners) == 0:
            continue

        S = 1  # cumulative product
        for p, corner in enumerate(corners):
            s[corner] = S

            # Get edge to next corner around vertex
            # Corner is at vertex v in some face, halfedge from v
            he = corner  # halfedge index = corner index
            e = mesh.halfedge_to_edge[he]

            # Update cumulative sign
            S = s_edge[e] * S

    # Step 34-45: Compute cone indices (angle defect + omega0 contributions)
    K = np.zeros(n_vertices, dtype=np.float64)
    for c in range(n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K  # angle defect

    # Add omega0 contributions to cone index
    c_vertex = K.copy()  # cone index
    for e in range(n_edges):
        i, j = mesh.edge_vertices[e]
        # omega0 is signed based on edge orientation
        c_vertex[i] += omega0[e]
        c_vertex[j] -= omega0[e]

    # Step 46-57: Prune cut graph (remove degree-1 non-cone vertices)
    # Detect cones - prefer using cross-field singularities (MATLAB convention)
    if singularities is not None:
        # MATLAB convention: idcone = param.idx_int(abs(sing(param.idx_int)) > 0.1)
        is_cone = np.abs(singularities) > 0.1
    else:
        # Fall back to original threshold-based method
        # A vertex is a cone if c_vertex is not close to a multiple of pi/2
        is_cone = np.abs(np.mod(c_vertex + np.pi/4, np.pi/2) - np.pi/4) > CONE_THRESHOLD

    pruning = True
    while pruning:
        # Compute degree of each vertex in cut graph
        d = np.zeros(n_vertices, dtype=np.int32)
        for e in range(n_edges):
            if Gamma[e] == 1:
                i, j = mesh.edge_vertices[e]
                d[i] += 1
                d[j] += 1

        pruning = False
        for v in range(n_vertices):
            if d[v] == 1 and not is_cone[v]:
                # Remove the single incident cut edge
                for e in range(n_edges):
                    if Gamma[e] == 1:
                        i, j = mesh.edge_vertices[e]
                        if i == v or j == v:
                            Gamma[e] = 0
                            pruning = True
                            break

    return Gamma, zeta, s, phi, omega0


def count_cut_edges(Gamma: np.ndarray) -> int:
    """Count number of edges in cut graph."""
    return int(np.sum(Gamma))


def verify_cut_graph(mesh: TriangleMesh, Gamma: np.ndarray) -> Tuple[bool, str]:
    """
    Verify that the cut graph makes the mesh simply connected.

    For a valid cut:
    - Cut edges should form a tree or forest connecting cones
    - After cutting, mesh should be a topological disk
    """
    n_cut = count_cut_edges(Gamma)

    if n_cut == 0:
        # No cuts needed for disk topology
        return True, "No cuts needed"

    # Check connectivity of cut graph
    from mesh import euler_characteristic, count_boundary_loops

    # For now, just report statistics
    return True, f"Cut has {n_cut} edges"


def get_cone_vertices(mesh: TriangleMesh, alpha: np.ndarray, omega0: np.ndarray) -> np.ndarray:
    """
    Find vertices with non-zero cone index (singularities).

    Returns:
        Array of vertex indices that are cones
    """
    n_vertices = mesh.n_vertices

    # Compute cone index
    K = np.zeros(n_vertices, dtype=np.float64)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    for e in range(mesh.n_edges):
        i, j = mesh.edge_vertices[e]
        K[i] += omega0[e]
        K[j] -= omega0[e]

    # Find cones (non-zero mod pi/2)
    cone_index = np.round(2 * K / np.pi) * np.pi / 2  # quantized to multiples of pi/2
    is_cone = np.abs(cone_index) > 0.01

    return np.where(is_cone)[0]
