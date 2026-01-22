"""
Cross field computation and parallel transport.

A cross field assigns four orthogonal directions to each face (with 4-fold symmetry).
We store one representative unit vector W per face, and compute angles ξ relative
to the first edge of each face.

Two implementations:
1. BFS propagation (simple, fast, but produces noisy fields)
2. Connection Laplacian (proper method from MATLAB repo - globally smooth)

The connection Laplacian method:
- Builds a complex-valued Laplacian with 4-fold symmetry
- Solves eigenvalue problem for smoothest field
- Applies heat-flow smoothing
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve

from mesh import TriangleMesh
from geometry import compute_all_face_bases, compute_corner_angles, compute_face_areas


def parallel_transport(
    v: np.ndarray,
    from_normal: np.ndarray,
    to_normal: np.ndarray
) -> np.ndarray:
    """
    Parallel transport a tangent vector v from one tangent plane to another.

    Uses rotation that aligns the normals. For tangent vectors on a surface,
    this approximates Levi-Civita connection.

    Args:
        v: vector in the tangent plane of from_normal
        from_normal: source face normal
        to_normal: target face normal

    Returns:
        Transported vector in target tangent plane
    """
    # Axis of rotation is cross product of normals
    axis = np.cross(from_normal, to_normal)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(from_normal, to_normal)

    if sin_angle < 1e-10:
        # Normals are parallel, no rotation needed
        if cos_angle > 0:
            return v.copy()
        else:
            # Normals are anti-parallel (should not happen in a good mesh)
            return -v

    # Normalize axis
    axis = axis / sin_angle

    # Rodrigues' rotation formula
    v_rot = (v * cos_angle +
             np.cross(axis, v) * sin_angle +
             axis * np.dot(axis, v) * (1 - cos_angle))

    return v_rot


def propagate_cross_field(
    mesh: TriangleMesh,
    seed_face: int = 0,
    seed_direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Propagate a cross field from a seed face using BFS and parallel transport.

    Args:
        mesh: Triangle mesh
        seed_face: Starting face index
        seed_direction: Initial direction in seed face (if None, uses first edge direction)

    Returns:
        W: |F| x 3 array of one representative cross field direction per face
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    W = np.zeros((mesh.n_faces, 3), dtype=np.float64)
    visited = np.zeros(mesh.n_faces, dtype=bool)

    # Initialize seed face
    if seed_direction is None:
        # Use first edge direction (T1)
        seed_direction = T1[seed_face]
    else:
        # Project onto tangent plane and normalize
        seed_direction = seed_direction - np.dot(seed_direction, N[seed_face]) * N[seed_face]
        seed_direction = seed_direction / (np.linalg.norm(seed_direction) + 1e-30)

    W[seed_face] = seed_direction
    visited[seed_face] = True

    # BFS propagation
    queue = deque([seed_face])

    while queue:
        f = queue.popleft()

        # Visit neighbors across each edge
        for local in range(3):
            neighbor = mesh.face_adjacent(f, local)
            if neighbor == -1 or visited[neighbor]:
                continue

            # Parallel transport W[f] to neighbor
            w_transported = parallel_transport(W[f], N[f], N[neighbor])

            # Project onto neighbor's tangent plane (should already be there, but numerical safety)
            w_transported = w_transported - np.dot(w_transported, N[neighbor]) * N[neighbor]
            norm = np.linalg.norm(w_transported)
            if norm > 1e-10:
                w_transported = w_transported / norm
            else:
                # Fallback to neighbor's T1
                w_transported = T1[neighbor]

            W[neighbor] = w_transported
            visited[neighbor] = True
            queue.append(neighbor)

    return W


def cross_field_to_angles(mesh: TriangleMesh, W: np.ndarray) -> np.ndarray:
    """
    Convert cross field vectors to angles ξ relative to first edge of each face.

    From Algorithm 1 line 9:
    ξ_ijk = atan2(<T2, W>, <T1, W>)

    Args:
        W: |F| x 3 cross field directions

    Returns:
        xi: |F| array of angles in radians
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    xi = np.zeros(mesh.n_faces, dtype=np.float64)
    for f in range(mesh.n_faces):
        xi[f] = np.arctan2(np.dot(T2[f], W[f]), np.dot(T1[f], W[f]))

    return xi


def angles_to_cross_field(mesh: TriangleMesh, xi: np.ndarray) -> np.ndarray:
    """
    Convert angles back to cross field vectors.

    W = cos(ξ) * T1 + sin(ξ) * T2

    Args:
        xi: |F| array of angles in radians

    Returns:
        W: |F| x 3 cross field directions
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    W = np.zeros((mesh.n_faces, 3), dtype=np.float64)
    for f in range(mesh.n_faces):
        W[f] = np.cos(xi[f]) * T1[f] + np.sin(xi[f]) * T2[f]

    return W


def compute_xi_per_halfedge(mesh: TriangleMesh, xi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Compute cross field angle relative to each halfedge direction.

    From Algorithm 2 lines 1-4:
    For face ijk with halfedges ij, jk, ki:
    - ξ_ij = ξ_ijk (angle relative to edge ij)
    - ξ_jk = ξ_ij - (π - α_j^ki)
    - ξ_ki = ξ_jk - (π - α_k^ij)

    Args:
        xi: |F| array of cross field angles (relative to first edge)
        alpha: |C| array of corner angles

    Returns:
        xi_halfedge: |H| array of angles relative to each halfedge
    """
    xi_halfedge = np.zeros(mesh.n_halfedges, dtype=np.float64)

    for f in range(mesh.n_faces):
        # Corner indices
        c0 = 3 * f + 0  # corner at i in face ijk
        c1 = 3 * f + 1  # corner at j
        c2 = 3 * f + 2  # corner at k

        # Halfedge indices
        he_ij = 3 * f + 0
        he_jk = 3 * f + 1
        he_ki = 3 * f + 2

        # ξ_ij is given
        xi_halfedge[he_ij] = xi[f]

        # ξ_jk = ξ_ij - (π - α_j^ki)
        # α_j^ki is the angle at vertex j, which is corner index c1
        xi_halfedge[he_jk] = xi_halfedge[he_ij] - (np.pi - alpha[c1])

        # ξ_ki = ξ_jk - (π - α_k^ij)
        # α_k^ij is the angle at vertex k, which is corner index c2
        xi_halfedge[he_ki] = xi_halfedge[he_jk] - (np.pi - alpha[c2])

    return xi_halfedge


def compute_smoothness_energy(mesh: TriangleMesh, W: np.ndarray) -> float:
    """
    Compute cross field smoothness energy (sum of squared angle differences across edges).

    This measures how well the cross field is aligned across edges.

    Returns:
        Total smoothness energy (lower is better)
    """
    N, T1, T2 = compute_all_face_bases(mesh)
    energy = 0.0

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            continue

        f0 = he0 // 3
        f1 = he1 // 3

        # Transport W[f0] to f1's tangent plane
        w0_transported = parallel_transport(W[f0], N[f0], N[f1])

        # Compute angle difference (mod π/2 for cross field)
        angle0 = np.arctan2(np.dot(T2[f1], w0_transported), np.dot(T1[f1], w0_transported))
        angle1 = np.arctan2(np.dot(T2[f1], W[f1]), np.dot(T1[f1], W[f1]))

        diff = angle1 - angle0
        # Wrap to [-π/4, π/4] for cross field
        diff = np.arctan2(np.sin(4 * diff), np.cos(4 * diff)) / 4

        energy += diff * diff

    return energy


def initialize_smooth_cross_field(mesh: TriangleMesh, n_iters: int = 100) -> np.ndarray:
    """
    Initialize a smooth cross field by propagation then local smoothing.

    Args:
        n_iters: Number of smoothing iterations

    Returns:
        W: |F| x 3 smoothed cross field
    """
    # Initial propagation
    W = propagate_cross_field(mesh)
    xi = cross_field_to_angles(mesh, W)

    N, T1, T2 = compute_all_face_bases(mesh)

    # Simple smoothing: average neighboring angles (considering 4-fold symmetry)
    for _ in range(n_iters):
        new_xi = xi.copy()

        for f in range(mesh.n_faces):
            # Collect neighbor angles, transported to face f
            angles = [xi[f]]  # include self

            for local in range(3):
                neighbor = mesh.face_adjacent(f, local)
                if neighbor == -1:
                    continue

                # Transport W[neighbor] to face f
                w_neighbor = np.cos(xi[neighbor]) * T1[neighbor] + np.sin(xi[neighbor]) * T2[neighbor]
                w_transported = parallel_transport(w_neighbor, N[neighbor], N[f])

                # Get angle in face f's frame
                angle_n = np.arctan2(np.dot(T2[f], w_transported), np.dot(T1[f], w_transported))

                # Find closest representative (4-fold symmetry)
                diff = angle_n - xi[f]
                diff_mod = np.arctan2(np.sin(4 * diff), np.cos(4 * diff)) / 4
                angles.append(xi[f] + diff_mod)

            # Average (simple approach, could use better weighting)
            new_xi[f] = np.mean(angles)

        xi = new_xi

    return angles_to_cross_field(mesh, xi)


# =============================================================================
# Connection Laplacian Method (from MATLAB repo)
# =============================================================================

def _signed_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> float:
    """
    Compute signed angle from vector u to vector v, with n as the rotation axis.
    Matches MATLAB: atan2(dot(cross(u,v), n), dot(u,v))
    """
    cross_uv = np.cross(u, v)
    return np.arctan2(np.dot(cross_uv, n), np.dot(u, v))


def compute_parallel_transport_angles(mesh: TriangleMesh) -> np.ndarray:
    """
    Compute parallel transport angles between adjacent faces.

    Uses the edge_vertices convention for consistent global orientation:
    - edge_vertices[e] = (v1, v2) where v1 < v2
    - This must match the d1d operator which uses the same convention
    - f_pos = face where edge goes v1 -> v2 (positive direction)
    - f_neg = face where edge goes v2 -> v1 (negative direction)
    - para_trans[e] = angle(edge, T1[f_neg]) - angle(edge, T1[f_pos])

    The constraint d1d * para_trans = K (Gaussian curvature) should hold.

    Returns:
        para_trans: |E| array of parallel transport angles
    """
    N, T1, T2 = compute_all_face_bases(mesh)
    para_trans = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 == -1 or he1 == -1:
            # Boundary edge
            continue

        # Get canonical edge direction from edge_vertices (v1 < v2)
        v1, v2 = mesh.edge_vertices[e]
        edge_vec = mesh.positions[v2] - mesh.positions[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Determine which face has positive edge direction (v1 -> v2)
        i0, j0 = mesh.halfedge_vertices(he0)

        if i0 == v1 and j0 == v2:
            # he0 goes in positive direction (v1 -> v2)
            f_pos = he0 // 3
            f_neg = he1 // 3
        else:
            # he1 goes in positive direction
            f_pos = he1 // 3
            f_neg = he0 // 3

        # Signed angle from edge to T1 in each face
        angle_pos = _signed_angle(edge_vec, T1[f_pos], N[f_pos])
        angle_neg = _signed_angle(edge_vec, T1[f_neg], N[f_neg])

        # Parallel transport: f_neg - f_pos (this order satisfies d1d * para_trans = K)
        para_trans[e] = _wrap_to_pi(angle_neg - angle_pos)

    return para_trans


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_edge_face_map(mesh: TriangleMesh) -> np.ndarray:
    """
    Build edge-to-face mapping with consistent orientation.

    Returns:
        E2T: |E| x 2 array where E2T[e] = [f0, f1] are the two adjacent faces
             (f1 = -1 for boundary edges)
    """
    E2T = np.zeros((mesh.n_edges, 2), dtype=np.int32)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        E2T[e, 0] = he0 // 3 if he0 != -1 else -1
        E2T[e, 1] = he1 // 3 if he1 != -1 else -1

    return E2T


def compute_smooth_cross_field(
    mesh: TriangleMesh,
    smoothing_iters: int = 20,
    power: int = 4,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a globally smooth cross field using the connection Laplacian method.

    This implements the approach from the MATLAB repo:
    1. Build complex-valued connection Laplacian with 4-fold symmetry
    2. Find smallest eigenvector (smoothest field)
    3. Apply heat-flow smoothing

    Args:
        mesh: Triangle mesh
        smoothing_iters: Number of heat flow iterations
        power: Symmetry order (4 for cross field)
        verbose: Print progress info

    Returns:
        W: |F| x 3 array of cross field directions
        xi: |F| array of cross field angles relative to first edge
    """
    nf = mesh.n_faces
    ne = mesh.n_edges

    # Get face areas for mass matrix
    areas = compute_face_areas(mesh)

    # Compute parallel transport angles
    para_trans = compute_parallel_transport_angles(mesh)

    # Get edge-face map
    E2T = compute_edge_face_map(mesh)

    # Identify interior edges
    ide_int = np.where((E2T[:, 0] >= 0) & (E2T[:, 1] >= 0))[0]

    # Compute cotangent weights for edges
    alpha = compute_corner_angles(mesh)
    cotan_weights = _compute_cotan_weights_for_connection(mesh, alpha)

    if verbose:
        print(f"[cross_field] Building connection Laplacian...")
        print(f"  Interior edges: {len(ide_int)}/{ne}")

    # Build complex-valued connection Laplacian
    # d0d_cplx has entries exp(i * power * rot / 2) and -exp(-i * power * rot / 2)
    # for edges connecting face pairs

    rot = para_trans[ide_int]

    # Build sparse matrix indices
    # Each interior edge contributes two entries per row (one per adjacent face)
    row_idx = np.concatenate([ide_int, ide_int])
    col_idx = np.concatenate([E2T[ide_int, 0], E2T[ide_int, 1]])

    # Complex values: exp(i*power*rot/2) for first face, -exp(-i*power*rot/2) for second
    vals = np.concatenate([
        np.exp(1j * power * rot / 2),
        -np.exp(-1j * power * rot / 2)
    ])

    d0d_cplx = sp.csr_matrix((vals, (row_idx, col_idx)), shape=(ne, nf), dtype=np.complex128)

    # Edge weight matrix (inverse cotangent weights = star1d in MATLAB)
    # Use 1/cotan for the connection Laplacian
    star1d_diag = np.zeros(ne)
    star1d_diag[cotan_weights > 1e-10] = 1.0 / cotan_weights[cotan_weights > 1e-10]
    star1d_diag[cotan_weights <= 1e-10] = 1e10  # Large value for small weights
    star1d = sp.diags(star1d_diag)

    # Connection Laplacian: Wcon = d0d_cplx^H * star1d * d0d_cplx
    Wcon = d0d_cplx.conj().T @ star1d @ d0d_cplx
    Wcon = (Wcon + Wcon.conj().T) / 2  # Ensure Hermitian

    # Mass matrix (face areas)
    star0d = sp.diags(areas)

    if verbose:
        print(f"[cross_field] Solving eigenvalue problem...")

    # Find smallest eigenvector (smoothest field)
    # Wcon is Hermitian, so we need to handle complex eigenvalue problem
    # For small meshes, use dense solver; for large, use sparse
    try:
        if nf < 2000:
            # Dense solver for small meshes - handles complex Hermitian properly
            from scipy.linalg import eigh
            Wcon_dense = Wcon.toarray()
            star0d_dense = star0d.toarray()

            # Solve generalized eigenvalue problem: Wcon @ z = lambda * star0d @ z
            eigenvalues, eigenvectors = eigh(Wcon_dense, star0d_dense)

            # Take the smallest eigenvector (first one, eigenvalues are sorted ascending)
            z = eigenvectors[:, 0].astype(np.complex128)
            dt = 20 * abs(eigenvalues[0]) if eigenvalues[0] != 0 else 1.0

            if verbose:
                print(f"  Smallest eigenvalues: {eigenvalues[:5]}")
                print(f"  Heat flow dt: {dt}")
        else:
            # For large meshes, use sparse solver with real approximation
            eigenvalues, eigenvectors = eigsh(
                Wcon.real,
                k=min(5, nf - 1),
                M=star0d,
                which='SM',
                tol=1e-6
            )
            z = eigenvectors[:, 0].astype(np.complex128)
            dt = 20 * abs(eigenvalues[0]) if len(eigenvalues) > 0 else 1.0

            if verbose:
                print(f"  Smallest eigenvalues: {eigenvalues[:3]}")
                print(f"  Heat flow dt: {dt}")

    except Exception as e:
        if verbose:
            print(f"  Eigensolve failed: {e}, using random initialization")
        z = np.random.randn(nf) + 1j * np.random.randn(nf)
        dt = 1.0

    # Normalize to unit complex numbers
    z = z / (np.abs(z) + 1e-30)

    # Heat flow smoothing
    if verbose:
        print(f"[cross_field] Heat flow smoothing ({smoothing_iters} iterations)...")

    A = Wcon + dt * star0d

    for i in range(smoothing_iters):
        rhs = dt * star0d @ z
        try:
            z = spsolve(A, rhs)
        except Exception:
            # If solve fails, just normalize
            pass
        z = z / (np.abs(z) + 1e-30)

    # Extract angles
    xi = np.angle(z) / power

    if verbose:
        print(f"[cross_field] Done. Angle range: [{np.min(xi):.3f}, {np.max(xi):.3f}]")

    # Convert to 3D vectors
    W = angles_to_cross_field(mesh, xi)

    return W, xi


def _compute_cotan_weights_for_connection(mesh: TriangleMesh, alpha: np.ndarray) -> np.ndarray:
    """
    Compute cotangent weights per edge for the connection Laplacian.

    w[e] = (cot(alpha_opposite_in_f0) + cot(alpha_opposite_in_f1)) / 2

    Returns:
        weights: |E| array
    """
    weights = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]

        if he0 != -1:
            # Opposite corner in face containing he0
            face = he0 // 3
            local = he0 % 3
            opposite_corner = 3 * face + (local + 2) % 3
            cot_val = 1.0 / np.tan(alpha[opposite_corner] + 1e-10)
            weights[e] += 0.5 * max(cot_val, 1e-6)  # Clamp to positive

        if he1 != -1:
            face = he1 // 3
            local = he1 % 3
            opposite_corner = 3 * face + (local + 2) % 3
            cot_val = 1.0 / np.tan(alpha[opposite_corner] + 1e-10)
            weights[e] += 0.5 * max(cot_val, 1e-6)

    return weights


def compute_cross_field_singularities(
    mesh: TriangleMesh,
    xi: np.ndarray,
    alpha: np.ndarray,
    power: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross field singularities (cone indices) at vertices.

    Uses MATLAB formula: sing = (d1d*(para_trans - omega)) / (2*pi)
    Since d1d*para_trans = K: sing = (K - d1d*omega) / (2*pi)

    where omega is the field rotation across each edge, computed with consistent
    edge_vertices orientation to match the d1d operator.

    For a cross-field with n-fold symmetry (power=n), the sum of cone indices
    equals the Euler characteristic chi.

    Returns:
        cone_indices: |V| array of cone indices (multiples of 1/power)
        is_singular: |V| boolean array
    """
    N, T1, T2 = compute_all_face_bases(mesh)

    # Compute angle defect (Gaussian curvature)
    K = np.zeros(mesh.n_vertices, dtype=np.float64)
    for c in range(mesh.n_corners):
        v = mesh.corner_vertex(c)
        K[v] += alpha[c]
    K = 2 * np.pi - K

    # Compute omega for each edge: field angle change + parallel transport
    # Must use edge_vertices convention for consistency with d1d
    omega = np.zeros(mesh.n_edges, dtype=np.float64)

    for e in range(mesh.n_edges):
        he0 = mesh.edge_to_halfedge[e, 0]
        he1 = mesh.edge_to_halfedge[e, 1]
        if he0 == -1 or he1 == -1:
            continue

        # Get canonical edge direction from edge_vertices (v1 < v2)
        v1, v2 = mesh.edge_vertices[e]
        edge_vec = mesh.positions[v2] - mesh.positions[v1]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            continue
        edge_vec = edge_vec / edge_len

        # Determine which face has positive edge direction (v1 -> v2)
        i0, j0 = mesh.halfedge_vertices(he0)
        if i0 == v1 and j0 == v2:
            f_pos = he0 // 3
            f_neg = he1 // 3
        else:
            f_pos = he1 // 3
            f_neg = he0 // 3

        # Angle of edge in each face's T1 basis
        angle_pos = _signed_angle(edge_vec, T1[f_pos], N[f_pos])
        angle_neg = _signed_angle(edge_vec, T1[f_neg], N[f_neg])

        # Parallel transport = angle_neg - angle_pos (from our fixed formula)
        para_trans_e = _wrap_to_pi(angle_neg - angle_pos)

        # Field angle change from f_pos to f_neg (matching d0d convention)
        # d0d*xi[e] = xi[f_neg] - xi[f_pos] (derivative along positive edge direction)
        d0d_xi = xi[f_neg] - xi[f_pos]

        # omega = wrap_4(d0d*xi + para_trans)
        omega[e] = _wrap_to_pi(power * (d0d_xi + para_trans_e)) / power

    # Apply d1d: (d1d*omega)[v] = sum over edges of sign * omega[e]
    # sign = +1 if v = v1 (start of edge), -1 if v = v2 (end of edge)
    d1d_omega = np.zeros(mesh.n_vertices, dtype=np.float64)
    for e in range(mesh.n_edges):
        v1, v2 = mesh.edge_vertices[e]
        d1d_omega[v1] += omega[e]
        d1d_omega[v2] -= omega[e]

    # Singularity formula: sing = (K - d1d*omega) / (2*pi)
    cone_index = (K - d1d_omega) / (2 * np.pi)

    # Round to nearest multiple of 1/power
    cone_indices_rounded = np.round(cone_index * power) / power

    # A vertex is singular if cone index is not 0
    is_singular = np.abs(cone_indices_rounded) > 0.01

    return cone_indices_rounded, is_singular
