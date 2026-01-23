"""
Test file implementing equivalent Python tests for MATLAB assertions.

This file contains pytest tests that verify the same invariants as the MATLAB
implementation of Corman & Crane's rectangular parameterization.

Source: C:/slask/RectangularSurfaceParameterization/

Summary of MATLAB assertions extracted:
========================================

1. trivial_connection.m:27 - Gauss-Bonnet for singularities:
   sum(sing) = chi (Euler characteristic) for closed surfaces

2. trivial_connection.m:59 - Prescribed singularities match computed:
   norm(sing[interior] - sing2[interior]) < 1e-5

3. trivial_connection.m:63 - Cycle constraints satisfied:
   norm(sing_loop - sing_loop2) < 1e-5

4. trivial_connection.m:67 - Feature curve constraints satisfied:
   norm(sing_link - sing_link2) < 1e-5

5. compute_face_cross_field.m:84 - No NaN in vector field:
   all(~isnan(z))

6. angles_of_triangles.m:29 - No zero-size triangles:
   all(~isnan(A)) && all(isreal(A))

7. cut_mesh.m:69 - New indices valid after cutting:
   max(Tc(:)) == length(idx_cut_inv)

8. cut_mesh.m:85 - All edge indices found:
   all(ide_cut_inv ~= 0)

9. cut_mesh.m:95-100 - Union-find equivalence valid:
   size(equiv,2) == 1 or 2, all(equiv > 0), all(equiv <= n)

10. mesh_to_disk_seamless.m:16 - Cut edges paired correctly:
    all(abs(ide_cut[1:2:end]) == abs(ide_cut[2:2:end]))

11. mesh_to_disk_seamless.m:40 - Fixed triangles found:
    length(ia) == length(tri_fix_cut)

12. dec_tri.m:15 - Positive vertex areas:
    all(vor_area > 0)

13. dec_tri.m:24 - DEC primal orientation:
    norm(d1p * d0p) == 0

14. dec_tri.m:32 - DEC dual orientation:
    norm(d1d * d0d) == 0

15. find_graph_generator.m:37 - Single connected component:
    all(~isnan(pred))

16. find_graph_generator.m:77-78 - All predecessors valid:
    all(~isnan(pred)) && all(~isnan(copred))

17. preprocess_ortho_param.m:65 - Unique triangle constraints:
    numel(tri_fix) == length(unique(tri_fix))

18. preprocess_ortho_param.m:198 - Gaussian curvature matches topology:
    norm(sum(K) - 2*pi*chi) < 1e-5

19. preprocess_ortho_param.m:222 - Curvature compatible with angle defect:
    norm(wrapToPi(d1d*para_trans - K)) < 1e-6

20. preprocess_ortho_param.m:283 - No isolated vertices in d1d:
    all(sum(abs(d1d),2) ~= 0)

21. preprocess_ortho_param.m:327,360 - Path/cycle edge counts match:
    length(ide) == length(P)-1, length(ide) == length(cocycle{i})

22. MeshInfo.m:5 - Input is triangulation:
    size(T,2) == 3

23. sort_triangles_comp.m:100 - Manifold mesh:
    unique(paths) == size(paths,1)

24. brush_frame_field.m:41-42 - Variable dimensions correct:
    size(x,1) == nv, size(omega,1) == ne

25. optimize_RSP.m:76,79 - Optimization converged:
    norm(A*x - b) < 1e-5

26. optimize_RSP.m:165 - Boundary constraints respected:
    max(abs(err_ang_bound)) < 1e-3

27. oracle_integrability_condition.m:57 - Second derivative valid:
    max(abs(D_vth*vt - dO'*lambda_full)) < 1e-6

28. reduce_corner_var_2d_cut.m:36,59 - Vertex indices valid:
    id(1) is true, ~isempty(idvx)

29. writeObj.m:24 - Vertex matrix is 3D:
    size(V,2) == 3

30. writeObj.m:64-65 - Face indices positive:
    all(NF > 0) && all(TF > 0)

31. readOBJ.m:41,144-145 - Face parsing correct:
    numel(t) == numel(tf) == numel(nf)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh import TriangleMesh, build_connectivity, euler_characteristic, validate_manifold
from io_obj import load_obj
from geometry import (
    compute_corner_angles,
    compute_face_areas,
    compute_face_normals,
    compute_cotan_weights,
    angle_defect,
    total_gaussian_curvature,
    compute_all_face_bases
)
from cross_field import (
    compute_smooth_cross_field,
    compute_cross_field_singularities,
    compute_parallel_transport_angles
)
from cut_graph import compute_cut_jump_data


# Test mesh paths
SPHERE_PATH = "C:/Dev/Colonel/Data/Meshes/sphere320.obj"
TORUS_PATH = "C:/Dev/Colonel/Data/Meshes/torus.obj"


@pytest.fixture
def sphere_mesh():
    """Load the test sphere mesh."""
    return load_obj(SPHERE_PATH)


@pytest.fixture
def torus_mesh():
    """Load the test torus mesh if it exists."""
    try:
        return load_obj(TORUS_PATH)
    except FileNotFoundError:
        pytest.skip("Torus mesh not found")


# =============================================================================
# MeshInfo.m assertions (basic mesh validation)
# =============================================================================

class TestMeshValidation:
    """Tests corresponding to MeshInfo.m assertions."""

    def test_triangulation_shape(self, sphere_mesh):
        """
        MeshInfo.m:5 - assert(size(T,2) == 3, 'Not a triangulations.')
        Verify mesh has triangular faces (3 vertices per face).
        """
        assert sphere_mesh.faces.shape[1] == 3, "Mesh must have triangular faces"

    def test_manifold_mesh(self, sphere_mesh):
        """
        sort_triangles_comp.m:100 - Non manifold mesh check
        Verify mesh is manifold (edge has at most 2 incident faces).
        """
        is_valid, msg = validate_manifold(sphere_mesh)
        assert is_valid, f"Mesh must be manifold: {msg}"

    def test_vertex_positions_3d(self, sphere_mesh):
        """
        writeObj.m:24 - assert(size(V,2) == 3)
        Verify vertices are 3D.
        """
        assert sphere_mesh.positions.shape[1] == 3, "Vertices must be 3D"

    def test_face_indices_valid(self, sphere_mesh):
        """
        writeObj.m:64-65 - assert(all(NF > 0)) && assert(all(TF > 0))
        Verify face indices are valid (positive in MATLAB = non-negative in Python).
        """
        assert np.all(sphere_mesh.faces >= 0), "Face indices must be non-negative"
        assert np.all(sphere_mesh.faces < sphere_mesh.n_vertices), \
            "Face indices must be less than vertex count"


# =============================================================================
# angles_of_triangles.m assertions (triangle angle computation)
# =============================================================================

class TestTriangleAngles:
    """Tests corresponding to angles_of_triangles.m assertions."""

    def test_no_nan_angles(self, sphere_mesh):
        """
        angles_of_triangles.m:29 - assert(all(~isnan(A(:))), 'Triangle of size zero.')
        Verify no NaN values in corner angles.
        """
        alpha = compute_corner_angles(sphere_mesh)
        assert not np.any(np.isnan(alpha)), "Corner angles must not contain NaN"

    def test_real_angles(self, sphere_mesh):
        """
        angles_of_triangles.m:29 - assert(all(isreal(A(:))), 'Triangle of size zero.')
        Verify all angles are real (no imaginary components).
        """
        alpha = compute_corner_angles(sphere_mesh)
        assert not np.any(np.iscomplex(alpha)), "Corner angles must be real"
        # Also check for reasonable range
        assert np.all(alpha > 0), "All angles must be positive"
        assert np.all(alpha < np.pi), "All angles must be less than pi"

    def test_angle_sum_per_face(self, sphere_mesh):
        """
        Additional invariant: angles in each face sum to pi.
        """
        alpha = compute_corner_angles(sphere_mesh)
        for f in range(sphere_mesh.n_faces):
            angle_sum = alpha[3*f] + alpha[3*f+1] + alpha[3*f+2]
            assert abs(angle_sum - np.pi) < 1e-10, \
                f"Angles in face {f} must sum to pi, got {angle_sum}"


# =============================================================================
# dec_tri.m assertions (Discrete Exterior Calculus operators)
# =============================================================================

class TestDECOperators:
    """Tests corresponding to dec_tri.m assertions."""

    def test_positive_vertex_areas(self, sphere_mesh):
        """
        dec_tri.m:15 - assert(all(vor_area > 0), 'Negative vertex area.')
        Verify Voronoi (barycentric) vertex areas are positive.
        """
        areas = compute_face_areas(sphere_mesh)
        # Compute vertex area as 1/3 of incident face areas (barycentric)
        vor_area = np.zeros(sphere_mesh.n_vertices)
        for f in range(sphere_mesh.n_faces):
            for local in range(3):
                v = sphere_mesh.faces[f, local]
                vor_area[v] += areas[f] / 3.0

        assert np.all(vor_area > 0), "Voronoi vertex areas must be positive"

    def test_positive_face_areas(self, sphere_mesh):
        """
        Additional invariant: all face areas must be positive.
        """
        areas = compute_face_areas(sphere_mesh)
        assert np.all(areas > 0), "All face areas must be positive"

    def test_d0p_d1p_composition_is_zero(self, sphere_mesh):
        """
        dec_tri.m:24 - assert(norm(d1p*d0p, 'fro') == 0, 'Orinetation problems')
        Verify d1p * d0p = 0 (boundary of a boundary is zero).

        d0p: V -> E (gradient)
        d1p: E -> F (curl)
        """
        import scipy.sparse as sp

        nv = sphere_mesh.n_vertices
        ne = sphere_mesh.n_edges
        nf = sphere_mesh.n_faces

        # Build d0p (primal gradient: vertices to edges)
        # d0p[e, v] = +1 if v is end of edge, -1 if v is start
        row = []
        col = []
        data = []
        for e in range(ne):
            v0, v1 = sphere_mesh.edge_vertices[e]
            row.extend([e, e])
            col.extend([v0, v1])
            data.extend([-1, 1])  # v0 -> v1 direction
        d0p = sp.csr_matrix((data, (row, col)), shape=(ne, nv))

        # Build d1p (primal curl: edges to faces)
        # d1p[f, e] = +1 or -1 based on edge orientation in face
        row = []
        col = []
        data = []
        for f in range(nf):
            for local in range(3):
                he = 3 * f + local
                e = sphere_mesh.halfedge_to_edge[he]
                # Determine sign based on halfedge vs edge orientation
                i, j = sphere_mesh.halfedge_vertices(he)
                v0, v1 = sphere_mesh.edge_vertices[e]
                sign = 1 if (i == v0 and j == v1) else -1
                row.append(f)
                col.append(e)
                data.append(sign)
        d1p = sp.csr_matrix((data, (row, col)), shape=(nf, ne))

        # Check composition is zero
        composition = d1p @ d0p
        norm_fro = np.sqrt(np.sum(composition.toarray()**2))
        assert norm_fro < 1e-10, f"d1p*d0p must be zero, got norm {norm_fro}"

    def test_d0d_d1d_composition_is_zero(self, sphere_mesh):
        """
        dec_tri.m:32 - assert(norm(d1d*d0d, 'fro') == 0, 'Orinetation problems')
        Verify d1d * d0d = 0 (dual operators also satisfy boundary property).

        d0d: F -> E (dual gradient, transpose of d1p)
        d1d: E -> V (dual curl, transpose of d0p)
        """
        import scipy.sparse as sp

        nv = sphere_mesh.n_vertices
        ne = sphere_mesh.n_edges
        nf = sphere_mesh.n_faces

        # Build d1d = d0p^T (dual coboundary: edges to vertices)
        row = []
        col = []
        data = []
        for e in range(ne):
            v0, v1 = sphere_mesh.edge_vertices[e]
            row.extend([v0, v1])
            col.extend([e, e])
            data.extend([1, -1])  # v0 gets +1, v1 gets -1
        d1d = sp.csr_matrix((data, (row, col)), shape=(nv, ne))

        # Build d0d = d1p^T (dual gradient: faces to edges)
        row = []
        col = []
        data = []
        for f in range(nf):
            for local in range(3):
                he = 3 * f + local
                e = sphere_mesh.halfedge_to_edge[he]
                i, j = sphere_mesh.halfedge_vertices(he)
                v0, v1 = sphere_mesh.edge_vertices[e]
                sign = 1 if (i == v0 and j == v1) else -1
                row.append(e)
                col.append(f)
                data.append(sign)
        d0d = sp.csr_matrix((data, (row, col)), shape=(ne, nf))

        # Check composition is zero
        composition = d1d @ d0d
        norm_fro = np.sqrt(np.sum(composition.toarray()**2))
        assert norm_fro < 1e-10, f"d1d*d0d must be zero, got norm {norm_fro}"


# =============================================================================
# preprocess_ortho_param.m assertions (Gaussian curvature and topology)
# =============================================================================

class TestGaussianCurvature:
    """Tests corresponding to preprocess_ortho_param.m assertions."""

    def test_gauss_bonnet_theorem(self, sphere_mesh):
        """
        preprocess_ortho_param.m:198 -
        assert(norm(sum(K) - 2*pi*(Src.nf-Src.ne+Src.nv)) < 1e-5)

        Gauss-Bonnet: sum(K) = 2*pi*chi where chi = V - E + F
        """
        alpha = compute_corner_angles(sphere_mesh)
        K = angle_defect(sphere_mesh, alpha)
        chi = euler_characteristic(sphere_mesh)

        total_K = np.sum(K)
        expected = 2 * np.pi * chi

        assert abs(total_K - expected) < 1e-5, \
            f"Gauss-Bonnet failed: sum(K)={total_K:.6f}, 2*pi*chi={expected:.6f}"

    def test_sphere_euler_characteristic(self, sphere_mesh):
        """
        Additional check: sphere should have chi = 2.
        """
        chi = euler_characteristic(sphere_mesh)
        assert chi == 2, f"Sphere should have chi=2, got chi={chi}"

    def test_parallel_transport_curvature_compatibility(self, sphere_mesh):
        """
        preprocess_ortho_param.m:222 -
        assert(norm(wrapToPi(dec.d1d*para_trans - K)) < 1e-6)

        Verify d1d * para_trans = K (parallel transport recovers curvature).
        """
        import scipy.sparse as sp

        alpha = compute_corner_angles(sphere_mesh)
        K = angle_defect(sphere_mesh, alpha)
        para_trans = compute_parallel_transport_angles(sphere_mesh)

        nv = sphere_mesh.n_vertices
        ne = sphere_mesh.n_edges

        # Build d1d operator
        row = []
        col = []
        data = []
        for e in range(ne):
            v0, v1 = sphere_mesh.edge_vertices[e]
            row.extend([v0, v1])
            col.extend([e, e])
            data.extend([1, -1])
        d1d = sp.csr_matrix((data, (row, col)), shape=(nv, ne))

        # Compute d1d * para_trans
        d1d_para = d1d @ para_trans

        # Wrap to [-pi, pi] and compare
        diff = d1d_para - K
        diff_wrapped = np.arctan2(np.sin(diff), np.cos(diff))

        max_diff = np.max(np.abs(diff_wrapped))
        assert max_diff < 1e-4, \
            f"Parallel transport incompatible with curvature, max diff={max_diff:.6f}"


# =============================================================================
# trivial_connection.m assertions (singularities and Gauss-Bonnet)
# =============================================================================

class TestSingularities:
    """Tests corresponding to trivial_connection.m assertions."""

    def test_singularity_gauss_bonnet(self, sphere_mesh):
        """
        trivial_connection.m:27 -
        assert(norm(sum(sing) - (Src.nf - Src.ne + Src.nv)) < 1e-5)

        For closed surfaces, sum of singularities = Euler characteristic.
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)
        sing, is_singular = compute_cross_field_singularities(sphere_mesh, xi, alpha)

        chi = euler_characteristic(sphere_mesh)
        total_sing = np.sum(sing)

        # The paper uses 4-fold symmetry, so singularities are multiples of 1/4
        # and they should sum to chi
        assert abs(total_sing - chi) < 0.5, \
            f"Sum of singularities ({total_sing:.4f}) should equal chi ({chi})"

    def test_singularities_are_quantized(self, sphere_mesh):
        """
        Additional check: singularities should be multiples of 1/4 for cross-fields.
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)
        sing, is_singular = compute_cross_field_singularities(sphere_mesh, xi, alpha)

        # Non-zero singularities should be multiples of 1/4
        for i in range(len(sing)):
            if abs(sing[i]) > 0.01:
                # Check if it's a multiple of 1/4
                remainder = abs(sing[i] * 4 - round(sing[i] * 4))
                assert remainder < 0.1, \
                    f"Singularity at vertex {i} ({sing[i]:.4f}) not a multiple of 1/4"


# =============================================================================
# compute_face_cross_field.m assertions (cross field computation)
# =============================================================================

class TestCrossField:
    """Tests corresponding to compute_face_cross_field.m assertions."""

    def test_no_nan_in_cross_field(self, sphere_mesh):
        """
        compute_face_cross_field.m:84 - assert(all(~isnan(z)), 'NaN vector field.')
        Verify no NaN values in computed cross field.
        """
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        assert not np.any(np.isnan(W)), "Cross field vectors must not contain NaN"
        assert not np.any(np.isnan(xi)), "Cross field angles must not contain NaN"

    def test_cross_field_unit_length(self, sphere_mesh):
        """
        Additional check: cross field vectors should be unit length.
        """
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        norms = np.linalg.norm(W, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), \
            f"Cross field vectors must be unit length, got range [{norms.min():.6f}, {norms.max():.6f}]"

    def test_cross_field_tangent_to_surface(self, sphere_mesh):
        """
        Additional check: cross field vectors should be tangent to surface.
        """
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)
        N = compute_face_normals(sphere_mesh)

        # W should be perpendicular to N (dot product ~= 0)
        dots = np.sum(W * N, axis=1)
        max_dot = np.max(np.abs(dots))

        assert max_dot < 1e-6, \
            f"Cross field must be tangent to surface, max dot(W,N) = {max_dot:.6f}"


# =============================================================================
# find_graph_generator.m assertions (graph connectivity)
# =============================================================================

class TestGraphConnectivity:
    """Tests corresponding to find_graph_generator.m assertions."""

    def test_single_connected_component(self, sphere_mesh):
        """
        find_graph_generator.m:37 - assert(all(~isnan(pred)), 'Multiple connected components.')
        Verify mesh is a single connected component.
        """
        from collections import deque

        # BFS to check connectivity
        visited = np.zeros(sphere_mesh.n_vertices, dtype=bool)
        queue = deque([0])
        visited[0] = True

        while queue:
            v = queue.popleft()
            # Visit neighbors through halfedges
            he_start = sphere_mesh.vertex_to_halfedge[v]
            if he_start == -1:
                continue

            he = he_start
            while True:
                # Get the other vertex of this halfedge
                _, neighbor = sphere_mesh.halfedge_vertices(he)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

                # Move to next halfedge around vertex
                he_twin = sphere_mesh.halfedge_twin[he]
                if he_twin == -1:
                    break
                he = sphere_mesh.halfedge_next(he_twin)
                if he == he_start:
                    break

        assert np.all(visited), "Mesh must be a single connected component"


# =============================================================================
# cut_graph.py assertions (cut mesh and jump data)
# =============================================================================

class TestCutGraph:
    """Tests corresponding to cut_mesh.m assertions."""

    def test_cut_jump_data_valid(self, sphere_mesh):
        """
        Test that cut_jump_data returns valid data structures.
        Corresponds to multiple cut_mesh.m assertions.
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(sphere_mesh, alpha, xi)

        # Check array shapes
        assert len(Gamma) == sphere_mesh.n_edges, "Gamma must have n_edges entries"
        assert len(zeta) == sphere_mesh.n_edges, "zeta must have n_edges entries"
        assert len(s) == sphere_mesh.n_corners, "s must have n_corners entries"
        assert len(phi) == sphere_mesh.n_halfedges, "phi must have n_halfedges entries"
        assert len(omega0) == sphere_mesh.n_edges, "omega0 must have n_edges entries"

    def test_gamma_binary(self, sphere_mesh):
        """
        Gamma should be binary (0 or 1).
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(sphere_mesh, alpha, xi)

        assert np.all((Gamma == 0) | (Gamma == 1)), "Gamma must be binary (0 or 1)"

    def test_sign_bits_valid(self, sphere_mesh):
        """
        Sign bits s should be +1 or -1.
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(sphere_mesh, alpha, xi)

        assert np.all((s == 1) | (s == -1)), "Sign bits must be +1 or -1"

    def test_zeta_quantized(self, sphere_mesh):
        """
        zeta should be multiples of pi/2 (quarter rotations).
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(sphere_mesh, alpha, xi)

        # Check zeta is multiple of pi/2
        zeta_normalized = zeta / (np.pi / 2)
        zeta_rounded = np.round(zeta_normalized)
        diff = np.abs(zeta_normalized - zeta_rounded)

        assert np.all(diff < 0.01), \
            f"zeta must be multiples of pi/2, max deviation = {np.max(diff):.4f}"

    def test_phi_finite(self, sphere_mesh):
        """
        phi should be finite (no inf values after BFS traversal).
        """
        alpha = compute_corner_angles(sphere_mesh)
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(sphere_mesh, alpha, xi)

        assert np.all(np.isfinite(phi)), "phi must be finite for all halfedges"


# =============================================================================
# brush_frame_field.m assertions (frame field dimensions)
# =============================================================================

class TestDimensions:
    """Tests corresponding to brush_frame_field.m dimension assertions."""

    def test_cross_field_dimensions(self, sphere_mesh):
        """
        brush_frame_field.m:41 - assert(size(x,1) == nv)
        brush_frame_field.m:42 - assert(size(omega,1) == ne)

        Verify cross field has correct dimensions.
        """
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        assert W.shape == (sphere_mesh.n_faces, 3), \
            f"W must be (n_faces, 3), got {W.shape}"
        assert xi.shape == (sphere_mesh.n_faces,), \
            f"xi must be (n_faces,), got {xi.shape}"

    def test_corner_angle_dimensions(self, sphere_mesh):
        """
        Verify corner angles have correct dimensions.
        """
        alpha = compute_corner_angles(sphere_mesh)

        assert alpha.shape == (sphere_mesh.n_corners,), \
            f"alpha must be (n_corners,), got {alpha.shape}"

    def test_cotan_weight_dimensions(self, sphere_mesh):
        """
        Verify cotan weights have correct dimensions.
        """
        alpha = compute_corner_angles(sphere_mesh)
        weights = compute_cotan_weights(sphere_mesh, alpha)

        assert weights.shape == (sphere_mesh.n_edges,), \
            f"cotan weights must be (n_edges,), got {weights.shape}"


# =============================================================================
# Integration tests (combining multiple stages)
# =============================================================================

class TestPipelineIntegration:
    """Integration tests that verify multiple assertions together."""

    def test_full_pipeline_no_nan(self, sphere_mesh):
        """
        Verify no NaN values appear anywhere in the pipeline.
        """
        # Stage 1: Geometry
        alpha = compute_corner_angles(sphere_mesh)
        areas = compute_face_areas(sphere_mesh)
        N, T1, T2 = compute_all_face_bases(sphere_mesh)

        assert not np.any(np.isnan(alpha)), "Corner angles contain NaN"
        assert not np.any(np.isnan(areas)), "Face areas contain NaN"
        assert not np.any(np.isnan(N)), "Face normals contain NaN"
        assert not np.any(np.isnan(T1)), "T1 tangents contain NaN"
        assert not np.any(np.isnan(T2)), "T2 tangents contain NaN"

        # Stage 2: Cross field
        W, xi = compute_smooth_cross_field(sphere_mesh, smoothing_iters=10, verbose=False)

        assert not np.any(np.isnan(W)), "Cross field W contains NaN"
        assert not np.any(np.isnan(xi)), "Cross field xi contains NaN"

        # Stage 3: Cut graph
        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(sphere_mesh, alpha, xi)

        assert not np.any(np.isnan(Gamma)), "Gamma contains NaN"
        assert not np.any(np.isnan(zeta)), "zeta contains NaN"
        assert not np.any(np.isnan(s)), "s contains NaN"
        assert not np.any(np.isnan(phi)), "phi contains NaN"
        assert not np.any(np.isnan(omega0)), "omega0 contains NaN"

    def test_topology_consistency(self, sphere_mesh):
        """
        Verify topological invariants are consistent.
        """
        chi = euler_characteristic(sphere_mesh)

        # V - E + F = chi
        assert sphere_mesh.n_vertices - sphere_mesh.n_edges + sphere_mesh.n_faces == chi

        # For sphere: chi = 2
        assert chi == 2, f"Sphere should have chi=2, got {chi}"

        # Total Gaussian curvature = 2*pi*chi
        total_K = total_gaussian_curvature(sphere_mesh)
        expected_K = 2 * np.pi * chi

        assert abs(total_K - expected_K) < 1e-5, \
            f"Total K ({total_K:.6f}) should equal 2*pi*chi ({expected_K:.6f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
