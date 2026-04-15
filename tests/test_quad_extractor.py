"""Tests for the quad extractor (libQEx replacement)."""

import numpy as np
import pytest
from pathlib import Path
from collections import defaultdict

from rectangular_surface_parameterization.utils.quad_extractor import (
    extract_quads, _barycentric_2d, _find_grid_points,
    _merge_grid_points, _form_quads,
)

GOLDEN = Path(__file__).parent / "golden_data"


# =========================================================================
# Helpers
# =========================================================================

def make_single_quad_uv():
    """Two triangles forming a quad in UV space spanning [0,1]x[0,1].

    UV layout:
        (0,1)---(1,1)
          | \\    |
          |  \\   |
          |   \\  |
          |    \\ |
        (0,0)---(1,0)
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],  # v0
        [1.0, 0.0, 0.0],  # v1
        [1.0, 1.0, 0.0],  # v2
        [0.0, 1.0, 0.0],  # v3
    ])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    uv_per_tri = np.array([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])
    return vertices, triangles, uv_per_tri


def make_2x2_grid_uv():
    """A 2x2 quad grid in UV space spanning [0,2]x[0,2].

    9 vertices, 8 triangles, should produce 4 quads.
    """
    vertices = []
    for j in range(3):
        for i in range(3):
            vertices.append([float(i), float(j), 0.0])
    vertices = np.array(vertices)

    triangles = []
    uv_per_tri = []
    for j in range(2):
        for i in range(2):
            v00 = j * 3 + i
            v10 = j * 3 + i + 1
            v01 = (j + 1) * 3 + i
            v11 = (j + 1) * 3 + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j)],
                               [float(i + 1), float(j + 1)]])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j + 1)],
                               [float(i), float(j + 1)]])

    return vertices, np.array(triangles), np.array(uv_per_tri)


def make_scaled_grid_uv(n=3, scale=1.0):
    """An nxn quad grid with UVs scaled by `scale`.

    Returns flat mesh at y=0 in 3D, with UV = 3D (x,z) * scale.
    """
    vertices = []
    for j in range(n + 1):
        for i in range(n + 1):
            vertices.append([float(i), 0.0, float(j)])
    vertices = np.array(vertices)

    triangles = []
    uv_per_tri = []
    for j in range(n):
        for i in range(n):
            v00 = j * (n + 1) + i
            v10 = j * (n + 1) + i + 1
            v01 = (j + 1) * (n + 1) + i
            v11 = (j + 1) * (n + 1) + i + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[i * scale, j * scale],
                               [(i + 1) * scale, j * scale],
                               [(i + 1) * scale, (j + 1) * scale]])
            uv_per_tri.append([[i * scale, j * scale],
                               [(i + 1) * scale, (j + 1) * scale],
                               [i * scale, (j + 1) * scale]])

    return vertices, np.array(triangles), np.array(uv_per_tri)


def make_cylinder_with_cut():
    """A cylinder (open tube) with a UV cut along one seam.

    The mesh is a 4-column cylinder. In 3D, column 0 and column 4 are the
    same vertices. In UV space, they have different U coordinates (0 and 4),
    simulating a UV cut.
    """
    n_cols = 4
    n_rows = 3
    # 3D: cylinder with radius 1, height 2
    angles = np.linspace(0, 2 * np.pi, n_cols + 1)  # 5 angles, first==last
    heights = np.linspace(0, 2, n_rows + 1)

    vertices_3d = []
    for j in range(n_rows + 1):
        for i in range(n_cols + 1):
            # Column n_cols wraps to column 0 in 3D
            a = angles[i % n_cols] if i < n_cols else angles[0]
            vertices_3d.append([np.cos(a), heights[j], np.sin(a)])
    vertices_3d = np.array(vertices_3d)

    triangles = []
    uv_per_tri = []
    for j in range(n_rows):
        for i in range(n_cols):
            stride = n_cols + 1
            v00 = j * stride + i
            v10 = j * stride + i + 1
            v01 = (j + 1) * stride + i
            v11 = (j + 1) * stride + i + 1

            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j)],
                               [float(i + 1), float(j + 1)]])
            uv_per_tri.append([[float(i), float(j)],
                               [float(i + 1), float(j + 1)],
                               [float(i), float(j + 1)]])

    return vertices_3d, np.array(triangles), np.array(uv_per_tri)


def check_manifold(quads):
    """Return (n_boundary, n_non_manifold) edge counts."""
    edge_count = defaultdict(int)
    for q in quads:
        for i in range(4):
            e = (min(q[i], q[(i + 1) % 4]), max(q[i], q[(i + 1) % 4]))
            edge_count[e] += 1
    boundary = sum(1 for c in edge_count.values() if c == 1)
    non_manifold = sum(1 for c in edge_count.values() if c > 2)
    return boundary, non_manifold


def check_inverted_faces(vertices, quads):
    """Count faces whose normal points toward the origin (for sphere-like meshes)."""
    inverted = 0
    for q in quads:
        center = vertices[q].mean(axis=0)
        e1 = vertices[q[1]] - vertices[q[0]]
        e2 = vertices[q[3]] - vertices[q[0]]
        normal = np.cross(e1, e2)
        if np.dot(normal, center) < 0:
            inverted += 1
    return inverted


# =========================================================================
# Barycentric coordinate tests
# =========================================================================

class TestBarycentric:
    def test_center_of_triangle(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])
        p = np.array([1 / 3, 1 / 3])
        l1, l2, l3 = _barycentric_2d(p, a, b, c)
        assert abs(l1 - 1 / 3) < 1e-10
        assert abs(l2 - 1 / 3) < 1e-10
        assert abs(l3 - 1 / 3) < 1e-10

    def test_at_vertex(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])
        l1, l2, l3 = _barycentric_2d(a, a, b, c)
        assert abs(l1 - 1.0) < 1e-10
        assert abs(l2) < 1e-10
        assert abs(l3) < 1e-10

    def test_outside_triangle(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([0.0, 1.0])
        p = np.array([1.0, 1.0])  # outside
        l1, l2, l3 = _barycentric_2d(p, a, b, c)
        assert l1 < 0  # Should have a negative coordinate

    def test_on_edge(self):
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 0.0])
        c = np.array([0.0, 2.0])
        p = np.array([1.0, 0.0])  # midpoint of a-b edge
        l1, l2, l3 = _barycentric_2d(p, a, b, c)
        assert l1 >= -1e-10 and l2 >= -1e-10 and l3 >= -1e-10


# =========================================================================
# Grid point finding tests
# =========================================================================

class TestFindGridPoints:
    def test_single_quad(self):
        V, T, uv = make_single_quad_uv()
        # UV spans [0,1]x[0,1], so integer points are the 4 corners
        gp = _find_grid_points(V, T, uv)
        assert (0, 0) in gp
        assert (1, 0) in gp
        assert (1, 1) in gp
        assert (0, 1) in gp
        assert len(gp) == 4

    def test_2x2_grid(self):
        V, T, uv = make_2x2_grid_uv()
        gp = _find_grid_points(V, T, uv)
        # UV spans [0,2]x[0,2], so 3x3 = 9 integer grid points
        assert len(gp) == 9
        for i in range(3):
            for j in range(3):
                assert (i, j) in gp

    def test_scaled_grid(self):
        V, T, uv = make_scaled_grid_uv(n=2, scale=0.5)
        gp = _find_grid_points(V, T, uv)
        # UV spans [0,1]x[0,1], so only corners: (0,0), (1,0), (0,1), (1,1)
        assert len(gp) == 4

    def test_3d_positions_via_interpolation(self):
        V, T, uv = make_2x2_grid_uv()
        gp = _find_grid_points(V, T, uv)
        # 3D positions should match UV positions (flat mesh at z=0)
        for (iu, iv), pos in gp.items():
            np.testing.assert_allclose(pos, [iu, iv, 0], atol=1e-10)

    def test_no_points_for_tiny_uv(self):
        V, T, uv = make_scaled_grid_uv(n=2, scale=0.3)
        gp = _find_grid_points(V, T, uv)
        # UV spans [0,0.6]x[0,0.6], only (0,0) is inside
        assert len(gp) == 1
        assert (0, 0) in gp


# =========================================================================
# Quad formation tests
# =========================================================================

class TestFormQuads:
    def test_single_quad(self):
        V, T, uv = make_single_quad_uv()
        gp = _find_grid_points(V, T, uv)
        merged, uv_to_m = _merge_grid_points(gp)
        quads = _form_quads(gp, uv_to_m)
        assert len(quads) == 1

    def test_2x2_grid(self):
        V, T, uv = make_2x2_grid_uv()
        gp = _find_grid_points(V, T, uv)
        merged, uv_to_m = _merge_grid_points(gp)
        quads = _form_quads(gp, uv_to_m)
        assert len(quads) == 4

    def test_no_quads_from_single_point(self):
        gp = {(0, 0): np.array([0.0, 0.0, 0.0])}
        merged, uv_to_m = _merge_grid_points(gp)
        quads = _form_quads(gp, uv_to_m)
        assert len(quads) == 0

    def test_three_corners_no_quad(self):
        gp = {
            (0, 0): np.array([0.0, 0.0, 0.0]),
            (1, 0): np.array([1.0, 0.0, 0.0]),
            (0, 1): np.array([0.0, 1.0, 0.0]),
        }
        merged, uv_to_m = _merge_grid_points(gp)
        quads = _form_quads(gp, uv_to_m)
        assert len(quads) == 0


# =========================================================================
# Merge tests
# =========================================================================

class TestMerge:
    def test_no_duplicates(self):
        gp = {
            (0, 0): np.array([0.0, 0.0, 0.0]),
            (1, 0): np.array([1.0, 0.0, 0.0]),
            (0, 1): np.array([0.0, 1.0, 0.0]),
        }
        merged, uv_to_m = _merge_grid_points(gp)
        assert len(merged) == 3

    def test_duplicates_merged(self):
        gp = {
            (0, 0): np.array([0.0, 0.0, 0.0]),
            (1, 0): np.array([1.0, 0.0, 0.0]),
            (5, 5): np.array([0.0, 0.0, 0.0]),  # same 3D as (0,0)
        }
        merged, uv_to_m = _merge_grid_points(gp)
        assert len(merged) == 2
        assert uv_to_m[(0, 0)] == uv_to_m[(5, 5)]

    def test_cylinder_cut_merge(self):
        V, T, uv = make_cylinder_with_cut()
        gp = _find_grid_points(V, T, uv)
        merged, uv_to_m = _merge_grid_points(gp)

        # Column 0 and column 4 share the same 3D positions
        # So (0,j) and (4,j) should merge
        n_before = len(gp)
        n_after = len(merged)
        assert n_after < n_before, "Cut vertices should be merged"

        # Check that (0,j) and (4,j) map to the same merged index
        for j in range(4):
            if (0, j) in uv_to_m and (4, j) in uv_to_m:
                assert uv_to_m[(0, j)] == uv_to_m[(4, j)], \
                    f"(0,{j}) and (4,{j}) should merge"


# =========================================================================
# Full extraction tests: synthetic
# =========================================================================

class TestExtractSynthetic:
    def test_single_quad(self):
        V, T, uv = make_single_quad_uv()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        assert len(qf) == 1
        assert len(set(qf[0])) == 4

    def test_2x2_grid_produces_4_quads(self):
        V, T, uv = make_2x2_grid_uv()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        assert len(qf) == 4
        boundary, non_manifold = check_manifold(qf)
        assert non_manifold == 0
        assert boundary == 8  # 2x2 grid has 8 boundary edges

    def test_larger_grid(self):
        V, T, uv = make_scaled_grid_uv(n=5, scale=1.0)
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        assert len(qf) == 25  # 5x5 quads
        boundary, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_cylinder_forms_quads_across_cut(self):
        V, T, uv = make_cylinder_with_cut()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        # 4 columns x 3 rows = 12 quads within the chart
        # Quads across the cut (column 0-4) need merge to work
        assert len(qf) >= 12
        boundary, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_no_degenerate_quads(self):
        V, T, uv = make_scaled_grid_uv(n=4, scale=1.0)
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            assert len(set(q)) == 4, f"Quad {i} has duplicate vertices: {q}"

    def test_vertex_positions_correct(self):
        V, T, uv = make_2x2_grid_uv()
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        # All vertices should be at integer positions on z=0 plane
        for i, v in enumerate(qv):
            assert abs(v[2]) < 1e-10, f"Vertex {i} z={v[2]}"
            assert abs(v[0] - round(v[0])) < 1e-10, f"Vertex {i} x={v[0]}"
            assert abs(v[1] - round(v[1])) < 1e-10, f"Vertex {i} y={v[1]}"

    def test_empty_uv_returns_empty(self):
        V = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=float)
        T = np.array([[0, 1, 2]])
        uv = np.array([[[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]])
        qv, qf, tf = extract_quads(V, T, uv, verbose=False, fill_holes=False)
        assert len(qf) == 0


# =========================================================================
# Full extraction tests: sphere320
# =========================================================================

class TestExtractSphere:
    @pytest.fixture
    def sphere_data(self):
        path = GOLDEN / "sphere320_param.npz"
        if not path.exists():
            pytest.skip("sphere320_param.npz not found")
        data = np.load(path)
        return data["vertices"], data["triangles"], data["uv_per_tri"]

    def test_produces_quads(self, sphere_data):
        V, T, uv = sphere_data
        uv_scaled = uv * 10.0
        qv, qf, tf = extract_quads(V, T, uv_scaled, verbose=False, fill_holes=False)
        assert len(qf) > 50, f"Expected >50 quads, got {len(qf)}"

    def test_no_non_manifold_edges(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        _, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_no_inverted_faces(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        inverted = check_inverted_faces(qv, qf)
        assert inverted == 0, f"{inverted} inverted faces"

    def test_no_degenerate_quads(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            assert len(set(q)) == 4, f"Quad {i} has duplicate vertices"

    def test_vertices_near_sphere(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        radii = np.linalg.norm(qv, axis=1)
        assert radii.min() > 0.9, f"Min radius {radii.min():.4f} too small"
        assert radii.max() < 1.1, f"Max radius {radii.max():.4f} too large"

    def test_no_excessively_long_edges(self, sphere_data):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            for j in range(4):
                d = np.linalg.norm(qv[q[j]] - qv[q[(j + 1) % 4]])
                assert d < 0.6, f"Quad {i} edge {j} length {d:.4f} too long"

    @pytest.mark.parametrize("scale", [5, 10, 15, 20])
    def test_more_quads_at_higher_scale(self, sphere_data, scale):
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(V, T, uv * scale, verbose=False, fill_holes=False)
        # Higher scale should produce more quads (more integer lines cross the mesh)
        assert len(qf) > 0

    def test_scale_10_vs_5_has_more_quads(self, sphere_data):
        V, T, uv = sphere_data
        _, qf5, _ = extract_quads(V, T, uv * 5, verbose=False, fill_holes=False)
        _, qf10, _ = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        assert len(qf10) > len(qf5)

    def test_hole_filling_small_holes_only(self, sphere_data):
        """Fill only small holes (singularity artifacts), not the cut boundary."""
        V, T, uv = sphere_data
        qv, qf, tf = extract_quads(
            V, T, uv * 10, verbose=False, fill_holes=True, max_hole_size=8)
        if tf is not None:
            for t in tf:
                for vi in t:
                    if vi < len(qv):
                        r = np.linalg.norm(qv[vi])
                        assert r > 0.8, f"Hole-fill vertex {vi} at r={r:.4f}"

    def test_large_hole_not_filled(self, sphere_data):
        """The cut boundary hole should not be filled (it's not a defect)."""
        V, T, uv = sphere_data
        _, qf_no_fill, _ = extract_quads(
            V, T, uv * 10, verbose=False, fill_holes=False)
        boundary_no_fill, _ = check_manifold(qf_no_fill)
        # The cut creates a large boundary; filling it would create bad geometry
        assert boundary_no_fill > 20, "Cut should create significant boundary"


# =========================================================================
# Full extraction tests: torus
# =========================================================================

class TestExtractTorus:
    @pytest.fixture
    def torus_data(self):
        path = GOLDEN / "torus_param.npz"
        if not path.exists():
            pytest.skip("torus_param.npz not found")
        data = np.load(path)
        return data["vertices"], data["triangles"], data["uv_per_tri"]

    def test_produces_quads(self, torus_data):
        V, T, uv = torus_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        assert len(qf) > 20

    def test_no_non_manifold(self, torus_data):
        V, T, uv = torus_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        _, non_manifold = check_manifold(qf)
        assert non_manifold == 0

    def test_no_degenerate(self, torus_data):
        V, T, uv = torus_data
        qv, qf, tf = extract_quads(V, T, uv * 10, verbose=False, fill_holes=False)
        for i, q in enumerate(qf):
            assert len(set(q)) == 4, f"Quad {i} degenerate"


# =========================================================================
# Edge cases and robustness
# =========================================================================

class TestRobustness:
    def test_grid_point_on_triangle_edge(self):
        """Integer point exactly on a shared edge should be found once."""
        V, T, uv = make_2x2_grid_uv()
        gp = _find_grid_points(V, T, uv)
        # (1,0) is on the edge between two triangles
        assert (1, 0) in gp
        # Should only appear once in the dict
        assert len(gp) == 9

    def test_grid_point_at_triangle_vertex(self):
        """Integer point at a triangle vertex should be found."""
        V, T, uv = make_single_quad_uv()
        gp = _find_grid_points(V, T, uv)
        assert (0, 0) in gp

    def test_different_merge_tolerances(self):
        """Tighter tolerance should merge fewer vertices."""
        V, T, uv = make_cylinder_with_cut()
        gp = _find_grid_points(V, T, uv)
        merged_tight, _ = _merge_grid_points(gp, tolerance=1e-10)
        merged_loose, _ = _merge_grid_points(gp, tolerance=0.1)
        assert len(merged_tight) >= len(merged_loose)

    def test_api_compatibility(self):
        """extract_quads should accept the same args as libqex_wrapper."""
        V, T, uv = make_2x2_grid_uv()
        # Should not raise with all optional args
        qv, qf, tf = extract_quads(
            V, T, uv,
            vertex_valences=None,
            fill_holes=True,
            max_hole_size=6,
            verbose=False,
            merge_tolerance=1e-6,
        )
        assert len(qf) == 4
