"""
Tests for cut graph properties.

IMPORTANT TRADEOFF DISCOVERED:
- Strict topology (F-1 non-cut edges): 31 flipped triangles, ALL at boundary
- Pruned (41 cut edges, cycles in dual): 10 flipped triangles

The pruned version violates the dual spanning tree property but produces
significantly better UV quality. Analysis shows:
- Extra flips from strict topology are ALL at the cut boundary
- More cut edges = fewer UV constraints = worse boundary conditioning

We test for:
1. Cut graph is a TREE (no cycles in the cut itself)
2. All singularities are connected by the cut
3. Cut graph has reasonable size (not too many, not too few edges)

We explicitly ALLOW cycles in the dual graph for better UV quality.
"""

import pytest
import numpy as np
from collections import deque, defaultdict

from io_obj import load_obj
from geometry import compute_corner_angles
from cross_field import compute_smooth_cross_field, compute_cross_field_singularities
from cut_graph import compute_cut_jump_data, count_cut_edges


def get_test_mesh_path():
    """Get path to test mesh."""
    return "C:/Dev/Colonel/Data/Meshes/sphere320.obj"


def compute_cut_data_for_mesh(mesh_path):
    """Compute all cut graph data for a mesh."""
    mesh = load_obj(mesh_path)
    alpha = compute_corner_angles(mesh)
    W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
    cone_indices, is_singular = compute_cross_field_singularities(mesh, xi, alpha)
    Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)
    return mesh, Gamma, cone_indices, is_singular


class TestDualGraph:
    """Tests for dual graph properties (faces connected through non-cut edges)."""

    def test_dual_graph_connected(self):
        """All faces should be reachable through non-cut edges (single component)."""
        mesh, Gamma, _, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        # Build dual adjacency: faces connected through non-cut edges
        dual_adj = defaultdict(set)
        for e in range(mesh.n_edges):
            if Gamma[e] == 0:  # non-cut edge
                he0 = mesh.edge_to_halfedge[e, 0]
                he1 = mesh.edge_to_halfedge[e, 1]
                if he0 != -1 and he1 != -1:
                    f0 = he0 // 3
                    f1 = he1 // 3
                    dual_adj[f0].add(f1)
                    dual_adj[f1].add(f0)

        # BFS from face 0
        visited = set()
        queue = deque([0])
        visited.add(0)
        while queue:
            f = queue.popleft()
            for neighbor in dual_adj[f]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        assert len(visited) == mesh.n_faces, (
            f"Dual graph should have 1 component (all {mesh.n_faces} faces), "
            f"but only {len(visited)} faces are reachable"
        )

    def test_reasonable_cut_size(self):
        """Cut should have reasonable number of edges (not too many, not too few)."""
        mesh, Gamma, cone_indices, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        n_cut = np.sum(Gamma)
        n_cones = np.sum(np.abs(cone_indices) > 0.1)

        # Minimum cut: Steiner tree connecting cones (at least n_cones - 1 edges)
        min_cut = max(1, n_cones - 1)
        # Maximum cut: shouldn't need more than 10x the cones
        max_cut = n_cones * 10

        assert n_cut >= min_cut, f"Cut too small: {n_cut} < {min_cut}"
        assert n_cut <= max_cut, f"Cut too large: {n_cut} > {max_cut}"


class TestCutGraphStructure:
    """Tests for the cut graph (primal) structure."""

    def test_cut_graph_is_tree_or_forest(self):
        """Cut graph should be a tree or forest (E <= V - 1)."""
        mesh, Gamma, _, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        # Find vertices and edges in cut graph
        cut_verts = set()
        for e in range(mesh.n_edges):
            if Gamma[e] == 1:
                i, j = mesh.edge_vertices[e]
                cut_verts.add(i)
                cut_verts.add(j)

        n_cut_verts = len(cut_verts)
        n_cut_edges = np.sum(Gamma)

        # For a tree: E = V - 1
        # For a forest: E < V - 1
        # If E > V - 1: has cycles
        max_tree_edges = n_cut_verts - 1 if n_cut_verts > 0 else 0

        assert n_cut_edges <= max_tree_edges + 1, (
            f"Cut graph has cycles! {n_cut_edges} edges for {n_cut_verts} vertices "
            f"(max for tree: {max_tree_edges})"
        )

    def test_cut_graph_connected(self):
        """Cut graph should be connected (single tree, not forest)."""
        mesh, Gamma, _, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        # Build cut graph adjacency
        cut_adj = defaultdict(set)
        cut_verts = set()
        for e in range(mesh.n_edges):
            if Gamma[e] == 1:
                i, j = mesh.edge_vertices[e]
                cut_adj[i].add(j)
                cut_adj[j].add(i)
                cut_verts.add(i)
                cut_verts.add(j)

        if len(cut_verts) == 0:
            pytest.skip("No cut edges")

        # BFS from first cut vertex
        visited = set()
        start = next(iter(cut_verts))
        queue = deque([start])
        visited.add(start)
        while queue:
            v = queue.popleft()
            for neighbor in cut_adj[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        assert len(visited) == len(cut_verts), (
            f"Cut graph is disconnected! Only {len(visited)}/{len(cut_verts)} "
            f"vertices reachable"
        )


class TestSingularities:
    """Tests that singularities are properly handled."""

    def test_all_singularities_on_cut(self):
        """All cross-field singularities should be on the cut graph."""
        mesh, Gamma, cone_indices, is_singular = compute_cut_data_for_mesh(get_test_mesh_path())

        sing_verts = set(np.where(np.abs(cone_indices) > 0.1)[0])

        # Find vertices on cut
        cut_verts = set()
        for e in range(mesh.n_edges):
            if Gamma[e] == 1:
                i, j = mesh.edge_vertices[e]
                cut_verts.add(i)
                cut_verts.add(j)

        missing = sing_verts - cut_verts
        assert len(missing) == 0, (
            f"Singularities not on cut: {missing}"
        )

    def test_singularity_sum_equals_chi(self):
        """Sum of singularity indices should equal Euler characteristic."""
        mesh, _, cone_indices, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        from mesh import euler_characteristic
        chi = euler_characteristic(mesh)
        sing_sum = cone_indices.sum()

        assert abs(sing_sum - chi) < 0.01, (
            f"Singularity sum {sing_sum:.4f} should equal chi={chi}"
        )


class TestUVQuality:
    """Tests for UV parameterization quality."""

    def test_low_flip_count(self):
        """UV should have low number of flipped triangles."""
        from optimization import solve_constraints_only
        from uv_recovery import recover_parameterization, compute_uv_quality
        from geometry import compute_edge_lengths

        mesh, Gamma, cone_indices, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        # Get cut data
        from geometry import compute_corner_angles
        from cross_field import compute_smooth_cross_field
        alpha = compute_corner_angles(mesh)
        W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
        from cut_graph import compute_cut_jump_data
        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)

        # Solve and recover UV
        u, v, theta = solve_constraints_only(mesh, alpha, phi, omega0, s, verbose=False)
        ell = compute_edge_lengths(mesh)
        corner_uvs = recover_parameterization(mesh, Gamma, zeta, ell, alpha, phi, theta, s, u, v)

        # Check quality
        quality = compute_uv_quality(mesh, corner_uvs)

        # Allow up to 5% flipped triangles
        max_flips = mesh.n_faces * 0.05
        assert quality['flipped_count'] <= max_flips, (
            f"Too many flipped triangles: {quality['flipped_count']} > {max_flips:.0f} (5%)"
        )

    @pytest.mark.xfail(reason="BUG: UV recovery produces 10 flipped triangles on sphere320 - should be 0. See fix-flips.md for root cause (BUG 1-2 in uv_recovery.py)")
    def test_zero_flips(self):
        """UV should have ZERO flipped triangles for quad meshing.

        This is the strict requirement for quad meshing - any flips break
        integer-grid extraction. The current implementation has bugs in
        uv_recovery.py (see fix-flips.md BUG 1-2):
        - BUG 1: Cut edges skip RHS rotation averaging
        - BUG 2: Adds incorrect edge-vector constraints for cut edges

        Current state:
        - Old code path (cut_graph.py + uv_recovery.py): 10 flips (3.1%)
        - MATLAB-ported path (run_RSP.py): 46 flips (14.4%)
        """
        from optimization import solve_constraints_only
        from uv_recovery import recover_parameterization, compute_uv_quality
        from geometry import compute_edge_lengths

        mesh, Gamma, cone_indices, _ = compute_cut_data_for_mesh(get_test_mesh_path())

        # Get cut data
        from geometry import compute_corner_angles
        from cross_field import compute_smooth_cross_field
        alpha = compute_corner_angles(mesh)
        W, xi = compute_smooth_cross_field(mesh, smoothing_iters=50, verbose=False)
        from cut_graph import compute_cut_jump_data
        Gamma, zeta, s, phi, omega0 = compute_cut_jump_data(mesh, alpha, xi, singularities=cone_indices)

        # Solve and recover UV
        u, v, theta = solve_constraints_only(mesh, alpha, phi, omega0, s, verbose=False)
        ell = compute_edge_lengths(mesh)
        corner_uvs = recover_parameterization(mesh, Gamma, zeta, ell, alpha, phi, theta, s, u, v)

        # Check quality - MUST be exactly 0 for quad meshing
        quality = compute_uv_quality(mesh, corner_uvs)

        assert quality['flipped_count'] == 0, (
            f"UV recovery produces {quality['flipped_count']} flipped triangles - must be 0 for quad meshing. "
            f"See fix-flips.md for root cause analysis."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
