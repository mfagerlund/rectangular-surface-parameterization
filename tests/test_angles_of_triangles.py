"""
Pytest tests for preprocessing/angles.py

Tests the angles_of_triangles function which computes corner angles
for triangles given vertex positions and triangle indices.

Run with: pytest tests/test_angles_of_triangles.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory and Preprocess to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Preprocess"))

from rectangular_surface_parameterization.preprocessing.angles_of_triangles import angles_of_triangles


# =============================================================================
# Test Fixtures - Known Triangles
# =============================================================================

@pytest.fixture
def equilateral_triangle():
    """Single equilateral triangle with side length 1."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return V, T


@pytest.fixture
def isosceles_right_triangle():
    """Isosceles right triangle: angles pi/2, pi/4, pi/4."""
    # Right angle at origin, legs along x and y axes
    V = np.array([
        [0.0, 0.0, 0.0],  # right angle here
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return V, T


@pytest.fixture
def right_triangle_345():
    """3-4-5 right triangle: angles pi/2, arctan(3/4), arctan(4/3)."""
    V = np.array([
        [0.0, 0.0, 0.0],  # right angle here
        [3.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return V, T


@pytest.fixture
def known_30_60_90_triangle():
    """30-60-90 triangle: angles pi/6, pi/3, pi/2."""
    # For a 30-60-90 triangle with hypotenuse 2:
    # - short leg = 1 (opposite 30 deg)
    # - long leg = sqrt(3) (opposite 60 deg)
    # 90 degree angle at origin
    V = np.array([
        [0.0, 0.0, 0.0],          # 90 deg angle
        [1.0, 0.0, 0.0],          # 60 deg angle
        [0.0, np.sqrt(3), 0.0],   # 30 deg angle
    ], dtype=np.float64)
    T = np.array([[0, 1, 2]], dtype=np.int32)
    return V, T


@pytest.fixture
def multiple_triangles():
    """Two triangles forming a unit square."""
    V = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2],  # lower-right triangle
        [0, 2, 3],  # upper-left triangle
    ], dtype=np.int32)
    return V, T


# =============================================================================
# Basic Angle Tests
# =============================================================================

class TestEquilateralTriangle:
    """Test equilateral triangle - all angles should be pi/3."""

    def test_all_angles_equal_pi_over_3(self, equilateral_triangle):
        """All three angles should be pi/3 (60 degrees)."""
        V, T = equilateral_triangle
        A = angles_of_triangles(V, T)

        expected = np.pi / 3
        assert A.shape == (1, 3), f"Expected shape (1, 3), got {A.shape}"

        for j in range(3):
            assert np.allclose(A[0, j], expected, atol=1e-10), \
                f"Angle {j}: expected {np.degrees(expected):.2f} deg, got {np.degrees(A[0, j]):.2f} deg"

    def test_angles_sum_to_pi(self, equilateral_triangle):
        """Sum of angles should be pi."""
        V, T = equilateral_triangle
        A = angles_of_triangles(V, T)

        angle_sum = np.sum(A[0])
        assert np.allclose(angle_sum, np.pi, atol=1e-10), \
            f"Expected angle sum pi, got {angle_sum}"


class TestIsoscelesRightTriangle:
    """Test isosceles right triangle - angles pi/2, pi/4, pi/4."""

    def test_right_angle_at_vertex_0(self, isosceles_right_triangle):
        """Vertex 0 should have angle pi/2 (90 degrees)."""
        V, T = isosceles_right_triangle
        A = angles_of_triangles(V, T)

        assert np.allclose(A[0, 0], np.pi / 2, atol=1e-10), \
            f"Expected 90 deg at vertex 0, got {np.degrees(A[0, 0]):.2f} deg"

    def test_base_angles_equal_pi_over_4(self, isosceles_right_triangle):
        """Vertices 1 and 2 should have angle pi/4 (45 degrees)."""
        V, T = isosceles_right_triangle
        A = angles_of_triangles(V, T)

        expected = np.pi / 4
        assert np.allclose(A[0, 1], expected, atol=1e-10), \
            f"Expected 45 deg at vertex 1, got {np.degrees(A[0, 1]):.2f} deg"
        assert np.allclose(A[0, 2], expected, atol=1e-10), \
            f"Expected 45 deg at vertex 2, got {np.degrees(A[0, 2]):.2f} deg"

    def test_angles_sum_to_pi(self, isosceles_right_triangle):
        """Sum of angles should be pi."""
        V, T = isosceles_right_triangle
        A = angles_of_triangles(V, T)

        angle_sum = np.sum(A[0])
        assert np.allclose(angle_sum, np.pi, atol=1e-10), \
            f"Expected angle sum pi, got {angle_sum}"


class TestRightTriangle345:
    """Test 3-4-5 right triangle with known angles."""

    def test_right_angle_at_vertex_0(self, right_triangle_345):
        """Vertex 0 should have angle pi/2 (90 degrees)."""
        V, T = right_triangle_345
        A = angles_of_triangles(V, T)

        assert np.allclose(A[0, 0], np.pi / 2, atol=1e-10), \
            f"Expected 90 deg at vertex 0, got {np.degrees(A[0, 0]):.2f} deg"

    def test_angle_at_vertex_1(self, right_triangle_345):
        """Vertex 1 should have angle arctan(4/3) ~ 53.13 degrees."""
        V, T = right_triangle_345
        A = angles_of_triangles(V, T)

        expected = np.arctan(4 / 3)
        assert np.allclose(A[0, 1], expected, atol=1e-10), \
            f"Expected {np.degrees(expected):.2f} deg at vertex 1, got {np.degrees(A[0, 1]):.2f} deg"

    def test_angle_at_vertex_2(self, right_triangle_345):
        """Vertex 2 should have angle arctan(3/4) ~ 36.87 degrees."""
        V, T = right_triangle_345
        A = angles_of_triangles(V, T)

        expected = np.arctan(3 / 4)
        assert np.allclose(A[0, 2], expected, atol=1e-10), \
            f"Expected {np.degrees(expected):.2f} deg at vertex 2, got {np.degrees(A[0, 2]):.2f} deg"

    def test_angles_sum_to_pi(self, right_triangle_345):
        """Sum of angles should be pi."""
        V, T = right_triangle_345
        A = angles_of_triangles(V, T)

        angle_sum = np.sum(A[0])
        assert np.allclose(angle_sum, np.pi, atol=1e-10), \
            f"Expected angle sum pi, got {angle_sum}"


class TestKnown30_60_90Triangle:
    """Test 30-60-90 triangle with well-known angles."""

    def test_90_degree_angle(self, known_30_60_90_triangle):
        """Vertex 0 should have 90 degree angle."""
        V, T = known_30_60_90_triangle
        A = angles_of_triangles(V, T)

        assert np.allclose(A[0, 0], np.pi / 2, atol=1e-10), \
            f"Expected 90 deg, got {np.degrees(A[0, 0]):.2f} deg"

    def test_60_degree_angle(self, known_30_60_90_triangle):
        """Vertex 1 should have 60 degree angle."""
        V, T = known_30_60_90_triangle
        A = angles_of_triangles(V, T)

        assert np.allclose(A[0, 1], np.pi / 3, atol=1e-10), \
            f"Expected 60 deg, got {np.degrees(A[0, 1]):.2f} deg"

    def test_30_degree_angle(self, known_30_60_90_triangle):
        """Vertex 2 should have 30 degree angle."""
        V, T = known_30_60_90_triangle
        A = angles_of_triangles(V, T)

        assert np.allclose(A[0, 2], np.pi / 6, atol=1e-10), \
            f"Expected 30 deg, got {np.degrees(A[0, 2]):.2f} deg"

    def test_angles_sum_to_pi(self, known_30_60_90_triangle):
        """Sum of angles should be pi."""
        V, T = known_30_60_90_triangle
        A = angles_of_triangles(V, T)

        angle_sum = np.sum(A[0])
        assert np.allclose(angle_sum, np.pi, atol=1e-10), \
            f"Expected angle sum pi, got {angle_sum}"


# =============================================================================
# Multiple Triangles Tests
# =============================================================================

class TestMultipleTriangles:
    """Test with multiple triangles."""

    def test_output_shape(self, multiple_triangles):
        """Output should have shape (n_triangles, 3)."""
        V, T = multiple_triangles
        A = angles_of_triangles(V, T)

        assert A.shape == (2, 3), f"Expected shape (2, 3), got {A.shape}"

    def test_each_triangle_sums_to_pi(self, multiple_triangles):
        """Each triangle's angles should sum to pi."""
        V, T = multiple_triangles
        A = angles_of_triangles(V, T)

        for i in range(A.shape[0]):
            angle_sum = np.sum(A[i])
            assert np.allclose(angle_sum, np.pi, atol=1e-10), \
                f"Triangle {i}: expected angle sum pi, got {angle_sum}"

    def test_right_triangles_have_right_angle(self, multiple_triangles):
        """Both triangles in the square should have a 90 degree angle."""
        V, T = multiple_triangles
        A = angles_of_triangles(V, T)

        for i in range(A.shape[0]):
            has_right_angle = np.any(np.isclose(A[i], np.pi / 2, atol=1e-10))
            assert has_right_angle, \
                f"Triangle {i} should have a 90 degree angle: {np.degrees(A[i])}"


class TestAllAnglesSumToPi:
    """Test that angles sum to pi for all fixture triangles."""

    def test_equilateral(self, equilateral_triangle):
        V, T = equilateral_triangle
        A = angles_of_triangles(V, T)
        assert np.allclose(np.sum(A, axis=1), np.pi, atol=1e-10)

    def test_isosceles_right(self, isosceles_right_triangle):
        V, T = isosceles_right_triangle
        A = angles_of_triangles(V, T)
        assert np.allclose(np.sum(A, axis=1), np.pi, atol=1e-10)

    def test_345_right(self, right_triangle_345):
        V, T = right_triangle_345
        A = angles_of_triangles(V, T)
        assert np.allclose(np.sum(A, axis=1), np.pi, atol=1e-10)

    def test_30_60_90(self, known_30_60_90_triangle):
        V, T = known_30_60_90_triangle
        A = angles_of_triangles(V, T)
        assert np.allclose(np.sum(A, axis=1), np.pi, atol=1e-10)

    def test_multiple(self, multiple_triangles):
        V, T = multiple_triangles
        A = angles_of_triangles(V, T)
        assert np.allclose(np.sum(A, axis=1), np.pi, atol=1e-10)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and numerical robustness."""

    def test_very_small_triangle(self):
        """Very small triangle should still compute correctly."""
        scale = 1e-8
        V = np.array([
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.5 * scale, np.sqrt(3)/2 * scale, 0.0],
        ], dtype=np.float64)
        T = np.array([[0, 1, 2]], dtype=np.int32)

        A = angles_of_triangles(V, T)

        # Should still be equilateral
        expected = np.pi / 3
        assert np.allclose(A[0], expected, atol=1e-8), \
            f"Small triangle angles: {np.degrees(A[0])}"

    def test_very_large_triangle(self):
        """Very large triangle should still compute correctly."""
        scale = 1e8
        V = np.array([
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.5 * scale, np.sqrt(3)/2 * scale, 0.0],
        ], dtype=np.float64)
        T = np.array([[0, 1, 2]], dtype=np.int32)

        A = angles_of_triangles(V, T)

        # Should still be equilateral
        expected = np.pi / 3
        assert np.allclose(A[0], expected, atol=1e-8), \
            f"Large triangle angles: {np.degrees(A[0])}"

    def test_3d_triangle_not_in_xy_plane(self):
        """Triangle not in XY plane should compute correctly."""
        # Equilateral triangle tilted in 3D
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.5, np.sqrt(3)/2, 0.5],
        ], dtype=np.float64)
        T = np.array([[0, 1, 2]], dtype=np.int32)

        A = angles_of_triangles(V, T)

        # Angles should still sum to pi
        angle_sum = np.sum(A[0])
        assert np.allclose(angle_sum, np.pi, atol=1e-10), \
            f"3D triangle angle sum: {angle_sum}"

    def test_all_angles_positive(self, equilateral_triangle, isosceles_right_triangle,
                                  right_triangle_345, multiple_triangles):
        """All computed angles should be positive."""
        for V, T in [equilateral_triangle, isosceles_right_triangle,
                     right_triangle_345, multiple_triangles]:
            A = angles_of_triangles(V, T)
            assert np.all(A > 0), f"All angles should be positive: {A}"

    def test_all_angles_less_than_pi(self, equilateral_triangle, isosceles_right_triangle,
                                      right_triangle_345, multiple_triangles):
        """All angles should be less than pi (interior angles)."""
        for V, T in [equilateral_triangle, isosceles_right_triangle,
                     right_triangle_345, multiple_triangles]:
            A = angles_of_triangles(V, T)
            assert np.all(A < np.pi), f"All angles should be < pi: {A}"


class TestDegenerateTriangles:
    """Test behavior with degenerate (flat) triangles."""

    def test_degenerate_collinear_points(self):
        """Collinear points (flat triangle) produces degenerate angles.

        When points are collinear, one angle becomes pi (180 degrees)
        and the others become 0. The sum still equals pi.
        """
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # Collinear - all on x-axis
        ], dtype=np.float64)
        T = np.array([[0, 1, 2]], dtype=np.int32)

        A = angles_of_triangles(V, T)

        # Angles sum to pi even for degenerate case
        assert np.allclose(np.sum(A[0]), np.pi, atol=1e-10), \
            f"Degenerate triangle angles should sum to pi: {A}"

        # One angle should be pi (180 degrees), others 0
        assert np.allclose(A[0, 1], np.pi, atol=1e-10), \
            f"Middle vertex should have 180 degree angle: {np.degrees(A[0])}"
        assert np.allclose(A[0, 0], 0.0, atol=1e-10), \
            f"End vertex should have 0 degree angle: {np.degrees(A[0])}"
        assert np.allclose(A[0, 2], 0.0, atol=1e-10), \
            f"End vertex should have 0 degree angle: {np.degrees(A[0])}"

    def test_nearly_degenerate_triangle(self):
        """Nearly degenerate (very thin) triangle should still work."""
        V = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-10, 0.0],  # Very thin but not quite degenerate
        ], dtype=np.float64)
        T = np.array([[0, 1, 2]], dtype=np.int32)

        A = angles_of_triangles(V, T)

        # Should still sum to pi (within numerical precision)
        angle_sum = np.sum(A[0])
        assert np.allclose(angle_sum, np.pi, atol=1e-6), \
            f"Nearly degenerate triangle angle sum: {angle_sum}"


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Test consistency properties."""

    def test_order_independence_within_triangle(self):
        """Angles should correspond to correct vertices regardless of face winding."""
        # Same triangle, different vertex orderings
        V = np.array([
            [0.0, 0.0, 0.0],  # right angle
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ], dtype=np.float64)

        # Triangle 0-1-2
        T1 = np.array([[0, 1, 2]], dtype=np.int32)
        A1 = angles_of_triangles(V, T1)

        # A[i, j] should be the angle at vertex T[i, j]
        # So A1[0, 0] should be angle at V[0] = right angle
        assert np.allclose(A1[0, 0], np.pi / 2, atol=1e-10), \
            f"Angle at vertex 0 should be 90 deg, got {np.degrees(A1[0, 0]):.2f}"

    def test_many_triangles_performance(self):
        """Test with many triangles for performance and correctness."""
        n_tris = 1000
        # Create many equilateral triangles
        V = []
        T = []
        for i in range(n_tris):
            offset = i * 3
            V.append([i * 2.0, 0.0, 0.0])
            V.append([i * 2.0 + 1.0, 0.0, 0.0])
            V.append([i * 2.0 + 0.5, np.sqrt(3)/2, 0.0])
            T.append([offset, offset + 1, offset + 2])

        V = np.array(V, dtype=np.float64)
        T = np.array(T, dtype=np.int32)

        A = angles_of_triangles(V, T)

        assert A.shape == (n_tris, 3)

        # All angles should be pi/3 for equilateral triangles
        expected = np.pi / 3
        assert np.allclose(A, expected, atol=1e-10), \
            f"Not all angles are pi/3 for equilateral triangles"

        # All should sum to pi
        assert np.allclose(np.sum(A, axis=1), np.pi, atol=1e-10), \
            "Not all triangles sum to pi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
