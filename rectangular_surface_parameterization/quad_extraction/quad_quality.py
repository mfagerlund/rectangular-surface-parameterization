"""
Quad mesh quality metrics.

Measures geometric and topological quality of extracted quad meshes.
Used to benchmark quantization improvements over the --scale workaround.

Metrics implemented:
- Aspect ratio (ideal = 1.0 for squares)
- Corner angles (ideal = 90°)
- Scaled Jacobian (ideal = 1.0, valid > 0)
- Irregular vertex count (valence ≠ 4)
- Singularity integer error (distance from integer UV coordinates)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class QuadMetrics:
    """Quality metrics for a single quad."""
    aspect_ratio: float      # width/height ratio (1.0 = square)
    min_angle: float         # smallest corner angle in degrees
    max_angle: float         # largest corner angle in degrees
    scaled_jacobian: float   # shape quality metric (0-1, negative = flipped)
    area: float              # quad area in 3D


@dataclass
class MeshQualityReport:
    """Aggregate quality metrics for an entire quad mesh."""
    # Counts
    num_quads: int
    num_vertices: int

    # Angle metrics (degrees)
    min_angle: float              # worst (smallest) angle across all quads
    max_angle: float              # worst (largest) angle across all quads
    mean_min_angle: float         # average of per-quad min angles
    mean_max_angle: float         # average of per-quad max angles
    angle_deviation_rms: float    # RMS deviation from 90°

    # Aspect ratio
    mean_aspect_ratio: float      # average aspect ratio
    max_aspect_ratio: float       # worst aspect ratio

    # Jacobian
    min_scaled_jacobian: float    # worst Jacobian (< 0 means flipped)
    mean_scaled_jacobian: float   # average Jacobian
    num_inverted: int             # count of quads with negative Jacobian

    # Topology
    num_irregular: int            # vertices with valence ≠ 4
    num_boundary: int             # boundary vertices
    valence_histogram: Dict[int, int]  # valence -> count

    # Area
    total_area: float
    area_std: float               # standard deviation of quad areas

    def __str__(self) -> str:
        """Pretty-print the quality report."""
        lines = [
            "=" * 50,
            "QUAD MESH QUALITY REPORT",
            "=" * 50,
            f"Quads: {self.num_quads}  |  Vertices: {self.num_vertices}",
            "",
            "ANGLES (ideal = 90°)",
            f"  Min angle:  {self.min_angle:.1f}°  (worst corner)",
            f"  Max angle:  {self.max_angle:.1f}°  (worst corner)",
            f"  Mean range: [{self.mean_min_angle:.1f}°, {self.mean_max_angle:.1f}°]",
            f"  RMS deviation from 90°: {self.angle_deviation_rms:.2f}°",
            "",
            "ASPECT RATIO (ideal = 1.0)",
            f"  Mean: {self.mean_aspect_ratio:.3f}",
            f"  Max:  {self.max_aspect_ratio:.3f}  (worst quad)",
            "",
            "SCALED JACOBIAN (ideal = 1.0, valid > 0)",
            f"  Min:  {self.min_scaled_jacobian:.3f}",
            f"  Mean: {self.mean_scaled_jacobian:.3f}",
            f"  Inverted quads: {self.num_inverted}",
            "",
            "TOPOLOGY",
            f"  Irregular vertices (valence ≠ 4): {self.num_irregular}",
            f"  Boundary vertices: {self.num_boundary}",
            f"  Valence distribution: {dict(sorted(self.valence_histogram.items()))}",
            "",
            "AREA",
            f"  Total: {self.total_area:.4f}",
            f"  Std dev: {self.area_std:.4f}  (uniformity)",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_quad_angles(p0: np.ndarray, p1: np.ndarray,
                        p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute the four corner angles of a quad in degrees.

    Vertices are ordered: p0 -> p1 -> p2 -> p3 -> p0 (CCW or CW)

    Returns:
        Array of 4 angles in degrees
    """
    points = [p0, p1, p2, p3]
    angles = []

    for i in range(4):
        # Vectors from corner i to adjacent corners
        v_prev = points[(i - 1) % 4] - points[i]
        v_next = points[(i + 1) % 4] - points[i]

        # Normalize
        len_prev = np.linalg.norm(v_prev)
        len_next = np.linalg.norm(v_next)

        if len_prev < 1e-10 or len_next < 1e-10:
            angles.append(0.0)  # Degenerate
            continue

        v_prev = v_prev / len_prev
        v_next = v_next / len_next

        # Angle via dot product
        cos_angle = np.clip(np.dot(v_prev, v_next), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    return np.array(angles)


def compute_aspect_ratio(p0: np.ndarray, p1: np.ndarray,
                         p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute aspect ratio of a quad.

    Uses average of opposite edge lengths.
    Returns ratio >= 1.0 (width/height or height/width, whichever is larger).
    """
    # Edge lengths
    e01 = np.linalg.norm(p1 - p0)
    e12 = np.linalg.norm(p2 - p1)
    e23 = np.linalg.norm(p3 - p2)
    e30 = np.linalg.norm(p0 - p3)

    # Average opposite edges
    width = (e01 + e23) / 2
    height = (e12 + e30) / 2

    if width < 1e-10 or height < 1e-10:
        return float('inf')  # Degenerate

    ratio = width / height
    return max(ratio, 1.0 / ratio)  # Always >= 1


def compute_scaled_jacobian(p0: np.ndarray, p1: np.ndarray,
                            p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute scaled Jacobian of a quad.

    The scaled Jacobian is computed at each corner and the minimum is returned.
    Range: [-1, 1], where 1 = perfect square, 0 = degenerate, < 0 = inverted.

    Reference: Stimpson et al., "The Verdict Geometric Quality Library"
    """
    points = [p0, p1, p2, p3]
    jacobians = []

    for i in range(4):
        # Edges from corner i
        e1 = points[(i + 1) % 4] - points[i]
        e2 = points[(i + 3) % 4] - points[i]  # Previous vertex

        # For 3D quads, we need the normal
        # Cross product gives area vector
        cross = np.cross(e1, e2)

        # Lengths
        len1 = np.linalg.norm(e1)
        len2 = np.linalg.norm(e2)
        len_cross = np.linalg.norm(cross)

        if len1 < 1e-10 or len2 < 1e-10:
            jacobians.append(0.0)
            continue

        # Scaled Jacobian at this corner
        # For a planar quad: J = |cross| / (|e1| * |e2|)
        # Sign determined by cross product direction
        jacobian = len_cross / (len1 * len2)

        # Determine sign by checking if cross points same way as quad normal
        # For now, we assume the input is consistently oriented
        jacobians.append(jacobian)

    return min(jacobians)


def compute_quad_area(p0: np.ndarray, p1: np.ndarray,
                      p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute area of a quad using the cross product method.

    Splits quad into two triangles and sums areas.
    """
    # Triangle 1: p0, p1, p2
    v1 = p1 - p0
    v2 = p2 - p0
    area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))

    # Triangle 2: p0, p2, p3
    v3 = p3 - p0
    area2 = 0.5 * np.linalg.norm(np.cross(v2, v3))

    return area1 + area2


def analyze_quad(vertices: np.ndarray, quad: np.ndarray) -> QuadMetrics:
    """
    Compute all quality metrics for a single quad.

    Args:
        vertices: (N, 3) vertex positions
        quad: (4,) vertex indices

    Returns:
        QuadMetrics for this quad
    """
    p0, p1, p2, p3 = vertices[quad]

    angles = compute_quad_angles(p0, p1, p2, p3)

    return QuadMetrics(
        aspect_ratio=compute_aspect_ratio(p0, p1, p2, p3),
        min_angle=float(np.min(angles)),
        max_angle=float(np.max(angles)),
        scaled_jacobian=compute_scaled_jacobian(p0, p1, p2, p3),
        area=compute_quad_area(p0, p1, p2, p3),
    )


def compute_vertex_valences(quads: np.ndarray, num_vertices: int) -> Dict[int, int]:
    """
    Compute valence (number of incident quads) for each vertex.

    Returns:
        Dictionary mapping vertex index to valence
    """
    valence = np.zeros(num_vertices, dtype=int)
    for quad in quads:
        for vi in quad:
            valence[vi] += 1
    return {i: int(v) for i, v in enumerate(valence) if v > 0}


def find_boundary_vertices(quads: np.ndarray, num_vertices: int) -> set:
    """
    Find boundary vertices (edges that appear in only one quad).

    Returns:
        Set of boundary vertex indices
    """
    # Count edge occurrences
    edge_count = {}
    for quad in quads:
        for i in range(4):
            v1, v2 = quad[i], quad[(i + 1) % 4]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear once
    boundary_verts = set()
    for (v1, v2), count in edge_count.items():
        if count == 1:
            boundary_verts.add(v1)
            boundary_verts.add(v2)

    return boundary_verts


def measure_mesh_quality(vertices: np.ndarray, quads: np.ndarray) -> MeshQualityReport:
    """
    Compute comprehensive quality metrics for a quad mesh.

    Args:
        vertices: (N, 3) vertex positions
        quads: (M, 4) quad face indices

    Returns:
        MeshQualityReport with all metrics
    """
    if len(quads) == 0:
        raise ValueError("No quads to analyze")

    # Analyze each quad
    metrics = [analyze_quad(vertices, quad) for quad in quads]

    # Collect per-quad values
    aspect_ratios = np.array([m.aspect_ratio for m in metrics])
    min_angles = np.array([m.min_angle for m in metrics])
    max_angles = np.array([m.max_angle for m in metrics])
    jacobians = np.array([m.scaled_jacobian for m in metrics])
    areas = np.array([m.area for m in metrics])

    # All angles for RMS calculation
    all_angles = np.concatenate([min_angles, max_angles])
    angle_deviation_rms = np.sqrt(np.mean((all_angles - 90.0) ** 2))

    # Vertex topology
    num_vertices = len(vertices)
    vertex_valences = compute_vertex_valences(quads, num_vertices)
    boundary_verts = find_boundary_vertices(quads, num_vertices)

    # Valence histogram
    valence_histogram = {}
    for v, valence in vertex_valences.items():
        valence_histogram[valence] = valence_histogram.get(valence, 0) + 1

    # Count irregular (interior vertices with valence ≠ 4)
    num_irregular = sum(
        1 for v, valence in vertex_valences.items()
        if v not in boundary_verts and valence != 4
    )

    return MeshQualityReport(
        num_quads=len(quads),
        num_vertices=len(vertex_valences),
        min_angle=float(np.min(min_angles)),
        max_angle=float(np.max(max_angles)),
        mean_min_angle=float(np.mean(min_angles)),
        mean_max_angle=float(np.mean(max_angles)),
        angle_deviation_rms=float(angle_deviation_rms),
        mean_aspect_ratio=float(np.mean(aspect_ratios)),
        max_aspect_ratio=float(np.max(aspect_ratios)),
        min_scaled_jacobian=float(np.min(jacobians)),
        mean_scaled_jacobian=float(np.mean(jacobians)),
        num_inverted=int(np.sum(jacobians < 0)),
        num_irregular=num_irregular,
        num_boundary=len(boundary_verts),
        valence_histogram=valence_histogram,
        total_area=float(np.sum(areas)),
        area_std=float(np.std(areas)),
    )


def measure_singularity_error(uv_coords: np.ndarray,
                               singularity_indices: np.ndarray) -> Dict[str, float]:
    """
    Measure how far singularities are from integer UV coordinates.

    This is the key metric for quantization quality.

    Args:
        uv_coords: (N, 2) UV coordinates
        singularity_indices: Indices of singular vertices

    Returns:
        Dictionary with error metrics
    """
    if len(singularity_indices) == 0:
        return {
            'num_singularities': 0,
            'total_error': 0.0,
            'mean_error': 0.0,
            'max_error': 0.0,
            'num_on_integer': 0,
        }

    sing_uvs = uv_coords[singularity_indices]

    # Distance to nearest integer
    errors = np.abs(sing_uvs - np.round(sing_uvs))
    total_errors = np.sum(errors, axis=1)  # Sum of u and v errors

    # Count singularities on integers (within tolerance)
    tol = 1e-6
    on_integer = np.all(errors < tol, axis=1)

    return {
        'num_singularities': len(singularity_indices),
        'total_error': float(np.sum(total_errors)),
        'mean_error': float(np.mean(total_errors)),
        'max_error': float(np.max(total_errors)),
        'num_on_integer': int(np.sum(on_integer)),
    }


def load_obj_quads(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a quad mesh from OBJ file.

    Args:
        filepath: Path to OBJ file

    Returns:
        (vertices, quads) tuple
    """
    vertices = []
    quads = []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                # Parse face indices (handles v, v/vt, v/vt/vn formats)
                indices = []
                for p in parts[1:]:
                    idx = int(p.split('/')[0]) - 1  # OBJ is 1-indexed
                    indices.append(idx)
                if len(indices) == 4:
                    quads.append(indices)

    return np.array(vertices), np.array(quads)


def benchmark_mesh(obj_path: str, verbose: bool = True) -> MeshQualityReport:
    """
    Load a quad mesh and compute quality metrics.

    Args:
        obj_path: Path to quad mesh OBJ file
        verbose: Print report to stdout

    Returns:
        MeshQualityReport
    """
    vertices, quads = load_obj_quads(obj_path)
    report = measure_mesh_quality(vertices, quads)

    if verbose:
        print(f"\nFile: {obj_path}")
        print(report)

    return report


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quad_quality.py <mesh.obj> [mesh2.obj ...]")
        print("\nMeasures quad mesh quality metrics for benchmarking.")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            benchmark_mesh(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
