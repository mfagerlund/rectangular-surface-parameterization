"""
Mesh preprocessing utilities to prepare meshes for the RSP pipeline.

Uses PyMeshLab to clean and remesh input meshes to meet pipeline requirements:
- Manifold geometry (each edge shared by exactly 2 triangles)
- Closed surface (no boundary edges)
- Well-shaped triangles (avoid very obtuse/skinny triangles)
- Consistent orientation

Usage:
    from Utils.preprocess_mesh import preprocess_mesh

    clean_path = preprocess_mesh("bunny.obj", "bunny_clean.obj")
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings


def preprocess_mesh(
    input_path: str,
    output_path: Optional[str] = None,
    target_edge_length: Optional[float] = None,
    remesh_iterations: int = 5,
    verbose: bool = True
) -> str:
    """
    Preprocess a mesh for the RSP pipeline.

    Parameters
    ----------
    input_path : str
        Path to input mesh file (OBJ, PLY, STL, etc.)
    output_path : str, optional
        Path for output mesh. If None, appends '_clean' to input name.
    target_edge_length : float, optional
        Target edge length for remeshing. If None, uses average edge length.
    remesh_iterations : int
        Number of remeshing iterations (default: 5)
    verbose : bool
        Print progress information

    Returns
    -------
    str
        Path to the cleaned mesh file.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    output_path = Path(output_path)

    if verbose:
        print(f"Preprocessing mesh: {input_path}")

    # Create MeshSet and load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))

    # Get initial stats
    mesh = ms.current_mesh()
    n_verts_initial = mesh.vertex_number()
    n_faces_initial = mesh.face_number()

    if verbose:
        print(f"  Initial: {n_verts_initial} vertices, {n_faces_initial} faces")

    # Step 1: Remove duplicate vertices and faces
    if verbose:
        print("  Removing duplicates...")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()

    # Step 2: Remove unreferenced vertices
    ms.meshing_remove_unreferenced_vertices()

    # Step 3: Remove zero-area faces
    ms.meshing_remove_null_faces()

    # Step 4: Repair non-manifold edges/vertices
    if verbose:
        print("  Repairing non-manifold geometry...")
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass  # May not be needed
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception:
        pass

    # Step 5: Close holes (small holes only to preserve features)
    if verbose:
        print("  Closing small holes...")
    try:
        ms.meshing_close_holes(maxholesize=30)
    except Exception:
        pass

    # Step 6: Re-orient faces consistently
    if verbose:
        print("  Re-orienting faces...")
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception:
        pass

    # Step 7: Isotropic remeshing for better triangle quality
    if verbose:
        print("  Remeshing for better triangle quality...")

    # Compute target edge length if not specified
    if target_edge_length is None:
        # Use bounding box diagonal / 100 as default
        bbox = mesh.bounding_box()
        diagonal = np.sqrt(
            (bbox.max()[0] - bbox.min()[0])**2 +
            (bbox.max()[1] - bbox.min()[1])**2 +
            (bbox.max()[2] - bbox.min()[2])**2
        )
        target_edge_length = diagonal / 50  # Reasonable default

    if verbose:
        print(f"    Target edge length: {target_edge_length:.6f}")

    try:
        # Try different API versions for remeshing
        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.AbsoluteValue(target_edge_length),
                iterations=remesh_iterations,
                adaptive=True
            )
        except AttributeError:
            # Newer PyMeshLab API
            ms.meshing_isotropic_explicit_remeshing(
                iterations=remesh_iterations,
                adaptive=True,
                targetlen=target_edge_length
            )
    except Exception as e:
        if verbose:
            print(f"    Remeshing failed: {e}")
            print("    Trying surface reconstruction instead...")

        # Fallback: use Poisson surface reconstruction
        try:
            ms.generate_surface_reconstruction_screened_poisson(depth=8)
        except Exception as e2:
            if verbose:
                print(f"    Surface reconstruction also failed: {e2}")

    # Step 8: Final cleanup
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Get final stats
    mesh = ms.current_mesh()
    n_verts_final = mesh.vertex_number()
    n_faces_final = mesh.face_number()

    if verbose:
        print(f"  Final: {n_verts_final} vertices, {n_faces_final} faces")

    # Save result
    ms.save_current_mesh(str(output_path))

    if verbose:
        print(f"  Saved to: {output_path}")

    return str(output_path)


def check_mesh_quality(mesh_path: str, verbose: bool = True) -> dict:
    """
    Check mesh quality and return diagnostic information.

    Parameters
    ----------
    mesh_path : str
        Path to mesh file.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    dict
        Dictionary with quality metrics and issues found.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))
    mesh = ms.current_mesh()

    result = {
        'vertices': mesh.vertex_number(),
        'faces': mesh.face_number(),
        'issues': [],
        'is_manifold': True,
        'is_closed': True,
        'has_consistent_orientation': True,
    }

    # Check for non-manifold edges
    try:
        ms.compute_selection_by_non_manifold_edges_per_face()
        n_non_manifold = mesh.selected_face_number()
        if n_non_manifold > 0:
            result['issues'].append(f"Non-manifold edges: {n_non_manifold} faces affected")
            result['is_manifold'] = False
        ms.set_selection_none()
    except Exception:
        pass

    # Check for boundary edges (holes)
    try:
        ms.compute_selection_by_border()
        n_border = mesh.selected_vertex_number()
        if n_border > 0:
            result['issues'].append(f"Boundary vertices: {n_border} (mesh has holes)")
            result['is_closed'] = False
        ms.set_selection_none()
    except Exception:
        pass

    # Compute quality metrics
    try:
        quality = ms.get_topological_measures()
        result['euler_characteristic'] = quality.get('euler_characteristic', None)
        result['genus'] = quality.get('genus', None)
        result['connected_components'] = quality.get('connected_components_number', None)
    except Exception:
        pass

    if verbose:
        print(f"Mesh: {mesh_path}")
        print(f"  Vertices: {result['vertices']}, Faces: {result['faces']}")
        print(f"  Manifold: {result['is_manifold']}, Closed: {result['is_closed']}")
        if result.get('genus') is not None:
            print(f"  Genus: {result['genus']}, Euler χ: {result['euler_characteristic']}")
        if result['issues']:
            print(f"  Issues:")
            for issue in result['issues']:
                print(f"    - {issue}")

    return result


def make_delaunay(mesh_path: str, output_path: Optional[str] = None, verbose: bool = True) -> str:
    """
    Apply intrinsic Delaunay triangulation to improve triangle quality.

    This flips edges to achieve a Delaunay triangulation, which helps
    avoid negative Voronoi areas.

    Parameters
    ----------
    mesh_path : str
        Path to input mesh.
    output_path : str, optional
        Path for output. If None, appends '_delaunay'.
    verbose : bool
        Print progress.

    Returns
    -------
    str
        Path to output mesh.
    """
    try:
        import pymeshlab
    except ImportError:
        raise ImportError("PyMeshLab required. Install with: pip install pymeshlab")

    mesh_path = Path(mesh_path)
    if output_path is None:
        output_path = mesh_path.parent / f"{mesh_path.stem}_delaunay{mesh_path.suffix}"

    if verbose:
        print(f"Applying Delaunay triangulation to: {mesh_path}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_path))

    # Apply Delaunay edge flipping
    try:
        ms.meshing_surface_delaunay_triangulation()
        if verbose:
            print("  Applied surface Delaunay triangulation")
    except Exception as e:
        if verbose:
            print(f"  Delaunay failed: {e}")
        # Fallback: just return original
        ms.save_current_mesh(str(output_path))
        return str(output_path)

    ms.save_current_mesh(str(output_path))

    if verbose:
        print(f"  Saved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess_mesh.py <input.obj> [output.obj]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Check quality first
    print("=" * 50)
    print("QUALITY CHECK (before)")
    print("=" * 50)
    check_mesh_quality(input_path)

    # Preprocess
    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)
    result_path = preprocess_mesh(input_path, output_path)

    # Check quality after
    print("\n" + "=" * 50)
    print("QUALITY CHECK (after)")
    print("=" * 50)
    check_mesh_quality(result_path)
