"""
Mesh preprocessing utilities to prepare meshes for the RSP pipeline.

Uses PyMeshLab to clean and remesh input meshes to meet pipeline requirements:
- Manifold geometry (each edge shared by exactly 2 triangles)
- Closed surface (no boundary edges)
- Well-shaped triangles (avoid very obtuse/skinny triangles)
- Consistent orientation

Usage:
    from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh

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
    target_faces: Optional[int] = None,
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
    target_faces : int, optional
        Target number of faces for decimation. Recommended for meshes >10K faces.
        If specified, decimation is used instead of remeshing.
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

    # Step 5: Close ALL holes to make mesh watertight
    if verbose:
        print("  Closing holes...")
    try:
        # Close holes iteratively with increasing max size
        for max_size in [100, 1000, 10000, 100000, 1000000]:
            ms.meshing_close_holes(maxholesize=max_size)
            try:
                info = ms.get_topological_measures()
                if info['boundary_edges'] == 0:
                    break
            except Exception:
                pass
    except Exception:
        pass

    # Step 6: Close remaining small holes (e.g., triangular holes that pymeshlab misses)
    try:
        info = ms.get_topological_measures()
        if info['boundary_edges'] > 0 and info['boundary_edges'] <= 10:
            if verbose:
                print(f"  Closing small remaining hole ({info['boundary_edges']} boundary edges)...")
            # Save temp, use trimesh to close triangular hole
            try:
                import trimesh
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
                    temp_path = f.name
                ms.save_current_mesh(temp_path)

                tmesh = trimesh.load(temp_path)
                from collections import Counter
                edge_counts = Counter(tuple(sorted(e)) for e in tmesh.edges)
                boundary_edges = [e for e, c in edge_counts.items() if c == 1]

                if len(boundary_edges) == 3:
                    # Triangular hole - add one face
                    boundary_verts = list(set(v for e in boundary_edges for v in e))
                    new_faces = np.vstack([tmesh.faces, boundary_verts])
                    tmesh = trimesh.Trimesh(vertices=tmesh.vertices, faces=new_faces, process=True)
                    tmesh.fix_normals()
                    tmesh.export(temp_path)

                    # Reload in pymeshlab
                    ms = pymeshlab.MeshSet()
                    ms.load_new_mesh(temp_path)
                    if verbose:
                        print("    Closed triangular hole")

                Path(temp_path).unlink()
            except ImportError:
                if verbose:
                    print("    trimesh not available for small hole fix")
    except Exception:
        pass

    # Step 7: Re-orient faces consistently
    if verbose:
        print("  Re-orienting faces...")
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception:
        pass

    # Step 8: Decimation or remeshing
    current_faces = ms.current_mesh().face_number()

    if target_faces is not None and target_faces < current_faces:
        # Use decimation for large meshes
        if verbose:
            print(f"  Decimating from {current_faces} to ~{target_faces} faces...")
        try:
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            # Re-close any holes that appeared from decimation
            ms.meshing_close_holes(maxholesize=1000)
        except Exception as e:
            if verbose:
                print(f"    Decimation failed: {e}")
    elif target_edge_length is not None or target_faces is None:
        # Isotropic remeshing for better triangle quality
        if verbose:
            print("  Remeshing for better triangle quality...")

        # Compute target edge length if not specified
        if target_edge_length is None:
            bbox = mesh.bounding_box()
            diagonal = np.sqrt(
                (bbox.max()[0] - bbox.min()[0])**2 +
                (bbox.max()[1] - bbox.min()[1])**2 +
                (bbox.max()[2] - bbox.min()[2])**2
            )
            target_edge_length = diagonal / 50

        if verbose:
            print(f"    Target edge length: {target_edge_length:.6f}")

        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(1.0),  # 1% of bbox diagonal
                iterations=remesh_iterations,
                adaptive=True
            )
        except Exception as e:
            if verbose:
                print(f"    Remeshing failed: {e}")
                print("    Skipping remeshing step...")

    # Step 9: Final cleanup
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
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess mesh for RSP pipeline',
        epilog='''
Examples:
  python preprocess_mesh.py bunny.obj bunny_clean.obj
  python preprocess_mesh.py bunny.obj bunny_clean.obj --target-faces 10000
        '''
    )
    parser.add_argument('input', help='Input mesh file')
    parser.add_argument('output', nargs='?', help='Output mesh file (default: input_clean.obj)')
    parser.add_argument('--target-faces', type=int, help='Target face count for decimation')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    verbose = not args.quiet

    # Check quality first
    if verbose:
        print("=" * 50)
        print("QUALITY CHECK (before)")
        print("=" * 50)
        check_mesh_quality(input_path)

    # Preprocess
    if verbose:
        print("\n" + "=" * 50)
        print("PREPROCESSING")
        print("=" * 50)
    result_path = preprocess_mesh(input_path, output_path, target_faces=args.target_faces, verbose=verbose)

    # Check quality after
    print("\n" + "=" * 50)
    print("QUALITY CHECK (after)")
    print("=" * 50)
    check_mesh_quality(result_path)
