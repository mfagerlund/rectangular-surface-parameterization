# Mesh I/O using trimesh library
# Replaces the original MATLAB-ported OBJ reader

import numpy as np
from typing import Tuple
import trimesh


def readOBJ(filename: str, quads: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a mesh file (OBJ, PLY, STL, OFF, etc.) using trimesh.

    Args:
        filename: path to mesh file
        quads: if True, preserve quad faces (not yet supported, will triangulate)

    Returns:
        V: vertices (#V, 3)
        F: face indices (#F, 3), 0-indexed
        UV: texture coordinates (#UV, 2) or empty array
        TF: face texture indices (#F, 3) or empty array, 0-indexed
        N: normals (#N, 3) or empty array
        NF: face normal indices (#F, 3) or empty array, 0-indexed
        SI: singularity info - empty array (not used)

    Note:
        - Supports any format trimesh can load (OBJ, PLY, STL, OFF, GLB, etc.)
        - Handles UTF-8 and other encodings automatically
        - Merges duplicate vertices by position (important for correct topology)
        - Quad faces are triangulated (quads=True not yet implemented)
    """
    # Load mesh with trimesh - it handles encoding automatically
    mesh = trimesh.load(filename, force='mesh', process=False)

    # Get vertices and faces
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    # Merge duplicate vertices by position
    # Trimesh expands vertices for per-face texture coords, but we need
    # the true mesh topology for geometry processing
    V, F = _merge_duplicate_vertices(V, F)

    # Handle texture coordinates if available
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        UV = np.asarray(mesh.visual.uv, dtype=np.float64)
        # Note: UV indices may not match after merging - return original for reference
        TF = np.zeros((F.shape[0], 3), dtype=np.int64)  # Placeholder
    else:
        UV = np.zeros((0, 2), dtype=np.float64)
        TF = np.zeros((0, 3), dtype=np.int64)

    # Handle normals if available
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        N = np.asarray(mesh.vertex_normals, dtype=np.float64)
        NF = np.zeros((F.shape[0], 3), dtype=np.int64)  # Placeholder
    else:
        N = np.zeros((0, 3), dtype=np.float64)
        NF = np.zeros((0, 3), dtype=np.int64)

    # SI (singularity info) - not used, return empty
    SI = np.zeros((0, 2), dtype=np.float64)

    return V, F, UV, TF, N, NF, SI


def _merge_duplicate_vertices(V: np.ndarray, F: np.ndarray,
                               decimals: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge vertices that have the same position.

    Trimesh often creates duplicate vertices when loading OBJ files with
    per-face texture coordinates. This function merges them back.

    Args:
        V: vertices (N, 3)
        F: faces (M, 3)
        decimals: precision for comparing vertex positions

    Returns:
        V_unique: unique vertices
        F_remapped: faces with updated indices
    """
    # Round to handle floating point comparison
    V_rounded = np.round(V, decimals=decimals)

    # Find unique vertices and mapping from old to new indices
    V_unique, inverse = np.unique(V_rounded, axis=0, return_inverse=True)

    # Remap face indices
    F_remapped = inverse[F]

    # Use original (non-rounded) vertices for the unique positions
    # Get one original vertex per unique position
    unique_indices = np.zeros(len(V_unique), dtype=np.int64)
    seen = np.zeros(len(V_unique), dtype=bool)
    for i, inv in enumerate(inverse):
        if not seen[inv]:
            unique_indices[inv] = i
            seen[inv] = True

    V_unique = V[unique_indices]

    return V_unique, F_remapped


def load_mesh(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple mesh loader - just vertices and faces.

    Args:
        filename: path to mesh file

    Returns:
        vertices: (#V, 3) vertex positions
        faces: (#F, 3) triangle indices, 0-indexed
    """
    V, F, *_ = readOBJ(filename)
    return V, F
