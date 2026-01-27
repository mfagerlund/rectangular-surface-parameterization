"""Lazy-loading wrapper for pyquantization (Coudert-Osmont et al., 2024).

Provides integer-grid quantization of seamless UV parameterizations.
"""

import numpy as np


_pyquantization = None


def _ensure_pyquantization():
    global _pyquantization
    if _pyquantization is not None:
        return _pyquantization
    try:
        import pyquantization
        _pyquantization = pyquantization
        return pyquantization
    except ImportError:
        raise ImportError(
            "pyquantization not installed.\n"
            "Install from source:  pip install C:/Dev/pyquantization\n"
            "Or from GitHub:       python scripts/install_pyquantization.py"
        )


def quantize_mesh(vertices, triangles, uv_coords, uv_triangles,
                  feature_edges=None, scale=-1.0, scale_auto=1.0,
                  mode="reembed"):
    """Quantize a seamless UV parameterization to integer grid.

    Parameters
    ----------
    vertices : ndarray, shape (n_verts, 3)
        3D vertex positions.
    triangles : ndarray, shape (n_tris, 3)
        Triangle face indices (0-based).
    uv_coords : ndarray, shape (n_uvs, 2)
        UV coordinates (texture vertices).
    uv_triangles : ndarray, shape (n_tris, 3)
        UV index per face corner (0-based).
    feature_edges : ndarray, shape (n_feat, 2), optional
        Hard edge vertex pairs (0-based). Empty array if None.
    scale : float
        Scale factor for UV coordinates. Use -1.0 for automatic.
    scale_auto : float
        Multiplier for automatic scale (only used when scale=-1.0).
    mode : str
        "reembed" (recommended), "imprint", or "decimate".

    Returns
    -------
    out_vertices : ndarray (n_out_verts, 3)
    out_faces : ndarray (n_out_faces, 3)
    out_uvs : ndarray (n_out_corners, 2)
    out_uv_triangles : ndarray (n_out_faces, 3)
    out_feature_edges : ndarray (n_out_feat, 2)
    """
    pq = _ensure_pyquantization()

    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    triangles = np.ascontiguousarray(triangles, dtype=np.int32)
    uv_coords = np.ascontiguousarray(uv_coords, dtype=np.float64)
    uv_triangles = np.ascontiguousarray(uv_triangles, dtype=np.int32)

    if feature_edges is None or len(feature_edges) == 0:
        feature_edges = np.zeros((0, 2), dtype=np.int32)
    else:
        feature_edges = np.ascontiguousarray(feature_edges, dtype=np.int32)

    return pq.quantize_mesh(
        vertices, triangles, uv_coords, uv_triangles, feature_edges,
        scale=scale, scale_auto=scale_auto, mode=mode
    )
