# === ISSUES ===
# - persistent (MATLAB): Python uses module-level dict for caching
# - cell arrays: Python uses dict with integer keys
# === END ISSUES ===

import numpy as np
from typing import Tuple, Optional, Dict, Any
from Preprocess.sort_triangles_comp import sort_triangles_comp

# Module-level cache (replaces MATLAB persistent variables)
# Cache keys are (mesh_id, vertex_idx) tuples to ensure cross-mesh isolation
_tri_ord_cache: Dict[Tuple[int, int], np.ndarray] = {}
_edge_ord_cache: Dict[Tuple[int, int], np.ndarray] = {}
_sign_edge_cache: Dict[Tuple[int, int], np.ndarray] = {}


def clear_cache():
    """Clear all cached results."""
    global _tri_ord_cache, _edge_ord_cache, _sign_edge_cache
    _tri_ord_cache.clear()
    _edge_ord_cache.clear()
    _sign_edge_cache.clear()


# function [tri_ord,edge_ord,sign_edge] = sort_triangles(idx, T, E2T, T2T, E2V, T2E)
#
# persistent tri_ord_cach edge_ord_cach sign_edge_cach;
# if isempty(tri_ord_cach)
#     nv = max(abs(T(:)));
#     tri_ord_cach = cell(nv,1);
#     edge_ord_cach = cell(nv,1);
#     sign_edge_cach = cell(nv,1);
# end
#
# if isempty(tri_ord_cach{idx})
#     [tri_ord,edge_ord,sign_edge] = sort_triangles_comp(idx, T, E2T, T2T, E2V, T2E);
#     tri_ord_cach{idx} = tri_ord;
#     edge_ord_cach{idx} = edge_ord;
#     sign_edge_cach{idx} = sign_edge;
# else
#     tri_ord = tri_ord_cach{idx};
#     edge_ord = edge_ord_cach{idx};
#     sign_edge = sign_edge_cach{idx};
# end

def sort_triangles(idx: int, T: np.ndarray, E2T: np.ndarray, T2T: np.ndarray,
                   E2V: np.ndarray, T2E: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort ring triangles around vertex idx with caching.

    Parameters
    ----------
    idx : int
        Vertex index (0-based in Python)
    T : ndarray (nf, 3)
        Triangle vertex indices (0-based)
    E2T : ndarray (ne, 4)
        Edge to triangle mapping: [tri1, tri2, sign1, sign2]
        -1 means boundary (no adjacent triangle)
    T2T : ndarray (nf, 3)
        Triangle to triangle adjacency
    E2V : ndarray (ne, 2)
        Edge to vertex mapping
    T2E : ndarray (nf, 3), optional
        Triangle to edge mapping

    Returns
    -------
    tri_ord : ndarray
        Ordered triangle indices around vertex
    edge_ord : ndarray
        Ordered edge indices
    sign_edge : ndarray
        Edge orientation signs
    """
    global _tri_ord_cache, _edge_ord_cache, _sign_edge_cache

    # Use (mesh_id, vertex_idx) as cache key to ensure cross-mesh isolation
    mesh_id = id(T)
    cache_key = (mesh_id, idx)

    if cache_key not in _tri_ord_cache:
        tri_ord, edge_ord, sign_edge = sort_triangles_comp(idx, T, E2T, T2T, E2V, T2E)
        _tri_ord_cache[cache_key] = tri_ord
        _edge_ord_cache[cache_key] = edge_ord
        _sign_edge_cache[cache_key] = sign_edge
    else:
        tri_ord = _tri_ord_cache[cache_key]
        edge_ord = _edge_ord_cache[cache_key]
        sign_edge = _sign_edge_cache[cache_key]

    return tri_ord, edge_ord, sign_edge
