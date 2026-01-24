# rectangular_surface_parameterization
# Rectangular (orthogonal) surface parameterization for quad meshing
#
# Python port of Corman & Crane SIGGRAPH 2025

from rectangular_surface_parameterization.core.mesh import MeshInfo
from rectangular_surface_parameterization.core.signed_edge_array import SignedEdgeArray
from rectangular_surface_parameterization.io.mesh_io import load_mesh, readOBJ

__version__ = "0.1.0"

__all__ = [
    "MeshInfo",
    "SignedEdgeArray",
    "load_mesh",
    "readOBJ",
]
