"""
QaWiTM: Quad Mesh Quantization Without a T-Mesh

A fresh implementation (in progress) of the algorithm from:

    Yoann Coudert-Osmont, David Desobry, Martin Heistermann, David Bommes,
    Nicolas Ray, Dmitry Sokolov. 2023.
    "Quad Mesh Quantization Without a T-Mesh."
    Computer Graphics Forum.

This is NOT a port - no source code was available. Implemented directly from
the paper's algorithm descriptions.

STATUS: SCAFFOLD - Core structure implemented but final solve needs work.

The algorithm snaps singularities to integer UV coordinates, producing
properly-formed irregular vertices in the extracted quad mesh.

Phases (implementation status):
1. Decimate mesh while preserving singularities (§4.1) - DONE
2. Optimize integer edge geometry via Dijkstra (§4.2) - DONE (basic)
3. Propagate constraints back to original mesh (§4.3) - DONE
4. Re-solve seamless map with integer constraints - NEEDS WORK

Current limitation:
The simple "snap singularities to integers" approach breaks the parameterization.
Full implementation requires re-solving RSP with integer constraints, which needs:
- Integration with the RSP optimization solver
- Proper constraint formulation (A @ U = omega)
- Constrained least-squares solve maintaining seamlessness

See QUANTIZATION.md for algorithm details.
"""

from .quantize import quantize_uv, QuantizationResult

__all__ = [
    "quantize_uv",
    "QuantizationResult",
]
