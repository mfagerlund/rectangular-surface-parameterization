---
oneliner: Python port of Corman & Crane's rectangular (orthogonal) mesh UV parameterization with quad extraction
tags: [mesh parameterization, uv layout, cross field, quad meshing, computational geometry, numpy, scipy, libqex]
stack: [Python, NumPy, SciPy]
generated: 2026-09-06
commit: bef5bea
placeholder: false
---
Line-by-line Python port of the MATLAB implementation of "Rectangular Surface Parameterization" (Corman & Crane, SIGGRAPH 2025), computing a cross-field-aligned orthogonal UV parameterization of a triangle mesh and extracting quad meshes via libQEx/QuadriFlow. Working pipeline validated against the original MATLAB code through Octave on several benchmark meshes (pig, torus, sphere, B36), with a CLI, visualization stages, and a growing test suite.
