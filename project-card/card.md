---
oneliner: Python port of the Corman/Crane algorithm for cross-field-aligned rectangular UV parameterization of triangle meshes
tags: [mesh parameterization, quad meshing, uv layout, cross field, computational geometry, python, numpy, scipy, siggraph]
stack: [Python, NumPy, SciPy]
generated: 2026-09-06
commit: bef5bea
placeholder: false
---
Computes an orthogonal, cross-field-aligned UV parameterization of a triangle mesh, then extracts a quad mesh from it via a bundled pure-Python libQEx replacement. Complete line-by-line Python/NumPy/SciPy port of the original MATLAB implementation, validated against it through Octave on multiple benchmark meshes (pig, torus, sphere, B36). Working state: CLI tools (`run_RSP.py`, `extract_quads.py`), test suite, and a docs gallery of example runs.
