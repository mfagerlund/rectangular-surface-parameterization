# Cross Fields Explained (from official MATLAB repo)

This note summarizes how the official MATLAB implementation computes cross fields for
Rectangular Surface Parameterization (Corman & Crane, 2025), with pointers to sources.

Repo:
- Online: https://github.com/etcorman/RectangularSurfaceParameterization
- Local clone: C:\slask\RectangularSurfaceParameterization

## Where the cross field is chosen

Entry point: run_RSP.m
- The option `frame_field_type` selects the method:
  - 'smooth'
  - 'curvature'
  - 'trivial'

Sources:
- C:\slask\RectangularSurfaceParameterization\run_RSP.m
- https://github.com/etcorman/RectangularSurfaceParameterization/blob/main/run_RSP.m

## 1) Smooth cross field (default)

Function:
- FrameField/compute_face_cross_field.m

What it does:
- Builds a complex-valued connection Laplacian for a 4-fold symmetric field.
- Applies alignment constraints at boundary/hard-edge constrained triangles.
- Initializes the field by solving a Poisson/eigen problem:
  - If constraints exist, solves a constrained linear system.
  - If no constraints, takes the smallest eigenvector.
- Smooths the field with heat-flow iterations on the connection Laplacian.
- Extracts field angles and computes singularities.

Key steps (see source for details):
- Connection Laplacian `Wcon = d0d_cplx' * star1d * d0d_cplx`
- Initialization via `eigs` or constrained solve
- Heat-flow smoothing: repeated solve of `(Wcon + dt * M) * z = rhs`

Sources:
- C:\slask\RectangularSurfaceParameterization\FrameField\compute_face_cross_field.m
- https://github.com/etcorman/RectangularSurfaceParameterization/blob/main/FrameField/compute_face_cross_field.m

## 2) Curvature-aligned cross field

Function:
- FrameField/compute_curvature_cross_field.m

What it does:
- Estimates a curvature tensor using dihedral angles (Cohen-Steiner & Morvan).
- Extracts principal directions per face.
- Initializes cross field from principal directions: `z = (dir_min/|dir_min|)^4`.
- Smooths the field using the same connection Laplacian approach.
- Extracts angles, computes singularities, and snaps frames to closest curvature direction.

Sources:
- C:\slask\RectangularSurfaceParameterization\FrameField\compute_curvature_cross_field.m
- https://github.com/etcorman/RectangularSurfaceParameterization/blob/main/FrameField/compute_curvature_cross_field.m

## 3) Trivial connection (user-specified singularities)

Function:
- FrameField/trivial_connection.m

What it does:
- Solves a quadratic program to find a connection `omega` that satisfies
  prescribed singularities and cycle/link constraints.
- Used when `frame_field_type = 'trivial'` and singularities are explicitly provided.

Sources:
- C:\slask\RectangularSurfaceParameterization\FrameField\trivial_connection.m
- https://github.com/etcorman/RectangularSurfaceParameterization/blob/main/FrameField/trivial_connection.m

## Important practical notes from the repo

- The cross field is not a simple BFS propagation. It is globally smoothed using
  a connection Laplacian with constraints.
- Boundary and hard-edge alignment constraints are built during preprocessing and
  strongly influence the field.

Relevant files:
- C:\slask\RectangularSurfaceParameterization\Preprocess\preprocess_ortho_param.m (constraints and mesh preprocessing)
- https://github.com/etcorman/RectangularSurfaceParameterization/tree/main/Preprocess
- C:\slask\RectangularSurfaceParameterization\FrameField\plot_frame_field.m (visualization)
- https://github.com/etcorman/RectangularSurfaceParameterization/blob/main/FrameField/plot_frame_field.m

## Mapping to this Python repo

Current Python implementation uses BFS propagation (cross_field.py) and does not
match the global smoothing used by the MATLAB reference. That difference likely
explains spurious singularities and fragmented cuts.

Python file:
- cross_field.py (BFS propagation)

## References in the paper (as mentioned in repo/paper)

The paper says pre-processing uses existing frame-field methods and does not
detail the solver. The MATLAB repo implements one such solver using a connection
Laplacian and heat-flow smoothing.

Suggested background sources (from the user notes and common field methods):
- Knoppel et al. 2013
- Ray et al. 2008
- Bommes et al. 2009

(Use these as background reading; the concrete implementation is in the MATLAB
repo sources linked above.)
