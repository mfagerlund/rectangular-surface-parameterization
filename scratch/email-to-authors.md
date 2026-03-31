Subject: Python port of your Rectangular Surface Parameterization code

Dear Dr. Corman and Prof. Crane,

I've been working on a Python/NumPy/SciPy port of your Rectangular Surface Parameterization MATLAB code. The repo is currently private and I wanted to reach out before making it public, in case you have any concerns or feedback.

The port covers the full pipeline (preprocessing, cross field computation, optimization, seamless parameterization). I validated it against your MATLAB code by running it through GNU Octave 10.3.0 — all three benchmark meshes (pig, B36, SquareMyles) produce matching singularity structures, zero flipped triangles, and successful convergence.

I also created Python bindings (via pybind11) for Yoann Coudert-Osmont's quantization code from your repository, packaged as a separate pip-installable library called pyquantization. This gives the Python port access to the same integer-grid quantization pipeline as the MATLAB version, with no commercial dependencies.

Your paper and repository are cited throughout, and the README links back to your original work. Both projects use AGPL-3.0, matching your licensing.

I'd be happy to share access to the private repo if you'd like to take a look before I make it public. No rush at all — I'm in no hurry to release.

Best regards,
Mattias Fagerlund
