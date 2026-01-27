"""Compare Python RSP outputs against Octave (MATLAB) reference outputs."""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rectangular_surface_parameterization.io.read_obj import readOBJ
from rectangular_surface_parameterization.core.mesh_info import MeshInfo, mesh_info
from rectangular_surface_parameterization.preprocessing.dec import dec_tri
from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param
from rectangular_surface_parameterization.cross_field.face_field import compute_face_cross_field
from rectangular_surface_parameterization.optimization.reduce_corner_var import reduce_corner_var_2d
from rectangular_surface_parameterization.optimization.reduce_corner_var_cut import reduce_corner_var_2d_cut
from rectangular_surface_parameterization.optimization.reduction import reduction_from_ff2d
from rectangular_surface_parameterization.optimization.solver import optimize_RSP
from rectangular_surface_parameterization.parameterization.integrate import parametrization_from_scales
from rectangular_surface_parameterization.parameterization.seamless import mesh_to_disk_seamless
from rectangular_surface_parameterization.utils.extract_scale import extract_scale_from_param

from dataclasses import dataclass
from typing import Optional

@dataclass
class EnergyWeight:
    w_conf_ar: float = 0.5
    w_gradv: float = 1e-2
    w_ang: Optional[float] = None
    w_ratio: Optional[float] = None
    aspect_ratio: Optional[np.ndarray] = None
    ang_dir: Optional[np.ndarray] = None


def run_python_pipeline(mesh_path, ifhardedge=False):
    """Run full Python RSP pipeline and return intermediates."""
    X, T = readOBJ(mesh_path)[:2]

    # Rescale
    cross_prod = np.cross(X[T[:, 0]] - X[T[:, 1]], X[T[:, 0]] - X[T[:, 2]])
    area_tot = np.sum(np.sqrt(np.sum(cross_prod ** 2, axis=1))) / 2
    X = X / np.sqrt(area_tot)

    Src = mesh_info(X, T)
    dec = dec_tri(Src)

    param, Src, dec = preprocess_ortho_param(Src, dec, True, ifhardedge, 40)

    omega, ang, sing = compute_face_cross_field(Src, param, dec, 10)

    if len(param.ide_bound) > 0:
        Edge_jump, v2t, base_tri = reduce_corner_var_2d_cut(Src, param.ide_bound)
    else:
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)
    k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

    weight = EnergyWeight(w_conf_ar=0.5, w_gradv=1e-2)
    u = np.zeros(Src.num_vertices)
    v = np.zeros(Src.num_vertices)

    result = optimize_RSP(omega, ang, u, v, Src, param, dec, Reduction,
                          'distortion', weight, False, 200)

    disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(
        Src, param, result.angn, sing, k21, True, True, ifhardedge)
    Xp, dX = parametrization_from_scales(
        Src, disk_mesh, dec_cut, param, result.angn, result.om,
        result.ut, result.vt, Align, Rot)

    disto = extract_scale_from_param(Xp, Src.vertices, Src.triangles, param, disk_mesh.triangles, result.angn)[0]

    return {
        'ang': ang,
        'sing': sing,
        'omega': omega,
        'u': result.u,
        'v': result.v,
        'angn': result.angn,
        'om': result.om,
        'Xp': Xp,
        'Src': Src,
        'disto': disto,
    }


def compare(name, py_val, oct_val, rtol=1e-6, atol=1e-8):
    """Compare two arrays and print statistics."""
    if py_val.shape != oct_val.shape:
        print(f"  {name}: SHAPE MISMATCH py={py_val.shape} oct={oct_val.shape}")
        return False

    diff = np.abs(py_val - oct_val)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = diff / (np.abs(oct_val) + 1e-15)
    max_rel = np.max(rel_diff)

    close = np.allclose(py_val, oct_val, rtol=rtol, atol=atol)
    status = "PASS" if close else "DIFF"

    print(f"  {name:12s}: {status}  max_abs={max_diff:.2e}  mean_abs={mean_diff:.2e}  max_rel={max_rel:.2e}  shape={py_val.shape}")
    return close


def main():
    octave_dir = Path("C:/Dev/RectangularSurfaceParameterization-matlab/Results")
    mesh_path = "C:/Dev/RectangularSurfaceParameterization-matlab/Mesh/B36.obj"

    print("Running Python pipeline on B36 (smooth, no hardedge, distortion)...")
    py = run_python_pipeline(mesh_path, ifhardedge=False)

    print(f"\nPython results:")
    print(f"  Vertices: {py['Src'].num_vertices}, Faces: {py['Src'].num_faces}")
    print(f"  Singularities: {(py['sing'] > 1/8).sum()} pos, {(py['sing'] < -1/8).sum()} neg")
    print(f"  Flipped: {(py['disto'].detJ <= 0).sum()}")
    print(f"  UV range: [{py['Xp'][:,0].min():.6f}, {py['Xp'][:,0].max():.6f}] x [{py['Xp'][:,1].min():.6f}, {py['Xp'][:,1].max():.6f}]")

    print(f"\nLoading Octave reference data from {octave_dir}...")
    oct_ang = np.loadtxt(octave_dir / "B36_octave_ang.txt")
    oct_sing = np.loadtxt(octave_dir / "B36_octave_sing.txt")
    oct_omega = np.loadtxt(octave_dir / "B36_octave_omega.txt")
    oct_u = np.loadtxt(octave_dir / "B36_octave_u.txt")
    oct_v = np.loadtxt(octave_dir / "B36_octave_v.txt")
    oct_Xp = np.loadtxt(octave_dir / "B36_octave_Xp.txt")

    print(f"\nOctave results:")
    print(f"  Singularities: {(oct_sing > 1/8).sum()} pos, {(oct_sing < -1/8).sum()} neg")
    print(f"  UV range: [{oct_Xp[:,0].min():.6f}, {oct_Xp[:,0].max():.6f}] x [{oct_Xp[:,1].min():.6f}, {oct_Xp[:,1].max():.6f}]")

    print(f"\n--- Comparison ---")

    # Cross field (before optimization)
    compare("ang", py['ang'], oct_ang)
    compare("sing", py['sing'], oct_sing)
    compare("omega", py['omega'], oct_omega)

    # Optimization results
    compare("u", py['u'], oct_u, rtol=1e-4)
    compare("v", py['v'], oct_v, rtol=1e-4)

    # UV coordinates - may differ by global translation/rotation
    if py['Xp'].shape == oct_Xp.shape:
        compare("Xp", py['Xp'], oct_Xp, rtol=1e-4)

        # Also try with translation alignment
        py_centered = py['Xp'] - py['Xp'].mean(axis=0)
        oct_centered = oct_Xp - oct_Xp.mean(axis=0)
        compare("Xp_centered", py_centered, oct_centered, rtol=1e-4)
    else:
        print(f"  Xp: SHAPE MISMATCH py={py['Xp'].shape} oct={oct_Xp.shape}")
        print(f"  (Different cut mesh topology - comparing UV ranges instead)")
        py_range = py['Xp'].max(axis=0) - py['Xp'].min(axis=0)
        oct_range = oct_Xp.max(axis=0) - oct_Xp.min(axis=0)
        print(f"  Python UV extent: {py_range[0]:.6f} x {py_range[1]:.6f}")
        print(f"  Octave UV extent: {oct_range[0]:.6f} x {oct_range[1]:.6f}")


if __name__ == '__main__':
    main()
