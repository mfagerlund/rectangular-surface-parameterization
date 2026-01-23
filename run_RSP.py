# === ISSUES ===
# - trisurf: visualization uses matplotlib (matplotlib.pyplot.triplot or mpl_toolkits.mplot3d)
# - contains: use 'x in str' or str.find()
# - QuantizationYoann: external C++ executable, not ported - quantization disabled
# === END ISSUES ===

# % Rectangle Surface Parametrization
# % Corman and Crane, 2025

"""
Rectangle Surface Parametrization - Corman and Crane, 2025

Converts MATLAB main entry point (run_RSP.m) to Python CLI.

Usage:
    python run_RSP.py mesh.obj [options]

Example:
    python run_RSP.py C:/Dev/Colonel/Data/Meshes/sphere320.obj -o output/
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import converted modules
from Utils.readOBJ import readOBJ
from Utils.writeObj import writeObj
from Utils.save_param import save_param
from Utils.extract_scale_from_param import extract_scale_from_param
from Utils.visualize_uv import save_uv_visualization, visualize_run_RSP_result

from Preprocess.MeshInfo import MeshInfo, mesh_info
from Preprocess.dec_tri import dec_tri
from Preprocess.preprocess_ortho_param import preprocess_ortho_param

from FrameField.trivial_connection import trivial_connection
from FrameField.compute_face_cross_field import compute_face_cross_field
from FrameField.brush_frame_field import brush_frame_field
from FrameField.plot_frame_field import plot_frame_field

from Orthotropic.reduce_corner_var_2d import reduce_corner_var_2d
from Orthotropic.reduction_from_ff2d import reduction_from_ff2d
from Orthotropic.optimize_RSP import optimize_RSP

from ComputeParam.parametrization_from_scales import parametrization_from_scales
from ComputeParam.mesh_to_disk_seamless import mesh_to_disk_seamless


@dataclass
class Weight:
    """Weight parameters for energy optimization."""
    w_conf_ar: float = 0.5        # 0: area-preserving -- 0.5: isometry -- 1: conformal
    w_ang: float = 1.0            # Weight direction energy (for alignment)
    w_ratio: float = 1.0          # Weight aspect ratio energy (for alignment)
    w_gradv: float = 1e-2         # Weight regularization on v
    aspect_ratio: Optional[np.ndarray] = None   # Target aspect ratio (for curvature)
    ang_dir: Optional[np.ndarray] = None        # Target direction (for curvature)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Rectangle Surface Parametrization - Corman and Crane, 2025',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_RSP.py mesh.obj
  python run_RSP.py mesh.obj -o output/ --frame-field smooth
  python run_RSP.py mesh.obj --no-boundary --no-hardedge
  python run_RSP.py mesh.obj --energy chebyshev
        """
    )

    parser.add_argument('mesh', type=str, help='Path to input OBJ mesh file')
    parser.add_argument('-o', '--output', type=str, default='Results/',
                        help='Output directory (default: Results/)')

    # Frame field options
    parser.add_argument('--frame-field', type=str, default='smooth',
                        choices=['curvature', 'smooth', 'trivial'],
                        help='Frame field type (default: smooth)')
    parser.add_argument('--no-hardedge', action='store_true',
                        help='Disable hard edge constraints')
    parser.add_argument('--no-boundary', action='store_true',
                        help='Disable boundary alignment')
    parser.add_argument('--no-seamless', action='store_true',
                        help='Disable seamlessness constraint')
    parser.add_argument('--quantization', action='store_true',
                        help='Enable integer quantization (requires external QuantizationYoann binary)')

    # Energy options
    parser.add_argument('--energy', type=str, default='distortion',
                        choices=['distortion', 'chebyshev', 'alignment'],
                        help='Energy type (default: distortion)')
    parser.add_argument('--w-conf-ar', type=float, default=0.5,
                        help='Conformal/area ratio weight for distortion energy (default: 0.5)')
    parser.add_argument('--w-gradv', type=float, default=1e-2,
                        help='Regularization weight on v (default: 1e-2)')

    # Solver options
    parser.add_argument('--itmax', type=int, default=200,
                        help='Maximum solver iterations (default: 200)')
    parser.add_argument('--dihedral-tol', type=float, default=40.0,
                        help='Dihedral angle threshold for hard edges in degrees (default: 40)')

    # Visualization
    parser.add_argument('--plot', action='store_true',
                        help='Show visualization plots')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save UV visualization PNGs to output directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # % Options
    # mesh_name = 'B36';
    # frame_field_type = 'smooth';
    # ifhardedge   = true;
    # ifboundary   = true;
    # ifseamless_const = true;
    # ifquantization = true;
    # energy_type = 'distortion';

    mesh_path = Path(args.mesh)
    mesh_name = mesh_path.stem

    frame_field_type = args.frame_field
    ifhardedge = not args.no_hardedge
    ifboundary = not args.no_boundary
    ifseamless_const = not args.no_seamless
    ifquantization = args.quantization

    energy_type = args.energy
    tol_dihedral_deg = args.dihedral_tol
    itmax = args.itmax
    ifplot = args.plot
    verbose = args.verbose

    path_save = args.output

    # Create output directory if needed
    os.makedirs(path_save, exist_ok=True)

    # % Energy weights
    # if strcmp(energy_type, 'distortion')
    #     weight.w_conf_ar = 0.5;
    # end
    # if strcmp(energy_type, 'alignment')
    #     weight.w_ang   = 1;
    #     weight.w_ratio = 1;
    # end
    # weight.w_gradv = 1e-2;

    weight = Weight()
    if energy_type == 'distortion':
        weight.w_conf_ar = args.w_conf_ar
    if energy_type == 'alignment':
        weight.w_ang = 1.0
        weight.w_ratio = 1.0
    weight.w_gradv = args.w_gradv

    # %% Load mesh
    # [X,T] = readOBJ([path_data, mesh_name, '.obj']);

    if verbose:
        print(f"Loading mesh: {mesh_path}")

    X, T, *_ = readOBJ(str(mesh_path))

    if verbose:
        print(f"  Vertices: {X.shape[0]}, Faces: {T.shape[0]}")

    # % Rescale: area equals one
    # area_tot = sum(sqrt(sum(cross(X(T(:,1),:) - X(T(:,2),:), X(T(:,1),:) - X(T(:,3),:),2).^2,2)))/2;
    # X = X/sqrt(area_tot);

    # Compute total area
    e1 = X[T[:, 0], :] - X[T[:, 1], :]
    e2 = X[T[:, 0], :] - X[T[:, 2], :]
    cross_prod = np.cross(e1, e2)
    area_tot = np.sum(np.sqrt(np.sum(cross_prod ** 2, axis=1))) / 2

    # Rescale so total area = 1
    scale_factor = np.sqrt(area_tot)
    X = X / scale_factor

    if verbose:
        print(f"  Rescaled by factor: {1/scale_factor:.4f} (total area -> 1)")

    # % Preprocess geometry
    # Src = MeshInfo(X, T);
    # dec = dec_tri(Src);
    # [param,Src,dec] = preprocess_ortho_param(Src, dec, ifboundary, ifhardedge, 40);

    if verbose:
        print("Computing mesh connectivity...")

    Src = mesh_info(X, T)
    dec = dec_tri(Src)

    if verbose:
        print(f"  Edges: {Src.ne}")
        print("Preprocessing for orthotropic parameterization...")

    param, Src, dec = preprocess_ortho_param(Src, dec, ifboundary, ifhardedge, tol_dihedral_deg)

    if verbose:
        print(f"  Hard edges: {len(param.ide_hard)}")
        print(f"  Boundary edges: {len(param.ide_bound)}")
        print(f"  Fixed edges: {len(param.ide_fix)}")

    # % Plot constraint edges (visualization)
    # col = zeros(Src.nv,1); col(Src.E2V(param.ide_fix,:)) = 1;
    # figure; trisurf(...); title('Constraint')

    if ifplot:
        col = np.zeros(Src.nv)
        if len(param.ide_fix) > 0:
            col[Src.E2V[param.ide_fix, :].flatten()] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(Src.X[:, 0], Src.X[:, 1], Src.X[:, 2], triangles=Src.T,
                        cmap='coolwarm', edgecolor='k', linewidth=0.1)
        ax.set_title('Constraint Edges')
        plt.show()

    # %% Compute initial cross field
    # if strcmp(frame_field_type, 'curvature')
    #     [omega,ang,sing,kappa,Curv] = compute_curvature_cross_field(Src, param, dec, 30, 1e-1);
    #     weight.aspect_ratio = ...
    #     weight.ang_dir = ang;
    # elseif strcmp(frame_field_type, 'smooth')
    #     [omega,ang,sing] = compute_face_cross_field(Src, param, dec, 10);
    # elseif strcmp(frame_field_type, 'trivial')
    #     sing = zeros(Src.nv,1);
    #     sing(param.idx_bound) = round(2*param.K(param.idx_bound)/pi)/4;
    #     om_cycle = param.Icycle*param.para_trans;
    #     om_cycle = om_cycle - 2*pi*round(4*om_cycle/(2*pi))/4;
    #     om_link = param.Ilink*param.para_trans;
    #     om_link = om_link - 2*pi*round(4*om_link/(2*pi))/4;
    #     [omega,ang,sing] = trivial_connection(Src, param, dec, ifboundary, ifhardedge, sing);
    # else
    #     error('Cross field option unavailable.')
    # end

    if verbose:
        print(f"Computing {frame_field_type} cross field...")

    if frame_field_type == 'curvature':
        # Import compute_curvature_cross_field if needed
        from FrameField.compute_curvature_cross_field import compute_curvature_cross_field

        omega, ang, sing, kappa, Curv = compute_curvature_cross_field(Src, param, dec, 30, 1e-1)

        # % Target aspect ratio
        # weight.aspect_ratio = ((abs(kappa(:,1)) + 1e-5)./(abs(kappa(:,2)) + 1e-5));
        # t = exp(5);
        # weight.aspect_ratio = max(min(weight.aspect_ratio, t), 1/t);

        weight.aspect_ratio = (np.abs(kappa[:, 0]) + 1e-5) / (np.abs(kappa[:, 1]) + 1e-5)
        t = np.exp(5)
        weight.aspect_ratio = np.clip(weight.aspect_ratio, 1/t, t)

        # % Target direction
        # weight.ang_dir = ang;

        weight.ang_dir = ang.copy()

    elif frame_field_type == 'smooth':
        omega, ang, sing = compute_face_cross_field(Src, param, dec, 10)

    elif frame_field_type == 'trivial':
        # % Vertex singularity index
        # sing = zeros(Src.nv,1);
        # sing(param.idx_bound) = round(2*param.K(param.idx_bound)/pi)/4;

        sing = np.zeros(Src.nv)
        if len(param.idx_bound) > 0:
            # param.K may have extended vertices, use param.Kt for vertex curvature
            sing[param.idx_bound] = np.round(2 * param.Kt[param.idx_bound] / np.pi) / 4

        # % Singularity index of non-contractible cycles
        # om_cycle = param.Icycle*param.para_trans;
        # om_cycle = om_cycle - 2*pi*round(4*om_cycle/(2*pi))/4;

        om_cycle = param.Icycle @ param.para_trans
        om_cycle = om_cycle - 2 * np.pi * np.round(4 * om_cycle / (2 * np.pi)) / 4

        # % Singularity index between disconnected constraints
        # om_link = param.Ilink*param.para_trans;
        # om_link = om_link - 2*pi*round(4*om_link/(2*pi))/4;

        om_link = param.Ilink @ param.para_trans
        om_link = om_link - 2 * np.pi * np.round(4 * om_link / (2 * np.pi)) / 4

        # % Trivial connection
        # [omega,ang,sing] = trivial_connection(Src, param, dec, ifboundary, ifhardedge, sing);

        omega, ang, sing = trivial_connection(Src, param, dec, ifboundary, ifhardedge, sing, om_cycle, om_link)

    else:
        raise ValueError(f'Cross field option unavailable: {frame_field_type}')

    # Count singularities
    id_sing_p = np.sum(sing > 1/8)
    id_sing_m = np.sum(sing < -1/8)

    if verbose:
        print(f"  Positive singularities: {id_sing_p}")
        print(f"  Negative singularities: {id_sing_m}")
        print(f"  Total singularities: {id_sing_p + id_sing_m}")

    # % Plot frame field
    # plot_frame_field(1, Src, param, ang, sing);
    # title('Init frame field');

    if ifplot:
        plot_frame_field(1, Src, param, ang, sing)
        plt.title('Initial Frame Field')
        plt.show()

    # %% Compute cross field jumps and build reduction matrix for v
    # [Edge_jump,v2t,base_tri] = reduce_corner_var_2d(Src);
    # [k21,Reduction] = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t);

    if verbose:
        print("Computing cross field jumps...")

    Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)
    k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

    # %% Optimize integrability condition
    # itmax = 200;
    # ifplot = false;
    # u = zeros(Src.nv,1);
    # v = zeros(Src.nv,1);
    # [u,v,ut,vt,om,angn,flag] = optimize_RSP(omega, ang, u, v, Src, param, dec, Reduction, energy_type, weight, ifplot, itmax);

    if verbose:
        print(f"Optimizing integrability ({energy_type} energy, max {itmax} iterations)...")

    u = np.zeros(Src.nv)
    v = np.zeros(Src.nv)

    result = optimize_RSP(omega, ang, u, v, Src, param, dec, Reduction, energy_type, weight, ifplot, itmax)

    u = result.u
    v = result.v
    ut = result.ut
    vt = result.vt
    om = result.om
    angn = result.angn
    flag = result.flag

    if verbose:
        if flag == 0:
            print(f"  Converged in {result.it} iterations")
        elif flag == 1:
            print(f"  Reached max iterations ({itmax})")
        else:
            print(f"  Line search failed at iteration {result.it}")

    # %% Compute parametrization
    # [SrcCut,dec_cut,Align,Rot] = mesh_to_disk_seamless(Src, param, angn, sing, k21, ifseamless_const, ifboundary, ifhardedge);
    # [Xp,dX] = parametrization_from_scales(Src, SrcCut, dec_cut, param, angn, om, ut, vt, Align, Rot);

    if verbose:
        print("Computing parametrization...")

    SrcCut, dec_cut, Align, Rot = mesh_to_disk_seamless(Src, param, angn, sing, k21, ifseamless_const, ifboundary, ifhardedge)
    Xp, dX = parametrization_from_scales(Src, SrcCut, dec_cut, param, angn, om, ut, vt, Align, Rot)

    if verbose:
        print(f"  Cut mesh vertices: {SrcCut.nv}, faces: {SrcCut.nf}")

    # %% Extract distortion metrics
    # disto = extract_scale_from_param(Xp, Src.X, Src.T, param, SrcCut.T, angn);
    # curl_dX = sqrt(sum((dec_cut.d1p*dX).^2,2))./Src.area;

    disto, ut_out, theta_out, u_tri = extract_scale_from_param(Xp, Src.X, Src.T, param, SrcCut.T, angn)

    curl_dX_vec = dec_cut.d1p @ dX
    curl_dX = np.sqrt(np.sum(curl_dX_vec ** 2, axis=1)) / Src.area

    # Count flipped triangles
    n_flipped = np.sum(disto.detJ <= 0)

    if verbose:
        print(f"  Flipped triangles: {n_flipped}")
        print(f"  Max integrability error: {np.max(curl_dX):.2e}")

    # Save visualization PNGs
    if args.save_viz:
        if verbose:
            print(f"Saving visualizations to {path_save}...")
        visualize_run_RSP_result(Src, SrcCut, Xp, disto, output_dir=path_save)

    # %% Plot results
    if ifplot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # subplot(2,2,1): Integrability
        ax = axes[0, 0]
        ax.tripcolor(SrcCut.X[:, 0], SrcCut.X[:, 1], SrcCut.T,
                     np.log10(curl_dX + 1e-16), shading='flat', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title('Integrability (log10)')
        plt.colorbar(ax.collections[0], ax=ax)

        # subplot(2,2,2): Param
        ax = axes[0, 1]
        ax.tripcolor(Xp[:, 0], Xp[:, 1], SrcCut.T,
                     np.arange(SrcCut.nf), shading='flat', cmap='tab20')
        ax.set_aspect('equal')
        ax.set_title('Parametrization')

        # subplot(2,2,3): log area
        ax = axes[1, 0]
        ax.tripcolor(SrcCut.X[:, 0], SrcCut.X[:, 1], SrcCut.T,
                     np.log10(disto.area + 1e-16), shading='flat', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title('log10(area)')
        plt.colorbar(ax.collections[0], ax=ax)

        # subplot(2,2,4): log conformal
        ax = axes[1, 1]
        ax.tripcolor(SrcCut.X[:, 0], SrcCut.X[:, 1], SrcCut.T,
                     np.abs(np.log10(disto.conf + 1e-16)), shading='flat', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title('|log10(conformal)|')
        plt.colorbar(ax.collections[0], ax=ax)

        plt.tight_layout()
        plt.show()

        # % Plot singularities
        # col = zeros(Src.nf,1); col(disto.detJ <= 0) = 1;
        # id_sing_p = sing > 1/8;
        # id_sing_m = sing <-1/8;

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        col = np.zeros(Src.nf)
        col[disto.detJ <= 0] = 1

        ax.plot_trisurf(Src.X[:, 0], Src.X[:, 1], Src.X[:, 2], triangles=Src.T,
                        cmap='coolwarm', edgecolor='none', alpha=0.7)

        id_sing_p_mask = sing > 1/8
        id_sing_m_mask = sing < -1/8

        if np.any(id_sing_p_mask):
            ax.scatter(Src.X[id_sing_p_mask, 0], Src.X[id_sing_p_mask, 1], Src.X[id_sing_p_mask, 2],
                       c='red', s=100, marker='o', label='Positive singularity')
        if np.any(id_sing_m_mask):
            ax.scatter(Src.X[id_sing_m_mask, 0], Src.X[id_sing_m_mask, 1], Src.X[id_sing_m_mask, 2],
                       c='blue', s=100, marker='o', label='Negative singularity')

        n_sing = np.sum(id_sing_p_mask) + np.sum(id_sing_m_mask)
        ax.set_title(f'{n_sing} singularities')
        ax.legend()
        plt.show()

    # %% Save mesh
    # % Rotate UVs by 45 deg in case of Chebyshev net
    # if contains(energy_type, 'cheby')
    #     r = [1,1;-1,1]*(sqrt(2)/2);
    #     UV = Xp*r;
    # else
    #     UV = Xp;
    # end

    if 'cheby' in energy_type:
        # Rotate by 45 degrees
        r = np.array([[1, 1], [-1, 1]]) * (np.sqrt(2) / 2)
        UV = Xp @ r
    else:
        UV = Xp

    # % Desactivate quantization in known case of failure
    # if ifquantization && ~isempty(param.ide_bound) && ~ifboundary
    #     warning('Boundary alignment is disactivated.');
    #     warning('The quantization step does not support free boundaries.');
    #     ifquantization = false;
    # end

    if ifquantization and len(param.ide_bound) > 0 and not ifboundary:
        print("Warning: Boundary alignment is deactivated.")
        print("Warning: The quantization step does not support free boundaries.")
        ifquantization = False

    # if ifquantization && ~ifseamless_const
    #     warning('The seamlessness constraint is disactivated.');
    #     warning('The quantization needs an exact seamless map as input.');
    #     ifquantization = false;
    # end

    if ifquantization and not ifseamless_const:
        print("Warning: The seamlessness constraint is deactivated.")
        print("Warning: The quantization needs an exact seamless map as input.")
        ifquantization = False

    # if ifquantization && any(disto.detJ < 0)
    #     warning('The parametrization is not locally injective.');
    #     warning('The quantization step will fail.');
    #     ifquantization = false;
    # end

    if ifquantization and np.any(disto.detJ < 0):
        print("Warning: The parametrization is not locally injective.")
        print("Warning: The quantization step will fail.")
        ifquantization = False

    # % Save parametrization
    # save_param(ifquantization, path_save, mesh_name, sqrt(area_tot)*Src.X, Src.T, UV, SrcCut.T, sing, Src.E2V(param.ide_hard,:));

    if verbose:
        print(f"Saving parametrization to {path_save}...")

    # Rescale X back to original scale
    X_original = scale_factor * Src.X

    # Get hard edge vertices (may be empty)
    if len(param.ide_hard) > 0:
        E2V_hardedge = Src.E2V[param.ide_hard, :]
    else:
        E2V_hardedge = np.zeros((0, 2), dtype=int)

    save_param(ifquantization, path_save, mesh_name, X_original, Src.T, UV, SrcCut.T, sing, E2V_hardedge)

    if verbose:
        print("Done!")
        print(f"Output files:")
        print(f"  {path_save}{mesh_name}_param.obj")
        print(f"  {path_save}{mesh_name}_pos.obj (positive singularities)")
        print(f"  {path_save}{mesh_name}_neg.obj (negative singularities)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
