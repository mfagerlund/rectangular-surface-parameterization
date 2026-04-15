

"""
Rectangle Surface Parametrization - Corman and Crane, 2025

Converts MATLAB main entry point (run_RSP.m) to Python CLI.

Usage:
    python run_RSP.py mesh.obj [options]

Example:
    python run_RSP.py Mesh/sphere320.obj -o output/
"""


# For the original line-by-line MATLAB translation with interleaved comments,
# see commit 7d1aab4 or https://github.com/mfagerlund/rectangular-surface-parameterization/tree/7d1aab4

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Import converted modules
from rectangular_surface_parameterization.io.read_obj import readOBJ
from rectangular_surface_parameterization.io.save_param import save_param
from rectangular_surface_parameterization.utils.extract_scale import extract_scale_from_param
from rectangular_surface_parameterization.io.visualize import save_uv_visualization, visualize_run_RSP_result

from rectangular_surface_parameterization.core.mesh_info import MeshInfo, mesh_info
from rectangular_surface_parameterization.preprocessing.dec import dec_tri
from rectangular_surface_parameterization.preprocessing.preprocess import preprocess_ortho_param

from rectangular_surface_parameterization.cross_field.trivial_connection import trivial_connection
from rectangular_surface_parameterization.cross_field.face_field import compute_face_cross_field
from rectangular_surface_parameterization.cross_field.brush_field import brush_frame_field
from rectangular_surface_parameterization.cross_field.plot import plot_frame_field

from rectangular_surface_parameterization.optimization.reduce_corner_var import reduce_corner_var_2d
from rectangular_surface_parameterization.optimization.reduce_corner_var_cut import reduce_corner_var_2d_cut
from rectangular_surface_parameterization.optimization.reduction import reduction_from_ff2d
from rectangular_surface_parameterization.optimization.solver import optimize_RSP

from rectangular_surface_parameterization.parameterization.integrate import parametrization_from_scales
from rectangular_surface_parameterization.parameterization.seamless import mesh_to_disk_seamless

# Stage visualization functions
from rectangular_surface_parameterization.utils.verify_pipeline import (
    verify_geometry, verify_principal_curvature, verify_cross_field, verify_cut_graph,
    verify_optimization, verify_uv_recovery
)


def parse_visualize_stages(viz_arg: str) -> set:
    """Parse --visualize argument into set of stage numbers."""
    if not viz_arg or viz_arg.lower() == 'none':
        return set()
    stages = set()
    for part in viz_arg.split(','):
        part = part.strip()
        if part.isdigit():
            stage = int(part)
            if 1 <= stage <= 5:
                stages.add(stage)
    return stages


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
                        help='Enable integer quantization (requires pyquantization package)')

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
    parser.add_argument('--hard-edges', type=str, default=None,
                        help='File with explicit hard edge vertex pairs (one "v1 v2" per line, 0-indexed). '
                             'These edges will be treated as hard edges in addition to dihedral-detected ones.')

    # Visualization
    parser.add_argument('--plot', action='store_true',
                        help='Show interactive matplotlib plots')
    parser.add_argument('--visualize', type=str, nargs='?', const='1,2,3,4,5', default='1,2,3,4,5',
                        metavar='STAGES',
                        help='Stages to visualize as comma-separated list (default: 1,2,3,4,5 = all). '
                             'Use --visualize "" or --visualize none to disable. '
                             'Stages: 1=geometry, 2=cross_field, 3=cut_graph, 4=optimization, 5=uv_recovery')
    parser.add_argument('--principal-curvature', action='store_true',
                        help='Generate principal curvature visualizations (k1, k2, Gaussian, Mean, directions). Off by default.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse visualization stages
    viz_stages = parse_visualize_stages(args.visualize)

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

    # Load explicit hard edges if provided
    Ehard2V = None
    hard_edge_path = Path(args.hard_edges) if args.hard_edges else None

    path_save = args.output

    # Create output directory if needed
    os.makedirs(path_save, exist_ok=True)

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

    # [X,T] = readOBJ([path_data, mesh_name, '.obj']);

    if verbose:
        print(f"Loading mesh: {mesh_path}")

    X, T, *_ = readOBJ(str(mesh_path))

    if verbose:
        print(f"  Vertices: {X.shape[0]}, Faces: {T.shape[0]}")

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

    # Src = MeshInfo(X, T);
    # dec = dec_tri(Src);
    # [param,Src,dec] = preprocess_ortho_param(Src, dec, ifboundary, ifhardedge, 40);

    if verbose:
        print("Computing mesh connectivity...")

    Src = mesh_info(X, T)
    dec = dec_tri(Src)

    # Resolve position-based hard edges to vertex index pairs
    if hard_edge_path is not None:
        if not hard_edge_path.exists():
            print(f"Error: hard edges file not found: {hard_edge_path}")
            return
        lines = hard_edge_path.read_text().splitlines()
        use_positions = any('# positions' in l for l in lines)

        if use_positions:
            # Position-based format: x1 y1 z1 x2 y2 z2
            from scipy.spatial import cKDTree
            tree = cKDTree(X)
            # Use adaptive tolerance: 50% of mean nearest-neighbor distance
            nn_dists = tree.query(X, k=2)[0][:, 1]  # distance to 2nd nearest (self=0)
            pos_tol = 0.5 * np.mean(nn_dists)
            edges = []
            n_missed = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    p1 = np.array([float(parts[0]), float(parts[1]), float(parts[2])]) / scale_factor
                    p2 = np.array([float(parts[3]), float(parts[4]), float(parts[5])]) / scale_factor
                    d1, i1 = tree.query(p1)
                    d2, i2 = tree.query(p2)
                    if d1 < pos_tol and d2 < pos_tol and i1 != i2:
                        edges.append([i1, i2])
                    else:
                        n_missed += 1
            if edges:
                Ehard2V = np.array(edges, dtype=int)
            if verbose:
                print(f"Loaded {len(edges)} hard edges from positions ({n_missed} unmatched) from {hard_edge_path}")
        else:
            # Index-based format: v1 v2
            edges = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    edges.append([int(parts[0]), int(parts[1])])
            if edges:
                Ehard2V = np.array(edges, dtype=int)
            if verbose:
                print(f"Loaded {len(edges)} hard edges (vertex indices) from {hard_edge_path}")

    if verbose:
        print(f"  Edges: {Src.num_edges}")
        print("Preprocessing for orthotropic parameterization...")

    param, Src, dec = preprocess_ortho_param(Src, dec, ifboundary, ifhardedge, tol_dihedral_deg, Ehard2V=Ehard2V)

    if verbose:
        print(f"  Hard edges: {len(param.ide_hard)}")
        print(f"  Boundary edges: {len(param.ide_bound)}")
        print(f"  Fixed edges: {len(param.ide_fix)}")

    # Stage 1 visualization: Geometry
    if 1 in viz_stages:
        if verbose:
            print("Generating Stage 1 visualizations (geometry)...")
        verify_geometry(Src, Path(path_save))

    # Principal curvature visualization (off by default)
    if args.principal_curvature:
        if verbose:
            print("Generating principal curvature visualizations...")
        verify_principal_curvature(Src, param, Path(path_save))

    # col = zeros(Src.num_vertices,1); col(Src.edge_to_vertex(param.ide_fix,:)) = 1;
    # figure; trisurf(...); title('Constraint')

    if ifplot:
        col = np.zeros(Src.num_vertices)
        if len(param.ide_fix) > 0:
            col[Src.edge_to_vertex[param.ide_fix, :].flatten()] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(Src.vertices[:, 0], Src.vertices[:, 1], Src.vertices[:, 2], triangles=Src.triangles,
                        cmap='coolwarm', edgecolor='k', linewidth=0.1)
        ax.set_title('Constraint Edges')
        plt.show()

    # if strcmp(frame_field_type, 'curvature')
    #     [omega,ang,sing,kappa,Curv] = compute_curvature_cross_field(Src, param, dec, 30, 1e-1);
    #     weight.aspect_ratio = ...
    #     weight.ang_dir = ang;
    # elseif strcmp(frame_field_type, 'smooth')
    #     [omega,ang,sing] = compute_face_cross_field(Src, param, dec, 10);
    # elseif strcmp(frame_field_type, 'trivial')
    #     sing = zeros(Src.num_vertices,1);
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
        from rectangular_surface_parameterization.cross_field.curvature_field import compute_curvature_cross_field

        omega, ang, sing, kappa, Curv = compute_curvature_cross_field(Src, param, dec, 30, 1e-1)

        # weight.aspect_ratio = ((abs(kappa(:,1)) + 1e-5)./(abs(kappa(:,2)) + 1e-5));
        # t = exp(5);
        # weight.aspect_ratio = max(min(weight.aspect_ratio, t), 1/t);

        weight.aspect_ratio = (np.abs(kappa[:, 0]) + 1e-5) / (np.abs(kappa[:, 1]) + 1e-5)
        t = np.exp(5)
        weight.aspect_ratio = np.clip(weight.aspect_ratio, 1/t, t)

        # weight.ang_dir = ang;

        weight.ang_dir = ang.copy()

    elif frame_field_type == 'smooth':
        omega, ang, sing = compute_face_cross_field(Src, param, dec, 10)

    elif frame_field_type == 'trivial':
        # sing = zeros(Src.num_vertices,1);
        # sing(param.idx_bound) = round(2*param.K(param.idx_bound)/pi)/4;

        sing = np.zeros(Src.num_vertices)
        if len(param.idx_bound) > 0:
            # param.K may have extended vertices, use param.Kt for vertex curvature
            sing[param.idx_bound] = np.round(2 * param.Kt[param.idx_bound] / np.pi) / 4

        # om_cycle = param.Icycle*param.para_trans;
        # om_cycle = om_cycle - 2*pi*round(4*om_cycle/(2*pi))/4;

        om_cycle = param.Icycle @ param.para_trans
        om_cycle = om_cycle - 2 * np.pi * np.round(4 * om_cycle / (2 * np.pi)) / 4

        # om_link = param.Ilink*param.para_trans;
        # om_link = om_link - 2*pi*round(4*om_link/(2*pi))/4;

        om_link = param.Ilink @ param.para_trans
        om_link = om_link - 2 * np.pi * np.round(4 * om_link / (2 * np.pi)) / 4

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

    # Stage 2 visualization: Cross Field
    if 2 in viz_stages:
        if verbose:
            print("Generating Stage 2 visualizations (cross field)...")
        verify_cross_field(Src, param, ang, sing, Path(path_save))

    # plot_frame_field(1, Src, param, ang, sing);
    # title('Init frame field');

    if ifplot:
        plot_frame_field(1, Src, param, ang, sing)
        plt.title('Initial Frame Field')
        plt.show()

    # [Edge_jump,v2t,base_tri] = reduce_corner_var_2d(Src);
    # [k21,Reduction] = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t);

    if verbose:
        print("Computing cross field jumps...")

    # Use cut version for meshes with boundaries
    if len(param.ide_bound) > 0:
        # For open meshes, treat boundary edges as cut edges
        Edge_jump, v2t, base_tri = reduce_corner_var_2d_cut(Src, param.ide_bound)
    else:
        Edge_jump, v2t, base_tri = reduce_corner_var_2d(Src)
    k21, Reduction = reduction_from_ff2d(Src, param, ang, omega, Edge_jump, v2t)

    # Stage 3 visualization: Cut Graph
    if 3 in viz_stages:
        if verbose:
            print("Generating Stage 3 visualizations (cut graph)...")
        verify_cut_graph(Src, k21, sing, param, Path(path_save))

    # itmax = 200;
    # ifplot = false;
    # u = zeros(Src.num_vertices,1);
    # v = zeros(Src.num_vertices,1);
    # [u,v,ut,vt,om,angn,flag] = optimize_RSP(omega, ang, u, v, Src, param, dec, Reduction, energy_type, weight, ifplot, itmax);

    if verbose:
        print(f"Optimizing integrability ({energy_type} energy, max {itmax} iterations)...")

    u = np.zeros(Src.num_vertices)
    v = np.zeros(Src.num_vertices)

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

    # Stage 4 visualization: Optimization
    if 4 in viz_stages:
        if verbose:
            print("Generating Stage 4 visualizations (optimization)...")
        verify_optimization(Src, ut, vt, angn, dec, Path(path_save))

    # [disk_mesh,dec_cut,Align,Rot] = mesh_to_disk_seamless(Src, param, angn, sing, k21, ifseamless_const, ifboundary, ifhardedge);
    # [Xp,dX] = parametrization_from_scales(Src, disk_mesh, dec_cut, param, angn, om, ut, vt, Align, Rot);

    if verbose:
        print("Computing parametrization...")

    disk_mesh, dec_cut, Align, Rot = mesh_to_disk_seamless(Src, param, angn, sing, k21, ifseamless_const, ifboundary, ifhardedge)
    Xp, dX = parametrization_from_scales(Src, disk_mesh, dec_cut, param, angn, om, ut, vt, Align, Rot)

    if verbose:
        print(f"  Cut mesh vertices: {disk_mesh.num_vertices}, faces: {disk_mesh.num_faces}")

    # disto = extract_scale_from_param(Xp, Src.vertices, Src.triangles, param, disk_mesh.triangles, angn);
    # curl_dX = sqrt(sum((dec_cut.d1p*dX).^2,2))./Src.area;

    disto, ut_out, theta_out, u_tri = extract_scale_from_param(Xp, Src.vertices, Src.triangles, param, disk_mesh.triangles, angn)

    curl_dX_vec = dec_cut.d1p @ dX
    curl_dX = np.sqrt(np.sum(curl_dX_vec ** 2, axis=1)) / Src.area

    # Count flipped triangles
    n_flipped = np.sum(disto.detJ <= 0)

    if verbose:
        print(f"  Flipped triangles: {n_flipped}")
        print(f"  Max integrability error: {np.max(curl_dX):.2e}")

    # Stage 5 visualization: UV Recovery
    if 5 in viz_stages:
        if verbose:
            print("Generating Stage 5 visualizations (UV recovery)...")
        verify_uv_recovery(Xp, disk_mesh.triangles, disto.detJ, Path(path_save))

    if ifplot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # subplot(2,2,1): Integrability
        ax = axes[0, 0]
        ax.tripcolor(disk_mesh.vertices[:, 0], disk_mesh.vertices[:, 1], disk_mesh.triangles,
                     np.log10(curl_dX + 1e-16), shading='flat', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title('Integrability (log10)')
        plt.colorbar(ax.collections[0], ax=ax)

        # subplot(2,2,2): Param
        ax = axes[0, 1]
        ax.tripcolor(Xp[:, 0], Xp[:, 1], disk_mesh.triangles,
                     np.arange(disk_mesh.num_faces), shading='flat', cmap='tab20')
        ax.set_aspect('equal')
        ax.set_title('Parametrization')

        # subplot(2,2,3): log area
        ax = axes[1, 0]
        ax.tripcolor(disk_mesh.vertices[:, 0], disk_mesh.vertices[:, 1], disk_mesh.triangles,
                     np.log10(disto.area + 1e-16), shading='flat', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title('log10(area)')
        plt.colorbar(ax.collections[0], ax=ax)

        # subplot(2,2,4): log conformal
        ax = axes[1, 1]
        ax.tripcolor(disk_mesh.vertices[:, 0], disk_mesh.vertices[:, 1], disk_mesh.triangles,
                     np.abs(np.log10(disto.conf + 1e-16)), shading='flat', cmap='viridis')
        ax.set_aspect('equal')
        ax.set_title('|log10(conformal)|')
        plt.colorbar(ax.collections[0], ax=ax)

        plt.tight_layout()
        plt.show()

        # col = zeros(Src.num_faces,1); col(disto.detJ <= 0) = 1;
        # id_sing_p = sing > 1/8;
        # id_sing_m = sing <-1/8;

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        col = np.zeros(Src.num_faces)
        col[disto.detJ <= 0] = 1

        ax.plot_trisurf(Src.vertices[:, 0], Src.vertices[:, 1], Src.vertices[:, 2], triangles=Src.triangles,
                        cmap='coolwarm', edgecolor='none', alpha=0.7)

        id_sing_p_mask = sing > 1/8
        id_sing_m_mask = sing < -1/8

        if np.any(id_sing_p_mask):
            ax.scatter(Src.vertices[id_sing_p_mask, 0], Src.vertices[id_sing_p_mask, 1], Src.vertices[id_sing_p_mask, 2],
                       c='red', s=100, marker='o', label='Positive singularity')
        if np.any(id_sing_m_mask):
            ax.scatter(Src.vertices[id_sing_m_mask, 0], Src.vertices[id_sing_m_mask, 1], Src.vertices[id_sing_m_mask, 2],
                       c='blue', s=100, marker='o', label='Negative singularity')

        n_sing = np.sum(id_sing_p_mask) + np.sum(id_sing_m_mask)
        ax.set_title(f'{n_sing} singularities')
        ax.legend()
        plt.show()

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

    # save_param(ifquantization, path_save, mesh_name, sqrt(area_tot)*Src.vertices, Src.triangles, UV, disk_mesh.triangles, sing, Src.edge_to_vertex(param.ide_hard,:));

    if verbose:
        print(f"Saving parametrization to {path_save}...")

    # Rescale X back to original scale
    X_original = scale_factor * Src.vertices

    # Get hard edge vertices (may be empty)
    if len(param.ide_hard) > 0:
        E2V_hardedge = Src.edge_to_vertex[param.ide_hard, :]
    else:
        E2V_hardedge = np.zeros((0, 2), dtype=int)

    save_param(ifquantization, path_save, mesh_name, X_original, Src.triangles, UV, disk_mesh.triangles, sing, E2V_hardedge)

    if verbose:
        print("Done!")
        print(f"Output files:")
        print(f"  {path_save}{mesh_name}_param.obj")
        print(f"  {path_save}{mesh_name}_pos.obj (positive singularities)")
        print(f"  {path_save}{mesh_name}_neg.obj (negative singularities)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
