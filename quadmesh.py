#!/usr/bin/env python3
"""
QuadMesh - Complete quad meshing pipeline.

Rectangular Surface Parameterization (Corman & Crane, SIGGRAPH 2025)
with libQEx quad extraction.

Usage:
    python quadmesh.py mesh.obj
    python quadmesh.py mesh.obj -o output/ --scale 50
    python quadmesh.py mesh.obj --no-render --no-preview
"""

import click
import numpy as np
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def print_banner():
    """Print welcome banner."""
    click.secho("""
╔═══════════════════════════════════════════════════════════════╗
║  QuadMesh - Rectangular Surface Parameterization Pipeline     ║
║  Corman & Crane (SIGGRAPH 2025) + libQEx                      ║
╚═══════════════════════════════════════════════════════════════╝
""", fg='cyan')


def status(msg, **kwargs):
    click.secho(f"  → {msg}", **kwargs)


def success(msg):
    click.secho(f"  ✓ {msg}", fg='green')


def warn(msg):
    click.secho(f"  ! {msg}", fg='yellow')


def error(msg):
    click.secho(f"  ✗ {msg}", fg='red')


@click.command()
@click.argument('mesh', type=click.Path(exists=True))
@click.option('-o', '--output', 'output_dir', type=click.Path(), default='Results',
              help='Output directory [default: Results/]')
@click.option('--scale', type=float, default=50.0,
              help='UV scale for quad density (higher = more quads) [default: 50]')
@click.option('--target-faces', type=int, default=None,
              help='Decimate large meshes to this face count [default: auto]')
@click.option('--frame-field', type=click.Choice(['smooth', 'curvature', 'trivial']),
              default='smooth', help='Frame field type [default: smooth]')
@click.option('--no-hardedge', is_flag=True,
              help='Disable hard edge constraints (helps with complex geometry)')
@click.option('--no-preprocess', is_flag=True,
              help='Skip mesh preprocessing (hole filling, cleanup)')
@click.option('--no-quads', is_flag=True,
              help='Skip quad extraction (parameterization only)')
@click.option('--no-render', is_flag=True,
              help='Skip rendering PNG previews')
@click.option('--no-uv-export', is_flag=True,
              help='Skip UV data JSON export')
@click.option('--no-singularities', is_flag=True,
              help='Skip singularity OBJ export (use --no-sing)')
@click.option('-v', '--verbose', is_flag=True,
              help='Verbose output')
@click.option('-q', '--quiet', is_flag=True,
              help='Minimal output (errors only)')
def main(mesh, output_dir, scale, target_faces, frame_field,
         no_hardedge, no_preprocess, no_quads, no_render, no_uv_export,
         no_singularities, verbose, quiet):
    """
    Generate a quad mesh from a triangle mesh.

    MESH is the input triangle mesh (OBJ format).

    \b
    Pipeline stages:
      1. Preprocess    - Clean mesh, fill holes, optionally decimate
      2. Parameterize  - Compute UV coordinates via RSP algorithm
      3. Extract       - Convert UV grid to quad mesh via libQEx
      4. Export        - Save quads, UVs, singularities, renders

    \b
    Output files:
      {name}_clean.obj       - Preprocessed mesh (if preprocessing enabled)
      {name}_param.obj       - Parameterized mesh with UVs
      {name}_quads.obj       - Final quad mesh (+ hole-fill triangles)
      {name}_quads.json      - UV data for reuse (vertices, UVs, quads, tris)
      {name}_pos.obj         - Positive singularities (valence > 4)
      {name}_neg.obj         - Negative singularities (valence < 4)
      {name}_render.png      - 3D preview render
      {name}_uv.png          - UV layout visualization

    \b
    Examples:
      python quadmesh.py bunny.obj
      python quadmesh.py bunny.obj -o output/ --scale 100
      python quadmesh.py large_mesh.obj --target-faces 10000
      python quadmesh.py mesh.obj --no-render --no-singularities
    """
    if not quiet:
        print_banner()

    # Setup paths
    project_root = Path(__file__).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(mesh)
    mesh_name = mesh_path.stem

    outputs = {}
    stats = {
        'input_mesh': str(mesh_path),
        'timestamp': datetime.now().isoformat(),
        'scale': scale,
    }

    try:
        # ─────────────────────────────────────────────────────────────
        # Stage 1: Preprocessing
        # ─────────────────────────────────────────────────────────────
        if not quiet:
            click.secho("\n[1/4] Preprocessing", fg='blue', bold=True)

        working_mesh = mesh_path

        if no_preprocess:
            status("Skipped (--no-preprocess)")
        else:
            try:
                sys.path.insert(0, str(project_root))
                from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh
                import pymeshlab

                # Check mesh size
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(mesh_path))
                n_faces = ms.current_mesh().face_number()
                n_verts = ms.current_mesh().vertex_number()

                if not quiet:
                    status(f"Input: {n_verts:,} vertices, {n_faces:,} faces")

                # Auto-decimate large meshes
                if target_faces is None and n_faces > 20000:
                    target_faces = 10000
                    if not quiet:
                        status(f"Large mesh, decimating to {target_faces:,} faces")

                clean_path = output_dir / f"{mesh_name}_clean.obj"
                preprocess_mesh(str(mesh_path), str(clean_path),
                               target_faces=target_faces, verbose=verbose)

                working_mesh = clean_path
                outputs['clean'] = str(clean_path)
                stats['preprocessed'] = True

                if not quiet:
                    success(f"Saved: {clean_path.name}")

            except ImportError:
                warn("PyMeshLab not installed, skipping preprocessing")
                stats['preprocessed'] = False

        # ─────────────────────────────────────────────────────────────
        # Stage 2: Parameterization (RSP)
        # ─────────────────────────────────────────────────────────────
        if not quiet:
            click.secho("\n[2/4] Parameterization (RSP)", fg='blue', bold=True)
            status("Running RSP algorithm...")

        # Build run_RSP.py command
        rsp_cmd = [
            sys.executable, str(project_root / "run_RSP.py"),
            str(working_mesh),
            "-o", str(output_dir),
            "--frame-field", frame_field,
        ]
        if no_hardedge:
            rsp_cmd.append("--no-hardedge")
        if verbose:
            rsp_cmd.append("-v")
        if not no_render:
            rsp_cmd.append("--save-viz")

        result = subprocess.run(rsp_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            error("RSP parameterization failed")
            if verbose:
                click.echo(result.stderr)
            return 1

        if verbose:
            click.echo(result.stdout)

        # Find param file
        param_name = working_mesh.stem
        param_path = output_dir / f"{param_name}_param.obj"
        if not param_path.exists():
            error(f"Param file not found: {param_path}")
            return 1

        outputs['param'] = str(param_path)

        # Singularity files
        pos_path = output_dir / f"{param_name}_pos.obj"
        neg_path = output_dir / f"{param_name}_neg.obj"
        if pos_path.exists():
            outputs['singularities_pos'] = str(pos_path)
        if neg_path.exists():
            outputs['singularities_neg'] = str(neg_path)

        if not quiet:
            success(f"Saved: {param_path.name}")

        # ─────────────────────────────────────────────────────────────
        # Stage 3: Quad Extraction (libQEx)
        # ─────────────────────────────────────────────────────────────
        if not no_quads:
            if not quiet:
                click.secho("\n[3/4] Quad Extraction (libQEx)", fg='blue', bold=True)

            sys.path.insert(0, str(project_root))
            from rectangular_surface_parameterization.io.read_obj import readOBJ
            from rectangular_surface_parameterization.utils.libqex_wrapper import extract_quads, save_quad_obj

            # Load parameterized mesh
            V, F, UV, TF, *_ = readOBJ(str(param_path))

            if UV.shape[0] == 0:
                error("No UV coordinates in param file")
                return 1

            # Build per-triangle UVs
            n_tris = F.shape[0]
            uv_per_tri = np.zeros((n_tris, 3, 2), dtype=np.float64)
            for i in range(n_tris):
                for j in range(3):
                    uv_per_tri[i, j, :] = UV[TF[i, j], :2]

            uv_per_tri *= scale

            if not quiet:
                status(f"Extracting quads (scale={scale})...")

            try:
                quad_verts, quad_faces, tri_faces = extract_quads(
                    V, F, uv_per_tri, verbose=verbose
                )

                n_quads = len(quad_faces)
                n_tris_fill = len(tri_faces) if tri_faces is not None else 0

                stats['quads'] = n_quads
                stats['hole_fill_tris'] = n_tris_fill

                if n_quads == 0:
                    warn("No quads extracted - try increasing --scale")
                else:
                    # Save quad mesh
                    quads_path = output_dir / f"{mesh_name}_quads.obj"
                    save_quad_obj(quads_path, quad_verts, quad_faces, tri_faces)
                    outputs['quads'] = str(quads_path)

                    if not quiet:
                        success(f"Saved: {quads_path.name} ({n_quads:,} quads, {n_tris_fill} hole-fill tris)")

                    # Export UV data as JSON
                    if not no_uv_export:
                        json_path = output_dir / f"{mesh_name}_quads.json"
                        uv_data = {
                            'format': 'quadmesh_uv_v1',
                            'description': 'Quad mesh with UV parameterization data',
                            'generator': 'Corman-Crane RSP (SIGGRAPH 2025) + libQEx',
                            'source_mesh': str(mesh_path),
                            'scale': scale,
                            'quad_vertices': quad_verts.tolist(),
                            'quad_faces': [list(map(int, q)) for q in quad_faces],
                            'hole_fill_triangles': [list(map(int, t)) for t in tri_faces] if tri_faces is not None else [],
                            'parameterization': {
                                'vertices_3d': V.tolist(),
                                'triangles': F.tolist(),
                                'uv_coordinates': UV.tolist(),
                                'uv_triangle_indices': TF.tolist(),
                            },
                            'stats': stats
                        }
                        with open(json_path, 'w') as f:
                            json.dump(uv_data, f)
                        outputs['uv_json'] = str(json_path)

                        if not quiet:
                            success(f"Saved: {json_path.name}")

            except Exception as e:
                error(f"Quad extraction failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        else:
            if not quiet:
                click.secho("\n[3/4] Quad Extraction", fg='blue', bold=True)
                status("Skipped (--no-quads)")

        # ─────────────────────────────────────────────────────────────
        # Stage 4: Rendering
        # ─────────────────────────────────────────────────────────────
        if not no_render and 'quads' in outputs:
            if not quiet:
                click.secho("\n[4/4] Rendering", fg='blue', bold=True)

            try:
                from rectangular_surface_parameterization.utils.render_quads import read_quad_obj, render_quad_mesh

                render_path = output_dir / f"{mesh_name}_render.png"
                vertices, quads, tris = read_quad_obj(outputs['quads'])
                render_quad_mesh(vertices, quads, tris,
                                title=f"{mesh_name} ({len(quads):,} quads)",
                                output=str(render_path))
                outputs['render'] = str(render_path)

                if not quiet:
                    success(f"Saved: {render_path.name}")

            except Exception as e:
                warn(f"Rendering failed: {e}")

        elif not no_render and not quiet:
            click.secho("\n[4/4] Rendering", fg='blue', bold=True)
            status("Skipped (no quads to render)")

        # ─────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────
        if not quiet:
            click.secho("\n" + "═" * 63, fg='cyan')
            click.secho("  Output Files", fg='cyan', bold=True)
            click.secho("═" * 63, fg='cyan')

            for key, path in outputs.items():
                p = Path(path)
                if p.exists():
                    size = p.stat().st_size
                    if size > 1024 * 1024:
                        size_str = f"{size / 1024 / 1024:.1f} MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    click.echo(f"  {key:20} {p.name:30} {size_str:>10}")

            click.secho("═" * 63 + "\n", fg='cyan')

        return 0

    except Exception as e:
        error(f"Pipeline failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
