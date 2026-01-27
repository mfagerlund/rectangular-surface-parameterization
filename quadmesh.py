#!/usr/bin/env python3
"""
QuadMesh - Complete quad meshing pipeline.

Rectangular Surface Parameterization (Corman & Crane, SIGGRAPH 2025)
with optional UV quantization (Coudert-Osmont et al., 2024) and libQEx quad extraction.

Usage:
    python quadmesh.py mesh.obj
    python quadmesh.py mesh.obj -o output/ --scale 50
    python quadmesh.py mesh.obj --no-quantize   # skip quantization
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
+---------------------------------------------------------------+
|  QuadMesh - Rectangular Surface Parameterization Pipeline      |
|  Corman & Crane (SIGGRAPH 2025) + Quantization + libQEx        |
+---------------------------------------------------------------+
""", fg='cyan')


def status(msg, **kwargs):
    click.secho(f"  > {msg}", **kwargs)


def success(msg):
    click.secho(f"  OK {msg}", fg='green')


def warn(msg):
    click.secho(f"  ! {msg}", fg='yellow')


def error(msg):
    click.secho(f"  X {msg}", fg='red')


def load_mesh_with_uvs(obj_path):
    """Load OBJ file and return vertices, triangles, UV coords, and UV face indices."""
    vertices = []
    uvs = []
    faces_v = []
    faces_uv = []

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt' and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f' and len(parts) >= 4:
                fv = []
                fuv = []
                for p in parts[1:4]:
                    indices = p.split('/')
                    fv.append(int(indices[0]) - 1)
                    fuv.append(int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1)
                faces_v.append(fv)
                faces_uv.append(fuv)

    V = np.array(vertices, dtype=np.float64)
    T = np.array(faces_v, dtype=np.int32)
    UV = np.array(uvs, dtype=np.float64)
    TUV = np.array(faces_uv, dtype=np.int32)
    return V, T, UV, TUV


def load_feature_edges_from_obj(obj_path):
    """Load feature/hard edges from OBJ 'l' lines."""
    edges = []
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'l' and len(parts) >= 3:
                edges.append([int(parts[1]) - 1, int(parts[2]) - 1])
    if edges:
        return np.array(edges, dtype=np.int32)
    return np.zeros((0, 2), dtype=np.int32)


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
              help='Disable hard edge constraints')
@click.option('--no-preprocess', is_flag=True,
              help='Skip mesh preprocessing')
@click.option('--no-quantize', is_flag=True,
              help='Skip UV quantization (use raw RSP UVs directly)')
@click.option('--quantize-mode', type=click.Choice(['reembed', 'imprint', 'decimate']),
              default='reembed', help='Quantization output mode [default: reembed]')
@click.option('--quantize-scale', type=float, default=-1.0,
              help='Quantization scale (-1 for auto) [default: -1]')
@click.option('--quantize-scale-auto', type=float, default=1.0,
              help='Multiplier for auto quantization scale [default: 1.0]')
@click.option('--no-quads', is_flag=True,
              help='Skip quad extraction (parameterization only)')
@click.option('--no-render', is_flag=True,
              help='Skip rendering PNG previews')
@click.option('--no-uv-export', is_flag=True,
              help='Skip UV data JSON export')
@click.option('--no-singularities', is_flag=True,
              help='Skip singularity OBJ export')
@click.option('-v', '--verbose', is_flag=True,
              help='Verbose output')
@click.option('-q', '--quiet', is_flag=True,
              help='Minimal output (errors only)')
def main(mesh, output_dir, scale, target_faces, frame_field,
         no_hardedge, no_preprocess, no_quantize, quantize_mode,
         quantize_scale, quantize_scale_auto,
         no_quads, no_render, no_uv_export,
         no_singularities, verbose, quiet):
    """
    Generate a quad mesh from a triangle mesh.

    MESH is the input triangle mesh (OBJ format).

    \b
    Pipeline stages:
      1. Preprocess    - Clean mesh, fill holes, optionally decimate
      2. Parameterize  - Compute UV coordinates via RSP algorithm
      3. Quantize      - Snap UVs to integer grid (pyquantization)
      4. Extract       - Convert UV grid to quad mesh via libQEx
      5. Export        - Save quads, UVs, singularities, renders

    \b
    Output files:
      {name}_clean.obj       - Preprocessed mesh (if preprocessing enabled)
      {name}_param.obj       - Parameterized mesh with UVs
      {name}_quantiz.obj     - Quantized parameterization (integer UVs)
      {name}_quads.obj       - Final quad mesh (from quantized UVs if available)
      {name}_quads_raw.obj   - Quad mesh from raw UVs (for comparison)
      {name}_render.png      - 3D preview render
      {name}_uv.png          - UV layout visualization

    \b
    Examples:
      python quadmesh.py bunny.obj
      python quadmesh.py bunny.obj -o output/ --scale 100
      python quadmesh.py mesh.obj --no-quantize   # raw UVs only
      python quadmesh.py mesh.obj --quantize-mode imprint
    """
    if not quiet:
        print_banner()

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

    n_stages = 5
    stage = 0

    try:
        # =============================================================
        # Stage 1: Preprocessing
        # =============================================================
        stage += 1
        if not quiet:
            click.secho(f"\n[{stage}/{n_stages}] Preprocessing", fg='blue', bold=True)

        working_mesh = mesh_path

        if no_preprocess:
            status("Skipped (--no-preprocess)")
        else:
            try:
                sys.path.insert(0, str(project_root))
                from rectangular_surface_parameterization.utils.preprocess_mesh import preprocess_mesh
                import pymeshlab

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(str(mesh_path))
                n_faces = ms.current_mesh().face_number()
                n_verts = ms.current_mesh().vertex_number()

                if not quiet:
                    status(f"Input: {n_verts:,} vertices, {n_faces:,} faces")

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

        # =============================================================
        # Stage 2: Parameterization (RSP)
        # =============================================================
        stage += 1
        if not quiet:
            click.secho(f"\n[{stage}/{n_stages}] Parameterization (RSP)", fg='blue', bold=True)
            status("Running RSP algorithm...")

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

        result = subprocess.run(rsp_cmd, capture_output=not verbose, text=True)

        if result.returncode != 0:
            error("RSP parameterization failed")
            if not verbose and result.stderr:
                click.echo(result.stderr)
            return 1

        if verbose and result.stdout:
            click.echo(result.stdout)

        param_name = working_mesh.stem
        param_path = output_dir / f"{param_name}_param.obj"
        if not param_path.exists():
            error(f"Param file not found: {param_path}")
            return 1

        outputs['param'] = str(param_path)
        if not quiet:
            success(f"Saved: {param_path.name}")

        # =============================================================
        # Stage 3: Quantization (pyquantization)
        # =============================================================
        stage += 1
        quantiz_path = None

        if not quiet:
            click.secho(f"\n[{stage}/{n_stages}] UV Quantization", fg='blue', bold=True)

        if no_quantize:
            status("Skipped (--no-quantize)")
        else:
            try:
                from rectangular_surface_parameterization.utils.quantization_wrapper import quantize_mesh
                from rectangular_surface_parameterization.io.write_obj import writeObj

                status(f"Quantizing UVs (mode={quantize_mode})...")

                V, T, UV, TUV = load_mesh_with_uvs(str(param_path))
                feature_edges = load_feature_edges_from_obj(str(param_path))

                out_verts, out_faces, out_uvs, out_uv_tris, out_feats = quantize_mesh(
                    V, T, UV, TUV,
                    feature_edges=feature_edges,
                    scale=quantize_scale,
                    scale_auto=quantize_scale_auto,
                    mode=quantize_mode
                )

                quantiz_path = output_dir / f"{param_name}_quantiz.obj"
                writeObj(str(quantiz_path), out_verts, out_faces,
                         out_uvs.reshape(-1, 2), out_uv_tris,
                         None, None,
                         out_feats if len(out_feats) > 0 else None)

                outputs['quantized'] = str(quantiz_path)
                stats['quantized'] = True
                stats['quantize_mode'] = quantize_mode

                if not quiet:
                    success(f"Saved: {quantiz_path.name} "
                            f"({out_verts.shape[0]} verts, {out_faces.shape[0]} faces)")

            except ImportError:
                warn("pyquantization not installed. Skipping quantization.")
                warn("Install with: pip install pyquantization")
                stats['quantized'] = False
            except Exception as e:
                warn(f"Quantization failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                stats['quantized'] = False

        # =============================================================
        # Stage 4: Quad Extraction (libQEx)
        # =============================================================
        stage += 1
        if not quiet:
            click.secho(f"\n[{stage}/{n_stages}] Quad Extraction (libQEx)", fg='blue', bold=True)

        if no_quads:
            status("Skipped (--no-quads)")
        else:
            sys.path.insert(0, str(project_root))
            from rectangular_surface_parameterization.utils.libqex_wrapper import extract_quads, save_quad_obj

            # Extract from quantized mesh (primary) if available
            # Quantized UVs are already integer-grid aligned, so use scale=1
            # (the --scale flag only applies to raw UV extraction)
            if quantiz_path and quantiz_path.exists():
                quant_extract_scale = 1.0
                status(f"Extracting quads from QUANTIZED UVs (scale={quant_extract_scale})...")
                try:
                    V_q, T_q, UV_q, TUV_q = load_mesh_with_uvs(str(quantiz_path))
                    n_tris = T_q.shape[0]
                    uv_per_tri_q = np.zeros((n_tris, 3, 2), dtype=np.float64)
                    for i in range(n_tris):
                        for j in range(3):
                            uv_per_tri_q[i, j, :] = UV_q[TUV_q[i, j], :2]
                    uv_per_tri_q *= quant_extract_scale

                    quad_verts, quad_faces, tri_faces = extract_quads(
                        V_q, T_q, uv_per_tri_q, verbose=verbose
                    )
                    n_quads = len(quad_faces)
                    n_tris_fill = len(tri_faces) if tri_faces is not None else 0

                    if n_quads > 0:
                        quads_path = output_dir / f"{mesh_name}_quads.obj"
                        save_quad_obj(quads_path, quad_verts, quad_faces, tri_faces)
                        outputs['quads'] = str(quads_path)
                        stats['quads'] = n_quads
                        stats['quads_source'] = 'quantized'
                        if not quiet:
                            success(f"Saved: {quads_path.name} "
                                    f"({n_quads:,} quads, {n_tris_fill} hole-fill tris) [from quantized]")
                    else:
                        warn("No quads from quantized UVs - try increasing --scale")
                except Exception as e:
                    warn(f"Quad extraction from quantized mesh failed: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()

            # Also extract from raw UVs for comparison (or as primary if no quantization)
            status(f"Extracting quads from RAW UVs (scale={scale})...")
            try:
                V_r, T_r, UV_r, TUV_r = load_mesh_with_uvs(str(param_path))
                n_tris = T_r.shape[0]
                uv_per_tri_r = np.zeros((n_tris, 3, 2), dtype=np.float64)
                for i in range(n_tris):
                    for j in range(3):
                        uv_per_tri_r[i, j, :] = UV_r[TUV_r[i, j], :2]
                uv_per_tri_r *= scale

                quad_verts_r, quad_faces_r, tri_faces_r = extract_quads(
                    V_r, T_r, uv_per_tri_r, verbose=verbose
                )
                n_quads_r = len(quad_faces_r)
                n_tris_fill_r = len(tri_faces_r) if tri_faces_r is not None else 0

                if n_quads_r > 0:
                    if quantiz_path and quantiz_path.exists():
                        raw_quads_path = output_dir / f"{mesh_name}_quads_raw.obj"
                    else:
                        raw_quads_path = output_dir / f"{mesh_name}_quads.obj"
                    save_quad_obj(raw_quads_path, quad_verts_r, quad_faces_r, tri_faces_r)
                    outputs['quads_raw'] = str(raw_quads_path)
                    stats['quads_raw'] = n_quads_r
                    if not quiet:
                        label = "[from raw UVs]" if quantiz_path else ""
                        success(f"Saved: {raw_quads_path.name} "
                                f"({n_quads_r:,} quads, {n_tris_fill_r} hole-fill tris) {label}")

                    # If no quantized quads, use raw as primary
                    if 'quads' not in outputs:
                        outputs['quads'] = str(raw_quads_path)
                        stats['quads'] = n_quads_r
                        stats['quads_source'] = 'raw'
                else:
                    warn("No quads from raw UVs - try increasing --scale")
            except Exception as e:
                warn(f"Quad extraction from raw UVs failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

            # UV data export
            if not no_uv_export and 'quads' in outputs:
                json_path = output_dir / f"{mesh_name}_quads.json"
                uv_data = {
                    'format': 'quadmesh_uv_v2',
                    'generator': 'Corman-Crane RSP + pyquantization + libQEx',
                    'source_mesh': str(mesh_path),
                    'scale': scale,
                    'quantized': stats.get('quantized', False),
                    'stats': stats
                }
                with open(json_path, 'w') as f:
                    json.dump(uv_data, f, indent=2)
                outputs['uv_json'] = str(json_path)
                if not quiet:
                    success(f"Saved: {json_path.name}")

        # =============================================================
        # Stage 5: Rendering
        # =============================================================
        stage += 1
        if not quiet:
            click.secho(f"\n[{stage}/{n_stages}] Rendering", fg='blue', bold=True)

        if no_render:
            status("Skipped (--no-render)")
        elif 'quads' in outputs:
            try:
                from rectangular_surface_parameterization.utils.render_quads import read_quad_obj, render_quad_mesh

                render_path = output_dir / f"{mesh_name}_render.png"
                vertices, quads, tris = read_quad_obj(outputs['quads'])
                src = "quantized" if stats.get('quads_source') == 'quantized' else 'raw'
                render_quad_mesh(vertices, quads, tris,
                                title=f"{mesh_name} ({len(quads):,} quads, {src})",
                                output=str(render_path))
                outputs['render'] = str(render_path)
                if not quiet:
                    success(f"Saved: {render_path.name}")

                # Also render raw quads for comparison if both exist
                if 'quads_raw' in outputs and outputs.get('quads_raw') != outputs.get('quads'):
                    render_raw_path = output_dir / f"{mesh_name}_render_raw.png"
                    verts_r, quads_r, tris_r = read_quad_obj(outputs['quads_raw'])
                    render_quad_mesh(verts_r, quads_r, tris_r,
                                    title=f"{mesh_name} ({len(quads_r):,} quads, raw)",
                                    output=str(render_raw_path))
                    outputs['render_raw'] = str(render_raw_path)
                    if not quiet:
                        success(f"Saved: {render_raw_path.name}")

            except Exception as e:
                warn(f"Rendering failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        else:
            status("Skipped (no quads to render)")

        # =============================================================
        # Summary
        # =============================================================
        if not quiet:
            click.secho("\n" + "=" * 63, fg='cyan')
            click.secho("  Output Files", fg='cyan', bold=True)
            click.secho("=" * 63, fg='cyan')

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

            if 'quads' in stats and 'quads_raw' in stats:
                click.secho(f"\n  Comparison: {stats['quads']} quads (quantized) vs "
                            f"{stats['quads_raw']} quads (raw)", fg='yellow')

            click.secho("=" * 63 + "\n", fg='cyan')

        return 0

    except Exception as e:
        error(f"Pipeline failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
