#!/usr/bin/env python3
"""
Test the QaWiTM quantization implementation.

Runs quantization on a parameterized mesh and compares quality before/after.
"""

import sys
import os
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def load_param_mesh(obj_path: str):
    """Load a parameterized mesh with UVs."""
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vt':
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == 'f':
                # Parse face with UVs: v1/vt1 v2/vt2 v3/vt3
                face = []
                face_uv = []
                for p in parts[1:]:
                    if '/' in p:
                        v_idx, vt_idx = p.split('/')[:2]
                        face.append(int(v_idx) - 1)
                        if vt_idx:
                            face_uv.append(int(vt_idx) - 1)
                    else:
                        face.append(int(p) - 1)

                if len(face) == 3:
                    faces.append(face)
                    if len(face_uv) == 3:
                        face_uvs.append(face_uv)

    vertices = np.array(vertices)
    uvs = np.array(uvs) if uvs else None
    faces = np.array(faces)

    # Build UV per triangle
    if uvs is not None and face_uvs:
        face_uvs = np.array(face_uvs)
        uv_per_triangle = uvs[face_uvs]
    else:
        uv_per_triangle = None

    return vertices, faces, uv_per_triangle


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test QaWiTM quantization")
    parser.add_argument("mesh", help="Parameterized mesh (_param.obj)")
    parser.add_argument("--scale", type=float, default=10.0, help="Target scale")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    print(f"Loading mesh: {args.mesh}")
    vertices, triangles, uv_per_triangle = load_param_mesh(args.mesh)

    if uv_per_triangle is None:
        print("Error: Mesh has no UV coordinates")
        sys.exit(1)

    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")
    print(f"  UV range: [{uv_per_triangle.min():.2f}, {uv_per_triangle.max():.2f}]")

    # Import quantization
    from rectangular_surface_parameterization.quad_extraction.quantization import (
        quantize_uv
    )

    # Run quantization
    print(f"\nRunning quantization (target scale: {args.scale})...")

    try:
        result = quantize_uv(
            vertices, triangles, uv_per_triangle,
            target_scale=args.scale,
            verbose=args.verbose
        )

        print(f"\n{result}")

    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
