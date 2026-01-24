#!/usr/bin/env python3
"""
Rename .T to .triangles ONLY for known mesh variable names.
Does NOT touch numpy transpose operations.
"""

import re
from pathlib import Path


# Only rename .T when preceded by these known mesh variable names
MESH_VARS = [
    'mesh', 'Src', 'disk_mesh', 'info', 'SrcCut_raw',
    'tetrahedron', 'cube_triangulated', 'simple_mesh',
    'single_triangle', 'two_triangles'
]


def find_python_files(root: Path):
    files = []
    for f in root.rglob("*.py"):
        if "__pycache__" in str(f) or "venv" in str(f):
            continue
        files.append(f)
    return sorted(files)


def rename_mesh_T(filepath: Path, dry_run: bool = True):
    content = filepath.read_text(encoding="utf-8")
    original = content

    count = 0
    for var in MESH_VARS:
        # Match var.T with word boundary after T (not var.T2E, etc.)
        pattern = re.compile(rf'\b{re.escape(var)}\.T\b')
        matches = pattern.findall(content)
        count += len(matches)
        if not dry_run:
            content = pattern.sub(f'{var}.triangles', content)

    if not dry_run and content != original:
        filepath.write_text(content, encoding="utf-8")

    return count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--path", default=".")
    args = parser.parse_args()

    root = Path(args.path)
    files = find_python_files(root)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Renaming <mesh>.T -> <mesh>.triangles")
    print(f"Mesh variables: {', '.join(MESH_VARS)}")
    print(f"Searching {len(files)} files...\n")

    total = 0
    for f in files:
        count = rename_mesh_T(f, args.dry_run)
        if count > 0:
            print(f"{f.relative_to(root)}: {count}")
            total += count

    print(f"\n{'Would change' if args.dry_run else 'Changed'}: {total} occurrences")


if __name__ == "__main__":
    main()
