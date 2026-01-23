#!/usr/bin/env python3
"""
Convert MATLAB .m files to .py files with all lines commented out.

Usage:
    python convert_matlab.py <relative_path_to_file>
    python convert_matlab.py ComputeParam/cut_mesh.m
    python convert_matlab.py --all  # Convert all .m files

The script:
1. Reads from C:/Slask/RectangularSurfaceParameterization/<path>
2. Writes to C:/Dev/Corman-Crane/<path> (with .py extension)
3. Prefixes every line with "# " to comment it out
"""

import sys
import os
from pathlib import Path

MATLAB_ROOT = Path("C:/Slask/RectangularSurfaceParameterization")
OUTPUT_ROOT = Path("C:/Dev/Corman-Crane")


def convert_file(rel_path: str) -> None:
    """Convert a single MATLAB file to commented Python."""
    # Normalize path
    rel_path = rel_path.replace("\\", "/")

    # Handle .m extension
    if rel_path.endswith(".m"):
        src_rel = rel_path
        dst_rel = rel_path[:-2] + ".py"
    elif rel_path.endswith(".py"):
        src_rel = rel_path[:-3] + ".m"
        dst_rel = rel_path
    else:
        src_rel = rel_path + ".m"
        dst_rel = rel_path + ".py"

    src_path = MATLAB_ROOT / src_rel
    dst_path = OUTPUT_ROOT / dst_rel

    if not src_path.exists():
        print(f"ERROR: Source file not found: {src_path}")
        return

    # Create output directory if needed
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if destination already exists
    if dst_path.exists():
        print(f"WARNING: Destination already exists: {dst_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Skipped.")
            return

    # Read source and comment all lines
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    commented_lines = []
    for line in lines:
        # Preserve line endings but add comment prefix
        if line.endswith("\n"):
            commented_lines.append("# " + line[:-1] + "\n")
        else:
            commented_lines.append("# " + line)

    # Write output
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(commented_lines)

    print(f"Converted: {src_rel} -> {dst_rel}")
    print(f"  {len(lines)} lines commented")


def list_all_matlab_files() -> list:
    """List all .m files in the MATLAB repo."""
    files = []
    for path in MATLAB_ROOT.rglob("*.m"):
        rel = path.relative_to(MATLAB_ROOT)
        files.append(str(rel).replace("\\", "/"))
    return sorted(files)


def convert_all() -> None:
    """Convert all MATLAB files."""
    files = list_all_matlab_files()
    print(f"Found {len(files)} MATLAB files:\n")
    for i, f in enumerate(files, 1):
        print(f"  {i:2}. {f}")
    print()

    response = input("Convert all? [y/N]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    for f in files:
        convert_file(f)

    print(f"\nDone! Converted {len(files)} files.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable MATLAB files:")
        for f in list_all_matlab_files():
            print(f"  {f}")
        return

    arg = sys.argv[1]

    if arg == "--all":
        convert_all()
    elif arg == "--list":
        for f in list_all_matlab_files():
            print(f)
    else:
        convert_file(arg)


if __name__ == "__main__":
    main()
