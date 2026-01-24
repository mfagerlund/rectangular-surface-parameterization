#!/usr/bin/env python3
"""
Script to rename attribute accesses in Python files.

Usage:
    python scripts/rename_attribute.py --old nv --new num_vertices --dry-run
    python scripts/rename_attribute.py --old nv --new num_vertices
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files, excluding __pycache__ and venv."""
    files = []
    for f in root.rglob("*.py"):
        if "__pycache__" in str(f) or "venv" in str(f) or ".venv" in str(f):
            continue
        files.append(f)
    return sorted(files)


def rename_attribute(
    filepath: Path,
    old_attr: str,
    new_attr: str,
    dry_run: bool = True
) -> Tuple[int, List[str]]:
    """
    Rename attribute access .old_attr to .new_attr

    Matches patterns like:
        mesh.num_vertices -> mesh.num_vertices
        Src.num_vertices -> Src.num_vertices

    Does NOT match:
        nv = 10  (standalone variable)
    """
    content = filepath.read_text(encoding="utf-8")

    # Match .attr followed by word boundary (not another letter/digit)
    # This ensures .num_vertices matches but .nv_something doesn't
    regex = re.compile(rf'\.{re.escape(old_attr)}\b')

    # Find all matches for counting
    matches = list(regex.finditer(content))
    count = len(matches)

    if count == 0:
        return 0, []

    # Get preview of changes (first 5 unique lines)
    changed_lines = []
    for match in matches[:10]:
        start = content.rfind('\n', 0, match.start()) + 1
        end = content.find('\n', match.end())
        if end == -1:
            end = len(content)
        line = content[start:end].strip()
        if line not in changed_lines:
            changed_lines.append(line)
        if len(changed_lines) >= 5:
            break

    if not dry_run:
        new_content = regex.sub(f'.{new_attr}', content)
        filepath.write_text(new_content, encoding="utf-8")

    return count, changed_lines


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rename attributes in Python files")
    parser.add_argument("--old", required=True, help="Old attribute name")
    parser.add_argument("--new", required=True, help="New attribute name")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--path", default=".", help="Root path to search")
    args = parser.parse_args()

    root = Path(args.path)
    files = find_python_files(root)

    total_count = 0
    files_changed = 0

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Renaming '.{args.old}' -> '.{args.new}'")
    print(f"Searching {len(files)} Python files...\n")

    for filepath in files:
        count, preview = rename_attribute(
            filepath,
            args.old,
            args.new,
            dry_run=args.dry_run
        )
        if count > 0:
            rel_path = filepath.relative_to(root)
            print(f"{rel_path}: {count} occurrences")
            for line in preview[:3]:
                print(f"    {line[:80]}...")
            total_count += count
            files_changed += 1

    print(f"\n{'Would change' if args.dry_run else 'Changed'}: {total_count} occurrences in {files_changed} files")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
