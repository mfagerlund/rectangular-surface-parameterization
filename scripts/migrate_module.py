#!/usr/bin/env python3
"""
Migrate a Python module to a new location and update all imports.

Usage:
    python scripts/migrate_module.py Preprocess/connectivity.py rectangular_surface_parameterization/preprocessing/connectivity.py --dry-run
    python scripts/migrate_module.py Preprocess/connectivity.py rectangular_surface_parameterization/preprocessing/connectivity.py
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def path_to_module(path: str) -> str:
    """Convert file path to module name."""
    # Remove .py extension and convert slashes to dots
    module = path.replace('\\', '/').replace('/', '.')
    if module.endswith('.py'):
        module = module[:-3]
    return module


def find_python_files(root: str, exclude_dirs: set = None) -> list:
    """Find all Python files in directory."""
    if exclude_dirs is None:
        exclude_dirs = {'__pycache__', '.git', 'venv', '.venv', 'node_modules'}

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Modify dirnames in-place to skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for f in filenames:
            if f.endswith('.py'):
                files.append(os.path.join(dirpath, f))

    return files


def update_imports_in_file(filepath: str, old_module: str, new_module: str, dry_run: bool = True) -> bool:
    """Update imports in a single file. Returns True if changes were made."""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Pattern 1: from old_module import X
    # e.g., from rectangular_surface_parameterization.preprocessing.connectivity import connectivity
    pattern1 = rf'from\s+{re.escape(old_module)}\s+import'
    replacement1 = f'from {new_module} import'
    content = re.sub(pattern1, replacement1, content)

    # Pattern 2: import old_module
    # e.g., import Preprocess.connectivity
    pattern2 = rf'^import\s+{re.escape(old_module)}\s*$'
    replacement2 = f'import {new_module}'
    content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)

    # Pattern 3: from old_parent import module_name
    # e.g., from rectangular_surface_parameterization.preprocessing import connectivity -> from new_parent import connectivity
    old_parts = old_module.rsplit('.', 1)
    new_parts = new_module.rsplit('.', 1)
    if len(old_parts) == 2 and len(new_parts) == 2:
        old_parent, old_name = old_parts
        new_parent, new_name = new_parts
        if old_name == new_name:
            pattern3 = rf'from\s+{re.escape(old_parent)}\s+import\s+{re.escape(old_name)}\b'
            replacement3 = f'from {new_parent} import {new_name}'
            content = re.sub(pattern3, replacement3, content)

    # Pattern 4: Relative imports within same package
    # e.g., from .connectivity import X (if we're in same package)
    # This is more complex and depends on the file's location

    if content != original:
        if dry_run:
            print(f"  Would update: {filepath}")
            # Show diff
            for i, (old_line, new_line) in enumerate(zip(original.split('\n'), content.split('\n'))):
                if old_line != new_line:
                    print(f"    - {old_line}")
                    print(f"    + {new_line}")
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Updated: {filepath}")
        return True

    return False


def migrate_module(source: str, dest: str, dry_run: bool = True) -> None:
    """Migrate a module from source to dest, updating all imports."""

    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')

    old_module = path_to_module(source)
    new_module = path_to_module(dest)

    print(f"Migrating: {old_module} -> {new_module}")
    print(f"  Source: {source}")
    print(f"  Dest:   {dest}")
    print(f"  Dry run: {dry_run}")
    print()

    # Check source exists
    if not os.path.exists(source):
        print(f"ERROR: Source file does not exist: {source}")
        return

    # Find all Python files
    py_files = find_python_files('.')
    print(f"Found {len(py_files)} Python files to check")
    print()

    # Update imports in all files
    updated_count = 0
    for filepath in py_files:
        if update_imports_in_file(filepath, old_module, new_module, dry_run):
            updated_count += 1

    print()
    print(f"Files to update: {updated_count}")

    # Move the file
    if not dry_run:
        # Ensure destination directory exists
        dest_dir = os.path.dirname(dest)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        # Copy file (keep original for now)
        shutil.copy2(source, dest)
        print(f"Copied: {source} -> {dest}")
        print(f"NOTE: Original file kept at {source}. Delete manually after verification.")
    else:
        print(f"Would copy: {source} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description='Migrate a Python module to a new location')
    parser.add_argument('source', help='Source file path (e.g., Preprocess/connectivity.py)')
    parser.add_argument('dest', help='Destination file path (e.g., rectangular_surface_parameterization/preprocessing/connectivity.py)')
    parser.add_argument('--dry-run', action='store_true', default=False, help='Show what would be done without making changes')

    args = parser.parse_args()
    migrate_module(args.source, args.dest, args.dry_run)


if __name__ == '__main__':
    main()
