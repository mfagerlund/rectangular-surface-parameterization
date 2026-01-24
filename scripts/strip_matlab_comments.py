#!/usr/bin/env python3
"""
Strip MATLAB comments from Python files and add reference header.

This script removes:
- Lines starting with `# %` (MATLAB comment markers)
- `# === ISSUES ===` ... `# === END ISSUES ===` blocks

And adds a header referencing the original line-by-line translation commit.

Usage:
    python scripts/strip_matlab_comments.py --dry-run  # Preview changes
    python scripts/strip_matlab_comments.py            # Apply changes
"""

import re
import sys
from pathlib import Path

# Reference commit with line-by-line MATLAB translation
REFERENCE_COMMIT = "7d1aab4"
REPO_URL = "https://github.com/mfagerlund/rectangular-surface-parameterization"

HEADER_COMMENT = f"""\
# For the original line-by-line MATLAB translation with interleaved comments,
# see commit {REFERENCE_COMMIT} or {REPO_URL}/tree/{REFERENCE_COMMIT}
"""

# Directories to process
SOURCE_DIRS = ["Preprocess", "FrameField", "Orthotropic", "ComputeParam", "Utils"]

# Files to process at root level
ROOT_FILES = ["run_RSP.py", "extract_quads.py"]


def has_matlab_comments(content: str) -> bool:
    """Check if file has MATLAB comments."""
    return bool(re.search(r'^# %', content, re.MULTILINE) or
                re.search(r'^# === ISSUES ===', content, re.MULTILINE))


def strip_matlab_comments(content: str) -> str:
    """Remove MATLAB comments and ISSUES blocks."""
    lines = content.split('\n')
    result = []
    in_issues_block = False

    for line in lines:
        # Track ISSUES blocks
        if re.match(r'^# === ISSUES ===', line):
            in_issues_block = True
            continue
        if re.match(r'^# === END ISSUES ===', line):
            in_issues_block = False
            continue

        # Skip lines inside ISSUES block
        if in_issues_block:
            continue

        # Skip MATLAB comment lines (# % ...)
        if re.match(r'^(\s*)# %', line):
            continue

        # Skip standalone MATLAB comments that might have extra spacing
        if re.match(r'^\s*#\s*%', line):
            continue

        result.append(line)

    # Clean up excessive blank lines (more than 2 in a row)
    cleaned = '\n'.join(result)
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)

    return cleaned


def add_header(content: str) -> str:
    """Add reference header after initial docstring/imports."""
    lines = content.split('\n')

    # Check if header already exists
    if REFERENCE_COMMIT in content:
        return content

    # Find insertion point: after shebang, docstring, and initial comments
    insert_idx = 0
    in_docstring = False
    docstring_char = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip shebang
        if i == 0 and stripped.startswith('#!'):
            insert_idx = i + 1
            continue

        # Handle docstrings
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2 and len(stripped) > 3:
                    # Single-line docstring
                    insert_idx = i + 1
                else:
                    in_docstring = True
                continue
        else:
            if docstring_char in stripped:
                in_docstring = False
                insert_idx = i + 1
            continue

        # Skip initial comments (but not code)
        if stripped.startswith('#') or stripped == '':
            insert_idx = i + 1
            continue

        # Found first real code line
        break

    # Insert header
    lines.insert(insert_idx, '')
    lines.insert(insert_idx + 1, HEADER_COMMENT.rstrip())
    lines.insert(insert_idx + 2, '')

    return '\n'.join(lines)


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single file. Returns True if changes were made."""
    content = filepath.read_text(encoding='utf-8')

    if not has_matlab_comments(content):
        return False

    # Strip MATLAB comments
    new_content = strip_matlab_comments(content)

    # Add header
    new_content = add_header(new_content)

    if dry_run:
        print(f"Would modify: {filepath}")
        # Show stats
        old_lines = len(content.split('\n'))
        new_lines = len(new_content.split('\n'))
        print(f"  Lines: {old_lines} -> {new_lines} ({old_lines - new_lines} removed)")
    else:
        filepath.write_text(new_content, encoding='utf-8')
        print(f"Modified: {filepath}")

    return True


def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("=== DRY RUN MODE ===\n")

    root = Path(__file__).parent.parent
    modified_count = 0

    # Process source directories
    for dir_name in SOURCE_DIRS:
        dir_path = root / dir_name
        if not dir_path.exists():
            continue
        for py_file in dir_path.glob("*.py"):
            if process_file(py_file, dry_run):
                modified_count += 1

    # Process root files
    for filename in ROOT_FILES:
        filepath = root / filename
        if filepath.exists():
            if process_file(filepath, dry_run):
                modified_count += 1

    print(f"\n{'Would modify' if dry_run else 'Modified'} {modified_count} files")

    if dry_run:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
