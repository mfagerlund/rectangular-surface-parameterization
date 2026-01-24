#!/usr/bin/env python3
"""Rename connectivity field attributes using word boundaries."""

import re
import sys
from pathlib import Path

RENAMES = [
    (r'\.edge_to_vertex\b', '.edge_to_vertex'),
    (r'\.edge_to_triangle\b', '.edge_to_triangle'),
    (r'\.triangle_to_triangle\b', '.triangle_to_triangle'),
    (r'\.vertex_normals\b', '.vertex_normals'),
    (r'\.sq_edge_length\b', '.sq_edge_length'),
]

def process_file(path: Path, dry_run: bool = False) -> int:
    """Process a single file. Returns number of replacements."""
    try:
        content = path.read_text(encoding='utf-8')
    except:
        return 0
    
    original = content
    total = 0
    
    for pattern, replacement in RENAMES:
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, replacement, content)
            total += matches
    
    if total > 0 and not dry_run:
        path.write_text(content, encoding='utf-8')
    
    return total

def main():
    dry_run = '--dry-run' in sys.argv
    
    root = Path('.')
    py_files = list(root.rglob('*.py'))
    
    total = 0
    for f in py_files:
        if '.git' in str(f):
            continue
        count = process_file(f, dry_run)
        if count:
            print(f'{f}: {count} replacements')
            total += count
    
    print(f'\nTotal: {total} replacements')
    if dry_run:
        print('(dry run - no files modified)')

if __name__ == '__main__':
    main()
