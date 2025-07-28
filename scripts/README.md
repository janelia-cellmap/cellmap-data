# Development Scripts

This directory contains utility scripts for development and maintenance of the cellmap-data project.

## Scripts

### `cleanup_repo.py`

Repository cleanup utility that:

- Removes temporary files (coverage files, caches, etc.)
- Identifies potential duplicate files in PLANNING/
- Organizes development scripts
- Provides cleanup recommendations

Usage:

```bash
python scripts/cleanup_repo.py
```

## Guidelines

- Place development utilities and maintenance scripts in this directory
- Keep scripts focused and well-documented
- Update this README when adding new scripts
- Scripts should be safe to run multiple times (idempotent)
