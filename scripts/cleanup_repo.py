#!/usr/bin/env python3
"""
Repository cleanup script for cellmap-data.

This script helps maintain a clean repository by removing temporary files,
organizing development utilities, and identifying potential cleanup opportunities.
"""

import os
import glob
from pathlib import Path


def remove_temp_files():
    """Remove temporary and build files."""
    temp_patterns = [
        ".coverage*",
        "*.tmp",
        "*.temp",
        "*.log",
        "*.bak",
        "*.orig",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        "build/",
        "dist/",
        "*.egg-info/",
    ]

    removed_files = []

    for pattern in temp_patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            if os.path.exists(match):
                if os.path.isfile(match):
                    os.remove(match)
                    removed_files.append(match)
                elif os.path.isdir(match):
                    import shutil

                    shutil.rmtree(match)
                    removed_files.append(f"{match}/ (directory)")

    return removed_files


def find_potential_duplicates():
    """Find potential duplicate files in PLANNING directory."""
    planning_dir = Path("PLANNING")
    if not planning_dir.exists():
        return []

    files = list(planning_dir.glob("*.md"))
    potential_duplicates = []

    # Look for similar names
    for file1 in files:
        for file2 in files:
            if file1 != file2:
                name1 = file1.stem.lower()
                name2 = file2.stem.lower()

                # Check for similar names (simple similarity check)
                if (name1 in name2 or name2 in name1) and abs(
                    len(name1) - len(name2)
                ) < 5:
                    potential_duplicates.append((file1, file2))

    return potential_duplicates


def organize_scripts():
    """Move utility scripts to scripts directory."""
    script_files = [
        "convert_docstrings.py",
        "systematic_test_fix_plan.py",
        "critical_test_status_report.py",
    ]

    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)

    moved_files = []
    for script in script_files:
        if Path(script).exists():
            Path(script).rename(scripts_dir / script)
            moved_files.append(script)

    return moved_files


def main():
    """Run repository cleanup."""
    print("🧹 Starting repository cleanup...")

    # Remove temporary files
    removed = remove_temp_files()
    if removed:
        print(f"✅ Removed {len(removed)} temporary files:")
        for file in removed:
            print(f"   - {file}")
    else:
        print("✅ No temporary files to remove")

    # Find potential duplicates
    duplicates = find_potential_duplicates()
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} potential duplicate files:")
        for file1, file2 in duplicates:
            print(f"   - {file1.name} vs {file2.name}")
        print("   Review these manually to determine if they can be consolidated")
    else:
        print("\n✅ No obvious duplicate files found")

    # Organize scripts
    moved = organize_scripts()
    if moved:
        print(f"\n✅ Moved {len(moved)} utility scripts to scripts/:")
        for file in moved:
            print(f"   - {file}")
    else:
        print("\n✅ No utility scripts to organize")

    print("\n🎉 Repository cleanup complete!")
    print("\nRecommendations:")
    print("- Run this script periodically to maintain cleanliness")
    print("- Consider adding more patterns to .gitignore if needed")
    print("- Review PLANNING/ directory periodically for consolidation opportunities")


if __name__ == "__main__":
    main()
