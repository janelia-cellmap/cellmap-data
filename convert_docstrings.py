#!/usr/bin/env python3
"""
Quick docstring converter from NumPy-style to Google-style format.

This script automatically converts docstrings from NumPy format to Google format
according to the CONTRIBUTING.md standards.
"""

import re
import os
from pathlib import Path


def convert_numpy_to_google(content: str) -> str:
    """Convert NumPy-style docstrings to Google-style format."""

    # Pattern for Parameters section
    params_pattern = r"(\s*)Parameters\s*\n\s*-+\s*\n((?:\s*[^:]+\s*:\s*[^\n]*(?:\n\s*[^\S\n]*[^\n]*)*\n?)*)"

    def replace_params(match):
        indent = match.group(1)
        params_text = match.group(2)

        # Parse individual parameters
        param_items = []
        current_param = ""

        for line in params_text.split("\n"):
            if line.strip():
                if " : " in line:
                    if current_param:
                        param_items.append(current_param.strip())
                    current_param = line.strip()
                else:
                    current_param += " " + line.strip()

        if current_param:
            param_items.append(current_param.strip())

        # Convert to Google style
        google_params = []
        for param in param_items:
            if " : " in param:
                parts = param.split(" : ", 1)
                name_part = parts[0].strip()
                desc_part = parts[1].strip()

                # Clean up parameter name (remove type info)
                param_name = name_part.split(",")[0].strip()

                # Simplify description
                desc_lines = desc_part.split("\n")
                clean_desc = " ".join(
                    line.strip() for line in desc_lines if line.strip()
                )

                google_params.append(f"{indent}    {param_name}: {clean_desc}")

        result = f"{indent}Args:\n" + "\n".join(google_params)
        return result

    # Replace Parameters sections
    content = re.sub(params_pattern, replace_params, content, flags=re.MULTILINE)

    # Pattern for Returns section
    returns_pattern = r"(\s*)Returns\s*\n\s*-+\s*\n((?:\s*[^\n]*\n?)*?)(?=\n\s*[A-Z][a-z]+\s*\n\s*-+|$)"

    def replace_returns(match):
        indent = match.group(1)
        returns_text = match.group(2).strip()

        # Simplify returns description
        lines = [line.strip() for line in returns_text.split("\n") if line.strip()]
        if lines:
            # Skip type-only lines, focus on description
            desc_lines = [line for line in lines if not re.match(r"^\w+\s*$", line)]
            if desc_lines:
                clean_desc = " ".join(desc_lines)
            else:
                clean_desc = " ".join(lines)

            return f"{indent}Returns:\n{indent}    {clean_desc}"
        return ""

    # Replace Returns sections
    content = re.sub(returns_pattern, replace_returns, content, flags=re.MULTILINE)

    # Pattern for Examples section (remove dashes)
    examples_pattern = r"(\s*)Examples\s*\n\s*-+\s*\n"
    content = re.sub(examples_pattern, r"\1Examples:\n", content, flags=re.MULTILINE)

    # Pattern for Raises section (remove dashes)
    raises_pattern = r"(\s*)Raises\s*\n\s*-+\s*\n"
    content = re.sub(raises_pattern, r"\1Raises:\n", content, flags=re.MULTILINE)

    # Pattern for Notes section (remove dashes)
    notes_pattern = r"(\s*)Notes\s*\n\s*-+\s*\n"
    content = re.sub(notes_pattern, r"\1Notes:\n", content, flags=re.MULTILINE)

    return content


def convert_file(file_path: Path):
    """Convert a single file from NumPy to Google style."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file contains NumPy-style docstrings
        if "Parameters\n" in content and "---" in content:
            print(f"Converting {file_path}")

            converted = convert_numpy_to_google(content)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(converted)

            print(f"✅ Converted {file_path}")
        else:
            print(f"⏭️  Skipping {file_path} (no NumPy-style docstrings found)")

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


def main():
    """Convert all relevant Python files in the project."""
    src_dir = Path("src/cellmap_data")

    if not src_dir.exists():
        print("Error: src/cellmap_data directory not found")
        return

    # Find all Python files
    python_files = list(src_dir.rglob("*.py"))

    print(f"Found {len(python_files)} Python files")
    print("Converting NumPy-style docstrings to Google-style...\n")

    for file_path in python_files:
        if file_path.name != "__init__.py":  # Skip __init__ files
            convert_file(file_path)

    print("\n🎉 Conversion complete!")
    print("Please run tests to verify the changes.")


if __name__ == "__main__":
    main()
