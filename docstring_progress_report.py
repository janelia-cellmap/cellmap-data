#!/usr/bin/env python3
"""
Week 4 Day 3-4 Docstring Standardization Progress Report

This script analyzes the current state of docstring standardization across
the CellMap-Data codebase and generates a progress report for Google-style conversion.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


def analyze_docstring_format(file_path: str) -> Dict[str, Any]:
    """Analyze docstring format in a Python file.

    Returns:
        Dictionary with analysis results including Google-style vs NumPy-style usage.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return {"error": str(e)}

    # Count different docstring patterns
    google_args = len(re.findall(r"^\s*Args:\s*$", content, re.MULTILINE))
    google_returns = len(re.findall(r"^\s*Returns:\s*$", content, re.MULTILINE))
    numpy_params = len(
        re.findall(r"^\s*Parameters\s*\n\s*-+\s*$", content, re.MULTILINE)
    )
    numpy_returns = len(re.findall(r"^\s*Returns\s*\n\s*-+\s*$", content, re.MULTILINE))

    # Count docstrings with examples
    examples = len(re.findall(r"^\s*Examples\s*\n\s*-+\s*$", content, re.MULTILINE))

    # Count functions/classes with docstrings
    functions = len(re.findall(r"^def ", content, re.MULTILINE))
    classes = len(re.findall(r"^class ", content, re.MULTILINE))

    return {
        "google_style": google_args + google_returns,
        "numpy_style": numpy_params + numpy_returns,
        "examples": examples,
        "functions": functions,
        "classes": classes,
        "google_args": google_args,
        "google_returns": google_returns,
        "numpy_params": numpy_params,
        "numpy_returns": numpy_returns,
    }


def generate_progress_report() -> str:
    """Generate comprehensive progress report for docstring standardization."""

    src_dir = Path("src/cellmap_data")
    if not src_dir.exists():
        return "Error: src/cellmap_data directory not found"

    # Categorize files by priority
    core_files = ["dataset.py", "image.py", "dataloader.py"]

    transform_files = [
        "transforms/augment/gaussian_noise.py",
        "transforms/augment/random_contrast.py",
        "transforms/augment/random_gamma.py",
        "transforms/augment/gaussian_blur.py",
        "transforms/augment/nan_to_num.py",
        "transforms/augment/binarize.py",
        "transforms/augment/normalize.py",
    ]

    secondary_files = [
        "dataset_writer.py",
        "multidataset.py",
        "datasplit.py",
        "utils/figs.py",
        "utils/misc.py",
        "utils/error_handling.py",
    ]

    report = ["# Week 4 Day 3-4 Docstring Standardization Progress Report\n"]

    def analyze_category(files: List[str], category_name: str) -> Tuple[int, int, int]:
        """Analyze a category of files and return totals."""
        total_google = 0
        total_numpy = 0
        total_examples = 0

        report.append(f"## {category_name}\n")

        for file in files:
            file_path = src_dir / file
            if file_path.exists():
                analysis = analyze_docstring_format(str(file_path))
                if "error" not in analysis:
                    google_count = analysis["google_style"]
                    numpy_count = analysis["numpy_style"]
                    examples_count = analysis["examples"]

                    total_google += google_count
                    total_numpy += numpy_count
                    total_examples += examples_count

                    # Determine status
                    if numpy_count == 0 and google_count > 0:
                        status = "✅ COMPLETED (Google-style)"
                    elif numpy_count > 0 and google_count == 0:
                        status = "🔄 NEEDS CONVERSION (NumPy-style)"
                    elif numpy_count > 0 and google_count > 0:
                        status = "⚠️ MIXED FORMAT"
                    else:
                        status = "❓ MINIMAL DOCS"

                    report.append(f"- **{file}**: {status}")
                    report.append(
                        f"  - Google-style: {google_count}, NumPy-style: {numpy_count}, Examples: {examples_count}"
                    )
                else:
                    report.append(f"- **{file}**: ❌ ERROR - {analysis['error']}")

        report.append("")
        return total_google, total_numpy, total_examples

    # Analyze each category
    core_g, core_n, core_e = analyze_category(core_files, "Phase 1: Core Classes")
    trans_g, trans_n, trans_e = analyze_category(
        transform_files, "Phase 2: Transform Modules"
    )
    sec_g, sec_n, sec_e = analyze_category(
        secondary_files, "Phase 3: Secondary Modules"
    )

    # Summary statistics
    total_google = core_g + trans_g + sec_g
    total_numpy = core_n + trans_n + sec_e
    total_examples = core_e + trans_e + sec_e

    report.append("## Summary Statistics\n")
    report.append(f"- **Total Google-style docstrings**: {total_google}")
    report.append(f"- **Total NumPy-style docstrings**: {total_numpy}")
    report.append(f"- **Total with Examples sections**: {total_examples}")
    report.append(
        f"- **Conversion progress**: {total_google}/{total_google + total_numpy} ({100 * total_google / max(1, total_google + total_numpy):.1f}%)"
    )

    report.append("\n## Completion Status\n")

    if core_n == 0:
        report.append("✅ **Phase 1 (Core Classes)**: COMPLETED")
    else:
        report.append(f"🔄 **Phase 1 (Core Classes)**: {core_n} NumPy-style remaining")

    if trans_n == 0:
        report.append("✅ **Phase 2 (Transform Modules)**: COMPLETED")
    else:
        report.append(
            f"🔄 **Phase 2 (Transform Modules)**: {trans_n} NumPy-style remaining"
        )

    if sec_n == 0:
        report.append("✅ **Phase 3 (Secondary Modules)**: COMPLETED")
    else:
        report.append(
            f"🔄 **Phase 3 (Secondary Modules)**: {sec_n} NumPy-style remaining"
        )

    # Next steps
    report.append("\n## Next Steps\n")
    if total_numpy > 0:
        report.append(
            "1. Continue converting remaining NumPy-style docstrings to Google format"
        )
        report.append("2. Add missing Examples sections to public APIs")
        report.append("3. Verify all parameter descriptions are concise and clear")
    else:
        report.append("1. ✅ All docstrings converted to Google-style format!")
        report.append("2. Run documentation generation tests")
        report.append("3. Verify all examples execute correctly")

    return "\n".join(report)


if __name__ == "__main__":
    print(generate_progress_report())
