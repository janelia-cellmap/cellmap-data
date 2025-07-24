#!/usr/bin/env python3
"""
Quick test to verify parameter migration for raw_path -> input_path
"""
import warnings
import sys
import os

# Add src to path to import cellmap_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cellmap_data import CellMapDataset


def test_input_path_parameter():
    """Test that input_path parameter works correctly"""
    # Test with input_path (new parameter)
    try:
        dataset = CellMapDataset(
            input_path="/test/input/path",
            target_path="/test/target/path",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            target_arrays={
                "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
            },
            classes=["test_class"],
        )
        print("✓ input_path parameter works correctly")
        print(f"  dataset.input_path: {dataset.input_path}")
        print(
            f"  dataset.raw_path: {dataset.raw_path}"
        )  # Should be the same for backward compatibility
    except Exception as e:
        print(f"✗ input_path parameter failed: {e}")
        return False

    return True


def test_raw_path_deprecation():
    """Test that raw_path shows deprecation warning but still works"""
    # Test with raw_path (deprecated parameter)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            dataset = CellMapDataset(
                raw_path="/test/raw/path",  # Using deprecated parameter
                target_path="/test/target/path",
                input_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                target_arrays={
                    "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
                },
                classes=["test_class"],
            )

            # Check that deprecation warning was issued
            if len(w) > 0 and issubclass(w[0].category, DeprecationWarning):
                print("✓ raw_path shows deprecation warning correctly")
                print(f"  Warning message: {w[0].message}")
                print(f"  dataset.input_path: {dataset.input_path}")
                print(f"  dataset.raw_path: {dataset.raw_path}")
                return True
            else:
                print("✗ raw_path did not show deprecation warning")
                return False

        except Exception as e:
            print(f"✗ raw_path parameter failed: {e}")
            return False


def test_both_parameters_error():
    """Test that providing both parameters raises an error"""
    try:
        dataset = CellMapDataset(
            input_path="/test/input/path",
            raw_path="/test/raw/path",  # Should cause an error
            target_path="/test/target/path",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            target_arrays={
                "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
            },
            classes=["test_class"],
        )
        print("✗ Providing both parameters should have raised an error")
        return False
    except ValueError as e:
        if "Cannot specify both" in str(e):
            print("✓ Providing both parameters correctly raises ValueError")
            print(f"  Error message: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_missing_input_path_error():
    """Test that missing input_path raises an error"""
    try:
        dataset = CellMapDataset(
            # No input_path or raw_path provided
            target_path="/test/target/path",
            input_arrays={"test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}},
            target_arrays={
                "test": {"shape": [100, 100, 100], "scale": [1.0, 1.0, 1.0]}
            },
            classes=["test_class"],
        )
        print("✗ Missing input_path should have raised an error")
        return False
    except ValueError as e:
        if "'input_path' parameter is required" in str(e):
            print("✓ Missing input_path correctly raises ValueError")
            print(f"  Error message: {e}")
            return True
        else:
            print(f"✗ Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("Testing parameter migration: raw_path -> input_path")
    print("=" * 50)

    tests = [
        test_input_path_parameter,
        test_raw_path_deprecation,
        test_both_parameters_error,
        test_missing_input_path_error,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\nRunning {test.__name__}:")
        if test():
            passed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All parameter migration tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
