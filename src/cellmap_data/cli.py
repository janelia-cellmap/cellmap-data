"""
Command-line interface for cellmap-data.
"""

import argparse
import json
from cellmap_data.validation import ConfigValidator


def validate_config(file_path: str):
    """
    Validates a configuration file.
    """
    try:
        with open(file_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return

    if "dataset" in config and "batch_size" in config:
        print("Validating DataLoader config...")
        if ConfigValidator.validate_dataloader_config(config):
            print("DataLoader configuration is valid.")
        else:
            print("DataLoader configuration is invalid.")
    elif "raw_path" in config and "input_arrays" in config:
        print("Validating Dataset config...")
        if ConfigValidator.validate_dataset_config(config):
            print("Dataset configuration is valid.")
        else:
            print("Dataset configuration is invalid.")
    else:
        print("Could not determine config type (Dataset or DataLoader).")


def main():
    """
    Main function for the CLI.
    """
    parser = argparse.ArgumentParser(description="cellmap-data CLI tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a configuration file."
    )
    validate_parser.add_argument(
        "file_path", type=str, help="Path to the configuration file to validate."
    )

    args = parser.parse_args()

    if args.command == "validate":
        validate_config(args.file_path)


if __name__ == "__main__":
    main()
