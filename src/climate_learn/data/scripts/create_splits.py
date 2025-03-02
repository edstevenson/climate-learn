"""
Command-line script for creating train/val/test splits from processed climate data.

This script provides a convenient way to create data splits from already processed
climate data stored in .npz format.

Example usage:
    # Create 80/20 train/val split
    python -m climate_learn.data.scripts.create_splits /path/to/data --split 0.8 0.2
    
    # Create 70/15/15 train/val/test split
    python -m climate_learn.data.scripts.create_splits /path/to/data --split 0.7 0.15 0.15
"""

import argparse
import sys
from climate_learn.data.utils import create_data_splits


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from processed climate data."
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing processed data with a 'train' subdirectory"
    )
    parser.add_argument(
        "--split",
        nargs="+",
        type=float,
        required=True,
        help="Proportions for data splits (must sum to 1.0). "
             "Use 2 values for train/val split or 3 values for train/val/test split."
    )
    
    args = parser.parse_args()
    
    # Validate split proportions
    if len(args.split) not in [2, 3]:
        print(f"Error: Split must have length 2 or 3, got {len(args.split)}")
        sys.exit(1)
    
    if abs(sum(args.split) - 1.0) > 1e-10:
        print(f"Error: Split proportions must sum to 1, got {args.split}")
        sys.exit(1)
    
    # Create data splits
    try:
        create_data_splits(args.data_dir, args.split)
        print(f"Successfully created data splits in {args.data_dir}")
    except Exception as e:
        print(f"Error creating data splits: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 