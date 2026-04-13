from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class merging is currently disabled in this project version."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Unused for now.")
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    print("Class merging is currently disabled.")
    print("Active experiments are:")
    print("  - baseline_no_aug")
    print("  - brightness_aug")
    print("  - moderate_aug")


if __name__ == "__main__":
    main()