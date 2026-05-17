from __future__ import annotations

import argparse
import json
import sys

from src.training.device import device_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PyTorch device and CUDA availability.")
    parser.add_argument("--device", default="auto", help="Device to resolve: auto, cpu, cuda, cuda:<index>.")
    args = parser.parse_args()

    try:
        print(json.dumps(device_report(args.device), indent=2))
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
