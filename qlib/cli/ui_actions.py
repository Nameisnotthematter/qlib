# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Small, read-only actions used by the local Qlib UI."""

import argparse
import sys
from pathlib import Path


def environment_summary(data_dir: Path) -> None:
    import qlib

    calendars = data_dir / "calendars"
    instruments = data_dir / "instruments"
    features = data_dir / "features"

    print(f"Qlib version: {qlib.__version__}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Data directory: {data_dir}")
    print(f"Data directory exists: {'yes' if data_dir.is_dir() else 'no'}")
    print(f"Calendars available: {'yes' if calendars.is_dir() else 'no'}")
    print(f"Instrument lists available: {'yes' if instruments.is_dir() else 'no'}")
    print(f"Feature files available: {'yes' if features.is_dir() else 'no'}")


def preview_data(data_dir: Path) -> None:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(data_dir), region=REG_CN)
    calendar = D.calendar(freq="day")
    if len(calendar) == 0:
        raise RuntimeError("No daily calendar is available in the selected data directory.")

    end_time = calendar[-1]
    start_time = calendar[max(0, len(calendar) - 10)]
    data = D.features(
        ["SH600000"],
        ["$open", "$high", "$low", "$close", "$volume"],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    print(f"Latest available trading date: {end_time.date()}")
    print("\nSH600000 recent daily data:\n")
    print(data.tail(5).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a built-in Qlib UI action.")
    parser.add_argument("action", choices=("environment", "preview-data"))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    if args.action == "environment":
        environment_summary(data_dir)
    else:
        preview_data(data_dir)


if __name__ == "__main__":
    main()
