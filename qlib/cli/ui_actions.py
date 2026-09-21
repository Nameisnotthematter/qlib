# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constrained local actions used by the Qlib UI."""

import argparse
import json
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


def market_data(
    data_dir: Path,
    instruments_json: str,
    fields_json: str,
    start_date: str,
    end_date: str,
    limit: int,
    region: str = "cn",
) -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US
    from qlib.data import D

    instruments = json.loads(instruments_json)
    fields = json.loads(fields_json)
    qlib.init(provider_uri=str(data_dir), region=REG_US if region == "us" else REG_CN)
    data = D.features(
        instruments,
        fields,
        start_time=start_date,
        end_time=end_date,
        freq="day",
    )
    print(data.tail(limit).to_string())


def list_instruments(
    data_dir: Path,
    market: str,
    start_date: str = None,
    end_date: str = None,
    limit: int = 30,
) -> None:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(data_dir), region=REG_CN)
    instruments = D.list_instruments(
        D.instruments(market),
        start_time=start_date,
        end_time=end_date,
        freq="day",
        as_list=True,
    )
    print(json.dumps({"market": market, "count": len(instruments), "instruments": instruments[:limit]}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a built-in Qlib UI action.")
    parser.add_argument(
        "action", choices=("environment", "preview-data", "market-data", "list-instruments", "import-artifact")
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--import-root", type=Path)
    parser.add_argument("--expected-digest")
    parser.add_argument("--instruments-json")
    parser.add_argument("--fields-json")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--market")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--region", choices=("cn", "us"), default="cn")
    args = parser.parse_args()

    if args.action == "import-artifact":
        if not args.artifact_path or not args.artifact_root or not args.import_root or not args.expected_digest:
            parser.error(
                "import-artifact requires --artifact-path, --artifact-root, --import-root, and --expected-digest"
            )
        from qlib.cli.ui_artifacts import import_artifact

        target = import_artifact(
            args.artifact_path, args.artifact_root, args.import_root, expected_digest=args.expected_digest
        )
        print(json.dumps({"status": "imported", "dataset": str(target)}, ensure_ascii=False))
        return

    if args.data_dir is None:
        parser.error(f"{args.action} requires --data-dir")
    data_dir = args.data_dir.expanduser().resolve()

    if args.action == "environment":
        environment_summary(data_dir)
    elif args.action == "preview-data":
        preview_data(data_dir)
    elif args.action == "market-data":
        market_data(
            data_dir,
            args.instruments_json,
            args.fields_json,
            args.start_date,
            args.end_date,
            args.limit,
            args.region,
        )
    else:
        list_instruments(data_dir, args.market, args.start_date, args.end_date, args.limit)


if __name__ == "__main__":
    main()
