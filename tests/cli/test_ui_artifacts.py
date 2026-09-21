# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from qlib.cli.ui_artifacts import ArtifactValidationError, import_artifact, inspect_artifact
from qlib.cli.ui_agent import QlibToolbox, ToolProposal


def _write_artifact(root: Path, artifact_id: str = "ETH-USDT_20210714_20260714") -> Path:
    artifact = root / artifact_id
    artifact.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-13", "2026-07-14"], utc=True),
            "open": [1700.0, 1777.02],
            "high": [1800.0, 1794.51],
            "low": [1690.0, 1774.05],
            "close": [1775.0, 1788.98],
            "volume": [100.0, 59.9041],
        }
    )
    frame.to_parquet(artifact / "data.parquet", index=False)
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "asset_class": "crypto",
                "dataset_kind": "ohlcv",
                "date_range": {"start": "2021-07-14", "end": "2026-07-14"},
                "fields": ["open", "high", "low", "close", "volume"],
                "frequency": "1d",
                "instrument_identity": "ETH/USDT",
                "source": "ccxt",
                "timezone": "UTC",
                "validation_result": "passed",
                "warnings": ["research data only"],
                "paths": {"raw": "/private/outside.raw", "processed_data": "/stale/data.parquet"},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "validation_report.json").write_text(
        json.dumps({"status": "passed", "issues": []}), encoding="utf-8"
    )
    checksums = []
    for name in ("data.parquet", "manifest.json", "validation_report.json"):
        digest = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        checksums.append(f"{digest}  {name}")
    (artifact / "checksums.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return artifact


def test_inspect_artifact_reads_verified_single_asset_summary(tmp_path):
    root = tmp_path / "processed"
    artifact = _write_artifact(root)

    summary = inspect_artifact(artifact, allowed_root=root)

    assert summary["artifact_id"] == "ETH-USDT_20210714_20260714"
    assert summary["instrument"] == "ETH-USDT"
    assert summary["rows"] == 2
    assert summary["actual_date_range"] == {"start": "2026-07-13", "end": "2026-07-14"}
    assert summary["fields"] == ["open", "high", "low", "close", "volume"]
    assert summary["missing_values"] == {}
    assert summary["validation"] == "passed"
    assert summary["checksum_verified"] is True
    assert "paths" not in summary


def test_inspect_artifact_rejects_path_outside_allowed_root(tmp_path):
    root = tmp_path / "processed"
    root.mkdir()
    outside = _write_artifact(tmp_path / "outside")

    with pytest.raises(ArtifactValidationError, match="允许的数据目录"):
        inspect_artifact(outside, allowed_root=root)


def test_inspect_artifact_rejects_checksum_mismatch(tmp_path):
    root = tmp_path / "processed"
    artifact = _write_artifact(root)
    (artifact / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="校验和"):
        inspect_artifact(artifact, allowed_root=root)


def test_import_artifact_creates_idempotent_qlib_dataset(tmp_path):
    root = tmp_path / "processed"
    artifact = _write_artifact(root)
    import_root = tmp_path / "imported"

    first = import_artifact(artifact, allowed_root=root, import_root=import_root)
    second = import_artifact(artifact, allowed_root=root, import_root=import_root)

    assert first == second
    assert (first / "calendars" / "day.txt").is_file()
    assert (first / "instruments" / "all.txt").read_text().startswith("ETH-USDT\t")
    assert (first / "features" / "eth-usdt" / "close.day.bin").is_file()
    metadata = json.loads((first / "artifact.json").read_text())
    assert metadata["artifact_id"] == "ETH-USDT_20210714_20260714"
    assert metadata["rows"] == 2


def test_toolbox_import_requires_confirmation_and_binds_validated_path(tmp_path):
    root = tmp_path / "processed"
    artifact = _write_artifact(root)
    toolbox = QlibToolbox(
        tmp_path,
        tmp_path / "cn_data",
        python="python",
        artifact_root=root,
        import_root=tmp_path / "imported",
    )

    proposal = toolbox.execute("prepare_data_artifact_import", {"path": str(artifact)})

    assert isinstance(proposal, ToolProposal)
    assert proposal.command[:4] == ("python", "-m", "qlib.cli.ui_actions", "import-artifact")
    assert str(artifact.resolve()) in proposal.command
    assert "--expected-digest" in proposal.command
    assert "1827" not in proposal.description
    assert "2 行" in proposal.description


def test_import_rejects_data_changed_after_confirmation(tmp_path):
    root = tmp_path / "processed"
    artifact = _write_artifact(root)

    with pytest.raises(ArtifactValidationError, match="确认时的摘要"):
        import_artifact(
            artifact,
            allowed_root=root,
            import_root=tmp_path / "imported",
            expected_digest="0" * 64,
        )
