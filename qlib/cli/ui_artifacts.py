# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validated access to single-asset Data Agent artifacts for the local UI."""

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ARTIFACT_ROOT = Path("~/Documents/quant_outputs/artifacts/processed").expanduser()
DEFAULT_IMPORT_ROOT = Path("~/.qlib/qlib_data/imported").expanduser()
MAX_METADATA_BYTES = 1024 * 1024
MAX_CHECKSUM_BYTES = 64 * 1024
MAX_DATA_BYTES = 256 * 1024 * 1024
MAX_ROWS = 5_000_000
MAX_COLUMNS = 64
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,63}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DATA_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "value",
    "volume",
    "simple_return",
    "log_return",
    "equity_curve",
    "cumulative_return",
    "drawdown",
}


class ArtifactValidationError(ValueError):
    """A local artifact is outside the contract accepted by the UI."""


@dataclass(frozen=True)
class ArtifactData:
    path: Path
    manifest: dict
    validation_report: dict
    frame: pd.DataFrame
    instrument: str
    data_digest: str

    def summary(self) -> dict:
        timestamps = self.frame["timestamp"]
        missing = {key: int(value) for key, value in self.frame.isna().sum().items() if value}
        manifest_fields = [str(value).lower() for value in self.manifest.get("fields", [])]
        fields = [field for field in manifest_fields if field in self.frame.columns]
        close_field = "close" if "close" in self.frame.columns else "value"
        close = self.frame[close_field]
        issues = self.validation_report.get("issues", [])
        issue_summary = []
        if isinstance(issues, list):
            for issue in issues[:20]:
                if isinstance(issue, dict):
                    issue_summary.append(
                        {
                            "severity": str(issue.get("severity", ""))[:32],
                            "code": str(issue.get("code", ""))[:80],
                            "message": str(issue.get("message", ""))[:240],
                        }
                    )
        return {
            "artifact_id": self.manifest["artifact_id"],
            "instrument": self.instrument,
            "asset_class": str(self.manifest.get("asset_class", "unknown"))[:64],
            "dataset_kind": str(self.manifest.get("dataset_kind", "unknown"))[:64],
            "source": str(self.manifest.get("source", "unknown"))[:80],
            "frequency": self.manifest["frequency"],
            "timezone": self.manifest["timezone"],
            "rows": len(self.frame),
            "fields": fields,
            "available_numeric_fields": [column for column in self.frame.columns if column != "timestamp"],
            "actual_date_range": {
                "start": timestamps.iloc[0].date().isoformat(),
                "end": timestamps.iloc[-1].date().isoformat(),
            },
            "missing_values": missing,
            "close_statistics": {
                "first": float(close.iloc[0]),
                "last": float(close.iloc[-1]),
                "minimum": float(close.min()),
                "maximum": float(close.max()),
            },
            "validation": self.validation_report["status"],
            "validation_issues": issue_summary,
            "warnings": [str(value)[:300] for value in self.manifest.get("warnings", [])[:20]],
            "checksum_verified": True,
            "data_sha256": self.data_digest,
        }


def inspect_artifact(path: Path, allowed_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict:
    """Return a compact, verified summary without exposing rows or external manifest paths."""

    return load_artifact(path, allowed_root).summary()


def load_artifact(path: Path, allowed_root: Path = DEFAULT_ARTIFACT_ROOT) -> ArtifactData:
    root = Path(allowed_root).expanduser().resolve(strict=True)
    supplied = Path(path).expanduser()
    _reject_symlink(supplied)
    resolved = supplied.resolve(strict=True)
    artifact = resolved if resolved.is_dir() else resolved.parent
    if artifact.parent != root:
        raise ArtifactValidationError(f"路径不在允许的数据目录中：{root}")
    _safe_file(artifact, directory=True)

    paths = {
        "manifest.json": artifact / "manifest.json",
        "validation_report.json": artifact / "validation_report.json",
        "checksums.txt": artifact / "checksums.txt",
        "data.parquet": artifact / "data.parquet",
    }
    for file_path in paths.values():
        _safe_file(file_path)
    _check_size(paths["manifest.json"], MAX_METADATA_BYTES)
    _check_size(paths["validation_report.json"], MAX_METADATA_BYTES)
    _check_size(paths["checksums.txt"], MAX_CHECKSUM_BYTES)
    _check_size(paths["data.parquet"], MAX_DATA_BYTES)

    checksums = _read_checksums(paths["checksums.txt"])
    for name in ("manifest.json", "validation_report.json", "data.parquet"):
        expected = checksums.get(name)
        if expected is None or _sha256(paths[name]) != expected:
            raise ArtifactValidationError(f"文件校验和不匹配或缺失：{name}")

    manifest = _read_json(paths["manifest.json"])
    report = _read_json(paths["validation_report.json"])
    artifact_id = str(manifest.get("artifact_id", ""))
    if not _ARTIFACT_ID.fullmatch(artifact_id) or artifact_id != artifact.name:
        raise ArtifactValidationError("artifact_id 无效或与目录名不一致。")
    if manifest.get("frequency") != "1d" or manifest.get("timezone") != "UTC":
        raise ArtifactValidationError("当前仅支持 UTC 时区的 1d 单资产数据。")
    if manifest.get("validation_result") != "passed" or report.get("status") != "passed":
        raise ArtifactValidationError("Data Agent 校验未通过，拒绝读取。")

    instrument = _instrument(manifest.get("instrument_identity"))
    frame = _read_parquet(paths["data.parquet"])
    _validate_frame(frame)
    return ArtifactData(artifact, manifest, report, frame, instrument, checksums["data.parquet"])


def imported_dataset_path(summary: dict, import_root: Path = DEFAULT_IMPORT_ROOT) -> Path:
    root = Path(import_root).expanduser().resolve()
    return root / f"{summary['artifact_id']}-{summary['data_sha256'][:12]}"


def import_artifact(
    path: Path,
    allowed_root: Path = DEFAULT_ARTIFACT_ROOT,
    import_root: Path = DEFAULT_IMPORT_ROOT,
    expected_digest: Optional[str] = None,
) -> Path:
    """Build and verify an immutable Qlib dataset, then publish it atomically."""

    artifact = load_artifact(path, allowed_root)
    if expected_digest is not None and artifact.data_digest != expected_digest:
        raise ArtifactValidationError("源数据与确认时的摘要不一致，请重新检查并确认。")
    summary = artifact.summary()
    root = Path(import_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = imported_dataset_path(summary, root)
    with _import_lock(root):
        if target.is_symlink():
            raise ArtifactValidationError("目标数据集不能是符号链接。")
        if target.exists():
            metadata = _read_json(target / "artifact.json")
            if metadata.get("data_sha256") != artifact.data_digest:
                raise ArtifactValidationError("目标数据集存在但源摘要不一致，拒绝覆盖。")
            return target

        temp_base = Path(tempfile.mkdtemp(prefix=".qlib-import-", dir=root))
        os.chmod(temp_base, 0o700)
        try:
            source_dir = temp_base / "source"
            dataset_dir = temp_base / "dataset"
            source_dir.mkdir(mode=0o700)
            source = artifact.frame.copy()
            output = pd.DataFrame({"date": source["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)})
            for field in ("open", "high", "low", "close", "volume"):
                if field in source:
                    output[field] = source[field].astype(float)
            if "close" not in output and "value" in source:
                output["close"] = source["value"].astype(float)
            csv_path = source_dir / f"{artifact.instrument}.csv"
            output.to_csv(csv_path, index=False)
            os.chmod(csv_path, 0o600)

            _write_qlib_dataset(dataset_dir, artifact.instrument, output)
            _verify_qlib_dataset(dataset_dir, artifact.instrument, output)

            metadata = {
                "artifact_id": summary["artifact_id"],
                "instrument": artifact.instrument,
                "rows": len(output),
                "frequency": "1d",
                "data_sha256": artifact.data_digest,
                "source_path": str(artifact.path),
                "fields": [column for column in output.columns if column != "date"],
            }
            metadata_path = dataset_dir / "artifact.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.chmod(metadata_path, 0o600)
            if _sha256(artifact.path / "data.parquet") != artifact.data_digest:
                raise ArtifactValidationError("源数据在导入过程中发生变化，请重新检查并确认。")
            os.replace(dataset_dir, target)
        finally:
            shutil.rmtree(temp_base, ignore_errors=True)
    return target


@contextmanager
def _import_lock(root: Path):
    import fcntl

    lock_path = root / ".import.lock"
    with lock_path.open("a+") as stream:
        os.chmod(lock_path, 0o600)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_qlib_dataset(dataset: Path, instrument: str, expected: pd.DataFrame) -> None:
    import qlib
    from qlib.constant import REG_US
    from qlib.data import D

    qlib.init(provider_uri=str(dataset), region=REG_US, clear_mem_cache=True)
    actual = D.features(
        [instrument],
        ["$close"],
        start_time=expected["date"].iloc[0],
        end_time=expected["date"].iloc[-1],
        freq="day",
    )
    if len(actual) != len(expected):
        raise ArtifactValidationError("Qlib 回读行数与源数据不一致。")
    values = actual["$close"].to_numpy(dtype=float)
    if not np.allclose(values, expected["close"].to_numpy(dtype=float), rtol=1e-5, equal_nan=True):
        raise ArtifactValidationError("Qlib 回读价格与源数据不一致。")


def _write_qlib_dataset(dataset: Path, instrument: str, frame: pd.DataFrame) -> None:
    from qlib.utils import code_to_fname

    calendars = dataset / "calendars"
    instruments = dataset / "instruments"
    features = dataset / "features" / code_to_fname(instrument).lower()
    calendars.mkdir(parents=True)
    instruments.mkdir()
    features.mkdir(parents=True)
    dates = frame["date"].dt.strftime("%Y-%m-%d")
    (calendars / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")
    (instruments / "all.txt").write_text(
        f"{instrument}\t{dates.iloc[0]}\t{dates.iloc[-1]}\n", encoding="utf-8"
    )
    for field in ("open", "high", "low", "close", "volume"):
        if field not in frame:
            continue
        values = np.hstack(([0.0], frame[field].to_numpy(dtype=float))).astype("<f")
        values.tofile(features / f"{field}.day.bin")


def _reject_symlink(path: Path) -> None:
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise ArtifactValidationError("不接受符号链接数据路径。")
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"数据路径不存在：{path}") from exc


def _safe_file(path: Path, directory: bool = False) -> None:
    _reject_symlink(path)
    info = path.stat()
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not expected:
        raise ArtifactValidationError(f"不是受支持的{'目录' if directory else '普通文件'}：{path.name}")
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ArtifactValidationError(f"文件所有者或写权限不安全：{path.name}")


def _check_size(path: Path, maximum: int) -> None:
    if path.stat().st_size > maximum:
        raise ArtifactValidationError(f"文件超过大小限制：{path.name}")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"JSON 文件无效：{path.name}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"JSON 顶层必须是对象：{path.name}")
    return value


def _read_checksums(path: Path) -> Dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or not _DIGEST.fullmatch(parts[0]):
            raise ArtifactValidationError("checksums.txt 格式无效。")
        name = parts[1]
        if name not in {"data.csv", "data.parquet", "interactive.html", "manifest.json", "validation_report.json"}:
            raise ArtifactValidationError("checksums.txt 包含不受支持的文件名。")
        result[name] = parts[0]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _instrument(value) -> str:
    symbol = str(value or "").upper().replace("/", "-")
    if not _SYMBOL.fullmatch(symbol) or ".." in symbol:
        raise ArtifactValidationError("instrument_identity 无效。")
    return symbol


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise ArtifactValidationError("data.parquet 无法读取。") from exc
    metadata = parquet.metadata
    if metadata.num_rows < 1 or metadata.num_rows > MAX_ROWS or metadata.num_columns > MAX_COLUMNS:
        raise ArtifactValidationError("data.parquet 的行数或列数超过限制。")
    schema = parquet.schema_arrow
    if "timestamp" not in schema.names:
        raise ArtifactValidationError("数据缺少 timestamp 字段。")
    if any(name != "timestamp" and name not in _DATA_FIELDS for name in schema.names):
        raise ArtifactValidationError("数据包含不受支持的字段。")
    for field in schema:
        valid = pa.types.is_timestamp(field.type) if field.name == "timestamp" else (
            pa.types.is_floating(field.type) or pa.types.is_integer(field.type)
        )
        if not valid:
            raise ArtifactValidationError(f"字段类型不受支持：{field.name}")
    return parquet.read().to_pandas()


def _validate_frame(frame: pd.DataFrame) -> None:
    if "close" not in frame and "value" not in frame:
        raise ArtifactValidationError("数据必须包含 close 或 value 字段。")
    timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamp.isna().any() or timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ArtifactValidationError("timestamp 必须有效、升序且不重复。")
    frame["timestamp"] = timestamp
    numeric = frame.drop(columns="timestamp")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ArtifactValidationError("数据包含无穷值。")
    price_fields = [field for field in ("open", "high", "low", "close", "value") if field in frame]
    if any((frame[field].dropna() <= 0).any() for field in price_fields):
        raise ArtifactValidationError("价格字段包含非正值。")
    if "volume" in frame and (frame["volume"].dropna() < 0).any():
        raise ArtifactValidationError("volume 包含负值。")
    if {"open", "high", "low", "close"}.issubset(frame.columns):
        maximum = frame[["open", "close", "low"]].max(axis=1)
        minimum = frame[["open", "close", "high"]].min(axis=1)
        if (frame["high"] < maximum).any() or (frame["low"] > minimum).any():
            raise ArtifactValidationError("OHLC 价格关系无效。")
