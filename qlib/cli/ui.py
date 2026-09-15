# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A dependency-free, local web interface for common Qlib workflows."""

import argparse
import ipaddress
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).with_name("ui_static")
DEFAULT_DATA_DIR = Path(os.getenv("QLIB_DATA_DIR", "~/.qlib/qlib_data/cn_data")).expanduser().resolve()
MAX_LOG_LINES = 500
CONFIRMATION_TTL_SECONDS = 300
SAFE_ACTION_CWD = Path(tempfile.gettempdir()).resolve()


@dataclass(frozen=True)
class Action:
    id: str
    title: str
    category: str
    description: str
    outcome: str
    duration: str
    command: Tuple[str, ...]
    cwd: Path
    keywords: Tuple[str, ...]

    def public_dict(self) -> dict:
        value = asdict(self)
        value["command"] = shlex.join(self.command)
        value["cwd"] = str(self.cwd)
        value.pop("keywords")
        return value

    def card_dict(self) -> dict:
        value = self.public_dict()
        value.pop("command")
        value.pop("cwd")
        return value


def build_actions(data_dir: Path = DEFAULT_DATA_DIR, python: str = sys.executable) -> Dict[str, Action]:
    actions = (
        Action(
            id="environment",
            title="检查运行环境",
            category="开始",
            description="确认 Qlib、Python 和本地行情数据是否已经就绪。",
            outcome="输出版本、数据目录和关键数据文件的可用状态。",
            duration="约 2 秒",
            command=(python, "-m", "qlib.cli.ui_actions", "environment", "--data-dir", str(data_dir)),
            cwd=SAFE_ACTION_CWD,
            keywords=("环境", "安装", "版本", "准备", "能用", "开始", "setup", "install", "version"),
        ),
        Action(
            id="preview-data",
            title="预览最新行情",
            category="数据",
            description="读取浦发银行最近 5 个交易日的开高低收和成交量。",
            outcome="展示数据日期范围与最新 5 行日频行情。",
            duration="约 5 秒",
            command=(python, "-m", "qlib.cli.ui_actions", "preview-data", "--data-dir", str(data_dir)),
            cwd=SAFE_ACTION_CWD,
            keywords=("数据", "行情", "股票", "价格", "成交量", "读取", "预览", "data", "price"),
        ),
        Action(
            id="data-health",
            title="检查数据质量",
            category="数据",
            description="扫描本地 Qlib 数据，检查缺失值、异常跳变和交易日完整性。",
            outcome="输出数据健康报告；发现问题时给出具体统计。",
            duration="通常 1–3 分钟",
            command=(
                python,
                str(REPO_ROOT / "scripts" / "check_data_health.py"),
                "check_data",
                "--qlib_dir",
                str(data_dir),
            ),
            cwd=REPO_ROOT,
            keywords=("质量", "健康", "缺失", "异常", "完整", "检查数据", "health", "missing"),
        ),
        Action(
            id="lightgbm-backtest",
            title="运行 LightGBM 回测",
            category="研究",
            description="使用 Alpha158 特征训练 LightGBM，并完成选股回测和风险分析。",
            outcome="生成模型预测、组合收益、最大回撤和信息比率等结果。",
            duration="通常 5–20 分钟",
            command=(
                python,
                "-m",
                "qlib.cli.run",
                "benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml",
            ),
            cwd=REPO_ROOT / "examples",
            keywords=("模型", "训练", "预测", "回测", "收益", "风险", "lightgbm", "alpha158", "backtest"),
        ),
    )
    return {action.id: action for action in actions}


def recommend_actions(query: str, actions: Optional[Dict[str, Action]] = None) -> dict:
    actions = actions or build_actions()
    normalized = query.strip().lower()
    if not normalized:
        return {
            "answer": "可以描述你想查看数据、检查环境、验证数据质量，或运行模型回测。",
            "action_ids": list(actions)[:3],
        }

    ranked = []
    for index, action in enumerate(actions.values()):
        score = sum(2 if keyword.lower() in normalized else 0 for keyword in action.keywords)
        if action.category.lower() in normalized:
            score += 1
        ranked.append((score, -index, action.id))
    ranked.sort(reverse=True)
    selected = [action_id for score, _, action_id in ranked if score > 0][:3]

    if not selected:
        return {
            "answer": "我暂时没找到完全匹配的任务。建议先检查环境，再预览行情数据。",
            "action_ids": ["environment", "preview-data"],
        }

    primary = actions[selected[0]]
    return {
        "answer": f"你可以先执行“{primary.title}”。它会{primary.outcome}",
        "action_ids": selected,
    }


@dataclass
class Run:
    id: str
    action_id: str
    status: str = "queued"
    output: List[str] = field(default_factory=list)
    return_code: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)

    def public_dict(self, action: Action) -> dict:
        return {
            "id": self.id,
            "action_id": self.action_id,
            "title": action.title,
            "status": self.status,
            "output": "".join(self.output),
            "return_code": self.return_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class ActionRunner:
    def __init__(self, actions: Optional[Dict[str, Action]] = None):
        self.actions = actions or build_actions()
        self.runs: Dict[str, Run] = {}
        self.confirmations: Dict[str, Tuple[str, float]] = {}
        self.lock = threading.Lock()

    def preview(self, action_id: str) -> dict:
        if action_id not in self.actions:
            raise KeyError(action_id)
        token = uuid.uuid4().hex
        with self.lock:
            self.confirmations[token] = (action_id, time.time() + CONFIRMATION_TTL_SECONDS)
        return {"confirmation_token": token, "action": self.actions[action_id].public_dict()}

    def start(self, action_id: str, confirmation_token: str) -> Run:
        if action_id not in self.actions:
            raise KeyError(action_id)
        with self.lock:
            confirmation = self.confirmations.pop(confirmation_token, None)
            if confirmation is None or confirmation[0] != action_id or confirmation[1] < time.time():
                raise PermissionError("A valid command preview is required before execution.")
            if any(run.status in ("queued", "running") for run in self.runs.values()):
                raise RuntimeError("Another action is already running.")
        run = Run(id=uuid.uuid4().hex, action_id=action_id)
        with self.lock:
            self.runs[run.id] = run
        threading.Thread(target=self._execute, args=(run,), daemon=True).start()
        return run

    def get(self, run_id: str) -> Optional[Run]:
        with self.lock:
            return self.runs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        run = self.get(run_id)
        if run is None or run.process is None or run.status != "running":
            return False
        run.process.terminate()
        return True

    def _execute(self, run: Run) -> None:
        action = self.actions[run.action_id]
        run.status = "running"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            run.process = subprocess.Popen(
                action.command,
                cwd=action.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=False,
            )
            assert run.process.stdout is not None
            for line in run.process.stdout:
                run.output.append(line)
                if len(run.output) > MAX_LOG_LINES:
                    del run.output[: len(run.output) - MAX_LOG_LINES]
            run.return_code = run.process.wait()
            run.status = "completed" if run.return_code == 0 else "failed"
        except Exception as exc:  # pragma: no cover - OS-level launch failures vary
            run.output.append(f"Unable to start action: {exc}\n")
            run.status = "failed"
            run.return_code = -1
        finally:
            run.finished_at = time.time()


class QlibUIHandler(BaseHTTPRequestHandler):
    runner = ActionRunner()

    def do_GET(self) -> None:
        if not self._has_local_host():
            self._json({"error": "Invalid host."}, HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
        if path == "/api/catalog":
            self._json({"actions": [action.card_dict() for action in self.runner.actions.values()]})
            return
        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/") :].split("/")[0]
            run = self.runner.get(run_id)
            if run is None:
                self._json({"error": "Run not found."}, HTTPStatus.NOT_FOUND)
            else:
                self._json(run.public_dict(self.runner.actions[run.action_id]))
            return
        self._static(path)

    def do_POST(self) -> None:
        if not self._has_local_host():
            self._json({"error": "Invalid host."}, HTTPStatus.FORBIDDEN)
            return
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except (json.JSONDecodeError, ValueError):
            self._json({"error": "Invalid JSON request."}, HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/recommend":
            self._json(recommend_actions(str(payload.get("query", "")), self.runner.actions))
            return
        if path.startswith("/api/actions/") and path.endswith("/preview"):
            action_id = path[len("/api/actions/") : -len("/preview")]
            try:
                self._json(self.runner.preview(action_id))
            except KeyError:
                self._json({"error": "Unknown action."}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/runs":
            try:
                run = self.runner.start(
                    str(payload.get("action_id", "")), str(payload.get("confirmation_token", ""))
                )
            except KeyError:
                self._json({"error": "Unknown action."}, HTTPStatus.BAD_REQUEST)
                return
            except PermissionError as exc:
                self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
                return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
                return
            self._json(run.public_dict(self.runner.actions[run.action_id]), HTTPStatus.ACCEPTED)
            return
        if path.startswith("/api/runs/") and path.endswith("/cancel"):
            run_id = path[len("/api/runs/") : -len("/cancel")]
            if self.runner.cancel(run_id):
                self._json({"status": "cancelling"})
            else:
                self._json({"error": "Running action not found."}, HTTPStatus.CONFLICT)
            return
        self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 16_384:
            raise ValueError("Request is too large.")
        return json.loads(self.rfile.read(length) or b"{}")

    def _has_local_host(self) -> bool:
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
        return host in ("localhost", "127.0.0.1", "::1")

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.css": ("app.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path not in files:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = files[path]
        body = (STATIC_ROOT / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if not is_loopback:
        raise ValueError("Qlib UI only accepts a loopback host (127.0.0.1, ::1, or localhost).")
    handler = type("BoundQlibUIHandler", (QlibUIHandler,), {"runner": ActionRunner()})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local Qlib web interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = create_server(args.host, args.port)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Qlib UI is ready at {url}")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
