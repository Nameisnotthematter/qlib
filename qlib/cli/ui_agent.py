# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OpenRouter-backed conversational agent for the local Qlib UI."""

import ast
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_KEY_PATH = Path("~/.config/qlib/openrouter.key").expanduser()
MAX_TOOL_OUTPUT = 40_000
MAX_HISTORY_MESSAGES = 24
MAX_TOOL_ROUNDS = 4


class AgentConfigurationError(RuntimeError):
    """The assistant is not configured with a usable API key."""


class AgentRequestError(RuntimeError):
    """OpenRouter could not complete the request."""


class ToolValidationError(ValueError):
    """A requested Qlib tool call contains invalid parameters."""


def load_openrouter_key(key_path: Path = DEFAULT_KEY_PATH) -> Optional[str]:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        if key_path.stat().st_mode & 0o077:
            raise AgentConfigurationError(f"API key file permissions are too broad: {key_path}. Run chmod 600 on it.")
        return key_path.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


class OpenRouterClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        key_loader: Callable[[], Optional[str]] = load_openrouter_key,
        transport: Callable = urlopen,
    ):
        self.model = model
        self.key_loader = key_loader
        self.transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.key_loader())

    def complete(self, messages: List[dict], tools: List[dict]) -> dict:
        api_key = self.key_loader()
        if not api_key:
            raise AgentConfigurationError(
                "OpenRouter API Key 尚未配置。请将新 Key 保存到 ~/.config/qlib/openrouter.key 并执行 chmod 600。"
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
            "max_tokens": 1600,
            "reasoning_effort": "low",
            "provider": {"require_parameters": True, "allow_fallbacks": True},
        }
        request = Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8787",
                "X-Title": "Qlib Workbench",
            },
            method="POST",
        )
        try:
            with self.transport(request, timeout=60) as response:
                result = json.loads(response.read())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", {}).get("message", str(exc))
            except (AttributeError, ValueError):
                detail = str(exc)
            raise AgentRequestError(f"OpenRouter 请求失败：{detail}") from exc
        except (OSError, URLError, ValueError) as exc:
            raise AgentRequestError(f"无法连接 OpenRouter：{exc}") from exc

        if isinstance(result, dict) and result.get("error"):
            error = result["error"]
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise AgentRequestError(f"OpenRouter 请求失败：{detail}")
        try:
            return result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentRequestError("OpenRouter 返回了无法识别的响应。") from exc


@dataclass(frozen=True)
class ToolProposal:
    title: str
    description: str
    duration: str
    command: Tuple[str, ...]
    cwd: Path


class QlibToolbox:
    _instrument_pattern = re.compile(r"^(SH|SZ|BJ)\d{6}$")
    _field_pattern = re.compile(r"^[A-Za-z0-9_$(),.+*/<>=!&| \-]+$")
    _markets = {"all", "csi300", "csi500", "csi800", "csi1000", "csiall"}
    _operators = {
        "Feature",
        "Ref",
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Max",
        "Min",
        "Rank",
        "Corr",
        "Cov",
        "Delta",
        "EMA",
        "WMA",
        "Log",
        "Abs",
        "Sign",
        "If",
        "Greater",
        "Less",
        "Count",
        "Quantile",
        "Slope",
        "Rsquare",
    }

    def __init__(self, repo_root: Path, data_dir: Path, python: str = sys.executable):
        self.repo_root = repo_root.resolve()
        self.data_dir = data_dir.expanduser().resolve()
        self.python = python

    def schemas(self) -> List[dict]:
        return [
            self._schema("inspect_environment", "检查 Qlib、Python 与本地数据是否已准备好。", {}),
            self._schema(
                "query_market_data",
                "查询一只或多只 A 股的行情或 Qlib 表达式/因子数据。",
                {
                    "instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10,
                        "description": "股票代码，例如 SH600000、SZ000001。",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 12,
                        "description": "Qlib 字段或表达式，例如 $close、Mean($close, 5)。",
                    },
                    "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD。"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                required=("instruments", "fields", "start_date", "end_date", "limit"),
            ),
            self._schema(
                "list_instruments",
                "查看指定股票池在给定日期范围内包含的股票代码。",
                {
                    "market": {"type": "string", "enum": sorted(self._markets)},
                    "start_date": {"type": ["string", "null"], "description": "可选，YYYY-MM-DD；不限制则为 null。"},
                    "end_date": {"type": ["string", "null"], "description": "可选，YYYY-MM-DD；不限制则为 null。"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                required=("market", "start_date", "end_date", "limit"),
            ),
            self._schema("list_workflows", "列出本地可运行的 Qlib YAML 模型与回测工作流。", {}),
            self._schema(
                "run_workflow",
                "准备运行指定 Qlib YAML 工作流。此操作必须由用户在 UI 中确认。",
                {"workflow_id": {"type": "string", "description": "list_workflows 返回的完整 workflow_id。"}},
                required=("workflow_id",),
            ),
            self._schema(
                "check_data_health",
                "准备扫描本地 Qlib 数据的缺失值、异常跳变和交易日完整性。此操作必须确认。",
                {},
            ),
        ]

    @staticmethod
    def _schema(name: str, description: str, properties: dict, required: Sequence[str] = ()) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required),
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, name: str, arguments: dict) -> Union[str, ToolProposal]:
        if name == "inspect_environment":
            self._require_no_arguments(arguments)
            return self._run(
                (
                    self.python,
                    "-m",
                    "qlib.cli.ui_actions",
                    "environment",
                    "--data-dir",
                    str(self.data_dir),
                )
            )
        if name == "query_market_data":
            return self._query_market_data(arguments)
        if name == "list_instruments":
            return self._list_instruments(arguments)
        if name == "list_workflows":
            self._require_no_arguments(arguments)
            return json.dumps({"workflows": sorted(self._workflows())[:200]}, ensure_ascii=False)
        if name == "run_workflow":
            return self._workflow_proposal(arguments)
        if name == "check_data_health":
            self._require_no_arguments(arguments)
            return ToolProposal(
                title="检查数据质量",
                description="扫描本地 Qlib 数据，检查缺失值、异常跳变和交易日完整性。",
                duration="通常 1–3 分钟",
                command=(
                    self.python,
                    str(self.repo_root / "scripts" / "check_data_health.py"),
                    "check_data",
                    "--qlib_dir",
                    str(self.data_dir),
                ),
                cwd=self.repo_root,
            )
        raise ToolValidationError(f"未知工具：{name}")

    @staticmethod
    def _require_no_arguments(arguments: dict) -> None:
        if arguments:
            raise ToolValidationError("该工具不接受参数。")

    def _query_market_data(self, arguments: dict) -> str:
        instruments = arguments.get("instruments")
        fields = arguments.get("fields")
        start_date = self._date(arguments.get("start_date"), "start_date")
        end_date = self._date(arguments.get("end_date"), "end_date")
        limit = self._limit(arguments.get("limit", 20))
        if start_date > end_date:
            raise ToolValidationError("start_date 不能晚于 end_date。")
        if not isinstance(instruments, list) or not 1 <= len(instruments) <= 10:
            raise ToolValidationError("instruments 必须包含 1–10 个股票代码。")
        instruments = [str(value).upper() for value in instruments]
        if any(not self._instrument_pattern.fullmatch(value) for value in instruments):
            raise ToolValidationError("股票代码必须类似 SH600000 或 SZ000001。")
        if not isinstance(fields, list) or not 1 <= len(fields) <= 12:
            raise ToolValidationError("fields 必须包含 1–12 个字段或表达式。")
        fields = [str(value).strip() for value in fields]
        if any(len(value) > 120 or not self._field_pattern.fullmatch(value) for value in fields):
            raise ToolValidationError("字段包含不支持的字符或长度超过限制。")
        for field in fields:
            self._validate_expression(field)
        return self._run(
            (
                self.python,
                "-m",
                "qlib.cli.ui_actions",
                "market-data",
                "--data-dir",
                str(self.data_dir),
                "--instruments-json",
                json.dumps(instruments),
                "--fields-json",
                json.dumps(fields),
                "--start-date",
                start_date.isoformat(),
                "--end-date",
                end_date.isoformat(),
                "--limit",
                str(limit),
            )
        )

    def _list_instruments(self, arguments: dict) -> str:
        market = str(arguments.get("market", "")).lower()
        if market not in self._markets:
            raise ToolValidationError(f"market 只能是：{', '.join(sorted(self._markets))}。")
        start_date = arguments.get("start_date")
        end_date = arguments.get("end_date")
        if start_date:
            start_date = self._date(start_date, "start_date").isoformat()
        if end_date:
            end_date = self._date(end_date, "end_date").isoformat()
        if start_date and end_date and start_date > end_date:
            raise ToolValidationError("start_date 不能晚于 end_date。")
        limit = self._limit(arguments.get("limit", 30))
        command = [
            self.python,
            "-m",
            "qlib.cli.ui_actions",
            "list-instruments",
            "--data-dir",
            str(self.data_dir),
            "--market",
            market,
            "--limit",
            str(limit),
        ]
        if start_date:
            command.extend(("--start-date", start_date))
        if end_date:
            command.extend(("--end-date", end_date))
        return self._run(tuple(command))

    def _workflow_proposal(self, arguments: dict) -> ToolProposal:
        workflow_id = str(arguments.get("workflow_id", ""))
        workflows = self._workflows()
        if workflow_id not in workflows:
            raise ToolValidationError("未知 workflow_id；请先调用 list_workflows。")
        return ToolProposal(
            title=f"运行工作流：{Path(workflow_id).stem}",
            description="训练模型并执行该 YAML 中配置的预测、策略与回测流程。",
            duration="通常 5–30 分钟",
            command=(self.python, "-m", "qlib.cli.run", workflow_id),
            cwd=self.repo_root / "examples",
        )

    def _workflows(self) -> Dict[str, Path]:
        examples_root = (self.repo_root / "examples").resolve()
        workflows = {}
        for path in examples_root.joinpath("benchmarks").glob("**/*.yaml"):
            resolved = path.resolve()
            if examples_root not in resolved.parents:
                continue
            workflows[str(resolved.relative_to(examples_root))] = resolved
        return workflows

    @staticmethod
    def _date(value, name: str) -> date:
        try:
            return date.fromisoformat(str(value))
        except ValueError as exc:
            raise ToolValidationError(f"{name} 必须使用 YYYY-MM-DD 格式。") from exc

    @staticmethod
    def _limit(value) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError("limit 必须是整数。") from exc
        if not 1 <= limit <= 100:
            raise ToolValidationError("limit 必须在 1–100 之间。")
        return limit

    def _run(self, command: Tuple[str, ...]) -> str:
        env = os.environ.copy()
        env.pop("OPENROUTER_API_KEY", None)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            result = subprocess.run(
                command,
                cwd=Path(os.getenv("TMPDIR", "/tmp")).resolve(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Qlib 查询超过 60 秒，请缩小日期或股票范围。") from exc
        output = result.stdout[-MAX_TOOL_OUTPUT:]
        if result.returncode != 0:
            raise RuntimeError(f"Qlib 工具执行失败：\n{output}")
        return output

    @classmethod
    def _validate_expression(cls, expression: str) -> None:
        from qlib.utils import parse_field

        try:
            tree = ast.parse(parse_field(expression), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ToolValidationError(f"无效的 Qlib 字段表达式：{expression}") from exc
        nodes = list(ast.walk(tree))
        allowed = (
            ast.Expression,
            ast.Call,
            ast.Attribute,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.BinOp,
            ast.UnaryOp,
            ast.BoolOp,
            ast.Compare,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.And,
            ast.Or,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        )
        if len(nodes) > 128 or any(not isinstance(node, allowed) for node in nodes):
            raise ToolValidationError(f"字段表达式包含不支持的语法：{expression}")
        for node in nodes:
            if isinstance(node, ast.Name) and node.id != "Operators":
                raise ToolValidationError(f"字段表达式包含不支持的名称：{node.id}")
            if isinstance(node, ast.Attribute):
                if not isinstance(node.value, ast.Name) or node.value.id != "Operators" or node.attr not in cls._operators:
                    raise ToolValidationError(f"字段表达式包含不支持的操作符：{node.attr}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Attribute) or node.keywords:
                    raise ToolValidationError("字段表达式只允许使用受支持的 Qlib 操作符和位置参数。")


SYSTEM_PROMPT = """你是本机 Qlib 量化研究助手。始终使用简体中文，直接回答用户的问题。
涉及本机数据、股票池、因子或工作流时必须调用提供的工具，不得编造结果。
不要生成或建议任意 Shell 命令、Python 代码、文件路径或未注册的 YAML。
查询行情时，从用户话语中提取股票代码、日期和 Qlib 字段；信息不足时只追问缺失参数。
运行工作流或数据健康检查需要工具返回确认卡片，由用户在 UI 中确认，你不能绕过确认。
工具报错时解释原因并建议合法参数。回答保持简洁。"""


class AssistantService:
    def __init__(
        self,
        client: OpenRouterClient,
        toolbox: QlibToolbox,
        proposal_handler: Callable[[ToolProposal], dict],
    ):
        self.client = client
        self.toolbox = toolbox
        self.proposal_handler = proposal_handler
        self.sessions: Dict[str, List[dict]] = {}
        self.lock = threading.Lock()

    def status(self) -> dict:
        return {"configured": self.client.configured, "model": self.client.model}

    def reset(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(self._session_id(session_id), None)

    def chat(self, session_id: str, user_message: str) -> dict:
        session_id = self._session_id(session_id)
        user_message = user_message.strip()
        if not user_message or len(user_message) > 4000:
            raise ValueError("消息长度必须在 1–4000 个字符之间。")
        with self.lock:
            history = list(self.sessions.get(session_id, []))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_message}]

        for _ in range(MAX_TOOL_ROUNDS):
            raw_message = self.client.complete(messages, self.toolbox.schemas())
            assistant_message = {"role": "assistant", "content": raw_message.get("content") or ""}
            tool_calls = raw_message.get("tool_calls") or []
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)
            if not tool_calls:
                answer = assistant_message["content"].strip()
                if not answer:
                    raise AgentRequestError("模型没有返回可显示的内容。")
                self._save(session_id, messages[1:])
                return {"type": "message", "message": answer, "session_id": session_id, "model": self.client.model}

            for tool_call in tool_calls:
                tool_id, name, arguments = self._tool_call(tool_call)
                try:
                    result = self.toolbox.execute(name, arguments)
                except (ToolValidationError, RuntimeError) as exc:
                    result = f"工具错误：{exc}"
                if isinstance(result, ToolProposal):
                    preview = self.proposal_handler(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": "已生成本地执行预览，正在等待用户确认。",
                        }
                    )
                    self._save(session_id, messages[1:])
                    return {
                        "type": "proposal",
                        "message": assistant_message["content"].strip() or f"我已准备“{result.title}”，请确认后执行。",
                        "proposal": preview,
                        "session_id": session_id,
                        "model": self.client.model,
                    }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result)[:MAX_TOOL_OUTPUT],
                    }
                )
        raise AgentRequestError("工具调用轮次过多，请缩小问题范围后重试。")

    @staticmethod
    def _session_id(value: str) -> str:
        value = str(value or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", value):
            raise ValueError("无效的会话标识。")
        return value

    @staticmethod
    def _tool_call(tool_call: dict) -> Tuple[str, str, dict]:
        try:
            tool_id = str(tool_call["id"])
            function = tool_call["function"]
            name = str(function["name"])
            arguments = json.loads(function.get("arguments") or "{}")
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentRequestError("模型返回了无效的工具调用。") from exc
        if not isinstance(arguments, dict):
            raise AgentRequestError("工具参数必须是 JSON 对象。")
        return tool_id, name, arguments

    def _save(self, session_id: str, messages: List[dict]) -> None:
        with self.lock:
            if len(self.sessions) >= 32 and session_id not in self.sessions:
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[session_id] = messages[-MAX_HISTORY_MESSAGES:]


def new_session_id() -> str:
    return uuid.uuid4().hex
