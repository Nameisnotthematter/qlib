# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from unittest.mock import patch

import pytest

from qlib.cli.ui_agent import (
    AgentConfigurationError,
    AgentRequestError,
    AssistantService,
    OpenRouterClient,
    QlibToolbox,
    ToolProposal,
    ToolValidationError,
    load_openrouter_key,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


class FakeClient:
    model = "openai/gpt-5.6-luna"
    configured = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((json.loads(json.dumps(messages)), tools))
        return self.responses.pop(0)


class FakeToolbox:
    def __init__(self, result="tool result"):
        self.result = result
        self.calls = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "inspect_environment"}}]

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


def test_openrouter_request_uses_luna_and_does_not_put_key_in_payload():
    captured = {}

    def transport(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "完成"}}]})

    client = OpenRouterClient(key_loader=lambda: "private-key", transport=transport)
    result = client.complete([{"role": "user", "content": "你好"}], [])
    payload = json.loads(captured["request"].data)

    assert result["content"] == "完成"
    assert payload["model"] == "openai/gpt-5.6-luna"
    assert "parallel_tool_calls" not in payload
    assert payload["stream"] is False
    assert payload["reasoning_effort"] == "low"
    assert payload["provider"] == {"require_parameters": True, "allow_fallbacks": True}
    assert "private-key" not in json.dumps(payload)
    assert captured["request"].headers["Authorization"] == "Bearer private-key"


def test_openrouter_treats_http_200_error_body_as_failure():
    client = OpenRouterClient(
        key_loader=lambda: "key",
        transport=lambda *_args, **_kwargs: FakeResponse({"error": {"message": "provider failed"}}),
    )

    with pytest.raises(AgentRequestError, match="provider failed"):
        client.complete([], [])


def test_key_file_requires_private_permissions(tmp_path):
    key_file = tmp_path / "openrouter.key"
    key_file.write_text("a-test-key\n")
    key_file.chmod(0o644)
    with patch.dict("qlib.cli.ui_agent.os.environ", {}, clear=True):
        with pytest.raises(AgentConfigurationError, match="chmod 600"):
            load_openrouter_key(key_file)

    key_file.chmod(0o600)
    with patch.dict("qlib.cli.ui_agent.os.environ", {}, clear=True):
        assert load_openrouter_key(key_file) == "a-test-key"


def test_assistant_preserves_tool_call_and_result_in_followup():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "inspect_environment", "arguments": "{}"},
    }
    client = FakeClient(
        [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "assistant", "content": "环境可用。"},
        ]
    )
    toolbox = FakeToolbox()
    service = AssistantService(client, toolbox, lambda _proposal: {})

    result = service.chat("session123", "检查环境")

    assert result["message"] == "环境可用。"
    assert toolbox.calls == [("inspect_environment", {})]
    followup = client.calls[1][0]
    assert followup[-2]["tool_calls"][0]["id"] == "call_1"
    assert followup[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "tool result"}


def test_proposal_is_registered_without_executing_command(tmp_path):
    proposal = ToolProposal("运行测试", "说明", "1 分钟", ("python", "safe.py"), tmp_path)
    tool_call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "run_workflow", "arguments": '{"workflow_id":"safe"}'},
    }
    client = FakeClient([{"role": "assistant", "content": "请确认", "tool_calls": [tool_call]}])
    toolbox = FakeToolbox(proposal)
    previews = []
    service = AssistantService(client, toolbox, lambda value: previews.append(value) or {"action": {"id": "a"}})

    result = service.chat("session123", "运行工作流")

    assert result["type"] == "proposal"
    assert previews == [proposal]


@pytest.fixture
def toolbox(tmp_path):
    examples = tmp_path / "examples" / "benchmarks" / "LightGBM"
    examples.mkdir(parents=True)
    (examples / "workflow.yaml").write_text("model: {}")
    return QlibToolbox(tmp_path, tmp_path / "data", python="python")


def test_toolbox_rejects_unknown_workflow_and_extra_parameters(toolbox):
    with pytest.raises(ToolValidationError, match="未知 workflow_id"):
        toolbox.execute("run_workflow", {"workflow_id": "../../bad.yaml"})
    with pytest.raises(ToolValidationError, match="不接受参数"):
        toolbox.execute("inspect_environment", {"command": "rm -rf /"})


@pytest.mark.parametrize("expression", ['().__class__', '__import__(os)', 'lambda:1'])
def test_toolbox_rejects_unsafe_expressions_before_subprocess(toolbox, expression):
    with patch.object(toolbox, "_run") as run:
        with pytest.raises(ToolValidationError):
            toolbox.execute(
                "query_market_data",
                {
                    "instruments": ["SH600000"],
                    "fields": [expression],
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                },
            )
    run.assert_not_called()


def test_toolbox_accepts_common_factor_expression(toolbox):
    with patch.object(toolbox, "_run", return_value="ok") as run:
        result = toolbox.execute(
            "query_market_data",
            {
                "instruments": ["SH600000"],
                "fields": ["$close / Ref($close, 1) - 1"],
                "start_date": "2026-01-01",
                "end_date": "2026-01-02",
            },
        )

    assert result == "ok"
    assert run.called
