# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from qlib.cli.ui import ActionRunner, Run, build_actions, create_server, recommend_actions


@pytest.fixture
def actions(tmp_path):
    return build_actions(data_dir=tmp_path / "data", python=sys.executable)


def test_recommend_actions_understands_chinese_intent(actions):
    result = recommend_actions("我想训练模型并回测收益和风险", actions)

    assert result["action_ids"][0] == "lightgbm-backtest"
    assert "LightGBM" in result["answer"]


def test_natural_language_never_changes_whitelisted_command(actions):
    expected = actions["environment"].command

    result = recommend_actions("运行; rm -rf / ../../other.yaml", actions)

    assert result["action_ids"] == ["environment", "preview-data"]
    assert actions["environment"].command == expected
    assert "rm" not in " ".join(expected)


def test_catalog_cards_do_not_expose_execution_details(actions):
    card = actions["preview-data"].card_dict()

    assert "command" not in card
    assert "cwd" not in card


def test_preview_token_is_required_and_single_use(actions):
    runner = ActionRunner(actions)

    with pytest.raises(PermissionError):
        runner.start("environment", "missing")

    preview = runner.preview("environment")
    with patch.object(runner, "_execute"):
        run = runner.start("environment", preview["confirmation_token"])

    assert run.action_id == "environment"
    with pytest.raises(PermissionError):
        runner.start("environment", preview["confirmation_token"])


def test_preview_token_cannot_be_used_for_another_action(actions):
    runner = ActionRunner(actions)
    token = runner.preview("environment")["confirmation_token"]

    with pytest.raises(PermissionError):
        runner.start("preview-data", token)


def test_only_one_action_can_run_at_a_time(actions):
    runner = ActionRunner(actions)
    runner.runs["active"] = Run(id="active", action_id="environment", status="running")
    token = runner.preview("preview-data")["confirmation_token"]

    with pytest.raises(RuntimeError, match="already running"):
        runner.start("preview-data", token)


def test_expired_preview_token_is_rejected(actions):
    runner = ActionRunner(actions)
    token = runner.preview("environment")["confirmation_token"]
    action_id, _ = runner.confirmations[token]
    runner.confirmations[token] = (action_id, time.time() - 1)

    with pytest.raises(PermissionError):
        runner.start("environment", token)


def test_all_actions_use_fixed_argument_lists_and_trusted_paths(actions):
    repo_root = Path(__file__).resolve().parents[2]
    config_root = (repo_root / "examples" / "benchmarks" / "LightGBM").resolve()

    for action in actions.values():
        assert isinstance(action.command, tuple)
        assert action.command[0] == sys.executable

    backtest_config = (actions["lightgbm-backtest"].cwd / actions["lightgbm-backtest"].command[-1]).resolve()
    assert backtest_config.is_file()
    assert config_root in backtest_config.parents


def test_server_rejects_non_loopback_binding():
    with pytest.raises(ValueError, match="loopback"):
        create_server("0.0.0.0", 0)


def test_http_api_requires_preview_token():
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        catalog = json.load(urlopen(base_url + "/api/catalog"))
        assert len(catalog["actions"]) == 4
        assert "command" not in catalog["actions"][0]

        request = Request(
            base_url + "/api/runs",
            data=json.dumps({"action_id": "environment"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request)
        assert exc_info.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
