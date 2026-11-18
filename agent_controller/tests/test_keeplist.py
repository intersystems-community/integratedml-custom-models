import json
import subprocess
import sys
from pathlib import Path

from agent_controller.live.keeplist_hook import handle, keep_list, load, settings_for
from agent_controller.live.make_task_repo import build
from agent_controller.live.run_matrix import install_keeplist, split_condition

HOOK = Path(__file__).resolve().parents[1] / "live" / "keeplist_hook.py"


def ev(cwd, name, **kw):
    return {"cwd": str(cwd), "hook_event_name": name, "session_id": "s", **kw}


def test_keep_list_survives_compaction(tmp_path):
    handle(ev(tmp_path, "UserPromptSubmit",
              prompt="Fix the bugs. Do not modify anything under tests/. Keep Python 3.8."))
    handle(ev(tmp_path, "PostToolUse", tool_name="Edit",
              tool_input={"file_path": "app/rules/a.py"}, tool_response={}))
    handle(ev(tmp_path, "PostToolUse", tool_name="Bash", tool_input={"command": "python -m pytest -q"},
              tool_response={"stdout": "FAILED tests/test_a.py::test_x\n1 failed, 9 passed"}))
    pre = handle(ev(tmp_path, "PreCompact", trigger="auto", custom_instructions=None))
    assert "Do not modify anything under tests/." in pre and "app/rules/a.py" in pre
    post = json.loads(handle(ev(tmp_path, "SessionStart", source="compact")))
    ctx = post["hookSpecificOutput"]["additionalContext"]
    assert post["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "FAILED tests/test_a.py::test_x" in ctx and "Fix the bugs." in ctx
    assert load(str(tmp_path))["compactions"] == 1
    assert handle(ev(tmp_path, "SessionStart", source="startup")) == ""


def test_hook_script_never_fails(tmp_path):
    for payload in ("not json", json.dumps({"hook_event_name": "PostToolUse", "cwd": str(tmp_path),
                                            "tool_name": "Edit", "tool_input": None})):
        p = subprocess.run([sys.executable, str(HOOK)], input=payload, text=True,
                           capture_output=True)
        assert p.returncode == 0


def test_keep_list_is_bounded(tmp_path):
    state = load(str(tmp_path))
    state["task"] = "x" * 50_000
    assert len(keep_list(state)) <= 8_000


def test_settings_and_conditions(tmp_path):
    s = settings_for("/h.py", python="py")
    assert s["hooks"]["SessionStart"][0]["matcher"] == "compact"
    assert s["hooks"]["PreCompact"][0]["hooks"][0]["command"] == "py /h.py"
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps({"hooks": {"Stop": []}}))
    install_keeplist(tmp_path)
    merged = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert "Stop" in merged["hooks"] and "PreCompact" in merged["hooks"]
    assert split_condition({"A": "1"}) == ({"A": "1"}, False)
    assert split_condition({"env": {"A": "1"}, "keeplist": True}) == ({"A": "1"}, True)


def test_generated_task_fails_until_fixed(tmp_path):
    info = build(tmp_path / "repo", seed=3, n_modules=8, n_bugs=3, filler=2)
    key = Path(info["answer_key"])
    assert key.exists() and not (tmp_path / "repo" / key.name).exists()
    p = subprocess.run([sys.executable, "check.py", str(key)], cwd=tmp_path / "repo",
                       capture_output=True, text=True)
    score = json.loads(p.stdout.strip().splitlines()[-1])
    assert p.returncode == 1 and not score["tests_pass"] and score["changes_recall"] == 0.0
    assert score["tests_untouched"] and score["py38"] and score["signatures_kept"]
