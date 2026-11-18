import json

from agent_controller.live.run_matrix import main, summarize


def test_dry_run_builds_one_command_per_task_condition_repeat(tmp_path, capsys):
    spec = {
        "tasks": [{"name": "t", "repo": str(tmp_path), "prompt": "fix it", "check": "true"}],
        "conditions": {"default": {}, "c200": {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000"}},
    }
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(spec))
    rows = main([str(path), "--repeats", "2", "--dry-run", "--budget-usd", "3"])
    assert [(r["condition"], r["repeat"]) for r in rows] == [
        ("default", 0), ("c200", 0), ("default", 1), ("c200", 1)]
    cmd = rows[0]["command"]
    assert "--output-format json" in cmd and "--max-budget-usd 3.0" in cmd
    assert rows[1]["env"] == {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000"}


def test_summary_table():
    rows = [
        {"condition": "a", "passed": True, "cost_usd": 2.0, "peak_context": 300_000, "compactions": 0},
        {"condition": "a", "passed": False, "cost_usd": 4.0, "peak_context": 500_000, "compactions": 1},
    ]
    table = summarize(rows)
    assert "| a | 2 | 1 | 3.00 | 400,000 | 1 |" in table
