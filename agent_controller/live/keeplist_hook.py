"""Keep-list controller for Claude Code, as a hook script (version 0, rule-based).

It watches a session and keeps a short list of what must survive compaction:
the task as first stated, constraints the user gave, files the agent changed,
and the latest test result. At compaction it hands that list to Claude Code
twice: as instructions for the summary (PreCompact) and as context injected
into the fresh window right after (SessionStart with source "compact").

Register it for UserPromptSubmit, PostToolUse, PreCompact and SessionStart
(see `settings_for`). State lives in `<project>/.keeplist/state.json`.

This version decides what to keep with fixed rules. It is the baseline a
learned gate would have to beat.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

MAX_CHARS = 8_000
CONSTRAINT = re.compile(
    r"\b(must|must not|never|don't|do not|always|only|required|keep|without|avoid)\b", re.I
)
TEST_CMD = re.compile(r"\b(pytest|unittest|tox|nox|npm (run )?test|jest|go test|cargo test|make test)\b")
FAIL_LINE = re.compile(r"(FAILED|ERROR|Error:|AssertionError|failed|error:)")
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _state_path(cwd: str) -> Path:
    return Path(cwd) / ".keeplist" / "state.json"


def load(cwd: str) -> Dict:
    p = _state_path(cwd)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"task": "", "constraints": [], "files": [], "tests": None, "compactions": 0}


def save(cwd: str, state: Dict) -> None:
    p = _state_path(cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=1))


def _text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(_text(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(_text(v) for v in obj)
    return ""


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*") for p in parts if p.strip()]


def on_prompt(state: Dict, prompt: str) -> None:
    if not state["task"]:
        state["task"] = prompt.strip()[:1_500]
    for s in _sentences(prompt):
        if CONSTRAINT.search(s) and 8 <= len(s) <= 300 and s not in state["constraints"]:
            state["constraints"].append(s)
    state["constraints"] = state["constraints"][-25:]


def on_tool(state: Dict, tool: str, tool_input: Dict, response) -> None:
    if tool in EDIT_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        if path:
            if path in state["files"]:
                state["files"].remove(path)
            state["files"].append(path)
            state["files"] = state["files"][-40:]
    elif tool == "Bash":
        cmd = str(tool_input.get("command", ""))
        if TEST_CMD.search(cmd):
            out = _text(response)
            fails = [l.strip() for l in out.splitlines() if FAIL_LINE.search(l)]
            tail = [l for l in out.strip().splitlines()[-3:]]
            state["tests"] = {
                "command": cmd[:300],
                "failing_lines": fails[:20],
                "summary": " | ".join(tail)[:400],
            }


def keep_list(state: Dict) -> str:
    lines = ["# Keep list (maintained outside the model; preserve all of it)"]
    if state["task"]:
        lines += ["", "## Task as originally stated", state["task"]]
    if state["constraints"]:
        lines += ["", "## Constraints the user stated"] + [f"- {c}" for c in state["constraints"]]
    if state["files"]:
        lines += ["", "## Files changed so far (most recent last)"] + [f"- {f}" for f in state["files"]]
    t = state.get("tests")
    if t:
        lines += ["", "## Latest test run", f"- command: {t['command']}", f"- result: {t['summary']}"]
        lines += [f"- {l}" for l in t["failing_lines"]]
    text = "\n".join(lines)
    return text[:MAX_CHARS]


def handle(event: Dict) -> str:
    """Process one hook event; return what to print on stdout."""
    cwd = event.get("cwd") or "."
    name = event.get("hook_event_name", "")
    state = load(cwd)
    out = ""
    if name == "UserPromptSubmit":
        on_prompt(state, str(event.get("prompt", "")))
    elif name == "PostToolUse":
        on_tool(state, str(event.get("tool_name", "")), event.get("tool_input") or {},
                event.get("tool_response"))
    elif name == "PreCompact":
        state["compactions"] += 1
        out = (
            "When summarizing, carry the following keep list into the summary verbatim, "
            "including every constraint and every changed file.\n\n" + keep_list(state)
        )
    elif name == "SessionStart" and event.get("source") == "compact":
        out = json.dumps({"hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": keep_list(state),
        }})
    save(cwd, state)
    return out


def settings_for(script: str, python: str = sys.executable) -> Dict:
    """Project settings.json fragment that registers this hook."""
    cmd = f"{python} {script}"
    h = [{"type": "command", "command": cmd, "timeout": 30}]
    return {"hooks": {
        "UserPromptSubmit": [{"hooks": h}],
        "PostToolUse": [{"matcher": "Edit|Write|MultiEdit|NotebookEdit|Bash", "hooks": h}],
        "PreCompact": [{"matcher": "auto", "hooks": h}, {"matcher": "manual", "hooks": h}],
        "SessionStart": [{"matcher": "compact", "hooks": h}],
    }}


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    try:
        out = handle(event)
    except Exception as e:  # a broken hook must never break the session
        print(f"keeplist hook error: {e}", file=sys.stderr)
        return
    if out:
        print(out)


if __name__ == "__main__":
    main()
