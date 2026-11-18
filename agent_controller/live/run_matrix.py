"""Run long Claude Code sessions under different context policies and record
whether the task passed and what it cost.

Each run gets a fresh git worktree of the task's repo at a fixed ref, runs
`claude -p` headless with the condition's environment, then runs the task's
check command (exit code 0 = pass). Results are appended to a JSONL file, one
line per run, with the cost Claude Code reports and context statistics from
the run's transcript.

    python -m agent_controller.live.run_matrix tasks.json --repeats 1 --dry-run
    python -m agent_controller.live.run_matrix tasks.json --repeats 1 --budget-usd 15

Runs cost real money. Start with --dry-run, then a pilot of one task.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..replay_transcript import load_session
from .keeplist_hook import settings_for

HOOK = Path(__file__).resolve().with_name("keeplist_hook.py")

PROJECTS = Path.home() / ".claude" / "projects"
DEFAULT_TOOLS = "Read Edit Write Glob Grep Bash"


def task_prompts(task: dict) -> List[str]:
    """The prompts of a task, in order. Later ones resume the same session."""
    if "prompt" in task:
        return [task["prompt"]]
    repo = Path(os.path.expanduser(task["repo"]))
    files = task.get("prompt_files") or [task["prompt_file"]]
    return [(repo / f).read_text() for f in files]


def split_condition(spec) -> tuple:
    """A condition is either a dict of env vars, or {"env": {...}, "keeplist": bool}."""
    if isinstance(spec, dict) and ("env" in spec or "keeplist" in spec):
        return dict(spec.get("env") or {}), bool(spec.get("keeplist"))
    return dict(spec or {}), False


def install_keeplist(tree: Path) -> None:
    """Register the keep-list hook in the worktree's project settings."""
    settings = tree / ".claude" / "settings.json"
    current = json.loads(settings.read_text()) if settings.exists() else {}
    hooks = current.setdefault("hooks", {})
    for event, groups in settings_for(str(HOOK))["hooks"].items():
        hooks.setdefault(event, []).extend(groups)
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(current, indent=2))


def claude_command(task: dict, args, prompt: str, resume: Optional[str] = None) -> List[str]:
    cmd = [
        args.claude, "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(task.get("max_turns", args.max_turns)),
        "--max-budget-usd", str(task.get("budget_usd", args.budget_usd)),
        "--allowedTools", args.allowed_tools,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.bare:
        cmd.append("--bare")
    if resume:
        cmd += ["--resume", resume]
    return cmd


def find_transcript(session_id: str) -> Optional[Path]:
    hits = list(PROJECTS.glob(f"*/{session_id}.jsonl"))
    return hits[0] if hits else None


def transcript_stats(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    s = load_session(path)
    if not s.requests:
        return {}
    return {
        "requests": len(s.requests),
        "peak_context": max(r.context for r in s.requests),
        "mean_context": sum(r.context for r in s.requests) / len(s.requests),
        "compactions": sum(r.reset for r in s.requests),
        # input cost in units of uncached input tokens, for comparing runs
        "input_cost_units": sum(r.actual_cost for r in s.requests),
    }


def run_one(task: dict, condition: str, spec, repeat: int, args) -> dict:
    env_extra, keeplist = split_condition(spec)
    repo = Path(os.path.expanduser(task["repo"])).resolve()
    work = Path(tempfile.mkdtemp(prefix=f"ccrun-{task['name']}-{condition}-"))
    tree = work / "repo"
    record = {"task": task["name"], "condition": condition, "repeat": repeat,
              "env": env_extra, "keeplist": keeplist, "started": time.time()}
    prompts = task_prompts(task)
    if args.dry_run:
        cmds = [claude_command(task, args, p, None if i == 0 else "<session_id>")
                for i, p in enumerate(prompts)]
        record["command"] = "\n   ".join(" ".join(shlex.quote(c) for c in cmd) for cmd in cmds)
        record["worktree"] = f"git -C {repo} worktree add --detach {tree} {task.get('ref', 'HEAD')}"
        shutil.rmtree(work, ignore_errors=True)
        return record
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(tree),
                    task.get("ref", "HEAD")], check=True, capture_output=True)
    try:
        if keeplist:
            install_keeplist(tree)
        env = {**os.environ, **env_extra}
        timeout = 60 * task.get("timeout_minutes", 90)
        t0 = time.time()
        session, cost, turns, phases = None, 0.0, 0, []
        for i, prompt in enumerate(prompts):
            cmd = claude_command(task, args, prompt, session if i else None)
            try:
                proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True,
                                      timeout=timeout)
                out, code = proc.stdout, proc.returncode
            except subprocess.TimeoutExpired as e:
                out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                code = "timeout"
            try:
                res = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
            except json.JSONDecodeError:
                res = {}
            phases.append({"exit": code, "session_id": res.get("session_id"),
                           "cost_usd": res.get("total_cost_usd"), "turns": res.get("num_turns"),
                           "is_error": res.get("is_error")})
            session = res.get("session_id") or session
            cost += res.get("total_cost_usd") or 0.0
            turns += res.get("num_turns") or 0
            if code != 0 or not session:
                break
        record["seconds"] = time.time() - t0
        record["phases"] = phases
        record["phases_completed"] = sum(1 for ph in phases if ph["exit"] == 0)
        record["claude_exit"] = phases[-1]["exit"] if phases else None
        record["session_id"] = session
        record["cost_usd"] = cost
        record["num_turns"] = turns
        record["is_error"] = any(ph.get("is_error") for ph in phases)
        check = subprocess.run(task["check"], shell=True, cwd=tree, capture_output=True,
                               text=True, timeout=60 * 20)
        record["passed"] = check.returncode == 0
        for line in reversed(check.stdout.strip().splitlines()):
            try:
                record["score"] = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        state = tree / ".keeplist" / "state.json"
        if state.exists():
            record["keeplist_state"] = json.loads(state.read_text())
        if record["session_id"]:
            record.update(transcript_stats(find_transcript(record["session_id"])))
    finally:
        if not args.keep_worktrees:
            subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(tree)],
                           capture_output=True)
            shutil.rmtree(work, ignore_errors=True)
    return record


def summarize(rows: List[dict]) -> str:
    by: Dict[str, List[dict]] = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(r)
    lines = ["| condition | runs | passed | tests pass | CHANGES.md recall | mean cost (USD) "
             "| mean peak context | compactions |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, rs in by.items():
        done = [r for r in rs if "passed" in r]
        if not done:
            continue
        cost = [r["cost_usd"] for r in done if r.get("cost_usd") is not None]
        peak = [r["peak_context"] for r in done if "peak_context" in r]
        comp = sum(r.get("compactions", 0) for r in done)
        scores = [r["score"] for r in done if isinstance(r.get("score"), dict)]
        tp = sum(bool(s.get("tests_pass")) for s in scores)
        rec = [s["changes_recall"] for s in scores if s.get("changes_recall") is not None]
        recall = f"{sum(rec) / len(rec):.2f}" if rec else "n/a"
        lines.append(
            f"| {cond} | {len(done)} | {sum(r['passed'] for r in done)} | {tp}/{len(scores)} "
            f"| {recall} "
            f"| {sum(cost) / len(cost) if cost else float('nan'):.2f} "
            f"| {sum(peak) / len(peak) if peak else float('nan'):,.0f} | {comp} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> List[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tasks", type=Path, help="JSON file with tasks and conditions")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only-task", default=None)
    ap.add_argument("--only-condition", default=None)
    ap.add_argument("--budget-usd", type=float, default=10.0, help="cap per run")
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--model", default=None)
    ap.add_argument("--allowed-tools", default=DEFAULT_TOOLS)
    ap.add_argument("--bare", action="store_true",
                    help="skip hooks, CLAUDE.md and MCP discovery for cleaner comparisons")
    ap.add_argument("--claude", default="claude", help="path to the claude binary")
    ap.add_argument("--out", type=Path, default=Path("live_results.jsonl"))
    ap.add_argument("--keep-worktrees", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print what would run")
    args = ap.parse_args(argv)

    spec = json.loads(args.tasks.read_text())
    conditions = spec.get("conditions") or {"default": {}}
    rows = []
    # interleave conditions within each repeat so drift over time hits all of them
    for rep in range(args.repeats):
        for task in spec["tasks"]:
            if args.only_task and task["name"] != args.only_task:
                continue
            for cond, env_extra in conditions.items():
                if args.only_condition and cond != args.only_condition:
                    continue
                rec = run_one(task, cond, env_extra, rep, args)
                rows.append(rec)
                if args.dry_run:
                    print(f"[{task['name']} / {cond} / {rep}]")
                    print("  env:", rec["env"] or "(none)", "| keep-list hook:", rec["keeplist"])
                    print("  ", rec["worktree"])
                    print("  ", rec["command"])
                else:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{task['name']} / {cond} / {rep}: passed={rec.get('passed')} "
                          f"cost=${rec.get('cost_usd')} peak={rec.get('peak_context')}")
    if not args.dry_run:
        print("\n" + summarize(rows))
    return rows


if __name__ == "__main__":
    main()
