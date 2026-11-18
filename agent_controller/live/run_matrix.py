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

PROJECTS = Path.home() / ".claude" / "projects"
DEFAULT_TOOLS = "Read Edit Write Glob Grep Bash"


def claude_command(task: dict, args) -> List[str]:
    cmd = [
        args.claude, "-p", task["prompt"],
        "--output-format", "json",
        "--max-turns", str(task.get("max_turns", args.max_turns)),
        "--max-budget-usd", str(task.get("budget_usd", args.budget_usd)),
        "--allowedTools", args.allowed_tools,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.bare:
        cmd.append("--bare")
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


def run_one(task: dict, condition: str, env_extra: Dict[str, str], repeat: int, args) -> dict:
    repo = Path(os.path.expanduser(task["repo"])).resolve()
    work = Path(tempfile.mkdtemp(prefix=f"ccrun-{task['name']}-{condition}-"))
    tree = work / "repo"
    record = {"task": task["name"], "condition": condition, "repeat": repeat,
              "env": env_extra, "started": time.time()}
    cmd = claude_command(task, args)
    if args.dry_run:
        record["command"] = " ".join(shlex.quote(c) for c in cmd)
        record["worktree"] = f"git -C {repo} worktree add --detach {tree} {task.get('ref', 'HEAD')}"
        shutil.rmtree(work, ignore_errors=True)
        return record
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(tree),
                    task.get("ref", "HEAD")], check=True, capture_output=True)
    try:
        env = {**os.environ, **env_extra}
        timeout = 60 * task.get("timeout_minutes", 90)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True,
                                  timeout=timeout)
            out = proc.stdout
            record["claude_exit"] = proc.returncode
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            record["claude_exit"] = "timeout"
        record["seconds"] = time.time() - t0
        try:
            res = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except json.JSONDecodeError:
            res = {}
        record["session_id"] = res.get("session_id")
        record["cost_usd"] = res.get("total_cost_usd")
        record["num_turns"] = res.get("num_turns")
        record["is_error"] = res.get("is_error")
        check = subprocess.run(task["check"], shell=True, cwd=tree, capture_output=True,
                               text=True, timeout=60 * 20)
        record["passed"] = check.returncode == 0
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
    lines = ["| condition | runs | passed | mean cost (USD) | mean peak context | compactions |",
             "|---|---|---|---|---|---|"]
    for cond, rs in by.items():
        done = [r for r in rs if "passed" in r]
        if not done:
            continue
        cost = [r["cost_usd"] for r in done if r.get("cost_usd") is not None]
        peak = [r["peak_context"] for r in done if "peak_context" in r]
        comp = sum(r.get("compactions", 0) for r in done)
        lines.append(
            f"| {cond} | {len(done)} | {sum(r['passed'] for r in done)} "
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
                    print("  env:", env_extra or "(none)")
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
