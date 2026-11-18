"""Generate a long, deterministic bug-fixing task for live runs.

The repo is a Python package of many modules with planted bugs. Each bug sits
in a helper module, while the failing test is on a caller in another module,
so the agent has to read across the code base and its context grows. The
prompt states constraints up front that only matter at the end, and asks for a
CHANGES.md that lists every fix: after a few compactions, that list is the
part most likely to be forgotten.

    python -m agent_controller.live.make_task_repo ~/ccruns/bugs-a --seed 1
    # writes ~/ccruns/bugs-a (the repo) and ~/ccruns/bugs-a.bugs.json (the answer key);
    # `python check.py <answer key>` prints a JSON score and exits 0 on a full pass

The repo is a git repository with one commit, so the harness can make worktrees.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, List

DOMAINS = ["ledger", "inventory", "pricing", "shipping", "billing", "payroll", "audit",
           "forecast", "loyalty", "returns", "tax", "catalog", "orders", "refunds",
           "invoices", "budget", "rates", "quotas", "metrics", "reports"]

# (template, buggy mutation) for helper bodies; x, y are ints
HELPERS = [
    ("return x + y", "return x - y"),
    ("return x * y", "return x * y + 1"),
    ("return max(x, y)", "return min(x, y)"),
    ("return sum(range(x, y + 1))", "return sum(range(x, y))"),
    ("return (x * 3 + y) % 97", "return (x * 3 + y) % 96"),
    ("return x if x > y else y - x", "return x if x >= y else y - x"),
    ("return abs(x - y) * 2", "return abs(x - y) * 3"),
    ("return x // (y if y else 1)", "return x // (y + 1 if y else 1)"),
]

PROMPT = textwrap.dedent("""\
    This repository has failing tests. Fix every bug so that `python -m pytest -q` passes.

    Constraints, all of which will be checked at the end:
    - Do not modify anything under tests/.
    - Do not change any function signature or rename anything.
    - Keep the code compatible with Python 3.8: no match statements and no `X | Y` type unions.
    - Do not add third-party dependencies or new imports.

    The failing tests point at functions in app/; the bugs are in the business rules under
    app/rules/, usually for a different module than the one the test names. When you are done, write CHANGES.md at the repository root with one line per
    bug you fixed, in the form `path/to/module.py: function_name`.
""")


def _func(name: str, body: str, doc: str) -> str:
    return f"def {name}(x: int, y: int) -> int:\n    \"\"\"{doc}\"\"\"\n    {body}\n"


def build(root: Path, seed: int, n_modules: int, n_bugs: int, filler: int) -> Dict:
    rng = random.Random(seed)
    pkg = root / "app"
    (pkg / "rules").mkdir(parents=True)
    (root / "tests").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "rules" / "__init__.py").write_text("")
    mods = [f"{DOMAINS[i % len(DOMAINS)]}_{i:02d}" for i in range(n_modules)]
    helpers: List[Dict] = []
    for m in mods:
        for k in range(4):
            tmpl, bug = rng.choice(HELPERS)
            helpers.append({"module": m, "name": f"{m.split('_')[0]}_rule_{k}", "body": tmpl,
                            "bug": bug, "buggy": False})
    for h in rng.sample(helpers, n_bugs):
        h["buggy"] = True
    # every rules module is used by exactly one other module (a derangement),
    # so every bug sits behind at least one test
    order = list(range(n_modules))
    while True:
        rng.shuffle(order)
        if all(i != j for i, j in enumerate(order)):
            break

    def ref(body: str, x: int, y: int) -> int:
        env: Dict = {}
        exec(f"def f(x, y):\n    {body}", env)
        return env["f"](x, y)

    def both(body: str, x: int, y: int) -> int:
        return ref(body, x, y) + ref(body, y, x)

    def pick(h: Dict) -> tuple:
        """Test inputs on which the planted bug changes the caller's result."""
        pairs = [(rng.randrange(2, 40), rng.randrange(2, 40)) for _ in range(200)]
        pairs += [(v, v) for v in range(2, 40)]
        for x, y in pairs:
            if both(h["body"], x, y) != both(h["bug"], x, y):
                return x, y
        raise AssertionError(f"bug in {h['name']} cannot be detected")

    callers: List[Dict] = []
    tests: Dict[str, List[str]] = {m: [] for m in mods}
    for i, m in enumerate(mods):
        rules = [f'"""Business rules for {m.replace("_", " ")}."""', "",
                 "from __future__ import annotations", ""]
        for h in (h for h in helpers if h["module"] == m):
            body = h["bug"] if h["buggy"] else h["body"]
            rules.append(_func(h["name"], body, f"Business rule {h['name']}."))
        for k in range(filler // 2):
            rules.append(_func(f"_retired_{m}_rule_{k}", f"return (x - {k}) * (y + {k % 5}) // 3",
                               "Retired rule kept for audit history; not used."))
        (pkg / "rules" / f"{m}.py").write_text("\n".join(rules) + "\n")
        src = [f'"""{m.replace("_", " ").title()} calculations."""', "",
               "from __future__ import annotations", ""]
        # padding: realistic-looking but unused code, so modules are long
        for k in range(filler):
            src.append(_func(f"_legacy_{m}_{k}", f"return (x + {k}) * (y + {k % 7})",
                             "Legacy rule kept for compatibility; not used by current callers."))
        # callers use helpers from another module
        other = mods[order[i]]
        targets = [h for h in helpers if h["module"] == other]
        imports = sorted({h["name"] for h in targets})
        src.insert(3, f"from app.rules.{other} import {', '.join(imports)}")
        for k, h in enumerate(targets):
            cname = f"{m.split('_')[0]}_total_{k}"
            src.append(_func(cname, f"return {h['name']}(x, y) + {h['name']}(y, x)",
                             f"Total combining {h['name']} both ways."))
            callers.append({"module": m, "name": cname, "helper": h})
        (pkg / f"{m}.py").write_text("\n".join(src) + "\n")

    for c in callers:
        h = c["helper"]
        x, y = pick(h)
        want = both(h["body"], x, y)
        tests[c["module"]].append(
            f"def test_{c['name']}():\n    assert {c['name']}({x}, {y}) == {want}\n")
    for m, ts in tests.items():
        names = sorted({t.split('(')[0][9:] for t in ts})
        head = f"from app.{m} import {', '.join(names)}\n\n" if names else ""
        (root / "tests" / f"test_{m}.py").write_text(head + "\n".join(ts))

    bugs = sorted(f"app/rules/{h['module']}.py: {h['name']}" for h in helpers if h["buggy"])
    key = root.parent / f"{root.name}.bugs.json"  # answer key, kept outside the repo
    key.write_text(json.dumps(bugs))
    (root / "check.py").write_text(CHECK)
    (root / "PROMPT.md").write_text(PROMPT)
    (root / ".gitignore").write_text(".keeplist/\n__pycache__/\n.claude/\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=task", "-c", "user.email=task@example.invalid",
                    "commit", "-qm", "task"], cwd=root, check=True)
    return {"repo": str(root), "answer_key": str(key), "bugs": len(bugs), "modules": n_modules}


CHECK = '''"""Score the task. Prints one JSON line; exits 0 only if everything passes."""
import ast, json, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parent
key = Path(sys.argv[1]) if len(sys.argv) > 1 else None
bugs = set(json.loads(key.read_text())) if key and key.exists() else set()
import os, shutil
for d in root.rglob("__pycache__"):  # stale bytecode can mask a same-size edit
    shutil.rmtree(d, ignore_errors=True)
t = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                   cwd=root, capture_output=True, text=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
tests_pass = t.returncode == 0
diff = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=root, capture_output=True,
                      text=True).stdout.split()
tests_untouched = not any(p.startswith("tests/") for p in diff)
py38 = True
sig_ok = True
for f in (root / "app").rglob("*.py"):
    src = f.read_text()
    try:
        ast.parse(src, feature_version=(3, 8))
    except SyntaxError:
        py38 = False
    new = {n.name: [a.arg for a in n.args.args] for n in ast.walk(ast.parse(src))
           if isinstance(n, ast.FunctionDef)}
    old_src = subprocess.run(["git", "show", f"HEAD:{f.relative_to(root).as_posix()}"], cwd=root,
                             capture_output=True, text=True).stdout
    if old_src:
        old = {n.name: [a.arg for a in n.args.args] for n in ast.walk(ast.parse(old_src))
               if isinstance(n, ast.FunctionDef)}
        if any(new.get(k) != v for k, v in old.items()):
            sig_ok = False
changes = root / "CHANGES.md"
listed = set()
if changes.exists():
    for line in changes.read_text().splitlines():
        line = line.strip(" -*`")
        for b in bugs:
            if b.split(": ")[0] in line and b.split(": ")[1] in line:
                listed.add(b)
recall = len(listed) / len(bugs) if bugs else None
score = {"tests_pass": tests_pass, "tests_untouched": tests_untouched, "py38": py38,
         "signatures_kept": sig_ok, "changes_recall": round(recall, 3)}
score["all_ok"] = tests_pass and tests_untouched and py38 and sig_ok and recall in (1.0, None)
print(json.dumps(score))
sys.exit(0 if score["all_ok"] else 1)
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dest", type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--modules", type=int, default=40)
    ap.add_argument("--bugs", type=int, default=24)
    ap.add_argument("--filler", type=int, default=60, help="padding functions per module")
    args = ap.parse_args()
    if args.dest.exists() and any(args.dest.iterdir()):
        raise SystemExit(f"{args.dest} exists and is not empty")
    info = build(args.dest, args.seed, args.modules, args.bugs, args.filler)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
