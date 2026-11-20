#!/usr/bin/env bash
# Pre-flight for live runs in a cloud environment:
#   1. claude can reach a model (API key or Bedrock credentials are set),
#   2. /compact works in headless mode, so the compaction conditions can run,
#   3. the pilot task repos exist.
# Costs a few cents. Run from the repository root:
#   bash agent_controller/live/cloud_check.sh
set -euo pipefail

echo "claude: $(claude --version 2>/dev/null || echo 'not found')"
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then echo "auth: ANTHROPIC_API_KEY is set";
elif [[ "${CLAUDE_CODE_USE_BEDROCK:-}" == "1" ]]; then echo "auth: Bedrock (region ${AWS_REGION:-unset})";
else echo "auth: no ANTHROPIC_API_KEY and CLAUDE_CODE_USE_BEDROCK is not 1"; exit 1; fi

work=$(mktemp -d)
cd "$work"
git init -q .
echo "A small scratch repo for a pre-flight check." > README.md

echo; echo "1) headless call"
first=$(claude -p "Read README.md and reply with one short sentence about it." --output-format json --max-turns 3)
sid=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1])["session_id"])' <<<"$first")
echo "   session $sid"

echo "2) /compact on the same session"
second=$(claude -p "/compact" --resume "$sid" --output-format json)
python3 -c 'import json,sys; r=json.loads(sys.stdin.read().strip().splitlines()[-1]); print("   result:", str(r.get("result",""))[:120].replace("\n"," "), "| cost $", r.get("total_cost_usd"))' <<<"$second"

cd - >/dev/null
python3 - "$sid" <<'PY'
import sys
from agent_controller.live.run_matrix import compaction_markers, find_transcript
from agent_controller.replay_transcript import load_session
path = find_transcript(sys.argv[1])
print("   transcript:", path)
markers = compaction_markers(path)
resets = sum(r.reset for r in load_session(path).requests) if path else 0
print(f"   compaction markers: {markers}, prompt-size drops: {resets}")
print("   => /compact works headless" if markers or resets else
      "   => no compaction seen; the harness should use fresh-session phases instead")
PY

echo; echo "3) pilot task repos"
for seed in 1 2; do
  name=$([[ $seed == 1 ]] && echo a || echo b)
  dest="$HOME/ccruns/bugs-$name"
  if [[ -d "$dest" ]]; then echo "   $dest exists"
  else python3 -m agent_controller.live.make_task_repo "$dest" --seed "$seed"; fi
done
rm -rf "$work"
echo; echo "Next: python3 -m agent_controller.live.run_matrix agent_controller/live/tasks.pilot.json --repeats 3 --budget-usd 10"
