# Agent controller prototype

A staged test of whether a small learned controller around a frozen LLM can
manage working memory better than hand-written rules. This folder is
self-contained (numpy and pytest only) and unrelated to the IntegratedML demos
around it; it is meant to move to its own repository.

## Sprint S0: is there headroom for a learned gate?

No LLM is involved. `sim.py` scripts a delayed-dependency task: facts arrive
with distractors and contradictions, only a few working slots are available,
and later queries ask for values seen much earlier. A gate policy decides, for
each arriving fact, which slot to overwrite or whether to drop it. A missed
query can fall back on a limited number of lookups in the durable store, which
returns the most recent claim about that key, true or not.

```bash
python -m agent_controller.run_s0 --out agent_controller/results_s0.json   # ~3 min
python -m pytest agent_controller/tests
```

### Result

Query accuracy on 3,000 held-out episodes, 2 retrievals per episode. All 95%
intervals are within ±0.8 points.

| Policy | 2 slots | 4 slots | 6 slots | 8 slots |
|---|---|---|---|---|
| No working memory | 16.8% | 16.8% | 16.8% | 16.8% |
| Recency | 31.7% | 47.6% | 63.4% | 76.6% |
| LRU | 32.1% | 50.3% | 68.3% | 81.4% |
| Type priority | 40.8% | 59.5% | 73.6% | 84.1% |
| Tuned heuristic (B2) | 55.0% | 76.5% | 89.2% | 94.4% |
| Informed ceiling | 75.3% | 91.3% | 97.1% | 99.1% |
| Hindsight oracle | 85.3% | 98.5% | 99.9% | 100.0% |
| **Informed − B2** | **+20.3** | **+14.8** | **+7.9** | **+4.7** |

S0 passes: with tight memory there are up to 15–20 points between the best
static rule and a gate that knew the task's hidden structure, well above the
brief's +5 point bar. The gap falls below 5 points at 8 slots, so a learned
gate can only pay off when working memory is scarce relative to the number of
live facts.

### How to read the reference rows

- **Tuned heuristic (B2)** is a fixed linear keep-score over observable
  features, with weights found by random search on 300 separate episodes. It is
  the baseline a learned gate has to beat.
- **Informed ceiling** is told which claims are true and how likely each key is
  to be asked about, but not the future. A learned gate has to infer both from
  noisy cues, so it will land somewhere below this row. The gap to B2 is an
  upper bound on what S1 can win, not a forecast.
- **Hindsight oracle** sees the whole episode. The part of its lead over the
  informed ceiling is luck about which questions get asked, and no policy can
  learn it.

### Limits of this check

- The task is synthetic and its parameters (`Config` in `sim.py`) are choices,
  not measurements of a real agent workload. Headroom here shows the benchmark
  can discriminate between gates; it does not show the same gap exists with a
  real LLM proposer. That is sprint S3.
- The oracle does not plan around the retrieval budget, so it is a strong
  reference rather than the exact optimum.
- Retrieval is automatic on a miss. Deciding when to retrieve, and when to
  stop, is not yet a gated action.

## Sprint S1: can a small learned gate capture the headroom?

`gate.py` trains a keep-score network (one hidden layer of 32 units, 22
observable inputs) in two stages: imitation of the informed policy with DAgger,
then outcome-driven improvement, where the rest of an episode is replayed once
per allowed action and the gate is pushed toward the choices that answered more
queries. It runs on numpy alone.

```bash
python -m agent_controller.run_s1 --out agent_controller/results_s1.json   # ~25 min on 4 cores
```

### Result

Same 3,000 held-out episodes as S0. Gains are paired differences against the
tuned heuristic, all within ±0.8 points. The learned gate is the mean of three
training seeds.

| Gate | 2 slots | vs B2 | 4 slots | vs B2 |
|---|---|---|---|---|
| Tuned heuristic (B2) | 55.0% | | 76.5% | |
| Learned gate, imitation only | 56.4% | +1.4 | 77.9% | +1.4 |
| **Learned gate (A1)** | **58.0%** | **+3.0** | **79.2%** | **+2.7** |
| Ablation: linear, same inputs | 55.3% | +0.3 | 76.3% | −0.2 |
| Ablation: network on B2's seven inputs | 54.2% | −0.8 | 75.2% | −1.3 |
| Diagnostic: told which claims are true | 59.5% | +4.5 | 81.1% | +4.6 |
| Diagnostic: told each key's query weight | 73.3% | +18.3 | 89.0% | +12.4 |
| Informed ceiling | 75.3% | +20.3 | 91.3% | +14.8 |

S1 is a weak pass. The learned gate beats the tuned heuristic reliably, but by
about 3 points, short of the brief's +5 bar, and it captures only 15–18% of
the S0 headroom.

What the other rows say:

- **Both ingredients are needed.** Richer inputs with a linear score, or a
  network on B2's inputs, do not beat B2. Outcome-driven improvement roughly
  doubles the gain from imitation alone.
- **The limit is information, not the policy.** Telling the gate each key's
  true query weight recovers 84–90% of the headroom; telling it which claims
  are true recovers 22–31%. The observable cues about which keys matter are
  too noisy for any gate to do much better here.
- **B2 is close to a learned gate already.** Its nine parameters were searched
  against outcomes on this same distribution. Against the hand-written rules
  in the S0 table (41–60% at these budgets) the learned gate is about 17–20
  points ahead.

### What this means for the benchmark

In this task a fact's worth is close to a weighted sum of its own features, so
a tuned linear rule is nearly as good as anything. The task has no dependency
chains, one static goal, automatic retrieval and no stop decision, and gates
are tuned and tested on the same distribution. It does not yet exercise what
the brief claims a learned controller is for.

## Coding sessions: offline replay

`replay_transcript.py` asks the S0 question for real coding sessions. It
replays Claude Code transcripts under different context policies, where a tool
result can be evicted from the prompt (leaving a short stub) and re-fetched if
needed, and reports input cost against keeping everything.

```bash
python -m agent_controller.replay_transcript ~/.claude/projects/*/*.jsonl
```

It runs locally on the standard library and prints aggregate numbers only.
Add `--diagnose` for parsing checks (prompt size percentiles, resets, cache
misses) and `--breakdown` for where context comes from (tool results, your
messages, the agent's text, tool calls and thinking, attachments) and how much
input cost cache misses add, grouped by the pause before the request.

- **Cost model.** Context sizes come from the recorded token usage. Prompt
  caching is modelled: an unchanged prefix is read at 0.1x, and everything
  after the first change is written again at 1.25x. Evicting an old result
  therefore costs a rewrite of what follows it, so eviction is cheap right
  after a result arrives and expensive once it is buried.
- **Policies.** Keep everything; truncate oldest results at half of peak
  context; a stale rule (superseded, or old and unused); and a hindsight
  reference that evicts only results never used again.
- **Misses.** A result evicted and later used is re-fetched and charged one
  extra request.

Limits: it measures cost, not whether the agent would still have succeeded on
a trimmed prompt. A result counts as used later only if the agent quotes a line
from it or edits the file it came from, which undercounts real use and so
flatters eviction. Only tool results are evictable; the agent's own earlier
output (for example large file writes) is not.

On the one transcript available while building this (the session that wrote
this code: 33 requests, peak 188k tokens) the replay of keep-everything is
within 1% of the recorded input cost. Tool results were only 12% of new
context there and the hindsight reference saved 8.5%, with no later use
detected. That session is not representative of a long coding run; the tool
needs real ones before the numbers mean anything.

## Live runs: does earlier compaction keep answers as good?

`live/run_matrix.py` runs the same long coding tasks under different context
policies with Claude Code headless (`claude -p`), checks each result with the
task's own test command, and records the cost Claude Code reports plus peak
context and compactions from the run's transcript. Each run gets a fresh git
worktree at a fixed ref; conditions are interleaved so drift over the day hits
all of them. Conditions are environment settings, for example
`CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000` (documented range 100k to 1M).

```bash
cp agent_controller/live/tasks.example.json tasks.json   # edit: your repos, prompts, checks
python -m agent_controller.live.run_matrix tasks.json --dry-run
python -m agent_controller.live.run_matrix tasks.json --only-task NAME --budget-usd 10
```

Runs spend real money and run tools on your machine with the allowed tool
list (Bash included by default), inside the worktree. `sawtooth.py` plots one
session's prompt size and estimates what compacting at a lower threshold would
have cost.

## Files

| File | Contents |
|---|---|
| `sim.py` | Episode generator, environment, observation and action mask |
| `policies.py` | Rule baselines, tuned scoring heuristic, informed ceiling, hindsight oracle |
| `trace.py` | Save a decision trace and replay it exactly |
| `run_s0.py` | Tunes B2 and prints the S0 comparison |
| `gate.py` | Features, keep-score network, imitation and outcome-driven training |
| `run_s1.py` | Trains the gate and its ablations and prints the S1 comparison |
| `replay_transcript.py` | Replays session transcripts under context eviction policies |
| `results_s0.json`, `results_s1.json` | Outputs of the two runs, including B2's tuned weights |

## Next

Rebuild the simulator along the lines of MEM1's multi-objective tasks before
testing the predictor: several objectives per episode, each needing a two-hop
chain; load scaled from 2 to 16 objectives at a fixed slot budget; gates
trained at low load and tested at high load; retrieve and stop as gated actions
with costs. If the learned gate still cannot clearly beat a tuned rule there,
the controller idea is in trouble.
