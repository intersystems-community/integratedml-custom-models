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

## Files

| File | Contents |
|---|---|
| `sim.py` | Episode generator, environment, observation and action mask |
| `policies.py` | Rule baselines, tuned scoring heuristic, informed ceiling, hindsight oracle |
| `trace.py` | Save a decision trace and replay it exactly |
| `run_s0.py` | Tunes B2 and prints the S0 comparison |
| `gate.py` | Features, keep-score network, imitation and outcome-driven training |
| `run_s1.py` | Trains the gate and its ablations and prints the S1 comparison |
| `results_s0.json`, `results_s1.json` | Outputs of the two runs, including B2's tuned weights |

## Next

Rebuild the simulator along the lines of MEM1's multi-objective tasks before
testing the predictor: several objectives per episode, each needing a two-hop
chain; load scaled from 2 to 16 objectives at a fixed slot budget; gates
trained at low load and tested at high load; retrieve and stop as gated actions
with costs. If the learned gate still cannot clearly beat a tuned rule there,
the controller idea is in trouble.
