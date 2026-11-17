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

## Files

| File | Contents |
|---|---|
| `sim.py` | Episode generator, environment, observation and action mask |
| `policies.py` | Rule baselines, tuned scoring heuristic, informed ceiling, hindsight oracle |
| `trace.py` | Save a decision trace and replay it exactly |
| `run_s0.py` | Tunes B2 and prints the comparison |
| `results_s0.json` | Output of the run above, including the tuned weights |

## Next: S1

Train a masked gate policy on the same observations, warm-started by imitating
the informed policy, and compare it with B2 at 2 and 4 slots. Stop or rethink
if it does not beat B2.
