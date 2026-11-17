"""Sprint S0: is there headroom for a learned gate?

Tunes a static scoring heuristic on training episodes, then compares it with
simple rules and a hindsight oracle on held-out episodes at the same slot and
retrieval budget.

    python -m agent_controller.run_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .policies import (
    SCORE_FEATURES,
    Informed,
    Oracle,
    ScoreHeuristic,
    lru,
    never_write,
    recency,
    type_priority,
)
from .sim import Config, Episode, PolicyFactory, generate_episode, run_episode

SOURCES = ("slot_ok", "slot_wrong", "retrieve_ok", "retrieve_wrong", "none")


def evaluate(
    cfg: Config,
    factory: PolicyFactory,
    episodes: Sequence[Episode],
    n_slots: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    acc = np.empty(len(episodes))
    counts = np.zeros(len(SOURCES))
    for i, ep in enumerate(episodes):
        res = run_episode(cfg, ep, factory(ep), n_slots)
        acc[i] = res.accuracy
        for row in res.trace:
            if row["kind"] != "query":
                continue
            if row["source"] == "none":
                name = "none"
            else:
                name = f"{row['source']}_{'ok' if row['correct'] else 'wrong'}"
            counts[SOURCES.index(name)] += 1
    return {"acc": acc, "sources": counts / counts.sum()}


def tune_heuristic(
    cfg: Config, episodes: Sequence[Episode], trials: int, seed: int
) -> ScoreHeuristic:
    """Random search, then one round of local perturbation around the best."""
    rng = np.random.default_rng(seed)

    def score(h: ScoreHeuristic) -> float:
        return float(evaluate(cfg, lambda ep: h, episodes)["acc"].mean())

    def sample(center: Optional[ScoreHeuristic], scale: float) -> ScoreHeuristic:
        if center is None:
            w = rng.normal(0.0, 1.0, len(SCORE_FEATURES))
            return ScoreHeuristic(
                tuple(w), float(rng.uniform(0, 1)), float(rng.uniform(0, 0.6))
            )
        w = np.array(center.w) + rng.normal(0.0, scale, len(SCORE_FEATURES))
        return ScoreHeuristic(
            tuple(w),
            float(np.clip(center.replace_conf + rng.normal(0, scale / 3), 0, 1)),
            float(np.clip(center.write_conf + rng.normal(0, scale / 3), 0, 1)),
        )

    best = ScoreHeuristic((1.0, 1.0, 0.5, -0.5, -1.0, 1.0, 1.0))  # hand-set start
    best_s = score(best)
    for phase, scale in ((None, 0.0), ("local", 0.3)):
        for _ in range(trials // 2):
            h = sample(best if phase else None, scale)
            s = score(h)
            if s > best_s:
                best, best_s = h, s
    return best


def mean_ci(x: np.ndarray) -> tuple:
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def main(argv: Optional[List[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=300, help="tuning episodes")
    ap.add_argument("--test", type=int, default=3000, help="held-out episodes")
    ap.add_argument("--trials", type=int, default=300, help="tuning trials")
    ap.add_argument("--slots", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--out", type=Path, default=None, help="write results JSON")
    args = ap.parse_args(argv)

    cfg = Config()
    train = [generate_episode(cfg, s) for s in range(args.train)]
    test = [generate_episode(cfg, 1_000_000 + s) for s in range(args.test)]

    results: dict = {"config": cfg.__dict__, "by_slots": {}}
    for k in args.slots:
        kcfg = Config(**{**cfg.__dict__, "n_slots": k})
        tuned = tune_heuristic(kcfg, train, args.trials, seed=k)
        policies: Dict[str, PolicyFactory] = {
            "no working memory": lambda ep: never_write,
            "recency": lambda ep: recency,
            "LRU": lambda ep: lru,
            "type priority": lambda ep: type_priority,
            "tuned heuristic (B2)": lambda ep: tuned,
            "informed ceiling": Informed,
            "hindsight oracle": Oracle,
        }
        evals = {name: evaluate(kcfg, f, test) for name, f in policies.items()}
        b2 = evals["tuned heuristic (B2)"]["acc"]
        gap, gap_half = mean_ci(evals["informed ceiling"]["acc"] - b2)
        ogap, ogap_half = mean_ci(evals["hindsight oracle"]["acc"] - b2)

        print(f"\n## {k} slots, {cfg.retrieval_budget} retrievals, {len(test)} episodes")
        print("| policy | accuracy | ±95% | slot ok | slot wrong | retr ok | retr wrong | unanswered |")
        print("|---|---|---|---|---|---|---|---|")
        rows = {}
        for name, e in evals.items():
            m, h = mean_ci(e["acc"])
            src = " | ".join(f"{v:.1%}" for v in e["sources"])
            print(f"| {name} | {m:.1%} | {h:.1%} | {src} |")
            rows[name] = {
                "accuracy": m,
                "ci95": h,
                "sources": dict(zip(SOURCES, e["sources"].tolist())),
            }
        print(f"\nLearnable headroom at most (informed − B2, paired): {gap:+.1%} ± {gap_half:.1%}")
        print(f"Hindsight gap (oracle − B2, paired): {ogap:+.1%} ± {ogap_half:.1%}")
        results["by_slots"][k] = {
            "policies": rows,
            "headroom": gap,
            "headroom_ci95": gap_half,
            "hindsight_gap": ogap,
            "hindsight_gap_ci95": ogap_half,
            "tuned": {
                "weights": dict(zip(SCORE_FEATURES, tuned.w)),
                "replace_conf": tuned.replace_conf,
                "write_conf": tuned.write_conf,
            },
        }

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
