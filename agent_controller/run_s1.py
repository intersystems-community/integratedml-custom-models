"""Sprint S1: can a small learned gate capture the headroom found in S0?

Trains a keep-score network by imitation, then by outcome-driven improvement,
and compares it with the S0 tuned heuristic on the same held-out episodes.
Ablations show how much comes from richer inputs, from the hidden layer, and
from each hidden quantity the gate has to infer.

    python -m agent_controller.run_s1
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np

from .gate import B2_FEATURES, OBSERVABLE, improve_gate, train_gate
from .policies import Informed, ScoreHeuristic
from .run_s0 import evaluate, mean_ci
from .sim import Config, generate_episode

HERE = Path(__file__).parent

# name -> (hidden units, input features, run outcome-driven improvement)
VARIANTS = {
    "learned gate, imitation only": (32, OBSERVABLE, False),
    "learned gate (A1)": (32, OBSERVABLE, True),
    "ablation: linear": (0, OBSERVABLE, True),
    "ablation: B2 inputs only": (32, B2_FEATURES, True),
    "diagnostic: told which claims are true": (
        32,
        OBSERVABLE + ("peek_is_true",),
        True,
    ),
    "diagnostic: told each key's query weight": (
        32,
        OBSERVABLE + ("peek_query_weight",),
        True,
    ),
}


def run_variant(job: tuple) -> tuple:
    name, slots, n_train, n_test, seed = job
    hidden, features, improve = VARIANTS[name]
    cfg = Config(n_slots=slots)
    train = [generate_episode(cfg, 2_000_000 + s) for s in range(n_train)]
    test = [generate_episode(cfg, 1_000_000 + s) for s in range(n_test)]
    t0 = time.time()
    gate = train_gate(cfg, train, hidden=hidden, features=features, seed=seed)
    if improve:
        improve_gate(cfg, train, gate, seed=seed)
    acc = evaluate(cfg, gate.for_episode, test)["acc"]
    return name, slots, seed, acc, time.time() - t0


def main(argv: Optional[List[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=int, default=2000, help="training episodes")
    ap.add_argument("--test", type=int, default=3000, help="held-out episodes")
    ap.add_argument("--slots", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--seeds", type=int, default=3, help="training seeds for A1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--s0", type=Path, default=HERE / "results_s0.json")
    ap.add_argument("--out", type=Path, default=None, help="write results JSON")
    args = ap.parse_args(argv)

    s0 = json.loads(args.s0.read_text())["by_slots"]
    jobs = []
    for k in args.slots:
        for name in VARIANTS:
            n_seeds = args.seeds if name == "learned gate (A1)" else 1
            jobs += [(name, k, args.train, args.test, s) for s in range(n_seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(run_variant, jobs))

    results: dict = {"by_slots": {}}
    for k in args.slots:
        cfg = Config(n_slots=k)
        test = [generate_episode(cfg, 1_000_000 + s) for s in range(args.test)]
        tuned = s0[str(k)]["tuned"]
        b2_policy = ScoreHeuristic(
            tuple(tuned["weights"].values()), tuned["replace_conf"], tuned["write_conf"]
        )
        b2 = evaluate(cfg, lambda ep: b2_policy, test)["acc"]
        ceiling = evaluate(cfg, Informed, test)["acc"]
        headroom = float((ceiling - b2).mean())

        print(f"\n## {k} slots, {len(test)} held-out episodes")
        print("| gate | accuracy | vs B2 (paired) | ±95% | share of headroom |")
        print("|---|---|---|---|---|")
        print(f"| tuned heuristic (B2) | {b2.mean():.1%} | | | |")
        rows = {"tuned heuristic (B2)": {"accuracy": float(b2.mean())}}
        for name in VARIANTS:
            accs = [a for n, s, _, a, _ in done if n == name and s == k]
            acc = np.mean(accs, axis=0)  # per-episode mean over training seeds
            gain, half = mean_ci(acc - b2)
            per_seed = [float(a.mean()) for a in accs]
            print(
                f"| {name} | {acc.mean():.1%} | {gain:+.1%} | {half:.1%} "
                f"| {gain / headroom:.0%} |"
            )
            rows[name] = {
                "accuracy": float(acc.mean()),
                "gain_vs_b2": gain,
                "gain_ci95": half,
                "share_of_headroom": gain / headroom,
                "per_seed_accuracy": per_seed,
            }
        print(f"| informed ceiling | {ceiling.mean():.1%} | {headroom:+.1%} | | 100% |")
        rows["informed ceiling"] = {
            "accuracy": float(ceiling.mean()),
            "gain_vs_b2": headroom,
        }
        results["by_slots"][k] = rows

    results["minutes_per_job"] = float(np.mean([d[4] for d in done]) / 60)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
