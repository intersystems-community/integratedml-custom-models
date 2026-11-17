"""Save an episode's decision trace and replay it exactly."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from .policies import replay_policy
from .sim import Config, Result, generate_episode, run_episode


def save_trace(
    path: Path, cfg: Config, result: Result, n_slots: Optional[int] = None
) -> None:
    header = {
        "config": asdict(cfg),
        "seed": result.seed,
        "n_slots": cfg.n_slots if n_slots is None else n_slots,
        "n_queries": result.n_queries,
        "n_correct": result.n_correct,
        "retrievals_used": result.retrievals_used,
    }
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in result.trace:
            f.write(json.dumps(row) + "\n")


def load_trace(path: Path) -> Tuple[dict, List[dict]]:
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    return lines[0], lines[1:]


def replay(header: dict, rows: List[dict]) -> Result:
    """Regenerate the episode from its seed and re-apply the recorded actions.

    Raises AssertionError if the outcome or any step differs from the record.
    """
    cfg = Config(**header["config"])
    episode = generate_episode(cfg, header["seed"])
    actions = [r["action"] for r in rows if r["kind"] == "fact"]
    result = run_episode(cfg, episode, replay_policy(actions), header["n_slots"])
    assert result.trace == rows, "replayed steps differ from the recorded trace"
    for field in ("n_queries", "n_correct", "retrievals_used"):
        assert getattr(result, field) == header[field], f"{field} differs on replay"
    return result
