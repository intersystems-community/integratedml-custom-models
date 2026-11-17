import dataclasses

import numpy as np
import pytest

from agent_controller.policies import (
    Informed,
    Oracle,
    ScoreHeuristic,
    lru,
    recency,
    type_priority,
)
from agent_controller.sim import Config, Env, Obs, Record, generate_episode, run_episode
from agent_controller.trace import load_trace, replay, save_trace

CFG = Config()
EPISODES = [generate_episode(CFG, s) for s in range(60)]
HEURISTIC = ScoreHeuristic((1.0, 1.0, 0.5, -0.5, -1.0, 1.0, 1.0))


def mean_acc(factory):
    return np.mean([run_episode(CFG, ep, factory(ep)).accuracy for ep in EPISODES])


def test_generation_is_deterministic():
    assert generate_episode(CFG, 7) == generate_episode(CFG, 7)
    assert generate_episode(CFG, 7) != generate_episode(CFG, 8)


def test_episode_shape():
    ep = EPISODES[0]
    assert len(ep.events) == CFG.n_events
    assert all(e.kind == "fact" for e in ep.events[: CFG.warmup])
    assert all(e.kind == "query" for e in ep.events[-CFG.final_queries :])


def test_observation_hides_ground_truth():
    hidden = {"is_true", "answer", "goal_topic", "query_weight"}
    for cls in (Obs, Record):
        assert not hidden & {f.name for f in dataclasses.fields(cls)}


def test_mask_is_enforced():
    env = Env(CFG, EPISODES[0])
    with pytest.raises(ValueError):
        env.step(CFG.n_slots + 1)


def test_held_key_restricts_actions_to_replace_or_drop():
    env = Env(CFG, EPISODES[0])
    while not env.done:
        if env.event.kind != "fact":
            env.step()
            continue
        obs = env.obs()
        held = obs.slot_of_key()
        if held is not None:
            assert obs.feasible == (held, obs.drop)
        env.step(recency(obs))


@pytest.mark.parametrize("policy", [recency, lru, type_priority, HEURISTIC])
def test_policies_respect_slot_budget(policy):
    res = run_episode(CFG, EPISODES[1], policy)
    assert 0 <= res.n_correct <= res.n_queries
    assert res.retrievals_used <= CFG.retrieval_budget


def test_reference_policies_are_ordered():
    heuristic = mean_acc(lambda ep: HEURISTIC)
    informed = mean_acc(Informed)
    oracle = mean_acc(Oracle)
    assert mean_acc(lambda ep: recency) < heuristic < informed < oracle


def test_trace_replays_exactly(tmp_path):
    ep = EPISODES[2]
    res = run_episode(CFG, ep, HEURISTIC)
    path = tmp_path / "trace.jsonl"
    save_trace(path, CFG, res)
    header, rows = load_trace(path)
    again = replay(header, rows)
    assert (again.n_correct, again.retrievals_used) == (
        res.n_correct,
        res.retrievals_used,
    )


def test_replay_detects_tampering(tmp_path):
    res = run_episode(CFG, EPISODES[2], HEURISTIC)
    path = tmp_path / "trace.jsonl"
    save_trace(path, CFG, res)
    header, rows = load_trace(path)
    header["n_correct"] += 1
    with pytest.raises(AssertionError):
        replay(header, rows)
