import dataclasses

import numpy as np

from agent_controller.gate import (
    FEATURES,
    N_FEAT,
    OBSERVABLE,
    GateNet,
    LearnedGate,
    allowed_actions,
    collect,
    collect_outcomes,
    featurize,
    improve_gate,
    train_gate,
)
from agent_controller.policies import Informed, recency
from agent_controller.sim import Config, Env, generate_episode, run_episode

CFG = Config()
TRAIN = [generate_episode(CFG, 5000 + s) for s in range(120)]
TEST = [generate_episode(CFG, 9000 + s) for s in range(120)]


def mean_acc(factory, episodes=TEST):
    return np.mean([run_episode(CFG, ep, factory(ep)).accuracy for ep in episodes])


def walk(episode, policy=recency):
    env = Env(CFG, episode)
    while not env.done:
        if env.event.kind == "fact":
            obs = env.obs()
            yield env, obs
            env.step(policy(obs))
        else:
            env.step()


def test_features_have_declared_shape_and_no_peek_by_default():
    for _, obs in walk(TEST[0]):
        x, present = featurize(obs)
        assert x.shape == (CFG.n_slots + 1, N_FEAT)
        assert present[-1]  # the candidate is always present
        assert not x[:, len(OBSERVABLE) :].any()
        assert np.isfinite(x).all()


def test_allowed_actions_are_feasible_and_fill_free_slots_first():
    for _, obs in walk(TEST[1]):
        allowed = allowed_actions(obs)
        assert set(allowed) <= set(obs.feasible)
        if obs.slot_of_key() is None and obs.first_empty() is not None:
            assert allowed == (obs.first_empty(), obs.drop)


def test_clone_does_not_disturb_the_original():
    for env, obs in walk(TEST[2]):
        before = (list(env.slots), dict(env.store), env.n_correct, env.t)
        stats_before = {k: list(v) for k, v in env.stats.values.items()}
        c = env.clone()
        c.step(obs.drop)
        while not c.done:
            c.step(recency(c.obs()) if c.event.kind == "fact" else None)
        assert before == (list(env.slots), dict(env.store), env.n_correct, env.t)
        assert stats_before == env.stats.values


def test_expert_labels_are_always_allowed():
    x, present, allowed, good = collect(CFG, TRAIN[:20], None)
    assert good.sum(1).tolist() == [1] * len(good)
    assert (good & ~allowed).sum() == 0


def test_training_step_reduces_imitation_loss():
    data = collect(CFG, TRAIN[:40], None)
    net = GateNet(seed=0)
    net.set_norm(data[0], data[1])
    first = net.train_step(*data, lr=3e-3)
    for _ in range(60):
        last = net.train_step(*data, lr=3e-3)
    assert last < first * 0.8


def test_outcome_data_records_a_zero_cost_best_action():
    gate = train_gate(CFG, TRAIN[:40], rounds=1, epochs=3)
    rng = np.random.default_rng(0)
    x, present, allowed, cost = collect_outcomes(CFG, TRAIN[:20], gate, 6, rng)
    assert (cost >= 0).all()
    assert (np.where(allowed, cost, np.inf).min(1) == 0).all()
    assert (cost.max(1) > 0).all()  # ties are skipped


def test_learned_gate_beats_recency_and_stays_below_ceiling():
    gate = train_gate(CFG, TRAIN, rounds=2, epochs=6)
    improve_gate(CFG, TRAIN, gate, iters=1)
    learned = mean_acc(gate.for_episode)
    assert mean_acc(lambda ep: recency) < learned < mean_acc(Informed)


def test_peeking_gate_is_bound_per_episode():
    net = GateNet(features=OBSERVABLE + ("peek_is_true",))
    gate = LearnedGate(net)
    assert net.peeks and gate.for_episode(TEST[0]).peek is not None
    plain = LearnedGate(GateNet())
    assert plain.for_episode(TEST[0]) is plain


def test_feature_names_are_unique():
    assert len(set(FEATURES)) == N_FEAT
    assert dataclasses.is_dataclass(CFG)
