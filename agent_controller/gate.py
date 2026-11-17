"""A small learned gate: a keep-score network trained by imitation (DAgger).

The network scores every record in play (each occupied slot and the arriving
candidate) from observable features. The gate discards the lowest-scoring one:
evicting that slot, or dropping the candidate. An empty slot scores zero, so a
candidate is written into free space only if its score is positive.

Training imitates `Informed`, which sees hidden ground truth. The learner sees
only the features below, so it has to infer what the expert was told.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .policies import Informed
from .sim import Config, Env, Episode, Obs, Record

FEATURES = (
    # the seven inputs of the tuned heuristic (B2)
    "confidence",
    "emphasis",
    "is_constraint",
    "is_hypothesis",
    "age",
    "was_queried",
    "topic_query_share",
    # evidence pooled over the durable store
    "value_mean_confidence",
    "value_mentions",
    "value_hypothesis_share",
    "value_is_latest_claim",
    "key_distinct_values",
    "key_mean_emphasis",
    "key_mentions",
    "key_constraint_share",
    "key_queries",
    "since_last_used",
    # episode context
    "topic_queries",
    "total_queries",
    "progress",
    "retrievals_left",
    "is_candidate",
    # diagnostic only: hidden ground truth, zero unless a gate is given a peek
    "peek_is_true",
    "peek_query_weight",
)
B2_FEATURES = FEATURES[:7]
PEEK_FEATURES = FEATURES[-2:]
OBSERVABLE = FEATURES[:-2]
N_FEAT = len(FEATURES)


def featurize(
    obs: Obs, peek: Optional[Informed] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Features for each slot and, last, the candidate; plus a presence mask.

    `peek` leaks hidden ground truth into the last two features. It exists
    only to measure how much each hidden quantity would be worth.
    """
    st = obs.stats
    total_q = sum(obs.topic_queries)
    progress = obs.step / obs.n_events
    records: Tuple[Optional[Record], ...] = obs.slots + (obs.candidate,)
    rows = []
    present = []
    last = len(records) - 1
    for i, r in enumerate(records):
        if r is None:
            rows.append([0.0] * N_FEAT)
            present.append(False)
            continue
        k = r.key
        n_val, conf_sum, n_hyp = st.values[(k, r.value)]
        mentions = st.mentions[k]
        rows.append(
            [
                r.confidence,
                r.emphasis,
                float(r.category == "constraint"),
                float(r.category == "hypothesis"),
                (obs.step - r.written_at) / obs.n_events,
                float(r.n_queries > 0),
                obs.topic_queries[r.topic] / max(1, total_q),
                conf_sum / n_val,
                math.log1p(n_val),
                n_hyp / n_val,
                float(st.latest[k] == r.value),
                math.log1p(st.distinct[k]),
                st.emphasis_sum[k] / mentions,
                math.log1p(mentions),
                st.constraints[k] / mentions,
                math.log1p(st.queries[k]),
                (obs.step - r.last_used) / obs.n_events,
                math.log1p(obs.topic_queries[r.topic]),
                math.log1p(total_q),
                progress,
                float(obs.retrievals_left),
                float(i == last),
                float(peek.is_true(r, obs.step)) if peek else 0.0,
                math.log(peek.weight(r.key)) if peek else 0.0,
            ]
        )
        present.append(True)
    return np.array(rows, dtype=np.float32), np.array(present)


def allowed_actions(obs: Obs) -> Tuple[int, ...]:
    """The gate's choices: never evict an occupied slot while one is free."""
    if len(obs.feasible) == 2:
        return obs.feasible  # key already held: replace or drop
    empty = obs.first_empty()
    if empty is not None:
        return (empty, obs.drop)
    return obs.feasible


class GateNet:
    """Keep-score model: one hidden ReLU layer, or linear when hidden == 0."""

    def __init__(
        self,
        hidden: int = 32,
        features: Sequence[str] = OBSERVABLE,
        seed: int = 0,
    ):
        self.peeks = any(f in PEEK_FEATURES for f in features)
        rng = np.random.default_rng(seed)
        self.cols = np.array([f in features for f in FEATURES], dtype=np.float32)
        self.mu = np.zeros(N_FEAT, dtype=np.float32)
        self.sd = np.ones(N_FEAT, dtype=np.float32)
        if hidden:
            self.params = [
                rng.normal(0, math.sqrt(2 / N_FEAT), (N_FEAT, hidden)),
                np.zeros(hidden),
                rng.normal(0, math.sqrt(1 / hidden), hidden),
                np.zeros(1),
            ]
        else:
            self.params = [np.zeros(N_FEAT), np.zeros(1)]
        self.params = [p.astype(np.float32) for p in self.params]
        self._adam = [(np.zeros_like(p), np.zeros_like(p)) for p in self.params]
        self._t = 0

    def set_norm(self, x: np.ndarray, present: np.ndarray) -> None:
        rows = x[present]
        self.mu = rows.mean(0)
        self.sd = rows.std(0) + 1e-6

    def _forward(self, x: np.ndarray):
        z = (x - self.mu) / self.sd * self.cols
        if len(self.params) == 2:
            return z @ self.params[0] + self.params[1], z, None
        w1, b1, w2, b2 = self.params
        h = np.maximum(z @ w1 + b1, 0)
        return h @ w2 + b2, z, h

    def score(self, x: np.ndarray, present: np.ndarray) -> np.ndarray:
        return self._forward(x)[0] * present

    def train_step(
        self, x, present, allowed, target, lr: float, regret: bool = False
    ) -> float:
        """One Adam step over the allowed discard choices.

        Imitation (default): `target` marks the expert's choice and the loss is
        its negative log probability. With `regret=True`, `target` holds each
        choice's shortfall against the best one and the loss is the expected
        shortfall under the policy.
        """
        s, z, h = self._forward(x)
        logits = np.where(allowed, -(s * present), -1e9)
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        n = len(target)
        if regret:
            expected = (p * target).sum(1, keepdims=True)
            loss = expected.mean()
            g = p * (target - expected)
        else:
            mass = (p * target).sum(1, keepdims=True) + 1e-12
            loss = -np.log(mass).mean()
            g = p - p * target / mass
        ds = -(g / n) * present
        if h is None:
            grads = [np.tensordot(ds, z, 2), ds.sum(keepdims=True).ravel()]
        else:
            w2 = self.params[2]
            dh = ds[..., None] * w2 * (h > 0)
            grads = [
                np.tensordot(z, dh, ([0, 1], [0, 1])),
                dh.sum((0, 1)),
                np.tensordot(ds, h, 2),
                ds.sum(keepdims=True).ravel(),
            ]
        self._t += 1
        for p_, g_, (m, v) in zip(self.params, grads, self._adam):
            m *= 0.9
            m += 0.1 * g_
            v *= 0.999
            v += 0.001 * g_ * g_
            m_hat = m / (1 - 0.9**self._t)
            v_hat = v / (1 - 0.999**self._t)
            p_ -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return float(loss)


Dataset = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _fit(net: GateNet, data: Dataset, epochs: int, lr: float, batch: int, rng) -> None:
    x, present, allowed, good = data
    for _ in range(epochs):
        order = rng.permutation(len(good))
        for i in range(0, len(good), batch):
            j = order[i : i + batch]
            net.train_step(x[j], present[j], allowed[j], good[j], lr)


class LearnedGate:
    """Policy wrapper: discard the allowed record with the lowest keep score."""

    def __init__(self, net: GateNet, peek: Optional[Informed] = None):
        self.net = net
        self.peek = peek

    def for_episode(self, episode: Episode) -> "LearnedGate":
        """Bind the gate to an episode. Only a peeking gate needs this."""
        return LearnedGate(self.net, Informed(episode)) if self.net.peeks else self

    def __call__(self, obs: Obs) -> int:
        x, present = featurize(obs, self.peek)
        return self.choose(x, present, allowed_actions(obs))

    def choose(self, x, present, allowed: Sequence[int]) -> int:
        s = self.net.score(x, present)
        return min(allowed, key=lambda a: s[a])


def collect(
    cfg: Config,
    episodes: Sequence[Episode],
    rollout: Optional[LearnedGate],
    force_peek: bool = False,
) -> Dataset:
    """Roll out `rollout` (or the expert if None) and label every state visited
    with the expert's action."""
    xs, ps, al, ys = [], [], [], []
    n = cfg.n_slots + 1
    peeks = force_peek or (rollout is not None and rollout.net.peeks)
    for ep in episodes:
        env = Env(cfg, ep)
        expert = Informed(ep)
        while not env.done:
            if env.event.kind != "fact":
                env.step()
                continue
            obs = env.obs()
            x, present = featurize(obs, expert if peeks else None)
            allowed = allowed_actions(obs)
            y = expert(obs)
            assert y in allowed
            mask = np.zeros(n, dtype=bool)
            mask[list(allowed)] = True
            good = np.zeros(n, dtype=bool)
            good[y] = True
            xs.append(x)
            ps.append(present)
            al.append(mask)
            ys.append(good)
            env.step(y if rollout is None else rollout.choose(x, present, allowed))
    return np.stack(xs), np.stack(ps), np.stack(al), np.stack(ys)


def train_gate(
    cfg: Config,
    episodes: Sequence[Episode],
    hidden: int = 32,
    features: Sequence[str] = OBSERVABLE,
    rounds: int = 4,
    epochs: int = 12,
    lr: float = 3e-3,
    batch: int = 512,
    seed: int = 0,
) -> LearnedGate:
    """DAgger: clone the expert, then retrain on states the learner reaches."""
    rng = np.random.default_rng(seed)
    net = GateNet(hidden, features, seed)
    gate = LearnedGate(net)
    data: List[Dataset] = []
    for r in range(rounds):
        data.append(collect(cfg, episodes, None if r == 0 else gate, net.peeks))
        merged = tuple(np.concatenate(d) for d in zip(*data))
        if r == 0:
            net.set_norm(merged[0], merged[1])
        _fit(net, merged, epochs, lr, batch, rng)
    return gate


def _suffix_return(env: Env, action: int, gate: LearnedGate) -> int:
    """Queries answered correctly from here on if `action` is taken now and
    the gate decides everything after."""
    c = env.clone()
    start = c.n_correct
    c.step(action)
    while not c.done:
        if c.event.kind == "fact":
            c.step(gate(c.obs()))
        else:
            c.step()
    return c.n_correct - start


def collect_outcomes(
    cfg: Config,
    episodes: Sequence[Episode],
    base: LearnedGate,
    steps_per_episode: int,
    rng,
) -> Dataset:
    """Roll out the gate; at sampled decisions, replay the rest of the episode
    once per allowed action and record each action's shortfall against the best.

    Decisions where every action ends the same carry no signal and are skipped.
    """
    xs, ps, al, cs = [], [], [], []
    n = cfg.n_slots + 1
    for ep in episodes:
        env = Env(cfg, ep)
        gate = base.for_episode(ep)
        fact_steps = [e.step for e in ep.events if e.kind == "fact"]
        chosen = set(
            rng.choice(
                fact_steps, min(steps_per_episode, len(fact_steps)), replace=False
            ).tolist()
        )
        while not env.done:
            if env.event.kind != "fact":
                env.step()
                continue
            obs = env.obs()
            x, present = featurize(obs, gate.peek)
            allowed = allowed_actions(obs)
            if obs.step in chosen:
                returns = [_suffix_return(env, a, gate) for a in allowed]
                best = max(returns)
                if min(returns) < best:
                    mask = np.zeros(n, dtype=bool)
                    mask[list(allowed)] = True
                    cost = np.zeros(n, dtype=np.float32)
                    cost[list(allowed)] = [best - q for q in returns]
                    xs.append(x)
                    ps.append(present)
                    al.append(mask)
                    cs.append(cost)
            env.step(gate.choose(x, present, allowed))
    return np.stack(xs), np.stack(ps), np.stack(al), np.stack(cs)


def improve_gate(
    cfg: Config,
    episodes: Sequence[Episode],
    gate: LearnedGate,
    iters: int = 4,
    steps_per_episode: int = 8,
    epochs: int = 4,
    lr: float = 3e-4,
    batch: int = 512,
    seed: int = 0,
) -> LearnedGate:
    """Outcome-driven policy improvement from counterfactual suffix replays.

    No expert and no hidden information: the only signal is how many queries
    each choice went on to answer in the episodes actually played.
    """
    rng = np.random.default_rng(seed + 1)
    data: List[Dataset] = []
    for _ in range(iters):
        data.append(collect_outcomes(cfg, episodes, gate, steps_per_episode, rng))
        x, present, allowed, cost = (np.concatenate(d) for d in zip(*data))
        for _ in range(epochs):
            order = rng.permutation(len(cost))
            for i in range(0, len(cost), batch):
                j = order[i : i + batch]
                gate.net.train_step(
                    x[j], present[j], allowed[j], cost[j], lr, regret=True
                )
    return gate
