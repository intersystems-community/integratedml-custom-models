"""Gate policies: simple rules, a tunable scoring heuristic, and a hindsight oracle.

Every policy maps an Obs to an action: a slot index to overwrite, or obs.drop.
Only the oracle is built from the Episode and so sees hidden ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .sim import Episode, Obs, Policy, Record

_TYPE_RANK = {"constraint": 2, "fact": 1, "hypothesis": 0}


def never_write(obs: Obs) -> int:
    """Floor: no working memory, retrieval only."""
    return obs.drop


def _write_or_evict(obs: Obs, keep_score) -> int:
    """Always accept the candidate; evict the slot with the lowest keep_score."""
    held = obs.slot_of_key()
    if held is not None:
        return held
    empty = obs.first_empty()
    if empty is not None:
        return empty
    return min(range(len(obs.slots)), key=lambda i: keep_score(obs.slots[i]))


def recency(obs: Obs) -> int:
    """Keep the most recently written facts."""
    return _write_or_evict(obs, lambda r: r.written_at)


def lru(obs: Obs) -> int:
    """Evict the least recently written-or-queried fact."""
    return _write_or_evict(obs, lambda r: r.last_used)


def type_priority(obs: Obs) -> int:
    """Protect constraints, then facts, then hypotheses; oldest goes first."""
    c = obs.candidate
    held = obs.slot_of_key()
    if held is not None:
        return held if c.category != "hypothesis" else obs.drop
    empty = obs.first_empty()
    if empty is not None:
        return empty
    i = min(
        range(len(obs.slots)),
        key=lambda j: (_TYPE_RANK[obs.slots[j].category], obs.slots[j].written_at),
    )
    worst = obs.slots[i]
    if _TYPE_RANK[c.category] < _TYPE_RANK[worst.category]:
        return obs.drop
    return i


# Order of the weights in ScoreHeuristic.w
SCORE_FEATURES = (
    "confidence",
    "emphasis",
    "is_constraint",
    "is_hypothesis",
    "age",
    "was_queried",
    "topic_query_share",
)


@dataclass(frozen=True)
class ScoreHeuristic:
    """Static rule: keep the records with the highest linear score.

    The candidate competes with the slots, so a low-scoring candidate is
    dropped. A candidate that contradicts a held key replaces it only above a
    confidence threshold. Weights are tuned offline and then fixed.
    """

    w: Tuple[float, ...]
    replace_conf: float = 0.5
    write_conf: float = 0.0

    def _score(self, r: Record, obs: Obs, total_q: int) -> float:
        w = self.w
        return (
            w[0] * r.confidence
            + w[1] * r.emphasis
            + w[2] * (r.category == "constraint")
            + w[3] * (r.category == "hypothesis")
            + w[4] * (obs.step - r.written_at) / obs.n_events
            + w[5] * (r.n_queries > 0)
            + w[6] * obs.topic_queries[r.topic] / total_q
        )

    def __call__(self, obs: Obs) -> int:
        c = obs.candidate
        held = obs.slot_of_key()
        if held is not None:
            if c.value == obs.slots[held].value:
                return held  # restatement: refresh
            return held if c.confidence >= self.replace_conf else obs.drop
        if c.confidence < self.write_conf:
            return obs.drop
        empty = obs.first_empty()
        if empty is not None:
            return empty
        total_q = max(1, sum(obs.topic_queries))
        best_i, best_s = obs.drop, self._score(c, obs, total_q)
        for i, r in enumerate(obs.slots):
            s = self._score(r, obs, total_q)
            if s < best_s:
                best_i, best_s = i, s
        return best_i


class Oracle:
    """Hindsight reference: Belady-style eviction using the full episode.

    A record is worth keeping until the next query on its key that its value
    will answer correctly. The oracle keeps the records whose next such use is
    soonest, and drops facts that will never answer a query correctly. It does
    not plan around the retrieval budget, so it is a strong reference rather
    than the exact optimum; the true ceiling is at least this high.
    """

    def __init__(self, episode: Episode):
        truth: Dict[int, int] = {}
        # per key: (step, value) pairs for queries, in order
        self._uses: Dict[int, List[Tuple[int, int]]] = {}
        for ev in episode.events:
            if ev.kind == "fact":
                if ev.is_true:
                    truth[ev.key] = ev.value
            else:
                self._uses.setdefault(ev.key, []).append((ev.step, ev.answer))

    def _next_use(self, key: int, value: int, step: int) -> float:
        for q_step, answer in self._uses.get(key, ()):
            if q_step > step and answer == value:
                return q_step
        return math.inf

    def __call__(self, obs: Obs) -> int:
        c = obs.candidate
        c_use = self._next_use(c.key, c.value, obs.step)
        held = obs.slot_of_key()
        if held is not None:
            r = obs.slots[held]
            return held if c_use < self._next_use(r.key, r.value, obs.step) else obs.drop
        if c_use == math.inf:
            return obs.drop
        empty = obs.first_empty()
        if empty is not None:
            return empty
        uses = [self._next_use(r.key, r.value, obs.step) for r in obs.slots]
        worst = max(range(len(uses)), key=uses.__getitem__)
        return worst if c_use < uses[worst] else obs.drop


class Informed:
    """Knows the hidden latents but not the future.

    It is told which claims are true and how likely each key is to be asked
    about, and keeps the true facts with the highest query weight. This is the
    ceiling for a gate that inferred the task's hidden structure perfectly, so
    what a learned gate can reach lies between the tuned heuristic and this.
    """

    def __init__(self, episode: Episode):
        self._w = episode.query_weight
        self._true = {
            (ev.key, ev.value) for ev in episode.events if ev.kind == "fact" and ev.is_true
        }
        self._events = episode.events

    def _current(self, key: int, step: int) -> int:
        value = -1
        for ev in self._events[: step + 1]:
            if ev.kind == "fact" and ev.key == key and ev.is_true:
                value = ev.value
        return value

    def __call__(self, obs: Obs) -> int:
        c = obs.candidate
        if c.value != self._current(c.key, obs.step):
            return obs.drop  # false claim
        held = obs.slot_of_key()
        if held is not None:
            return held
        empty = obs.first_empty()
        if empty is not None:
            return empty
        worst = min(range(len(obs.slots)), key=lambda i: self._w[obs.slots[i].key])
        return worst if self._w[c.key] > self._w[obs.slots[worst].key] else obs.drop


def replay_policy(actions: Sequence[int]) -> Policy:
    """Feeds back a recorded action sequence, one per fact event."""
    it = iter(actions)
    return lambda obs: next(it)
