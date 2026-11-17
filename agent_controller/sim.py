"""Synthetic delayed-dependency task with a bounded working memory.

No LLM is involved. A scripted "proposer" emits typed facts and queries; a gate
policy decides, for every arriving fact, whether to write it into one of a few
working slots (evicting what is there) or drop it. Every fact is also appended
to a durable episodic store that a query can fall back on, at a cost.

What makes the task non-trivial:
  * off-goal facts arrive more often than goal-relevant ones (distractors);
  * a key can be contradicted by a later claim, which is either a real update
    or a false claim, and the only cue is a noisy confidence score;
  * how likely a key is to be asked about depends on a hidden goal topic and a
    hidden per-key salience, neither of which is readable from the fact's type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

CATEGORIES = ("constraint", "fact", "hypothesis")


@dataclass(frozen=True)
class Config:
    n_topics: int = 4
    keys_per_topic: int = 4
    n_events: int = 40
    warmup: int = 8  # first events are always facts
    final_queries: int = 3  # last events are always queries
    p_query: float = 0.3  # chance an event after warmup is a query
    n_slots: int = 4  # working-memory budget
    retrieval_budget: int = 2  # episodic lookups allowed per episode
    off_goal_query_weight: float = 0.15  # query weight of off-goal keys
    goal_arrival_weight: float = 0.5  # arrival weight of goal keys (off-goal = 1)
    p_update: float = 0.15  # repeat mention is a real update
    p_false: float = 0.25  # repeat mention is a false claim
    emphasis_noise: float = 0.7  # noise on the observable salience cue

    @property
    def n_keys(self) -> int:
        return self.n_topics * self.keys_per_topic


@dataclass(frozen=True)
class Event:
    """One scripted event. Fields below the marker are hidden from policies."""

    step: int
    kind: str  # "fact" | "query"
    key: int
    topic: int
    value: int = -1
    category: str = ""
    confidence: float = 0.0
    emphasis: float = 0.0
    # --- hidden ground truth ---
    is_true: bool = False  # fact: the claimed value is the true one
    answer: int = -1  # query: the correct value


@dataclass(frozen=True)
class Episode:
    seed: int
    events: Tuple[Event, ...]
    goal_topic: int  # hidden
    query_weight: Tuple[float, ...] = ()  # hidden: relative chance a key is asked


@dataclass(frozen=True)
class Record:
    """What a policy may see about a candidate fact or an occupied slot."""

    key: int
    topic: int
    value: int
    category: str
    confidence: float
    emphasis: float
    written_at: int
    last_used: int
    n_queries: int = 0


@dataclass(frozen=True)
class Obs:
    """Policy input when a fact arrives. Contains no hidden fields."""

    step: int
    n_events: int
    candidate: Record
    slots: Tuple[Optional[Record], ...]
    topic_queries: Tuple[int, ...]  # queries seen so far, per topic
    retrievals_left: int
    feasible: Tuple[int, ...]  # slot index to overwrite, or len(slots) = drop

    @property
    def drop(self) -> int:
        return len(self.slots)

    def slot_of_key(self) -> Optional[int]:
        for i, r in enumerate(self.slots):
            if r is not None and r.key == self.candidate.key:
                return i
        return None

    def first_empty(self) -> Optional[int]:
        for i, r in enumerate(self.slots):
            if r is None:
                return i
        return None


def generate_episode(cfg: Config, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    topic_of = np.repeat(np.arange(cfg.n_topics), cfg.keys_per_topic)
    goal = int(rng.integers(cfg.n_topics))
    log_s = rng.normal(0.0, 1.0, cfg.n_keys)
    query_w = np.exp(log_s) * np.where(
        topic_of == goal, 1.0, cfg.off_goal_query_weight
    )
    arrive_w = np.where(topic_of == goal, cfg.goal_arrival_weight, 1.0)
    arrive_w = arrive_w / arrive_w.sum()

    truth: dict = {}
    next_value = 0
    events: List[Event] = []
    for t in range(cfg.n_events):
        want_query = t >= cfg.n_events - cfg.final_queries or (
            t >= cfg.warmup and rng.random() < cfg.p_query
        )
        if want_query and truth:
            keys = np.array(sorted(truth))
            p = query_w[keys] / query_w[keys].sum()
            k = int(rng.choice(keys, p=p))
            events.append(Event(t, "query", k, int(topic_of[k]), answer=truth[k]))
            continue

        k = int(rng.choice(cfg.n_keys, p=arrive_w))
        if k not in truth:
            is_true, value = True, next_value
            next_value += 1
            truth[k] = value
        else:
            u = rng.random()
            if u < cfg.p_update:
                is_true, value = True, next_value
                next_value += 1
                truth[k] = value
            elif u < cfg.p_update + cfg.p_false:
                is_true, value = False, next_value
                next_value += 1
            else:
                is_true, value = True, truth[k]

        p_con = float(np.clip(0.2 + 0.2 * log_s[k], 0.05, 0.7))
        if is_true:
            probs = (p_con, 0.85 - p_con, 0.15)
        else:
            probs = (p_con * 0.5, 0.4 - p_con * 0.5, 0.6)
        category = CATEGORIES[int(rng.choice(3, p=probs))]
        confidence = float(rng.beta(5, 2) if is_true else rng.beta(2, 4))
        emphasis = float(log_s[k] + rng.normal(0.0, cfg.emphasis_noise))
        events.append(
            Event(
                t,
                "fact",
                k,
                int(topic_of[k]),
                value=value,
                category=category,
                confidence=confidence,
                emphasis=emphasis,
                is_true=is_true,
            )
        )
    return Episode(
        seed=seed,
        events=tuple(events),
        goal_topic=goal,
        query_weight=tuple(float(x) for x in query_w),
    )


class Env:
    """Runs one episode. The caller supplies an action for each fact event."""

    def __init__(self, cfg: Config, episode: Episode, n_slots: Optional[int] = None):
        self.cfg = cfg
        self.ep = episode
        self.k = cfg.n_slots if n_slots is None else n_slots
        self.slots: List[Optional[Record]] = [None] * self.k
        self.store: dict = {}  # key -> most recent claimed value
        self.t = 0
        self.topic_queries = [0] * cfg.n_topics
        self.retrievals_left = cfg.retrieval_budget
        self.n_queries = 0
        self.n_correct = 0
        self.trace: List[dict] = []

    @property
    def done(self) -> bool:
        return self.t >= len(self.ep.events)

    @property
    def event(self) -> Event:
        return self.ep.events[self.t]

    def obs(self) -> Obs:
        ev = self.event
        assert ev.kind == "fact"
        cand = Record(
            ev.key,
            ev.topic,
            ev.value,
            ev.category,
            ev.confidence,
            ev.emphasis,
            written_at=ev.step,
            last_used=ev.step,
        )
        held = next(
            (i for i, r in enumerate(self.slots) if r is not None and r.key == ev.key),
            None,
        )
        feasible = (held, self.k) if held is not None else tuple(range(self.k + 1))
        return Obs(
            step=ev.step,
            n_events=len(self.ep.events),
            candidate=cand,
            slots=tuple(self.slots),
            topic_queries=tuple(self.topic_queries),
            retrievals_left=self.retrievals_left,
            feasible=feasible,
        )

    def step(self, action: Optional[int] = None) -> dict:
        ev = self.event
        if ev.kind == "fact":
            obs = self.obs()
            if action not in obs.feasible:
                raise ValueError(f"action {action} not in mask {obs.feasible}")
            self.store[ev.key] = ev.value  # durable store keeps everything
            if action != self.k:
                self.slots[action] = obs.candidate
            row = {
                "step": ev.step,
                "kind": "fact",
                "key": ev.key,
                "feasible": list(obs.feasible),
                "action": int(action),
            }
        else:
            self.n_queries += 1
            self.topic_queries[ev.topic] += 1
            held = next(
                (
                    i
                    for i, r in enumerate(self.slots)
                    if r is not None and r.key == ev.key
                ),
                None,
            )
            if held is not None:
                r = self.slots[held]
                source, answer = "slot", r.value
                self.slots[held] = Record(
                    r.key,
                    r.topic,
                    r.value,
                    r.category,
                    r.confidence,
                    r.emphasis,
                    r.written_at,
                    last_used=ev.step,
                    n_queries=r.n_queries + 1,
                )
            elif self.retrievals_left > 0:
                self.retrievals_left -= 1
                source, answer = "retrieve", self.store[ev.key]
            else:
                source, answer = "none", None
            correct = answer == ev.answer
            self.n_correct += int(correct)
            row = {
                "step": ev.step,
                "kind": "query",
                "key": ev.key,
                "source": source,
                "correct": bool(correct),
            }
        self.trace.append(row)
        self.t += 1
        return row


@dataclass
class Result:
    seed: int
    n_queries: int
    n_correct: int
    retrievals_used: int
    trace: List[dict]

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_queries


Policy = Callable[[Obs], int]
PolicyFactory = Callable[[Episode], Policy]


def run_episode(
    cfg: Config, episode: Episode, policy: Policy, n_slots: Optional[int] = None
) -> Result:
    env = Env(cfg, episode, n_slots)
    while not env.done:
        if env.event.kind == "fact":
            env.step(policy(env.obs()))
        else:
            env.step()
    return Result(
        episode.seed,
        env.n_queries,
        env.n_correct,
        cfg.retrieval_budget - env.retrievals_left,
        env.trace,
    )
