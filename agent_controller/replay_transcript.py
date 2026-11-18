"""Offline replay of Claude Code session transcripts under context policies.

Asks the S0 question for coding sessions: if tool results could be evicted
from the prompt (replaced by a short stub) and re-fetched when needed, how much
input cost is there to save, and how close do simple rules get to hindsight?

    python -m agent_controller.replay_transcript ~/.claude/projects/*/*.jsonl

Only aggregate numbers are printed; no transcript content leaves the machine.

What is modelled
  * Context size per request comes from the recorded token usage.
  * Prompt caching: an unchanged prefix is read at 0.1x; anything after the
    first change is written again at 1.25x. Evicting an old result therefore
    costs a rewrite of everything after it.
  * A result evicted and then used again is a miss: it is re-fetched (written
    again) and one extra request is charged.

What is not modelled
  * Whether the agent would still have succeeded with a trimmed context. The
    model never sees the trimmed prompt, so this measures cost only.
  * Uses that leave no trace. A result counts as used later only if the agent
    quotes a line from it or edits the file it came from, which undercounts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

READ, WRITE, PLAIN = 0.1, 1.25, 1.0  # price relative to an uncached input token
WRITE_1H = 2.0  # cache writes with a one-hour lifetime
STUB_TOKENS = 40  # what an evicted result leaves behind
IMAGE_TOKENS = 1500
CHARS_PER_TOKEN = 4.0
MIN_LINE = 30  # shortest quoted line that counts as evidence of use
EDIT_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}


# ---------------------------------------------------------------- parsing


@dataclass
class Item:
    """One tool result that entered the context."""

    uid: str
    tool: str
    path: Optional[str]
    born: int  # index of the first request that saw it
    tokens: int = 0
    lines: Set[str] = field(default_factory=set)
    uses: List[int] = field(default_factory=list)  # requests whose output used it
    superseded_at: Optional[int] = None  # file re-read or edited at this request


@dataclass
class Request:
    index: int
    context: int  # prompt tokens actually sent
    cache_read: int
    actual_cost: float
    plain: int = 0  # uncached, unwritten input tokens
    write_price: float = WRITE
    iterations: int = 1  # server-side iterations summed into one record
    fixed: int = 0  # new non-evictable tokens since the previous request
    items: List[Item] = field(default_factory=list)  # new tool results
    reset: bool = False  # context shrank: compaction, context editing or a restart
    cold: bool = False  # cache was not reused


@dataclass
class Session:
    name: str
    requests: List[Request]
    items: List[Item]


def _strings(obj) -> List[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _lines(texts: Sequence[str]) -> Set[str]:
    out = set()
    for t in texts:
        for line in t.splitlines():
            line = line.strip()
            if len(line) >= MIN_LINE:
                out.add(line)
    return out


def _result_text_and_tokens(content) -> tuple:
    if isinstance(content, str):
        return [content], int(len(content) / CHARS_PER_TOKEN)
    texts, tokens = [], 0
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
            tokens += int(len(block.get("text", "")) / CHARS_PER_TOKEN)
        elif block.get("type") in ("image", "document"):
            tokens += IMAGE_TOKENS
    return texts, tokens


def _usage_cost(u: dict) -> tuple:
    """Input cost of one usage record, and its price per written token."""
    read = u.get("cache_read_input_tokens", 0) or 0
    created = u.get("cache_creation_input_tokens", 0) or 0
    plain = u.get("input_tokens", 0) or 0
    split = u.get("cache_creation") or {}
    one_hour = split.get("ephemeral_1h_input_tokens", 0) or 0
    one_hour = min(one_hour, created)
    write_cost = (created - one_hour) * WRITE + one_hour * WRITE_1H
    price = write_cost / created if created else WRITE
    return read * READ + write_cost + plain * PLAIN, price


def load_session(path: Path) -> Session:
    """Turn a transcript into requests, tool-result items and later-use marks."""
    rows = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") in ("user", "assistant") and not row.get("isSidechain"):
                rows.append(row)

    requests: List[Request] = []
    items: List[Item] = []
    by_message: Dict[str, int] = {}  # API message id -> request index
    calls: Dict[str, dict] = {}  # tool_use id -> {"name", "input"}
    pending: List[Item] = []  # results waiting for the next request
    authored: Set[str] = set()  # lines the agent has written itself
    latest_read: Dict[str, Item] = {}

    for row in rows:
        msg = row.get("message") or {}
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []

        if row["type"] == "user":
            for b in blocks:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                call = calls.get(b.get("tool_use_id"), {})
                texts, tokens = _result_text_and_tokens(b.get("content"))
                path_ = (call.get("input") or {}).get("file_path")
                item = Item(
                    uid=b.get("tool_use_id", ""),
                    tool=call.get("name", "?"),
                    path=path_,
                    born=len(requests),
                    tokens=tokens,
                    lines=_lines(texts) - authored,
                )
                pending.append(item)
            continue

        usage = msg.get("usage") or {}
        mid = msg.get("id") or row.get("uuid")
        if mid not in by_message:
            iters = [
                i for i in (usage.get("iterations") or []) if isinstance(i, dict)
            ] or [usage]
            cost, wprice = 0.0, WRITE
            for it in iters:
                c, w = _usage_cost(it)
                cost += c
                wprice = w
            last = iters[-1]  # the prompt actually sent last
            read = last.get("cache_read_input_tokens", 0) or 0
            created = last.get("cache_creation_input_tokens", 0) or 0
            plain = last.get("input_tokens", 0) or 0
            ctx = read + created + plain
            if ctx == 0:
                continue  # synthetic message with no API call behind it
            r = Request(
                index=len(requests),
                context=ctx,
                cache_read=read,
                actual_cost=cost,
                plain=plain,
                write_price=wprice,
                iterations=len(iters),
            )
            prev = requests[-1].context if requests else 0
            # Any shrink means the client removed content (compaction, context
            # editing, a restart). What was removed is not recorded, so start
            # over from the recorded size.
            r.reset = bool(requests) and ctx < prev
            r.cold = bool(requests) and not r.reset and read < 0.5 * prev
            if r.reset:
                pending = []  # whatever was pending is inside the summary now
            delta = max(0, ctx - (0 if r.reset else prev))
            estimated = sum(i.tokens for i in pending)
            if estimated > delta and estimated > 0:  # calibrate to real usage
                for i in pending:
                    i.tokens = int(i.tokens * delta / estimated)
                estimated = sum(i.tokens for i in pending)
            r.fixed = ctx if r.reset else delta - estimated
            r.items = pending
            for i in pending:
                i.born = r.index
                if i.tool == "Read" and i.path:
                    old = latest_read.get(i.path)
                    if old is not None and old.superseded_at is None:
                        old.superseded_at = r.index
                    latest_read[i.path] = i
            items.extend(pending)
            pending = []
            by_message[mid] = r.index
            requests.append(r)
        step = by_message[mid]

        out_texts: List[str] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                calls[b.get("id")] = {"name": b.get("name"), "input": b.get("input")}
                out_texts += _strings(b.get("input"))
                path_ = (b.get("input") or {}).get("file_path")
                if b.get("name") in EDIT_TOOLS and path_ in latest_read:
                    src = latest_read[path_]
                    if step not in src.uses:
                        src.uses.append(step)
                    if src.superseded_at is None:
                        src.superseded_at = step + 1
            elif b.get("type") == "text":
                out_texts.append(b.get("text", ""))
        quoted = _lines(out_texts)
        if quoted:
            for item in items:
                if step not in item.uses and not quoted.isdisjoint(item.lines):
                    item.uses.append(step)
            authored |= quoted

    return Session(path.stem[:8], requests, items)


# ---------------------------------------------------------------- policies


@dataclass
class Seg:
    tokens: int
    item: Optional[Item] = None  # None for non-evictable content
    full: bool = True

    @property
    def size(self) -> int:
        return self.tokens if self.full else STUB_TOKENS


@dataclass
class View:
    """What a policy sees when the prompt for request `step` is assembled."""

    step: int
    n_requests: int  # total in the session; only the oracle may use it
    segs: List[Seg]
    cached_end: int  # segments before this index are in the cache, if warm
    cold: bool
    context: int
    write_price: float = WRITE

    def evictable(self) -> List[int]:
        return [
            i
            for i, s in enumerate(self.segs[: self.cached_end])
            if s.item is not None and s.full and s.tokens > STUB_TOKENS
        ]


Policy = Callable[[View], List[int]]


def keep_all(view: View) -> List[int]:
    return []


def truncate(budget: int) -> Policy:
    """Over budget: evict the oldest results until back at 70% of it."""

    def policy(view: View) -> List[int]:
        if view.context <= budget:
            return []
        out, ctx = [], view.context
        for i in view.evictable():
            if ctx <= 0.7 * budget:
                break
            out.append(i)
            ctx -= view.segs[i].tokens - STUB_TOKENS
        return out

    return policy


def best_cut(view: View, dead: List[int], remaining: float) -> List[int]:
    """Choose which `dead` results to evict, weighing the saved cache reads
    over `remaining` requests against rewriting everything after the cut."""
    if not dead:
        return []
    if view.cold:
        return dead  # the whole prompt is rewritten anyway
    dead_set = set(dead)
    best, best_net = [], 0.0
    saved = 0
    rewrite = 0
    chosen: List[int] = []
    for i in range(view.cached_end - 1, -1, -1):
        seg = view.segs[i]
        if i in dead_set:
            saved += seg.tokens - STUB_TOKENS
            rewrite += STUB_TOKENS
            chosen.append(i)
            net = saved * READ * remaining - rewrite * (view.write_price - READ)
            if net > best_net:
                best, best_net = list(chosen), net
        else:
            rewrite += seg.size
    return best


def stale_rule(age: int = 15, min_tokens: int = 500, horizon: float = 30) -> Policy:
    """Evict results that are superseded, or old and unused, when it pays for
    itself over an assumed number of remaining requests."""

    def policy(view: View) -> List[int]:
        dead = []
        for i in view.evictable():
            it = view.segs[i].item
            last = max([it.born] + [u for u in it.uses if u < view.step])
            stale = it.superseded_at is not None and it.superseded_at <= view.step
            if it.tokens >= min_tokens and (stale or view.step - last >= age):
                dead.append(i)
        return best_cut(view, dead, horizon)

    return policy


def hindsight(view: View) -> List[int]:
    """Reference: evict only results never used again, knowing the future."""
    dead = [
        i
        for i in view.evictable()
        if not any(u >= view.step for u in view.segs[i].item.uses)
    ]
    return best_cut(view, dead, view.n_requests - view.step)


# ---------------------------------------------------------------- replay


@dataclass
class Outcome:
    cost: float = 0.0
    peak: int = 0
    context_sum: int = 0
    evictions: int = 0
    evicted_tokens: int = 0
    misses: int = 0
    requests: int = 0


def replay(session: Session, policy: Policy) -> Outcome:
    """Re-price a session with `policy` evicting tool results.

    The recorded cache hit is reused: the uncached tail of each request is the
    same tokens as recorded, so only the prefix before the first eviction made
    at this request can be read from cache. With no evictions the replay
    reproduces the recorded input cost.
    """
    out = Outcome()
    segs: List[Seg] = []
    n = len(session.requests)
    for r in session.requests:
        if r.reset:
            segs = []
        cached_end = len(segs)
        if r.fixed:
            segs.append(Seg(r.fixed))
        segs += [Seg(i.tokens, i) for i in r.items]

        before = sum(s.size for s in segs)
        cold = r.cold or r.reset or r.index == 0
        view = View(r.index, n, segs, cached_end, cold, before, r.write_price)
        evict = policy(view)
        for i in evict:
            segs[i].full = False
            out.evictions += 1
            out.evicted_tokens += segs[i].tokens - STUB_TOKENS

        context = sum(s.size for s in segs)
        tail = max(0, r.context - r.cache_read)  # recorded uncached tokens
        cached = max(0, before - tail)
        if evict:
            cached = min(cached, sum(s.size for s in segs[: min(evict)]))
        written = max(0, context - cached)
        plain = min(r.plain, written)
        if r.index == 0:
            out.cost += r.actual_cost  # shared starting point
        else:
            out.cost += cached * READ + plain * PLAIN + (written - plain) * r.write_price
            # server-side iterations beyond the last are charged as recorded
            if r.iterations > 1:
                last = r.cache_read * READ + r.plain * PLAIN
                last += (r.context - r.cache_read - r.plain) * r.write_price
                out.cost += max(0.0, r.actual_cost - last)

        # the agent's output at this request may need something evicted
        for seg in list(segs):
            if seg.item is not None and not seg.full and r.index in seg.item.uses:
                out.misses += 1
                segs.append(Seg(seg.tokens, seg.item))  # re-fetched copy
                seg.item = None  # the stub stays behind as plain content
                seg.tokens, seg.full = STUB_TOKENS, True
                context += segs[-1].tokens
                out.cost += (context - segs[-1].tokens) * READ
                out.cost += segs[-1].tokens * r.write_price

        out.peak = max(out.peak, context)
        out.context_sum += context
        out.requests += 1
    return out


POLICIES: Dict[str, Callable[[Session], Policy]] = {
    "keep everything": lambda s: keep_all,
    "truncate at half of peak": lambda s: truncate(
        max(r.context for r in s.requests) // 2
    ),
    "stale rule": lambda s: stale_rule(),
    "hindsight reference": lambda s: hindsight,
}


def summarize(sessions: Sequence[Session]) -> dict:
    totals = {name: Outcome() for name in POLICIES}
    actual = 0.0
    for s in sessions:
        actual += sum(r.actual_cost for r in s.requests)
        for name, make in POLICIES.items():
            o, t = replay(s, make(s)), totals[name]
            for f in ("cost", "context_sum", "evictions", "evicted_tokens", "misses"):
                setattr(t, f, getattr(t, f) + getattr(o, f))
            t.requests += o.requests
            t.peak = max(t.peak, o.peak)
    reqs = [r for s in sessions for r in s.requests]
    ctx = sorted(r.context for r in reqs)

    def pct(q: float) -> int:
        return ctx[min(len(ctx) - 1, int(q * len(ctx)))]

    diagnostics = {
        "context_p50": pct(0.5),
        "context_p90": pct(0.9),
        "context_p99": pct(0.99),
        "context_max": ctx[-1],
        "requests_over_1m": sum(c > 1_000_000 for c in ctx),
        "multi_iteration_requests": sum(r.iterations > 1 for r in reqs),
        "resets": sum(r.reset for r in reqs),
        "cache_misses": sum(r.cold for r in reqs),
        "one_hour_write_share": sum(
            (r.write_price - WRITE) / (WRITE_1H - WRITE) * (r.context - r.cache_read)
            for r in reqs
        )
        / max(1, sum(r.context - r.cache_read for r in reqs)),
    }
    items = [i for s in sessions for i in s.items]
    result_tokens = sum(i.tokens for i in items)
    reused = sum(i.tokens for i in items if i.uses)
    return {
        "sessions": len(sessions),
        "requests": totals["keep everything"].requests,
        "tool_results": len(items),
        "result_tokens": result_tokens,
        "result_share_of_context": result_tokens
        / max(1, sum(r.fixed for s in sessions for r in s.requests) + result_tokens),
        "reused_share_of_result_tokens": reused / max(1, result_tokens),
        "actual_input_cost": actual,
        "diagnostics": diagnostics,
        "policies": {
            name: {
                "input_cost": o.cost,
                "vs_keep_everything": o.cost / totals["keep everything"].cost - 1,
                "mean_context": o.context_sum / max(1, o.requests),
                "peak_context": o.peak,
                "evictions": o.evictions,
                "misses": o.misses,
            }
            for name, o in totals.items()
        },
    }


def main(argv: Optional[List[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("transcripts", nargs="+", type=Path)
    ap.add_argument("--min-requests", type=int, default=20, help="skip short sessions")
    ap.add_argument("--out", type=Path, default=None, help="write summary JSON")
    ap.add_argument(
        "--diagnose", action="store_true", help="print aggregate parsing checks"
    )
    args = ap.parse_args(argv)

    sessions = [load_session(p) for p in args.transcripts]
    sessions = [s for s in sessions if len(s.requests) >= args.min_requests]
    if not sessions:
        raise SystemExit("no session with enough requests")
    res = summarize(sessions)

    print(
        f"{res['sessions']} session(s), {res['requests']} requests, "
        f"{res['tool_results']} tool results"
    )
    print(
        f"Tool results are {res['result_share_of_context']:.0%} of new context "
        f"tokens; {res['reused_share_of_result_tokens']:.0%} of those tokens are "
        f"in results with a detected later use."
    )
    keep = res["policies"]["keep everything"]["input_cost"]
    print(
        f"Cost model check: keep-everything replays at {keep / res['actual_input_cost']:.2f}x "
        f"the input cost actually recorded.\n"
    )
    print("| policy | input cost vs keep | mean context | peak context | evictions | misses |")
    print("|---|---|---|---|---|---|")
    for name, p in res["policies"].items():
        print(
            f"| {name} | {p['vs_keep_everything']:+.1%} | {p['mean_context']:,.0f} "
            f"| {p['peak_context']:,} | {p['evictions']} | {p['misses']} |"
        )
    if args.diagnose:
        d = res["diagnostics"]
        print("\nDiagnostics (recorded prompt size per request, in tokens):")
        print(
            f"  p50 {d['context_p50']:,}  p90 {d['context_p90']:,}  "
            f"p99 {d['context_p99']:,}  max {d['context_max']:,}"
        )
        print(f"  requests over 1M tokens: {d['requests_over_1m']}")
        print(f"  requests with several server-side iterations: {d['multi_iteration_requests']}")
        print(f"  context resets (compaction or restart): {d['resets']}")
        print(f"  requests that missed the cache: {d['cache_misses']}")
        print(f"  share of cache writes at the 1-hour price: {d['one_hour_write_share']:.0%}")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
