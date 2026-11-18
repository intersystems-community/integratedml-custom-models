import json

import pytest

from agent_controller.replay_transcript import (
    READ,
    STUB_TOKENS,
    WRITE,
    Item,
    Seg,
    View,
    best_cut,
    hindsight,
    keep_all,
    load_session,
    replay,
    stale_rule,
    summarize,
    truncate,
)

BIG = "\n".join(f"line {i:04d} of the original file contents here" for i in range(400))


def assistant(mid, context, read, blocks):
    usage = {
        "input_tokens": 0,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": context - read,
    }
    return {
        "type": "assistant",
        "message": {"id": mid, "usage": usage, "content": blocks},
    }


def tool_use(uid, name, **inp):
    return {"type": "tool_use", "id": uid, "name": name, "input": inp}


def result(uid, text):
    block = {"type": "tool_result", "tool_use_id": uid, "content": text}
    return {"type": "user", "message": {"content": [block]}}


def write(tmp_path, rows):
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def build(tmp_path, edit_at=None, n=30, reread_at=None):
    """Read a big file at request 0, then idle requests; optionally edit the
    file (quoting one of its lines) or read it again later."""
    big = len(BIG) // 4
    rows = [{"type": "user", "message": {"content": "start"}}]
    rows.append(assistant("m0", 10_000, 0, [tool_use("t0", "Read", file_path="/a.py")]))
    rows.append(result("t0", BIG))
    ctx = 10_000 + big + 100
    for k in range(1, n):
        prev = ctx - 100 if k > 1 else 10_000
        if k == edit_at:
            block = tool_use(
                f"t{k}",
                "Edit",
                file_path="/a.py",
                old_string="line 0200 of the original file contents here",
            )
        elif k == reread_at:
            block = tool_use(f"t{k}", "Read", file_path="/a.py")
        else:
            block = tool_use(f"t{k}", "Bash", command=f"echo {k}")
        rows.append(assistant(f"m{k}", ctx, prev, [block]))
        rows.append(result(f"t{k}", BIG if k == reread_at else "ok"))
        ctx += 100 + (big if k == reread_at else 0)
    return load_session(write(tmp_path, rows))


def test_parses_requests_items_and_calibrated_sizes(tmp_path):
    s = build(tmp_path)
    assert len(s.requests) == 30
    read = s.items[0]
    assert (read.tool, read.path, read.born) == ("Read", "/a.py", 1)
    assert read.tokens == pytest.approx(len(BIG) // 4, abs=2)
    assert not any(r.reset or r.cold for r in s.requests)


def test_keep_everything_matches_recorded_cost(tmp_path):
    s = build(tmp_path)
    actual = sum(r.actual_cost for r in s.requests)
    assert replay(s, keep_all).cost == pytest.approx(actual, rel=0.01)


def test_edit_marks_the_read_as_used_and_superseded(tmp_path):
    s = build(tmp_path, edit_at=20)
    assert s.items[0].uses == [20]
    assert s.items[0].superseded_at == 21


def test_reread_supersedes_the_earlier_read(tmp_path):
    s = build(tmp_path, reread_at=10)
    first, second = [i for i in s.items if i.tool == "Read"]
    assert first.superseded_at == second.born
    assert second.superseded_at is None


def test_hindsight_saves_cost_without_misses(tmp_path):
    s = build(tmp_path)
    keep, best = replay(s, keep_all), replay(s, hindsight)
    assert best.misses == 0 and best.evictions == 1
    assert best.cost < keep.cost and best.peak <= keep.peak


def test_hindsight_keeps_a_result_until_its_last_use(tmp_path):
    s = build(tmp_path, edit_at=20)
    best = replay(s, hindsight)
    assert best.misses == 0
    assert best.cost > replay(build(tmp_path), hindsight).cost


def test_evicting_a_needed_result_is_charged_as_a_miss(tmp_path):
    s = build(tmp_path, edit_at=20)
    eager = replay(s, truncate(budget=5_000))
    assert eager.misses == 1
    no_use = replay(build(tmp_path), truncate(budget=5_000))
    assert eager.cost > no_use.cost


def test_eviction_weighs_savings_against_rewriting_the_tail():
    old = Item("a", "Read", None, born=0, tokens=1_000)
    segs = [Seg(1_000, old), Seg(50_000)]
    view = View(step=5, n_requests=10, segs=segs, cached_end=2, cold=False, context=51_000)
    assert best_cut(view, [0], remaining=5) == []  # 50k rewrite outweighs it
    assert best_cut(view, [0], remaining=1_000) == [0]
    recent = View(5, 10, [Seg(50_000), Seg(1_000, old)], 2, False, 51_000)
    assert best_cut(recent, [1], remaining=5) == [1]  # nothing after it to rewrite
    cold = View(5, 10, segs, 2, True, 51_000)
    assert best_cut(cold, [0], remaining=1) == [0]  # prompt is rewritten anyway


def test_stale_rule_evicts_superseded_results(tmp_path):
    s = build(tmp_path, reread_at=5, n=60)
    out = replay(s, stale_rule(age=1000))
    assert out.evictions == 1 and out.misses == 0


def test_context_drop_is_treated_as_a_reset(tmp_path):
    rows = [
        assistant("m0", 50_000, 0, [tool_use("t0", "Bash", command="x")]),
        result("t0", "ok"),
        assistant("m1", 20_000, 0, [tool_use("t1", "Bash", command="y")]),
    ]
    s = load_session(write(tmp_path, rows))
    assert s.requests[1].reset and s.requests[1].fixed == 20_000
    assert replay(s, keep_all).cost == pytest.approx(50_000 * WRITE + 20_000 * WRITE)


def test_summary_reports_every_policy(tmp_path):
    res = summarize([build(tmp_path, edit_at=20)])
    assert res["policies"]["keep everything"]["vs_keep_everything"] == 0
    assert res["policies"]["hindsight reference"]["misses"] == 0
    assert 0 < res["reused_share_of_result_tokens"] <= 1
    assert READ < WRITE and STUB_TOKENS > 0


def test_any_shrink_rebuilds_from_the_recorded_size(tmp_path):
    rows = [
        assistant("m0", 50_000, 0, [tool_use("t0", "Bash", command="x")]),
        result("t0", BIG),
        assistant("m1", 60_000, 50_000, [tool_use("t1", "Bash", command="y")]),
        assistant("m2", 55_000, 40_000, [tool_use("t2", "Bash", command="z")]),
        assistant("m3", 56_000, 55_000, [tool_use("t3", "Bash", command="w")]),
    ]
    s = load_session(write(tmp_path, rows))
    assert s.requests[2].reset and not s.requests[3].reset
    out = replay(s, keep_all)
    assert out.peak == 60_000
    assert out.cost == pytest.approx(sum(r.actual_cost for r in s.requests))


def test_one_hour_writes_and_server_iterations_are_priced(tmp_path):
    usage = {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 1_000,
        "cache_creation": {"ephemeral_1h_input_tokens": 1_000},
    }
    second = dict(usage, cache_read_input_tokens=1_000, cache_creation_input_tokens=500,
                  cache_creation={"ephemeral_1h_input_tokens": 0})
    row = {
        "type": "assistant",
        "message": {
            "id": "m0",
            "usage": dict(usage, iterations=[usage, second]),
            "content": [],
        },
    }
    s = load_session(write(tmp_path, [row]))
    r = s.requests[0]
    assert r.context == 1_500 and r.iterations == 2
    assert r.actual_cost == pytest.approx(1_000 * 2.0 + 1_000 * READ + 500 * WRITE)


def test_composition_splits_growth_by_source(tmp_path):
    rows = [
        {"type": "user", "message": {"content": "please fix the bug " * 50}},
        assistant("m0", 10_000, 0, [tool_use("t0", "Bash", command="ls")]),
        result("t0", "x" * 4_000),
        assistant("m1", 12_000, 10_000, [{"type": "text", "text": "done " * 200}]),
        {"type": "user", "message": {"content": "thanks " * 100}},
        assistant("m2", 13_000, 12_000, []),
    ]
    s = load_session(write(tmp_path, rows))
    first, second, third = (r.composition for r in s.requests)
    assert first == {"start or after a reset": 10_000}
    assert second["tool results"] == 1_000 and sum(second.values()) == 2_000
    assert third["agent text"] == 250 and third["your messages"] == 175
    assert sum(third.values()) == 1_000
    shares = summarize([s])["composition"]
    assert sum(v["new"] for v in shares.values()) == pytest.approx(1.0)


def test_cache_misses_are_grouped_by_pause(tmp_path):
    rows = [
        assistant("m0", 10_000, 0, []),
        assistant("m1", 11_000, 10_000, []),
        assistant("m2", 12_000, 0, []),  # full rewrite after a long pause
    ]
    for row, t in zip(rows, ("10:00:00", "10:01:00", "12:00:00")):
        row["timestamp"] = f"2026-01-01T{t}Z"
    s = load_session(write(tmp_path, rows))
    c = summarize([s])["cache"]["misses_by_pause"]
    assert c["over 1 hour"]["requests"] == 1
    assert c["under 5 min"]["requests"] == 0
    assert c["over 1 hour"]["extra_cost"] == pytest.approx(11_000 * (WRITE - READ))
