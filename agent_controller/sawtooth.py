"""Context size over a session, and what earlier compaction would cost.

Two things, both from local Claude Code transcripts and both aggregate only:

  * An SVG plot of one session's prompt size per request (the "sawtooth"),
    split into the part read from cache and the part written or re-sent,
    with compactions and cache misses marked.
  * A what-if: compact whenever the prompt passes a threshold, and re-price
    every request. This counts input cost only. It cannot see whether the agent
    would still succeed with a summary in place of its history.

    python -m agent_controller.sawtooth ~/.claude/projects/*/*.jsonl
    python -m agent_controller.sawtooth --plot-session path/to/session.jsonl ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .replay_transcript import READ, Session, load_session, miss_extra

OUTPUT_PRICE = 5.0  # output tokens relative to an uncached input token
SUMMARY = 10_000  # tokens the compaction summary adds
KEEP = 20_000  # recent tokens kept verbatim after a compaction
THRESHOLDS = (100_000, 200_000, 400_000)


# ---------------------------------------------------------------- what-if


def compacted(session: Session, threshold: float) -> Dict[str, float]:
    """Re-price a session as if it compacted whenever the prompt passed
    `threshold`. Returns cost, compaction count and the prompt size series.

    Growth between requests and the recorded uncached tail (new tokens, or a
    cache miss) are taken from the transcript. A compaction costs one request
    that reads the cached prompt and writes a summary, after which the prompt
    is the starting prefix plus the summary plus recent tokens, written anew.
    """
    reqs = session.requests
    base = reqs[0].context
    ctx = base
    cost = reqs[0].actual_cost
    fresh = False
    n = 0
    series = [ctx]
    for prev, r in zip(reqs, reqs[1:]):
        if r.reset:
            ctx = min(ctx + max(0, r.context - prev.context), r.context)
        else:
            ctx += r.context - prev.context
        tail = max(0, r.context - r.cache_read)
        if fresh:
            cost += base * READ + (ctx - base) * r.write_price
            fresh = False
        else:
            cached = max(0, ctx - tail)
            cost += cached * READ + min(tail, ctx) * r.write_price
        series.append(ctx)
        # never compact a prompt that is already near its post-compaction size
        if ctx > max(threshold, 1.5 * (base + SUMMARY + KEEP)):
            cost += ctx * READ + SUMMARY * OUTPUT_PRICE
            ctx = min(ctx, base + SUMMARY + KEEP)
            fresh = True
            n += 1
    return {"cost": cost, "compactions": n, "series": series}


def what_if(sessions: Sequence[Session], thresholds=THRESHOLDS) -> List[dict]:
    keep = sum(compacted(s, float("inf"))["cost"] for s in sessions)
    actual = sum(r.actual_cost for s in sessions for r in s.requests)
    rows = [{"threshold": None, "vs_recorded": keep / actual - 1, "compactions": 0}]
    for t in thresholds:
        runs = [compacted(s, t) for s in sessions]
        rows.append(
            {
                "threshold": t,
                "vs_recorded": sum(r["cost"] for r in runs) / actual - 1,
                "compactions": sum(r["compactions"] for r in runs),
            }
        )
    return rows


# ---------------------------------------------------------------- plot


def typical(sessions: Sequence[Session], min_requests: int = 200) -> Session:
    """The session with the median request count among the long ones."""
    long = sorted(
        (s for s in sessions if len(s.requests) >= min_requests),
        key=lambda s: len(s.requests),
    ) or sorted(sessions, key=lambda s: len(s.requests))
    return long[len(long) // 2]


def svg_plot(session: Session, threshold: Optional[float] = 200_000) -> str:
    reqs = session.requests
    n = len(reqs)
    W, H, L, R, T, B = 1000, 420, 70, 20, 50, 60
    pw, ph = W - L - R, H - T - B
    what = compacted(session, threshold)["series"] if threshold else None
    top = max(max(r.context for r in reqs), 1)
    step = 100_000 if top > 300_000 else 50_000 if top > 100_000 else 20_000
    ymax = ((top // step) + 1) * step

    def x(i: float) -> float:
        return L + pw * i / max(1, n - 1)

    def y(v: float) -> float:
        return T + ph * (1 - v / ymax)

    def area(lo: List[float], hi: List[float]) -> str:
        pts = [f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(hi)]
        pts += [f"{x(i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lo)))]
        return " ".join(pts)

    read = [r.cache_read for r in reqs]
    full = [r.context for r in reqs]
    zero = [0] * n
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#0E1319"/>',
    ]
    for v in range(0, ymax + 1, step):
        out.append(
            f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            f'stroke="#26303C" stroke-width="1"/>'
            f'<text x="{L - 8}" y="{y(v) + 4:.1f}" fill="#A9B4C2" '
            f'text-anchor="end">{v // 1000}k</text>'
        )
    out.append(f'<polygon points="{area(zero, read)}" fill="#3987E5" opacity="0.85"/>')
    out.append(f'<polygon points="{area(read, full)}" fill="#D95926" opacity="0.95"/>')
    for i, (prev, r) in enumerate(zip(reqs, reqs[1:]), start=1):
        if r.reset:
            out.append(
                f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{T}" y2="{T + ph}" '
                f'stroke="#A9B4C2" stroke-dasharray="3,3" stroke-width="1"/>'
            )
        elif miss_extra(prev, r) > 0:
            out.append(
                f'<circle cx="{x(i):.1f}" cy="{y(r.context) - 6:.1f}" r="3" fill="#EEF1F5"/>'
            )
    if what:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(what))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#199E70" stroke-width="2"/>')
    hours = ""
    times = [r.time for r in reqs if r.time is not None]
    if len(times) > 1:
        hours = f", {(times[-1] - times[0]) / 3600:.1f} hours"
    out.append(
        f'<text x="{L}" y="24" fill="#EEF1F5" font-size="16" font-weight="bold">'
        f"Prompt size per request: one session ({n:,} requests{hours})</text>"
    )
    out.append(
        f'<text x="{L + pw / 2}" y="{T + ph + 18}" fill="#A9B4C2" text-anchor="middle">'
        f"request</text>"
    )
    legend = [
        ("#3987E5", "read from cache (0.1x)"),
        ("#D95926", "written or re-sent (1.25-2x)"),
    ]
    if what:
        legend.append(("#199E70", f"what-if: compact at {int(threshold) // 1000}k"))
    lx = L
    for color, label in legend:
        out.append(
            f'<rect x="{lx}" y="{H - 24}" width="12" height="12" fill="{color}"/>'
            f'<text x="{lx + 18}" y="{H - 14}" fill="#A9B4C2">{label}</text>'
        )
        lx += 30 + 7 * len(label)
    out.append(
        f'<line x1="{lx}" x2="{lx}" y1="{H - 26}" y2="{H - 12}" stroke="#A9B4C2" '
        f'stroke-dasharray="3,3"/><text x="{lx + 8}" y="{H - 14}" fill="#A9B4C2">'
        f"compaction</text>"
    )
    lx += 95
    out.append(
        f'<circle cx="{lx}" cy="{H - 18}" r="3" fill="#EEF1F5"/>'
        f'<text x="{lx + 8}" y="{H - 14}" fill="#A9B4C2">cache miss</text>'
    )
    out.append("</svg>")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("transcripts", nargs="+", type=Path)
    ap.add_argument("--min-requests", type=int, default=20)
    ap.add_argument("--plot", type=Path, default=Path("sawtooth.svg"), help="SVG to write")
    ap.add_argument("--plot-session", type=Path, default=None,
                    help="transcript to plot (default: a typical long session)")
    ap.add_argument("--plot-threshold", type=int, default=200_000,
                    help="what-if compaction line on the plot (0 for none)")
    args = ap.parse_args(argv)

    sessions = [load_session(p) for p in args.transcripts]
    sessions = [s for s in sessions if len(s.requests) >= args.min_requests]
    if not sessions:
        raise SystemExit("no session with enough requests")

    rows = what_if(sessions)
    print(f"{len(sessions)} session(s). Compaction what-if (input cost only; "
          f"summary {SUMMARY // 1000}k, {KEEP // 1000}k recent tokens kept):\n")
    print("| compact when the prompt passes | input cost vs recorded | compactions |")
    print("|---|---|---|")
    for row in rows:
        label = "never (replay check)" if row["threshold"] is None else f"{row['threshold'] // 1000}k"
        print(f"| {label} | {row['vs_recorded']:+.1%} | {row['compactions']:,} |")

    target = load_session(args.plot_session) if args.plot_session else typical(sessions)
    args.plot.write_text(svg_plot(target, args.plot_threshold or None))
    peak = max(r.context for r in target.requests)
    print(f"\nWrote {args.plot}: {len(target.requests):,} requests, peak prompt {peak:,} tokens.")
    return {"what_if": rows}


if __name__ == "__main__":
    main()
