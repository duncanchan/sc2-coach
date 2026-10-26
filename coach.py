#!/usr/bin/env python3
"""
SC2 Coach — basic LLM wrapper, v0.2.

What it does, in order:
  1. Parse a folder of .SC2Replay files with sc2reader.
  2. Compress each game to a small structured summary, then aggregate
     across the pack in Python.
  3. Send the summary + a coaching prompt to a model and print
     Top 3 fixes, each with a timestamp example, why it lost value,
     and one drill.

v0.2 event metrics:
  - avg_apm
  - Time based metrics: supply blocks, expansions, units lost
  - Supply blocks: consecutive PlayerStatsEvent samples where
    food_used >= food_made, starting at or after 1:00 real time,
    with at least 2 capped samples. Approx duration = span of the
    capped samples + one sample interval. Early 0:40 caps at the
    starting 15 supply are excluded by the 1:00 rule.
  - Expansions: UnitDoneEvent for Nexus / Hatchery / CommandCenter
    and its upgrades, owned by that player (the starting town hall
    is pre-placed, so these are expansions).
  - Units lost: UnitDiedEvent count for that player, plus the
    largest number of deaths in any 30-real-second window.
  - build_order_first_20 is deliberately NOT tracked in v0.2:
    it is noisy (workers / buildings / army / upgrades mixed),
    has no reference standard to judge against, and is not where
    recurring, fixable patterns live. Supply, APM by phase, and
    fight losses are.

Usage:
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python coach.py summarize --replays ~/path/to/replays --out summary.json
  .venv/bin/python coach.py coach --replays ~/path/to/replays
  .venv/bin/python coach.py coach --summary sample_summary.json

The model is local Ollama at http://localhost:11434. Set SC2_COACH_MODEL
(ex:: muse-glimmer:30b-mlx, ~19GB;

sample_summary.json is a LABELLED EXAMPLE. It exists
so the prompt/coaching flow can be tested end-to-end without a replay.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

SYSTEM_PROMPT = """You are a StarCraft II coach. Rules:
- Do the maths already done in the summary; do not recompute from raw events.
- Output exactly Top 3 fixes, ranked by value lost.
- For each fix: (1) the pattern with a timestamp example from the
  summary, (2) why it lost value in plain terms, (3) one concrete
  drill to fix it.
- Use only the APM, supply blocks, expansions and unit losses in the
  summary. If a field is missing or marked unknown, say so — do not
  invent a timestamp, a supply-block count, or an APM value.
- Keep it short: under 400 words total."""

ACTION_TYPES = {
    "SelectionEvent", "GetControlGroupEvent", "SetControlGroupEvent",
    "AddToControlGroupEvent", "DeleteControlGroupEvent",
    "TargetPointCommandEvent", "TargetUnitCommandEvent",
    "BasicCommandEvent", "UpdateTargetPointCommandEvent",
    "UpdateTargetUnitCommandEvent",
}
TOWNHALLS = {
    "Nexus", "Hatchery", "CommandCenter", "CommandCenterFlying",
    "OrbitalCommand", "PlanetaryFortress",
}


def _safe(obj, *names, default=None):
    for name in names:
        v = getattr(obj, name, None)
        if v is not None:
            return v
    return default


def _to_seconds(value):
    """sc2reader's game_length is a Length object, not an int.

    int(Length) raises TypeError, which previously made every real
    replay fail to parse (0 games, all in games_failed). Pull the
    .seconds attribute when present, otherwise fall back to int().
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    seconds = getattr(value, "seconds", None)
    if seconds is not None:
        try:
            return int(seconds)
        except Exception:
            pass
    try:
        return int(value)
    except Exception:
        return 0


def _fmt_time(seconds):
    try:
        s = int(seconds)
        return f"{s // 60}:{s % 60:02d}"
    except Exception:
        return "unknown"


def _player_metrics(replay, player_name, duration_real, scale):
    """Event-derived metrics for one player, in real seconds."""
    def real(game_sec):
        return int(round(game_sec * scale))

    # APM and APM by phase (real minutes).
    ev_secs = []
    for e in replay.game_events:
        if e.player and e.player.name == player_name and e.name in ACTION_TYPES:
            ev_secs.append(real(e.second))
    ev_secs.sort()
    minutes = duration_real / 60 if duration_real else 0
    avg_apm = round(len(ev_secs) / minutes, 1) if minutes else "unknown"
    phases = {}
    for label, start, end in (("0-5m", 0, 300), ("5-10m", 300, 600), ("10m+", 600, None)):
        if duration_real <= start:
            phases[label] = None
            continue
        window_end = duration_real if end is None else min(duration_real, end)
        window_mins = (window_end - start) / 60
        count = sum(1 for s in ev_secs if start <= s < window_end)
        phases[label] = round(count / window_mins, 1) if window_mins else None

    # Supply blocks from PlayerStatsEvent.
    stats = []
    for e in replay.tracker_events:
        if type(e).__name__ == "PlayerStatsEvent" and e.player and e.player.name == player_name:
            stats.append((real(e.second), float(e.food_used), float(e.food_made)))
    stats.sort()
    intervals = [b - a for (a, _, _), (b, _, _) in zip(stats, stats[1:]) if b > a]
    interval = sorted(intervals)[len(intervals) // 2] if intervals else 7
    blocks = []
    cur = None  # [start_real, end_real, samples]
    for sec, used, made in stats:
        if used >= made:
            if cur is None:
                cur = [sec, sec, 1]
            else:
                cur[1] = sec
                cur[2] += 1
        else:
            if cur is not None:
                blocks.append(cur)
                cur = None
    if cur is not None:
        blocks.append(cur)
    supply_blocks_raw = []
    for start, end, samples in blocks:
        if start < 60 or samples < 2:
            continue
        supply_blocks_raw.append({
            "start_sec": start,
            "end_sec": end,
            "capped_samples": samples,
            "approx_duration_seconds": int(end - start + interval),
        })
    # Longest blocks first, capped at 5, then back to time order.
    supply_blocks_raw = sorted(supply_blocks_raw, key=lambda b: b["approx_duration_seconds"], reverse=True)[:5]
    supply_blocks_raw.sort(key=lambda b: b["start_sec"])
    supply_blocks = [{
        "start": _fmt_time(b["start_sec"]),
        "end": _fmt_time(b["end_sec"]),
        "capped_samples": b["capped_samples"],
        "approx_duration_seconds": b["approx_duration_seconds"],
    } for b in supply_blocks_raw]
    total_blocked = sum(b["approx_duration_seconds"] for b in supply_blocks)

    # Expansions.
    expansions_raw = []
    for e in replay.tracker_events:
        if (type(e).__name__ == "UnitDoneEvent" and e.unit
                and getattr(e.unit, "owner", None) and e.unit.owner.name == player_name
                and e.unit.name in TOWNHALLS):
            expansions_raw.append((real(e.second), str(e.unit.name)))
    expansions_raw.sort()
    expansions = [{"time": _fmt_time(sec), "building": name} for sec, name in expansions_raw[:6]]

    # Units lost and biggest 30-real-second loss window.
    death_secs = []
    for e in replay.tracker_events:
        if (type(e).__name__ == "UnitDiedEvent" and e.unit
                and getattr(e.unit, "owner", None) and e.unit.owner.name == player_name):
            death_secs.append(real(e.second))
    death_secs.sort()
    biggest = None
    for s in death_secs:
        count = sum(1 for x in death_secs if s <= x < s + 30)
        if biggest is None or count > biggest[1]:
            biggest = (s, count)
    biggest_window = ({"start": _fmt_time(biggest[0]), "deaths_in_30s": biggest[1]}
                      if biggest else None)

    return {
        "avg_apm": avg_apm,
        "apm_by_phase": phases,
        "supply_blocks": supply_blocks,
        "supply_block_count": len(supply_blocks),
        "total_supply_blocked_seconds": total_blocked,
        "expansions": expansions,
        "units_lost": len(death_secs),
        "biggest_loss_window": biggest_window,
    }


def parse_replay(path):
    """Parse one replay with sc2reader, best-effort."""
    import sc2reader

    replay = sc2reader.load_replay(str(path), load_level=4)
    duration_real = _to_seconds(_safe(replay, "game_length", "length", default=0))
    frames = _safe(replay, "frames", default=0) or 0
    game_seconds = frames / 16 if frames else duration_real
    scale = (duration_real / game_seconds) if game_seconds else 1.0
    if not (0.5 <= scale <= 1.5):
        scale = 1.0
    players = []
    for p in replay.players:
        metrics = _player_metrics(replay, str(_safe(p, "name", default="unknown")),
                                  duration_real, scale)
        players.append({
            "name": str(_safe(p, "name", default="unknown")),
            "race": str(_safe(p, "play_race", "pick_race", default="unknown")),
            "result": str(_safe(p, "result", default="unknown")),
            **metrics,
        })
    return {
        "file": Path(path).name,
        "map": str(_safe(replay, "map_name", "map", default="unknown")),
        "duration": _fmt_time(duration_real),
        "duration_seconds": duration_real,
        "players": players,
    }


def summarize(replay_dir, me=None):
    paths = sorted(Path(replay_dir).glob("*.SC2Replay"))
    if not paths:
        raise SystemExit(f"No .SC2Replay files found in {replay_dir}")
    all_games, errors = [], []
    for path in paths:
        try:
            all_games.append(parse_replay(path))
        except Exception as e:  # keep one bad replay from killing the pack
            errors.append({"file": path.name, "error": str(e)[:200]})
    # --me is a full filter in v0.2: only games that player played in
    # are in the games list and in every aggregate (v0.1 filtered
    # only wins/losses, leaving duration/matchups unfiltered).
    if me:
        games = [g for g in all_games if any(p["name"] == me for p in g["players"])]
    else:
        games = all_games
    # Aggregate in Python, not in the model.
    wins = losses = 0
    apms, durations, block_counts, blocked_secs, units_lost = [], [], [], [], []
    matchups = Counter()
    for g in games:
        durations.append(g["duration_seconds"])
        races = [p["race"] for p in g["players"]]
        if len(races) == 2:
            matchups["v".join(races)] += 1
        for p in g["players"]:
            if me and p["name"] != me:
                continue
            if isinstance(p["avg_apm"], (int, float)):
                apms.append(p["avg_apm"])
            if p["result"] == "Win":
                wins += 1
            elif p["result"] == "Loss":
                losses += 1
            block_counts.append(p["supply_block_count"])
            blocked_secs.append(p["total_supply_blocked_seconds"])
            units_lost.append(p["units_lost"])
    def _mean(vals):
        return round(sum(vals) / len(vals), 1) if vals else "unknown"
    return {
        "games_parsed": len(games),
        "games_total_in_folder": len(all_games),
        "games_failed": errors,
        "player_filter": me,
        "aggregate": {
            "wins": wins,
            "losses": losses,
            "avg_apm": _mean(apms),
            "avg_duration": _fmt_time(sum(durations) / len(durations)) if durations else "unknown",
            "matchups": dict(matchups),
            "avg_supply_blocks_per_game": _mean(block_counts),
            "avg_supply_blocked_seconds_per_game": _mean(blocked_secs),
            "avg_units_lost_per_game": _mean(units_lost),
        },
        "games": games,
    }


def build_prompt(summary):
    body = json.dumps(summary, indent=2)[:40000]  # full 9-game v0.2 summary is ~20k chars; Glimmer context is 128k
    return (f"{SYSTEM_PROMPT}\n\n"
            "Here is the parsed replay-pack summary (already aggregated in Python):\n"
            f"{body}\n\n"
            "Coach on recurring patterns across these games. Top 3 fixes only.")


def call_ollama(prompt, model):
    if requests is None:
        raise SystemExit("requests not installed: pip install -r requirements.txt")
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="SC2 Coach — basic LLM wrapper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize")
    s.add_argument("--replays", required=True)
    s.add_argument("--me", default=None, help="Player to coach: filters games and aggregates to that player")
    s.add_argument("--out", default="summary.json")

    c = sub.add_parser("coach")
    c.add_argument("--replays", default=None)
    c.add_argument("--summary", default=None, help="Use an existing summary.json instead of parsing")
    c.add_argument("--me", default=None)
    c.add_argument("--model", default=None)
    c.add_argument("--no-llm", action="store_true", help="Print prompt + summary, do not call a model")

    args = ap.parse_args()
    if args.cmd == "summarize":
        summary = summarize(args.replays, me=args.me)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.out}: {summary['games_parsed']} games, "
              f"{summary['aggregate']['wins']}W-{summary['aggregate']['losses']}L")
        for failed in summary["games_failed"]:
            print(f"FAILED {failed['file']}: {failed['error']}", file=sys.stderr)
        return
    # coach
    if args.summary:
        summary = json.loads(Path(args.summary).read_text())
    else:
        if not args.replays:
            ap.error("coach needs --replays or --summary")
        summary = summarize(args.replays, me=args.me)
    prompt = build_prompt(summary)
    if args.no_llm:
        print(prompt)
        return
    model = args.model or os.environ.get("SC2_COACH_MODEL") or "muse-glimmer:30b-mlx"
    print(call_ollama(prompt, model))


if __name__ == "__main__":
    main()
