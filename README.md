# SC2 Coach (v0.2)

Parse SC2 replays, compute metrics in Python, and let a local LLM coach on patterns across games. The model never sees a replay binary, and never invents timestamps, APM, or supply blocks.

## What it computes

- Coach APM: non-camera actions per real minute (not official APM)
- APM by phase: 0–5m, 5–10m, 10m+
- Supply blocks: food used >= food made, at/after 1:00, at least 2 samples, with start/end and approx duration (real time, converted from game time at 1.4x)
- Expansions, units lost, and biggest 30s loss window
- `--me` filters to that player only, in games and aggregates

No build order: too noisy, no reference standard to judge against.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python coach.py summarize --replays ~/path/to/replays --me YourInGameName --out summary.json

# Local Ollama model, ~19GB. Verify with `ollama list`.
SC2_COACH_MODEL=muse-glimmer:30b-mlx .venv/bin/python coach.py coach --replays ~/path/to/replays --me YourInGameName

# No model? Print the prompt only.
.venv/bin/python coach.py coach --summary sample_summary.json --no-llm
```

## Tested on

13-file pro demo pack (PiGosaur 103, MaxPax vs Clem + Reynor vs MaxPax). Pro replays, not the author's games, and not in this repo.

`--me MaxPax`: 9 games, 6-3, avg APM 293.6, avg duration 12:10, 0.8 supply blocks / 18.1 blocked seconds per game.

## Limits

- Post-game only, not real-time
- Supply durations are approx; coach APM is not official APM
- No UI or training; Python does the maths, the LLM judges only
- Local Ollama only, no API keys
- Replays are never committed (`*.SC2Replay` is ignored)
- `sample_summary.json` is a labelled example, not a real game
