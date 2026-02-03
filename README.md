# Biome Simulation

Iterative biome simulation built in public (episode-by-episode). Current focus: seeded terrain generation using Perlin noise, rendered via a simple software 3D projection in pygame.

## Status

- Terrain: organic heightfield (warped fBm + ridged mountains) with clustered archipelago
- Water: global sea level + beaches + rivers (underwater view available)
- Rendering: pygame window + orbit camera + solid terrain (painter's algorithm)
- Tests: deterministic noise unit tests (pytest)

## Local Dev

Recommended: Python 3.12+ (pygame-ce supports newer versions; this repo uses `pygame-ce`)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

Run:

```bash
.venv/bin/python main.py
```

Tests:

```bash
.venv/bin/python -m pytest
```

## Controls

- `LMB drag` orbit camera
- `Mouse wheel` zoom
- `Arrow keys` orbit camera
- `R` regenerate terrain (new seed)
- `G` toggle wireframe overlay
- `U` toggle underwater view
- `P` cycle performance scaling
- `Esc` quit

## Repo Notes

- `plans/` and `content/` are intentionally gitignored (used for local planning + narration scripts).
- High-level targets live in `GOALS.md`.
