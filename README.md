# Biome Simulation

Iterative 3D biome simulation built in public (episode-by-episode). Current focus: seeded terrain generation using Perlin noise.

## Status

- Terrain: 3D mesh generated from seeded Perlin noise
- Rendering: Three.js scene with orbit controls
- Tests: deterministic noise unit tests (Vitest)

## Local Dev

Requirements: Node.js 20+ (recommended)

```bash
npm install
npm run dev
```

Other useful commands:

```bash
npm run typecheck
npm test
npm run build
npm run preview
```

## Controls

- `R` regenerate terrain (new seed)
- `G` toggle wireframe

## Repo Notes

- `plans/` and `content/` are intentionally gitignored (used for local planning + narration scripts).
- High-level targets live in `GOALS.md`.
