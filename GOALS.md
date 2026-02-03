# Project Goals

This repo builds a 3D-ish biome simulation in iterative episodes (YouTube-friendly), starting from terrain generation and scaling up to vegetation, wildlife, and ecosystem dynamics.

Implementation stack: Python + pygame (software 3D projection for early iterations).

## Core Goals

- Generate 3D terrain from Perlin noise (seeded, reproducible)
- Populate vegetation procedurally (noise + terrain constraints: elevation, slope, moisture)
- Simulate wildlife agents with traits (speed, vision, metabolism, temperament)
- Model basic needs and habits (hunger, thirst, rest, roaming, nesting)
- Implement food chains and predation (plants -> herbivores -> carnivores)
- Add herd / flock behavior (cohesion, separation, alignment) and threat response
- Make agents interact with terrain (navigation, slope limits, water avoidance/usage)
- Keep simulation performant (chunked rendering, spatial partitioning, fixed timestep)

## Episode Milestones (High Level)

1. Python/pygame scaffold + basic 3D renderer + reproducible seeded terrain
2. Terrain layers: waterline + temperature/moisture maps + biome classification
3. Vegetation pass 1: sprite/billboard placement via density maps + LOD by distance
4. Wildlife pass 1: agent loop (fixed timestep), needs, simple sensing + steering
5. Food chains: grazing + predation + energy budgets + stability controls
6. Herd behavior: cohesion/alignment/separation + threat response
7. Persistence: save/load seed + sim state + replay hooks
8. Tooling: debug overlays, time controls, population graphs

## Definition of “Done” (v1)

- Given a seed, the same terrain/vegetation layout is reproduced
- At least 3 species exist (plant, herbivore, carnivore)
- Food chain is functional and populations remain bounded for extended runs
- Herding is visible and meaningfully affects survival
- App runs smoothly at real-time speeds on a typical laptop
