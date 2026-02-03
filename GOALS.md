# Project Goals

This repo builds a 3D biome simulation in iterative episodes (YouTube-friendly), starting from terrain generation and scaling up to vegetation, wildlife, and ecosystem dynamics.

## Core Goals

- Generate 3D terrain from Perlin noise (seeded, reproducible)
- Populate vegetation procedurally (noise + terrain constraints: elevation, slope, moisture)
- Simulate wildlife agents with traits (speed, vision, metabolism, temperament)
- Model basic needs and habits (hunger, thirst, rest, roaming, nesting)
- Implement food chains and predation (plants -> herbivores -> carnivores)
- Add herd / flock behavior (cohesion, separation, alignment) and threat response
- Make agents interact with terrain (navigation, slope limits, water avoidance/usage)
- Keep simulation performant (instancing, spatial partitioning, fixed timestep)

## Episode Milestones (High Level)

1. Project scaffold + 3D renderer + reproducible seeded terrain
2. Terrain layers: waterline, biomes by temperature/moisture, vertex colors
3. Vegetation pass 1: instanced meshes, density maps, LOD strategy
4. Wildlife pass 1: agent loop, needs, movement, simple sensing
5. Predation + grazing + energy budgets + population stability controls
6. Herd behavior + territory + migration triggers
7. Persistence (save/load seed + world state) and replayability
8. Instrumentation: graphs, debugging tools, time controls

## Definition of “Done” (v1)

- Given a seed, the same terrain/vegetation layout is reproduced
- At least 3 species exist (plant, herbivore, carnivore)
- Food chain is functional and populations remain bounded for extended runs
- Herding is visible and meaningfully affects survival
- App runs smoothly at real-time speeds on a typical laptop
