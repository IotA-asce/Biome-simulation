from __future__ import annotations

from dataclasses import dataclass

from biome_sim.noise.perlin2d import FbmOptions, Perlin2D


@dataclass(frozen=True)
class TerrainParams:
    seed: int
    size: float = 220.0
    grid: int = 85
    amplitude: float = 28.0
    frequency: float = 2.2
    octaves: int = 5
    lacunarity: float = 2.0
    persistence: float = 0.5


class Terrain:
    def __init__(self, params: TerrainParams):
        self.params = params
        self._perlin = Perlin2D(params.seed)
        self._heights: list[list[float]] = []
        self.regenerate(params.seed)

    @property
    def heights(self) -> list[list[float]]:
        return self._heights

    def regenerate(self, seed: int) -> None:
        p = TerrainParams(
            seed=seed & 0xFFFFFFFF,
            size=self.params.size,
            grid=self.params.grid,
            amplitude=self.params.amplitude,
            frequency=self.params.frequency,
            octaves=self.params.octaves,
            lacunarity=self.params.lacunarity,
            persistence=self.params.persistence,
        )
        self.params = p
        self._perlin = Perlin2D(p.seed)

        opts = FbmOptions(
            octaves=p.octaves, lacunarity=p.lacunarity, persistence=p.persistence
        )
        g = p.grid
        half = p.size * 0.5
        step = p.size / (g - 1)

        heights: list[list[float]] = []
        for r in range(g):
            row: list[float] = []
            z = -half + r * step
            for c in range(g):
                x = -half + c * step
                nx = (x / p.size) * p.frequency
                nz = (z / p.size) * p.frequency
                h = self._perlin.fbm(nx + 1000.0, nz - 1000.0, opts) * p.amplitude
                row.append(h)
            heights.append(row)
        self._heights = heights

    def vertex(self, r: int, c: int) -> tuple[float, float, float]:
        p = self.params
        g = p.grid
        half = p.size * 0.5
        step = p.size / (g - 1)
        x = -half + c * step
        z = -half + r * step
        y = self._heights[r][c]
        return (x, y, z)
