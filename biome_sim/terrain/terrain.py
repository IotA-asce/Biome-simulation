from __future__ import annotations

from dataclasses import dataclass

from biome_sim.noise.perlin2d import FbmOptions, Perlin2D


@dataclass(frozen=True)
class TerrainParams:
    seed: int
    size: float = 220.0
    grid: int = 81
    amplitude: float = 42.0
    frequency: float = 2.2
    sea_level01: float = 0.48
    octaves: int = 5
    lacunarity: float = 2.0
    persistence: float = 0.5
    warp: float = 0.55
    river_threshold: int = 140
    river_carve: float = 0.28


class Terrain:
    def __init__(self, params: TerrainParams):
        self.params = params
        self._perlin = Perlin2D(params.seed)
        self._heights: list[list[float]] = []
        self._heights01: list[list[float]] = []
        self._x: list[float] = []
        self._z: list[float] = []
        self.height_min: float = 0.0
        self.height_max: float = 0.0
        self.river_edges: list[tuple[int, int, int]] = []
        self.regenerate(params.seed)

    @property
    def heights(self) -> list[list[float]]:
        return self._heights

    @property
    def heights01(self) -> list[list[float]]:
        return self._heights01

    @property
    def sea_level_y(self) -> float:
        # By construction: y = (height01 - sea_level01) * amplitude
        return 0.0

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

        self._x = [-half + c * step for c in range(g)]
        self._z = [-half + r * step for r in range(g)]

        heights01: list[list[float]] = []

        def clamp01(v: float) -> float:
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        def smoothstep(a: float, b: float, x: float) -> float:
            if a == b:
                return 0.0
            t = clamp01((x - a) / (b - a))
            return t * t * (3.0 - 2.0 * t)

        for r in range(g):
            row01: list[float] = []
            z = self._z[r]
            for c in range(g):
                x = self._x[c]
                nx = (x / p.size) * p.frequency
                nz = (z / p.size) * p.frequency

                # Domain warp: avoids the “perfect grid noise” look.
                wx = self._perlin.fbm(nx * 0.65 + 13.1, nz * 0.65 - 9.7, opts) * p.warp
                wz = self._perlin.fbm(nx * 0.65 - 7.4, nz * 0.65 + 11.9, opts) * p.warp
                nx2 = nx + wx
                nz2 = nz + wz

                continent = self._perlin.noise01(
                    nx2 * 0.38 + 1000.0, nz2 * 0.38 - 1000.0
                )
                hills = self._perlin.noise01(nx2 * 0.95 + 200.0, nz2 * 0.95 - 200.0)
                detail = self._perlin.noise01(nx2 * 2.2 - 40.0, nz2 * 2.2 + 40.0)

                # Ridged mountains masked to show up mostly on larger landmasses.
                mountain_mask = smoothstep(0.45, 0.78, continent)
                mountains = self._perlin.ridged_fbm(
                    nx2 * 1.35 + 520.0, nz2 * 1.35 - 520.0, opts
                )
                mountains *= mountain_mask

                # Shape curves: push up mountains, soften plains.
                hills = hills**1.35
                mountains = mountains**1.55

                h01 = 0.62 * continent + 0.22 * hills + 0.40 * mountains + 0.06 * detail
                # Gentle contrast.
                h01 = clamp01(h01)
                h01 = h01**1.10
                row01.append(h01)
            heights01.append(row01)

        self._heights01 = heights01

        heights: list[list[float]] = []
        min_y = 1e9
        max_y = -1e9
        for r in range(g):
            row: list[float] = []
            for c in range(g):
                y = (heights01[r][c] - p.sea_level01) * p.amplitude
                row.append(y)
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
            heights.append(row)

        self._heights = heights
        self.height_min = min_y
        self.height_max = max_y

        self._compute_rivers()

    def _compute_rivers(self) -> None:
        p = self.params
        g = p.grid
        n = g * g

        def idx(r: int, c: int) -> int:
            return r * g + c

        def rc(i: int) -> tuple[int, int]:
            return (i // g, i % g)

        h = self._heights
        down: list[int] = [-1] * n
        heights_flat: list[float] = [0.0] * n

        for r in range(g):
            for c in range(g):
                i = idx(r, c)
                heights_flat[i] = h[r][c]

        # Choose steepest downhill neighbor (8-neighborhood).
        for r in range(g):
            for c in range(g):
                i = idx(r, c)
                y0 = h[r][c]
                best = -1
                best_y = y0
                for dr in (-1, 0, 1):
                    rr = r + dr
                    if rr < 0 or rr >= g:
                        continue
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        cc = c + dc
                        if cc < 0 or cc >= g:
                            continue
                        y1 = h[rr][cc]
                        if y1 < best_y:
                            best_y = y1
                            best = idx(rr, cc)
                down[i] = best

        order = list(range(n))
        order.sort(key=lambda i: heights_flat[i], reverse=True)
        acc = [1] * n

        for i in order:
            di = down[i]
            if di != -1:
                acc[di] += acc[i]

        # Carve river beds a bit for visibility.
        carve = p.river_carve
        if carve > 0.0:
            for i in range(n):
                if acc[i] >= p.river_threshold:
                    r, c = rc(i)
                    if h[r][c] > self.sea_level_y + 0.25:
                        # Stronger carving for bigger rivers.
                        amt = carve * (
                            1.0 + (acc[i] / max(1, p.river_threshold)) * 0.35
                        )
                        h[r][c] -= amt

        # Recompute downhill links after carving.
        for r in range(g):
            for c in range(g):
                i = idx(r, c)
                y0 = h[r][c]
                best = -1
                best_y = y0
                for dr in (-1, 0, 1):
                    rr = r + dr
                    if rr < 0 or rr >= g:
                        continue
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        cc = c + dc
                        if cc < 0 or cc >= g:
                            continue
                        y1 = h[rr][cc]
                        if y1 < best_y:
                            best_y = y1
                            best = idx(rr, cc)
                down[i] = best

        heights_flat = [h[r][c] for r in range(g) for c in range(g)]
        order.sort(key=lambda i: heights_flat[i], reverse=True)
        acc = [1] * n
        for i in order:
            di = down[i]
            if di != -1:
                acc[di] += acc[i]

        edges: list[tuple[int, int, int]] = []
        for i in range(n):
            di = down[i]
            if di == -1:
                continue
            if acc[i] < p.river_threshold:
                continue
            r0, c0 = rc(i)
            if h[r0][c0] <= self.sea_level_y + 0.05:
                continue
            edges.append((i, di, acc[i]))

        self.river_edges = edges

    def vertex(self, r: int, c: int) -> tuple[float, float, float]:
        y = self._heights[r][c]
        return (self._x[c], y, self._z[r])

    def vertex_i(self, i: int) -> tuple[float, float, float]:
        g = self.params.grid
        r = i // g
        c = i % g
        return self.vertex(r, c)

    def height01(self, r: int, c: int) -> float:
        return self._heights01[r][c]
