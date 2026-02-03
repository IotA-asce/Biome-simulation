from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from biome_sim.core.prng import Mulberry32

from biome_sim.noise.perlin2d import FbmOptions, Perlin2D


@dataclass(frozen=True)
class TerrainParams:
    seed: int
    size: float = 220.0
    grid: int = 121
    amplitude: float = 42.0
    frequency: float = 2.2
    sea_level01: float = 0.48
    octaves: int = 5
    lacunarity: float = 2.0
    persistence: float = 0.5
    warp: float = 0.55

    # Controlled environment: one main island + several small islands.
    main_island_radius: float = 0.78
    main_island_sharpness: float = 1.65
    small_island_count: int = 7
    small_island_radius_min: float = 0.06
    small_island_radius_max: float = 0.14
    coast_wobble: float = 0.06
    river_threshold: int = 140
    river_carve: float = 0.28
    smooth_iters: int = 2
    smooth_strength: float = 0.55


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
        self.base_y: float = -80.0
        self.river_edges: list[tuple[int, int, int]] = []

        # Cached mesh data for rendering.
        self.vertices_flat: list[tuple[float, float, float]] = []
        self.h01_flat: list[float] = []
        self.tri_indices: list[tuple[int, int, int]] = []
        self.tri_normal: list[tuple[float, float, float]] = []
        self.tri_base_color: list[tuple[int, int, int]] = []
        self.tri_minmax_y: list[tuple[float, float]] = []
        self.tri_avg_y: list[float] = []
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
            sea_level01=self.params.sea_level01,
            octaves=self.params.octaves,
            lacunarity=self.params.lacunarity,
            persistence=self.params.persistence,
            warp=self.params.warp,
            river_threshold=self.params.river_threshold,
            river_carve=self.params.river_carve,
            smooth_iters=self.params.smooth_iters,
            smooth_strength=self.params.smooth_strength,
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

        rng = Mulberry32(p.seed ^ 0x9E3779B9)
        small_islands: list[tuple[float, float, float, float]] = []
        # island: (cx, cz, radius, strength) in normalized [-1, 1] coords.
        for _ in range(max(0, int(p.small_island_count))):
            for _attempt in range(20):
                cx = rng.random() * 2.0 - 1.0
                cz = rng.random() * 2.0 - 1.0
                d = sqrt(cx * cx + cz * cz)
                # Keep small islands away from the main island core.
                if d < p.main_island_radius * 0.82:
                    continue
                if d > 0.98:
                    continue
                rad = (
                    p.small_island_radius_min
                    + (p.small_island_radius_max - p.small_island_radius_min)
                    * rng.random()
                )
                strength = 0.55 + 0.65 * rng.random()
                small_islands.append((cx, cz, rad, strength))
                break

        for r in range(g):
            row01: list[float] = []
            z = self._z[r]
            for c in range(g):
                x = self._x[c]
                ux = x / half
                uz = z / half
                nx = (x / p.size) * p.frequency
                nz = (z / p.size) * p.frequency

                # Domain warp: avoids the “perfect grid noise” look.
                wx = self._perlin.fbm(nx * 0.65 + 13.1, nz * 0.65 - 9.7, opts) * p.warp
                wz = self._perlin.fbm(nx * 0.65 - 7.4, nz * 0.65 + 11.9, opts) * p.warp
                nx2 = nx + wx
                nz2 = nz + wz

                # Coastline wobble for less-perfect circles.
                coast_n = self._perlin.fbm(nx2 * 1.7 + 220.0, nz2 * 1.7 - 220.0, opts)
                d0 = sqrt(ux * ux + uz * uz) + coast_n * p.coast_wobble
                main = clamp01(1.0 - (d0 / max(1e-6, p.main_island_radius)))
                main = smoothstep(0.0, 1.0, main) ** p.main_island_sharpness

                small = 0.0
                for cx, cz, rad, strength in small_islands:
                    dx = ux - cx
                    dz = uz - cz
                    di = sqrt(dx * dx + dz * dz)
                    s = clamp01(1.0 - (di / max(1e-6, rad)))
                    s = smoothstep(0.0, 1.0, s) ** 1.35
                    s *= strength
                    if s > small:
                        small = s

                land = max(main, small)
                land = clamp01(land)
                land_mask = smoothstep(0.05, 0.85, land)

                hills = self._perlin.noise01(nx2 * 1.05 + 200.0, nz2 * 1.05 - 200.0)
                hills = hills**1.35
                detail = self._perlin.noise01(nx2 * 2.3 - 40.0, nz2 * 2.3 + 40.0)

                range_mask = self._perlin.noise01(
                    nx2 * 0.55 + 700.0, nz2 * 0.55 - 700.0
                )
                range_mask = smoothstep(0.56, 0.78, range_mask)
                mountains = self._perlin.ridged_fbm(
                    nx2 * 1.55 + 520.0, nz2 * 1.55 - 520.0, opts
                )
                mountains = (mountains**1.35) * range_mask

                # Ocean bathymetry (seabed variation) + land elevation.
                ocean = 1.0 - land_mask
                seabed = self._perlin.noise01(nx2 * 0.45 - 1100.0, nz2 * 0.45 + 1100.0)
                ocean_depth = (ocean**1.55) * (0.30 + 0.22 * seabed)
                ocean_h01 = p.sea_level01 - ocean_depth

                land_elev = land_mask * (
                    0.06 + 0.45 * hills + 0.62 * mountains + 0.06 * detail
                )
                h01 = ocean_h01 + land_elev
                h01 = clamp01(h01)
                row01.append(h01)
            heights01.append(row01)

        self._heights01 = heights01

        if p.smooth_iters > 0 and p.smooth_strength > 0.0:
            self._heights01 = self._smooth_heights01(
                self._heights01, iters=p.smooth_iters, strength=p.smooth_strength
            )

        heights: list[list[float]] = []
        min_y = 1e9
        max_y = -1e9
        for r in range(g):
            row: list[float] = []
            for c in range(g):
                y = (self._heights01[r][c] - p.sea_level01) * p.amplitude
                row.append(y)
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
            heights.append(row)

        self._heights = heights
        self.height_min = min_y
        self.height_max = max_y
        self.base_y = min(-p.amplitude * 1.35, self.height_min - p.amplitude * 0.20)

        self._compute_rivers()
        self._build_render_cache()

    def _smooth_heights01(
        self, src: list[list[float]], iters: int, strength: float
    ) -> list[list[float]]:
        # Weighted 3x3 blur kernel (interpolation-like smoothing).
        # Blend with original so mountains keep some sharpness.
        p = self.params
        g = p.grid

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

        out = [row[:] for row in src]
        k = clamp01(strength)
        for _ in range(max(0, int(iters))):
            tmp = [row[:] for row in out]
            for r in range(1, g - 1):
                for c in range(1, g - 1):
                    # Kernel:
                    # 1 2 1
                    # 2 4 2   / 16
                    # 1 2 1
                    s = (
                        tmp[r - 1][c - 1]
                        + 2.0 * tmp[r - 1][c]
                        + tmp[r - 1][c + 1]
                        + 2.0 * tmp[r][c - 1]
                        + 4.0 * tmp[r][c]
                        + 2.0 * tmp[r][c + 1]
                        + tmp[r + 1][c - 1]
                        + 2.0 * tmp[r + 1][c]
                        + tmp[r + 1][c + 1]
                    )
                    blurred = s / 16.0
                    orig = tmp[r][c]

                    # Preserve peaks more than lowlands.
                    preserve = smoothstep(0.62, 0.88, orig)
                    blend = k * (1.0 - 0.80 * preserve)
                    out[r][c] = clamp01(orig * (1.0 - blend) + blurred * blend)
        return out

    def sample_height(self, x: float, z: float) -> float:
        # Bilinear interpolation in the heightfield (smooth queries for agents).
        p = self.params
        g = p.grid
        half = p.size * 0.5

        fx = ((x + half) / p.size) * (g - 1)
        fz = ((z + half) / p.size) * (g - 1)

        if fx < 0.0:
            fx = 0.0
        if fz < 0.0:
            fz = 0.0
        if fx > g - 1.001:
            fx = g - 1.001
        if fz > g - 1.001:
            fz = g - 1.001

        x0 = int(fx)
        z0 = int(fz)
        x1 = min(g - 1, x0 + 1)
        z1 = min(g - 1, z0 + 1)

        tx = fx - x0
        tz = fz - z0

        h00 = self._heights[z0][x0]
        h10 = self._heights[z0][x1]
        h01 = self._heights[z1][x0]
        h11 = self._heights[z1][x1]

        hx0 = h00 + (h10 - h00) * tx
        hx1 = h01 + (h11 - h01) * tx
        return hx0 + (hx1 - hx0) * tz

    def water_depth(self, x: float, z: float) -> float:
        # Depth of water column above terrain (0 if land).
        h = self.sample_height(x, z)
        return max(0.0, self.sea_level_y - h)

    def _build_render_cache(self) -> None:
        p = self.params
        g = p.grid
        n = g * g

        def clamp01(v: float) -> float:
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        def lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t

        def lerp_col(
            a: tuple[int, int, int], b: tuple[int, int, int], t: float
        ) -> tuple[int, int, int]:
            tt = clamp01(t)
            return (
                int(lerp(a[0], b[0], tt)),
                int(lerp(a[1], b[1], tt)),
                int(lerp(a[2], b[2], tt)),
            )

        def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        def sub(
            a: tuple[float, float, float], b: tuple[float, float, float]
        ) -> tuple[float, float, float]:
            return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

        def cross(
            a: tuple[float, float, float], b: tuple[float, float, float]
        ) -> tuple[float, float, float]:
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        def normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
            nn = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            if nn == 0.0:
                return (0.0, 1.0, 0.0)
            return (v[0] / nn, v[1] / nn, v[2] / nn)

        def land_base_color(
            yavg: float, h01: float, nrm_y: float
        ) -> tuple[int, int, int]:
            sea = p.sea_level01
            slope = 1.0 - max(0.0, min(1.0, nrm_y))

            if h01 < sea + 0.018:
                base = (184, 173, 124)  # sand
            elif h01 < sea + 0.16:
                base = (44, 96, 52)  # grass
            elif h01 < 0.72:
                base = (54, 110, 62)  # upland
            elif h01 < 0.86:
                base = (112, 104, 92)  # rock
            else:
                base = (235, 240, 246)  # snow

            if slope > 0.20:
                base = lerp_col(base, (50, 48, 44), (slope - 0.20) * 1.2)

            if yavg < 8.0:
                base = lerp_col(
                    base, (128, 120, 92), clamp01((8.0 - yavg) / 18.0) * 0.25
                )
            return base

        verts: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n
        h01f: list[float] = [0.0] * n
        for r in range(g):
            for c in range(g):
                i = r * g + c
                y = self._heights[r][c]
                verts[i] = (self._x[c], y, self._z[r])
                h01f[i] = self._heights01[r][c]

        tri_idx: list[tuple[int, int, int]] = []
        tri_nrm: list[tuple[float, float, float]] = []
        tri_col: list[tuple[int, int, int]] = []
        tri_mm: list[tuple[float, float]] = []
        tri_avg: list[float] = []

        for r in range(g - 1):
            for c in range(g - 1):
                i00 = r * g + c
                i10 = r * g + (c + 1)
                i01 = (r + 1) * g + c
                i11 = (r + 1) * g + (c + 1)

                # Two triangles per cell. Alternate diagonal to reduce visible grid artifacts.
                if ((r + c) & 1) == 0:
                    tris = ((i00, i11, i10), (i00, i01, i11))
                else:
                    tris = ((i00, i01, i10), (i10, i01, i11))

                for ia, ib, ic in tris:
                    a = verts[ia]
                    b = verts[ib]
                    c0 = verts[ic]
                    u = sub(b, a)
                    v = sub(c0, a)
                    nrm0 = normalize(cross(u, v))

                    # Keep normals pointing upward for stable shading.
                    if nrm0[1] < 0.0:
                        ib, ic = ic, ib
                        b = verts[ib]
                        c0 = verts[ic]
                        u = sub(b, a)
                        v = sub(c0, a)
                        nrm0 = normalize(cross(u, v))

                    h01 = (h01f[ia] + h01f[ib] + h01f[ic]) / 3.0
                    yavg = (a[1] + b[1] + c0[1]) / 3.0
                    base = land_base_color(yavg, h01, nrm0[1])
                    miny = min(a[1], b[1], c0[1])
                    maxy = max(a[1], b[1], c0[1])
                    avgy = (a[1] + b[1] + c0[1]) / 3.0

                    tri_idx.append((ia, ib, ic))
                    tri_nrm.append(nrm0)
                    tri_col.append(base)
                    tri_mm.append((miny, maxy))
                    tri_avg.append(avgy)

        self.vertices_flat = verts
        self.h01_flat = h01f
        self.tri_indices = tri_idx
        self.tri_normal = tri_nrm
        self.tri_base_color = tri_col
        self.tri_minmax_y = tri_mm
        self.tri_avg_y = tri_avg

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
