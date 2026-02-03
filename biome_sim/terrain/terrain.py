from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

from biome_sim.core.prng import Mulberry32

from biome_sim.noise.perlin2d import FbmOptions, Perlin2D


@dataclass(frozen=True)
class TerrainParams:
    seed: int
    size: float = 220.0
    grid: int = 201
    amplitude: float = 58.0
    frequency: float = 2.45
    sea_level01: float = 0.5
    octaves: int = 6
    lacunarity: float = 2.0
    persistence: float = 0.5
    warp: float = 0.90

    # Archipelago clustering controls.
    archipelago_satellite_clusters: int = 3
    archipelago_warp: float = 0.14
    archipelago_edge_fade: float = 0.90
    archipelago_uplift: float = 0.0
    ocean_depth: float = 0.20

    river_threshold: int = 520
    river_carve: float = 0.34
    smooth_iters: int = 0
    smooth_strength: float = 0.0


class Terrain:
    def __init__(self, params: TerrainParams):
        self.params = params
        self._perlin = Perlin2D(params.seed)
        self._heights: list[list[float]] = []
        self._heights01: list[list[float]] = []
        self._slope01: list[list[float]] = []
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
    def slope01(self) -> list[list[float]]:
        return self._slope01

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
            archipelago_satellite_clusters=self.params.archipelago_satellite_clusters,
            archipelago_warp=self.params.archipelago_warp,
            archipelago_edge_fade=self.params.archipelago_edge_fade,
            archipelago_uplift=self.params.archipelago_uplift,
            ocean_depth=self.params.ocean_depth,
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

        opts_hi = opts
        opts_warp = FbmOptions(
            octaves=max(2, int(p.octaves) - 4),
            lacunarity=2.2,
            persistence=0.55,
        )

        rng = Mulberry32(p.seed ^ 0x9E3779B9)

        # Blob clusters (in normalized [-1, 1] coords) to encourage one large landmass
        # with nearby island clusters, while still letting Perlin shape the terrain.
        blobs: list[tuple[float, float, float, float]] = []
        # (cx, cz, radius, weight)
        main_blob_count = 9
        for _ in range(main_blob_count):
            ang = rng.random() * 2.0 * pi
            dist = (rng.random() ** 0.65) * 0.22
            cx = cos(ang) * dist
            cz = sin(ang) * dist
            rad = 0.48 + 0.30 * rng.random()
            wgt = 0.75 + 0.55 * rng.random()
            blobs.append((cx, cz, rad, wgt))

        for _ in range(max(0, int(p.archipelago_satellite_clusters))):
            ang = rng.random() * 2.0 * pi
            dist = 0.58 + 0.26 * rng.random()
            scx = cos(ang) * dist
            scz = sin(ang) * dist
            for _j in range(4):
                ang2 = rng.random() * 2.0 * pi
                dist2 = rng.random() * 0.18
                cx = scx + cos(ang2) * dist2
                cz = scz + sin(ang2) * dist2
                rad = 0.16 + 0.18 * rng.random()
                wgt = 0.55 + 0.70 * rng.random()
                blobs.append((cx, cz, rad, wgt))

        def blob_value(u: float, v: float, cx: float, cz: float, rad: float) -> float:
            d = sqrt((u - cx) * (u - cx) + (v - cz) * (v - cz))
            t = clamp01(1.0 - (d / max(1e-6, rad)))
            t = smoothstep(0.0, 1.0, t)
            return t

        for r in range(g):
            row01: list[float] = []
            z = self._z[r]
            uz = z / half
            nz = (z / p.size) * p.frequency

            for c in range(g):
                x = self._x[c]
                ux = x / half
                nx = (x / p.size) * p.frequency

                # Domain warp for terrain.
                wx = (
                    self._perlin.fbm(nx * 0.65 + 13.1, nz * 0.65 - 9.7, opts_warp)
                    * p.warp
                )
                wz = (
                    self._perlin.fbm(nx * 0.65 - 7.4, nz * 0.65 + 11.9, opts_warp)
                    * p.warp
                )
                nx2 = nx + wx
                nz2 = nz + wz

                # Cluster mask for "one big island + nearby clusters".
                cu = ux + (
                    self._perlin.fbm(ux * 2.1 + 91.0, uz * 2.1 - 73.0, opts_warp)
                    * p.archipelago_warp
                )
                cv = uz + (
                    self._perlin.fbm(ux * 2.1 - 31.0, uz * 2.1 + 51.0, opts_warp)
                    * p.archipelago_warp
                )
                cluster = 0.0
                for cx, cz, rad, wgt in blobs:
                    b = blob_value(cu, cv, cx, cz, rad) ** 1.35
                    b = clamp01(b * wgt)
                    cluster = 1.0 - (1.0 - cluster) * (1.0 - b)

                # Fade clusters near the map edge so the archipelago stays grouped.
                if p.archipelago_edge_fade < 1.0:
                    d_edge = sqrt(ux * ux + uz * uz)
                    t_edge = clamp01(
                        (d_edge - p.archipelago_edge_fade)
                        / (1.0 - p.archipelago_edge_fade)
                    )
                    edge = 1.0 - (t_edge * t_edge * (3.0 - 2.0 * t_edge))
                    cluster *= edge
                cluster = clamp01(cluster)
                # Treat this primarily as a land-distribution field (helps avoid dome-like islands).
                cluster = smoothstep(0.20, 0.60, cluster)

                # Terrain noise (generate height first; sea level applied afterwards).
                macro = self._perlin.fbm(
                    nx2 * 0.22 + 1000.0, nz2 * 0.22 - 1000.0, opts_warp
                )
                hills = self._perlin.fbm(
                    nx2 * 0.88 + 200.0, nz2 * 0.88 - 200.0, opts_hi
                )
                detail = self._perlin.fbm(nx2 * 3.25 - 40.0, nz2 * 3.25 + 40.0, opts_hi)

                range_n = self._perlin.noise01(nx2 * 0.42 + 700.0, nz2 * 0.42 - 700.0)
                range_mask = smoothstep(0.55, 0.80, range_n)
                ridge = self._perlin.ridged_fbm(
                    nx2 * 1.55 + 520.0, nz2 * 1.55 - 520.0, opts_hi
                )
                mountains = max(0.0, ridge - 0.35) ** 1.35
                mountains *= range_mask

                seabed = self._perlin.fbm(
                    nx2 * 0.35 - 1100.0, nz2 * 0.35 + 1100.0, opts_warp
                )

                # Turn the cluster field into an organic shoreline mask.
                coast_n = self._perlin.fbm(
                    nx2 * 0.85 + 333.0, nz2 * 0.85 - 333.0, opts_warp
                )
                cluster_w = clamp01(cluster + coast_n * 0.18)
                land_mask = smoothstep(0.45, 0.55, cluster_w)
                coast_mask = land_mask**1.45

                # Height-first: build a detailed heightfield, then sea-level decides water/land.
                base = (macro * 0.06) + (hills * 0.16) + (detail * 0.07)
                land_dh = coast_mask * (0.05 + base) + mountains * (land_mask * 0.20)

                ocean = 1.0 - land_mask
                ocean_dh = -(ocean**1.35) * p.ocean_depth + seabed * (0.05 * ocean)

                dh = land_dh + ocean_dh
                h01 = clamp01(p.sea_level01 + dh)
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

        self._compute_slope01()

        self._compute_rivers()
        self._build_render_cache()

    def _compute_slope01(self) -> None:
        p = self.params
        g = p.grid
        step = p.size / (g - 1)
        h = self._heights

        slope: list[list[float]] = []
        for r in range(g):
            row: list[float] = []
            for c in range(g):
                if c == 0:
                    dx = (h[r][c + 1] - h[r][c]) / step
                elif c == g - 1:
                    dx = (h[r][c] - h[r][c - 1]) / step
                else:
                    dx = (h[r][c + 1] - h[r][c - 1]) / (2.0 * step)

                if r == 0:
                    dz = (h[r + 1][c] - h[r][c]) / step
                elif r == g - 1:
                    dz = (h[r][c] - h[r - 1][c]) / step
                else:
                    dz = (h[r + 1][c] - h[r - 1][c]) / (2.0 * step)

                # Normal of heightfield ~ (-dx, 1, -dz); y component after normalize:
                # ny = 1 / sqrt(1 + dx^2 + dz^2)
                ny = 1.0 / sqrt(1.0 + dx * dx + dz * dz)
                s01 = 1.0 - ny
                if s01 < 0.0:
                    s01 = 0.0
                if s01 > 1.0:
                    s01 = 1.0
                row.append(s01)
            slope.append(row)
        self._slope01 = slope

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

    def sample_slope(self, x: float, z: float) -> float:
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

        s00 = self._slope01[z0][x0]
        s10 = self._slope01[z0][x1]
        s01 = self._slope01[z1][x0]
        s11 = self._slope01[z1][x1]

        sx0 = s00 + (s10 - s00) * tx
        sx1 = s01 + (s11 - s01) * tx
        return sx0 + (sx1 - sx0) * tz

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

            # Normalize elevation above sea into [0, 1] so bands stay stable
            # even if sea_level01 changes.
            t = 0.0
            if h01 > sea:
                t = (h01 - sea) / max(1e-6, 1.0 - sea)
                t = clamp01(t)

            if t < 0.04:
                base = (188, 176, 120)  # beach sand
            elif t < 0.28:
                base = (46, 104, 58)  # plains
            elif t < 0.52:
                base = (52, 114, 64)  # upland
            elif t < 0.78:
                base = (116, 108, 98)  # rock
            else:
                base = (236, 241, 247)  # snow

            # Expose rock on steep slopes.
            if slope > 0.25 and t > 0.10:
                base = lerp_col(base, (74, 70, 64), clamp01((slope - 0.25) * 1.6))

            # Slight warming near sea level.
            if yavg < 10.0:
                base = lerp_col(
                    base, (132, 122, 92), clamp01((10.0 - yavg) / 22.0) * 0.22
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
