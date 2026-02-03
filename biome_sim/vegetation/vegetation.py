from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from biome_sim.core.prng import Mulberry32
from biome_sim.noise.perlin2d import FbmOptions, Perlin2D
from biome_sim.terrain.terrain import TerrainParams


class TerrainLike(Protocol):
    params: TerrainParams

    def sample_height(self, x: float, z: float) -> float: ...

    def sample_slope(self, x: float, z: float) -> float: ...

    def water_depth(self, x: float, z: float) -> float: ...


class VegKind:
    TREE = 1
    BUSH = 2
    KELP = 3


@dataclass(frozen=True)
class VegetationParams:
    # Upper bounds; actual counts depend on density fields.
    trees_max: int = 5500
    bushes_max: int = 7000
    kelp_max: int = 4500

    # Placement thresholds.
    tree_slope_max: float = 0.55
    bush_slope_max: float = 0.70
    kelp_slope_max: float = 0.65

    # Height thresholds (in world y units).
    tree_min_y: float = 1.3
    bush_min_y: float = 0.35

    # Underwater depth range for kelp.
    kelp_min_depth: float = 1.4
    kelp_max_depth: float = 18.0


@dataclass(frozen=True)
class VegetationInstance:
    kind: int
    pos: tuple[float, float, float]
    scale: float
    tint: int


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _smoothstep(a: float, b: float, x: float) -> float:
    if a == b:
        return 0.0
    t = _clamp01((x - a) / (b - a))
    return t * t * (3.0 - 2.0 * t)


class VegetationField:
    def __init__(self, terrain: TerrainLike, params: VegetationParams | None = None):
        self.params = params or VegetationParams()
        self.instances: list[VegetationInstance] = []
        # cell_index -> list of instance indices (None for empty cells)
        self.cell_to_indices: list[list[int] | None] = []
        self.regenerate(terrain)

    def regenerate(self, terrain: TerrainLike) -> None:
        p = self.params
        tp = terrain.params

        self.instances = []
        self.cell_to_indices = []

        g = tp.grid
        cells = g - 1
        cell_count = cells * cells
        size = tp.size
        half = size * 0.5
        step = size / (g - 1)

        self.cell_to_indices = [None] * cell_count

        # Dedicated noise fields for vegetation, but deterministic given terrain seed.
        moist_noise = Perlin2D(tp.seed ^ 0xA17F31D9)
        dens_noise = Perlin2D(tp.seed ^ 0xC0FFEE77)
        opts = FbmOptions(octaves=4, lacunarity=2.1, persistence=0.52)

        def cell_index_for(x: float, z: float) -> int:
            cc = int(((x + half) / size) * cells)
            rr = int(((z + half) / size) * cells)
            if cc < 0:
                cc = 0
            elif cc >= cells:
                cc = cells - 1
            if rr < 0:
                rr = 0
            elif rr >= cells:
                rr = cells - 1
            return rr * cells + cc

        trees = 0
        bushes = 0
        kelp = 0

        # Iterate cells; decide vegetation based on height/slope/moisture.
        for r in range(cells):
            zc = -half + (r + 0.5) * step
            nz = (zc / size) * tp.frequency
            for c in range(cells):
                xc = -half + (c + 0.5) * step
                nx = (xc / size) * tp.frequency

                # Deterministic RNG per cell.
                cell_seed = (
                    tp.seed ^ (r * 73856093) ^ (c * 19349663) ^ 0xDEADBEEF
                ) & 0xFFFFFFFF
                rng = Mulberry32(cell_seed)

                # Jitter within the cell.
                jx = (rng.random() - 0.5) * step * 0.92
                jz = (rng.random() - 0.5) * step * 0.92
                x = xc + jx
                z = zc + jz

                y = terrain.sample_height(x, z)
                slope = terrain.sample_slope(x, z)

                # Normalized height above sea.
                h01 = (y / max(1e-6, tp.amplitude)) + tp.sea_level01
                above = 0.0
                if h01 > tp.sea_level01:
                    above = (h01 - tp.sea_level01) / max(1e-6, 1.0 - tp.sea_level01)
                    above = _clamp01(above)

                moist = moist_noise.noise01(nx * 0.60 + 88.0, nz * 0.60 - 88.0)
                dens = (
                    dens_noise.fbm(nx * 1.15 - 240.0, nz * 1.15 + 240.0, opts) + 1.0
                ) * 0.5
                dens = _clamp01(dens)

                # Prefer mid elevations; fade out on beaches and snow.
                mid = _smoothstep(0.05, 0.38, above) * (
                    1.0 - _smoothstep(0.62, 0.90, above)
                )
                fert = _clamp01(
                    (0.68 * moist + 0.32 * dens) * (0.35 + 0.65 * mid) * (1.0 - slope)
                )

                if y > 0.0:
                    # LAND
                    if (
                        trees < p.trees_max
                        and y >= p.tree_min_y
                        and slope <= p.tree_slope_max
                        and above >= 0.06
                        and above <= 0.78
                    ):
                        # Sparse on beaches, denser in fertile areas.
                        prob = fert * 0.18
                        if rng.random() < prob:
                            scale = 0.85 + rng.random() * 0.85
                            tint = int(rng.random() * 7)  # small variation bucket
                            idx = len(self.instances)
                            self.instances.append(
                                VegetationInstance(VegKind.TREE, (x, y, z), scale, tint)
                            )
                            cell = cell_index_for(x, z)
                            bucket = self.cell_to_indices[cell]
                            if bucket is None:
                                bucket = []
                                self.cell_to_indices[cell] = bucket
                            bucket.append(idx)
                            trees += 1

                    if (
                        bushes < p.bushes_max
                        and y >= p.bush_min_y
                        and slope <= p.bush_slope_max
                        and above <= 0.70
                    ):
                        prob = fert * 0.34
                        if rng.random() < prob:
                            # Small extra jitter so bushes don't line up with trees.
                            bx = x + (rng.random() - 0.5) * step * 0.30
                            bz = z + (rng.random() - 0.5) * step * 0.30
                            by = terrain.sample_height(bx, bz)
                            if by > 0.0:
                                bscale = 0.75 + rng.random() * 0.55
                                btint = int(rng.random() * 9)
                                idx = len(self.instances)
                                self.instances.append(
                                    VegetationInstance(
                                        VegKind.BUSH, (bx, by, bz), bscale, btint
                                    )
                                )
                                cell = cell_index_for(bx, bz)
                                bucket = self.cell_to_indices[cell]
                                if bucket is None:
                                    bucket = []
                                    self.cell_to_indices[cell] = bucket
                                bucket.append(idx)
                                bushes += 1
                else:
                    # OCEAN: kelp near shores (shallow-ish water).
                    if kelp >= p.kelp_max:
                        continue
                    depth = terrain.water_depth(x, z)
                    if (
                        depth >= p.kelp_min_depth
                        and depth <= p.kelp_max_depth
                        and slope <= p.kelp_slope_max
                    ):
                        # Kelp likes moisture (use dens noise as proxy).
                        prob = _clamp01((0.55 * dens + 0.45 * moist) * 0.45)
                        if rng.random() < prob:
                            scale = 0.8 + rng.random() * 0.9
                            tint = int(rng.random() * 8)
                            idx = len(self.instances)
                            self.instances.append(
                                VegetationInstance(VegKind.KELP, (x, y, z), scale, tint)
                            )
                            cell = cell_index_for(x, z)
                            bucket = self.cell_to_indices[cell]
                            if bucket is None:
                                bucket = []
                                self.cell_to_indices[cell] = bucket
                            bucket.append(idx)
                            kelp += 1
