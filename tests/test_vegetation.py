from __future__ import annotations


from biome_sim.terrain.terrain import Terrain, TerrainParams
from biome_sim.vegetation.vegetation import VegKind, VegetationField


def test_vegetation_is_deterministic_for_seed() -> None:
    tp = TerrainParams(seed=12345, grid=61)
    t1 = Terrain(tp)
    v1 = VegetationField(t1)

    t2 = Terrain(tp)
    v2 = VegetationField(t2)

    assert len(v1.instances) == len(v2.instances)

    # Spot-check a few instances with tolerant float compares.
    for i in range(min(50, len(v1.instances))):
        a = v1.instances[i]
        b = v2.instances[i]
        assert a.kind == b.kind
        assert a.scale == b.scale
        assert a.tint == b.tint
        for j in range(3):
            assert abs(a.pos[j] - b.pos[j]) < 1e-9


def test_vegetation_contains_land_and_water_plants() -> None:
    t = Terrain(TerrainParams(seed=7, grid=61))
    v = VegetationField(t)
    kinds = {inst.kind for inst in v.instances}
    assert VegKind.TREE in kinds or VegKind.BUSH in kinds
    assert VegKind.KELP in kinds
