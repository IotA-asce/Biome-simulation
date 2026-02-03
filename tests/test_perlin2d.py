from biome_sim.noise.perlin2d import FbmOptions, Perlin2D


def test_perlin_is_deterministic_for_seed() -> None:
    a = Perlin2D(12345)
    b = Perlin2D(12345)
    pts = [(0.0, 0.0), (0.1, 0.2), (2.345, -1.75), (10.01, 10.02)]
    opts = FbmOptions(octaves=5, lacunarity=2.0, persistence=0.5)
    for x, y in pts:
        assert a.noise(x, y) == b.noise(x, y)
        assert a.fbm(x, y, opts) == b.fbm(x, y, opts)


def test_perlin_stays_in_sane_range() -> None:
    p = Perlin2D(7)
    opts = FbmOptions(octaves=6, lacunarity=2.1, persistence=0.52)
    for i in range(200):
        x = i * 0.173
        y = i * 0.091
        n = p.noise(x, y)
        assert -1.01 <= n <= 1.01
        f = p.fbm(x, y, opts)
        assert -1.01 <= f <= 1.01
