from __future__ import annotations

from dataclasses import dataclass

from biome_sim.core.prng import Mulberry32, clamp_u32


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _grad(h: int, x: float, y: float) -> float:
    # 8 gradient directions based on lower bits.
    hh = h & 7
    u = x if hh < 4 else y
    v = y if hh < 4 else x
    return (u if (hh & 1) == 0 else -u) + (v if (hh & 2) == 0 else -v)


@dataclass(frozen=True)
class FbmOptions:
    octaves: int = 5
    lacunarity: float = 2.0
    persistence: float = 0.5


class Perlin2D:
    def __init__(self, seed: int):
        rng = Mulberry32(clamp_u32(seed))
        p = list(range(256))
        # Fisher-Yates shuffle
        for i in range(255, 0, -1):
            j = int(rng.random() * (i + 1))
            p[i], p[j] = p[j], p[i]
        self._perm = [p[i & 255] for i in range(512)]

    def noise(self, x: float, y: float) -> float:
        xi = int(x) & 255
        yi = int(y) & 255
        xf = x - int(x)
        yf = y - int(y)

        u = _fade(xf)
        v = _fade(yf)

        aa = self._perm[xi + self._perm[yi]]
        ab = self._perm[xi + self._perm[yi + 1]]
        ba = self._perm[xi + 1 + self._perm[yi]]
        bb = self._perm[xi + 1 + self._perm[yi + 1]]

        x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1.0, yf), u)
        x2 = _lerp(_grad(ab, xf, yf - 1.0), _grad(bb, xf - 1.0, yf - 1.0), u)

        # Roughly in [-1, 1]; normalize by 1/sqrt(2) to keep gradients sane.
        return _lerp(x1, x2, v) * 0.7071067811865475

    def fbm(self, x: float, y: float, opts: FbmOptions) -> float:
        octaves = max(1, int(opts.octaves))
        freq = 1.0
        amp = 1.0
        s = 0.0
        amp_sum = 0.0

        for _ in range(octaves):
            s += self.noise(x * freq, y * freq) * amp
            amp_sum += amp
            freq *= float(opts.lacunarity)
            amp *= float(opts.persistence)

        if amp_sum == 0.0:
            return 0.0
        return s / amp_sum
