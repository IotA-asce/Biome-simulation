from __future__ import annotations


class Mulberry32:
    def __init__(self, seed: int):
        self._a = seed & 0xFFFFFFFF

    def random_u32(self) -> int:
        self._a = (self._a + 0x6D2B79F5) & 0xFFFFFFFF
        t = (self._a ^ (self._a >> 15)) & 0xFFFFFFFF
        t = (t * ((1 | self._a) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t2 = (t ^ (t >> 7)) & 0xFFFFFFFF
        t2 = (t2 * ((61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF
        out = (t ^ t2) & 0xFFFFFFFF
        out = (out ^ (out >> 14)) & 0xFFFFFFFF
        return out

    def random(self) -> float:
        return self.random_u32() / 4294967296.0


def seed_from_string(s: str) -> int:
    # FNV-1a 32-bit
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def clamp_u32(x: int) -> int:
    return x & 0xFFFFFFFF
