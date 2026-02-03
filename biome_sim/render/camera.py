from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tan


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: tuple[float, float, float]) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _norm(v)
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


@dataclass
class OrbitCamera:
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.8
    pitch: float = 0.55
    distance: float = 140.0
    fov_deg: float = 60.0
    near: float = 0.5

    def position(self) -> tuple[float, float, float]:
        # Y-up orbit.
        x = self.target[0] + cos(self.yaw) * cos(self.pitch) * self.distance
        y = self.target[1] + sin(self.pitch) * self.distance
        z = self.target[2] + sin(self.yaw) * cos(self.pitch) * self.distance
        return (x, y, z)

    def project(
        self,
        p: tuple[float, float, float],
        viewport: tuple[int, int],
    ) -> tuple[float, float, float] | None:
        w, h = viewport
        cam_pos = self.position()

        forward = _normalize(_sub(self.target, cam_pos))
        world_up = (0.0, 1.0, 0.0)
        right = _normalize(_cross(forward, world_up))
        up = _cross(right, forward)

        d = _sub(p, cam_pos)
        cx = _dot(d, right)
        cy = _dot(d, up)
        cz = _dot(d, forward)

        if cz <= self.near:
            return None

        f = (h * 0.5) / tan((self.fov_deg * 0.5) * 3.141592653589793 / 180.0)
        sx = (w * 0.5) + (cx * f) / cz
        sy = (h * 0.5) - (cy * f) / cz
        return (sx, sy, cz)
