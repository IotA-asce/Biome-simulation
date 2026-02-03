from __future__ import annotations

import time

from math import sqrt

import pygame

from biome_sim.core.prng import Mulberry32
from biome_sim.render.camera import OrbitCamera
from biome_sim.terrain.terrain import Terrain, TerrainParams


def _random_seed() -> int:
    # pygame-ce includes a fast time module, but time.time_ns is enough.
    t = time.time_ns() & 0xFFFFFFFF
    # Mix in a PRNG step so consecutive calls differ more.
    return Mulberry32(t).random_u32()


def run() -> None:
    pygame.init()
    pygame.display.set_caption("Biome Simulation (Python + pygame)")

    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)

    seed = _random_seed()
    terrain = Terrain(TerrainParams(seed=seed))

    cam = OrbitCamera(
        target=(0.0, 0.0, 0.0), yaw=0.9, pitch=0.55, distance=165.0, fov_deg=62.0
    )

    show_wireframe = False
    dragging = False
    last_mouse = (0, 0)

    def regenerate() -> None:
        nonlocal seed
        seed = _random_seed()
        terrain.regenerate(seed)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    regenerate()
                elif event.key == pygame.K_g:
                    show_wireframe = not show_wireframe
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    cam.distance = max(30.0, cam.distance * 0.92)
                elif event.button == 5:
                    cam.distance = min(520.0, cam.distance * 1.08)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                lx, ly = last_mouse
                dx = mx - lx
                dy = my - ly
                last_mouse = event.pos

                cam.yaw += dx * 0.007
                cam.pitch -= dy * 0.007
                cam.pitch = max(-1.35, min(1.35, cam.pitch))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            cam.yaw -= 0.02
        if keys[pygame.K_RIGHT]:
            cam.yaw += 0.02
        if keys[pygame.K_UP]:
            cam.pitch = max(-1.35, cam.pitch - 0.02)
        if keys[pygame.K_DOWN]:
            cam.pitch = min(1.35, cam.pitch + 0.02)

        bg = (12, 18, 16)
        fog_col = (12, 18, 16)
        screen.fill(bg)
        w, h = screen.get_size()

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

        def col_mul(a: tuple[int, int, int], k: float) -> tuple[int, int, int]:
            kk = max(0.0, min(2.0, k))
            return (
                int(max(0, min(255, a[0] * kk))),
                int(max(0, min(255, a[1] * kk))),
                int(max(0, min(255, a[2] * kk))),
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
            n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            if n == 0.0:
                return (0.0, 1.0, 0.0)
            return (v[0] / n, v[1] / n, v[2] / n)

        def clip_poly_y(
            poly: list[tuple[float, float, float]], keep_above: bool
        ) -> list[tuple[float, float, float]]:
            # Clip polygon against plane y=0.
            if not poly:
                return []

            def inside(v: tuple[float, float, float]) -> bool:
                return v[1] >= 0.0 if keep_above else v[1] <= 0.0

            out: list[tuple[float, float, float]] = []
            prev = poly[-1]
            prev_in = inside(prev)

            for cur in poly:
                cur_in = inside(cur)
                if cur_in != prev_in:
                    y1 = prev[1]
                    y2 = cur[1]
                    if y2 != y1:
                        t = (0.0 - y1) / (y2 - y1)
                        x = prev[0] + t * (cur[0] - prev[0])
                        z = prev[2] + t * (cur[2] - prev[2])
                        out.append((x, 0.0, z))
                if cur_in:
                    out.append(cur)
                prev = cur
                prev_in = cur_in
            return out

        def land_color(
            y: float, h01: float, nrm: tuple[float, float, float]
        ) -> tuple[int, int, int]:
            sea = terrain.params.sea_level01
            slope = 1.0 - max(0.0, min(1.0, nrm[1]))

            # Bands in height01 space.
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

            # Darken steep slopes a bit.
            if slope > 0.20:
                base = lerp_col(base, (50, 48, 44), (slope - 0.20) * 1.2)

            # Slight warming at lower elevations.
            if y < 8.0:
                base = lerp_col(base, (128, 120, 92), clamp01((8.0 - y) / 18.0) * 0.25)
            return base

        def water_color(depth_hint: float) -> tuple[int, int, int]:
            shallow = (36, 98, 120)
            deep = (11, 38, 58)
            return lerp_col(shallow, deep, clamp01(depth_hint))

        light_dir = normalize((-0.35, 1.0, -0.25))

        g = terrain.params.grid
        # Project all vertices once.
        proj_ok = [False] * (g * g)
        proj = [(0.0, 0.0, 0.0)] * (g * g)
        world = [(0.0, 0.0, 0.0)] * (g * g)
        h01_flat = [0.0] * (g * g)

        for r in range(g):
            for c in range(g):
                i = r * g + c
                p0 = terrain.vertex(r, c)
                world[i] = p0
                h01_flat[i] = terrain.height01(r, c)
                s0 = cam.project(p0, (w, h))
                if s0 is None:
                    continue
                proj_ok[i] = True
                proj[i] = s0

        # Build draw list using a painter's algorithm (far -> near).
        draw_polys: list[
            tuple[float, list[tuple[float, float]], tuple[int, int, int]]
        ] = []

        fog_start = 110.0
        fog_end = 380.0

        def apply_fog(col: tuple[int, int, int], z: float) -> tuple[int, int, int]:
            t = clamp01((z - fog_start) / (fog_end - fog_start))
            return lerp_col(col, fog_col, t)

        for r in range(g - 1):
            for c in range(g - 1):
                i00 = r * g + c
                i10 = r * g + (c + 1)
                i01 = (r + 1) * g + c
                i11 = (r + 1) * g + (c + 1)

                # Two triangles per grid cell.
                tris = ((i00, i11, i10), (i00, i01, i11))
                for ia, ib, ic in tris:
                    pa, pb, pc = world[ia], world[ib], world[ic]
                    # Clip into land (y>=0) and water (y<=0), then draw both.
                    tri = [pa, pb, pc]
                    land_poly = clip_poly_y(tri, keep_above=True)
                    water_poly = clip_poly_y(tri, keep_above=False)

                    if len(land_poly) >= 3:
                        # Shade using triangle normal from original triangle.
                        u = sub(pb, pa)
                        v = sub(pc, pa)
                        nrm = normalize(cross(u, v))
                        h01 = (h01_flat[ia] + h01_flat[ib] + h01_flat[ic]) / 3.0
                        yavg = (pa[1] + pb[1] + pc[1]) / 3.0
                        base = land_color(yavg, h01, nrm)
                        diff = max(0.0, dot(nrm, light_dir))
                        intensity = 0.42 + 0.72 * diff
                        col = col_mul(base, intensity)

                        pts: list[tuple[float, float]] = []
                        zsum = 0.0
                        ok = True
                        for vv in land_poly:
                            s = cam.project(vv, (w, h))
                            if s is None:
                                ok = False
                                break
                            pts.append((s[0], s[1]))
                            zsum += s[2]
                        if ok:
                            zavg = zsum / len(pts)
                            draw_polys.append((zavg, pts, apply_fog(col, zavg)))

                    if len(water_poly) >= 3:
                        # Render water as a flat surface at y=0.
                        pts2: list[tuple[float, float]] = []
                        zsum2 = 0.0
                        ok2 = True
                        min_y = min(pa[1], pb[1], pc[1])
                        depth_hint = clamp01(
                            (-min_y) / max(1e-6, terrain.params.amplitude * 0.75)
                        )
                        colw = water_color(depth_hint)
                        colw = col_mul(colw, 0.95)
                        for vv in water_poly:
                            vflat = (vv[0], 0.0, vv[2])
                            s = cam.project(vflat, (w, h))
                            if s is None:
                                ok2 = False
                                break
                            pts2.append((s[0], s[1]))
                            zsum2 += s[2]
                        if ok2:
                            zavg2 = zsum2 / len(pts2)
                            draw_polys.append((zavg2, pts2, apply_fog(colw, zavg2)))

        draw_polys.sort(key=lambda it: it[0], reverse=True)
        for _, pts, col in draw_polys:
            pygame.draw.polygon(screen, col, pts)

        # Rivers: draw on top of terrain.
        river_col = (40, 140, 170)
        for i0, i1, flow in terrain.river_edges:
            p0 = terrain.vertex_i(i0)
            p1 = terrain.vertex_i(i1)
            if p0[1] <= 0.05:
                continue
            # Lift slightly so it doesn't Z-fight with the land.
            p0 = (p0[0], p0[1] + 0.18, p0[2])
            p1 = (p1[0], p1[1] + 0.18, p1[2])
            s0 = cam.project(p0, (w, h))
            s1 = cam.project(p1, (w, h))
            if s0 is None or s1 is None:
                continue
            zavg = (s0[2] + s1[2]) * 0.5
            width = 1
            if flow > terrain.params.river_threshold * 2:
                width = 2
            if flow > terrain.params.river_threshold * 4:
                width = 3
            colr = apply_fog(river_col, zavg)
            pygame.draw.line(screen, colr, (s0[0], s0[1]), (s1[0], s1[1]), width)

        if show_wireframe:
            line_color = (190, 235, 220)
            stride = 2
            for r in range(0, g, stride):
                for c in range(0, g, stride):
                    i0 = r * g + c
                    if not proj_ok[i0]:
                        continue
                    x0, y0, z0 = proj[i0]
                    fogt = clamp01((z0 - fog_start) / (fog_end - fog_start))
                    col = lerp_col(line_color, fog_col, fogt)
                    if c + stride < g:
                        i1 = r * g + (c + stride)
                        if proj_ok[i1]:
                            s1 = proj[i1]
                            pygame.draw.aaline(screen, col, (x0, y0), (s1[0], s1[1]))
                    if r + stride < g:
                        i2 = (r + stride) * g + c
                        if proj_ok[i2]:
                            s2 = proj[i2]
                            pygame.draw.aaline(screen, col, (x0, y0), (s2[0], s2[1]))

        # HUD
        fps = clock.get_fps()
        hud_lines = [
            f"seed: {seed}",
            f"grid: {terrain.params.grid} x {terrain.params.grid}",
            f"render: {'wireframe overlay' if show_wireframe else 'solid'}",
            "controls: drag LMB, wheel zoom, R regenerate, G wireframe, arrows orbit",
            f"fps: {fps:0.1f}",
        ]
        y = 10
        for line in hud_lines:
            surf = font.render(line, True, (230, 245, 238))
            screen.blit(surf, (10, y))
            y += 18

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
