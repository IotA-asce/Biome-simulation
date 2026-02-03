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
    show_underwater = False
    render_scale = 0.75
    render_surface: pygame.Surface | None = None
    water_layer: pygame.Surface | None = None
    dragging = False
    last_mouse = (0, 0)

    # Helpers (defined once; used in the render loop).
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

    # Cached terrain data for rendering (refreshed on regenerate).
    g = terrain.params.grid
    verts = terrain.vertices_flat
    tri_indices = terrain.tri_indices
    tri_normals = terrain.tri_normal
    tri_base = terrain.tri_base_color
    tri_mm = terrain.tri_minmax_y
    tri_avg_y = terrain.tri_avg_y

    proj_ok = [False] * (g * g)
    proj = [(0.0, 0.0, 0.0)] * (g * g)

    light_dir = normalize((-0.35, 1.0, -0.25))

    def regenerate() -> None:
        nonlocal seed
        nonlocal \
            g, \
            verts, \
            tri_indices, \
            tri_normals, \
            tri_base, \
            tri_mm, \
            tri_avg_y, \
            proj_ok, \
            proj
        seed = _random_seed()
        terrain.regenerate(seed)

        g = terrain.params.grid
        verts = terrain.vertices_flat
        tri_indices = terrain.tri_indices
        tri_normals = terrain.tri_normal
        tri_base = terrain.tri_base_color
        tri_mm = terrain.tri_minmax_y
        tri_avg_y = terrain.tri_avg_y

        proj_ok = [False] * (g * g)
        proj = [(0.0, 0.0, 0.0)] * (g * g)

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
                elif event.key == pygame.K_u:
                    show_underwater = not show_underwater
                elif event.key == pygame.K_p:
                    # Cycle render scale for performance.
                    if render_scale > 0.9:
                        render_scale = 0.75
                    elif render_scale > 0.7:
                        render_scale = 0.5
                    else:
                        render_scale = 1.0
                    render_surface = None
                    water_layer = None
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
        w, h = screen.get_size()

        rw = max(240, int(w * render_scale))
        rh = max(180, int(h * render_scale))
        if render_surface is None or render_surface.get_size() != (rw, rh):
            render_surface = pygame.Surface((rw, rh))
            water_layer = None

        draw_surf: pygame.Surface
        if render_scale >= 0.999:
            draw_surf = screen
            draw_surf.fill(bg)
            frame = cam.frame((w, h))
        else:
            draw_surf = render_surface
            draw_surf.fill(bg)
            frame = cam.frame((rw, rh))

        if show_underwater:
            if water_layer is None or water_layer.get_size() != draw_surf.get_size():
                water_layer = pygame.Surface(draw_surf.get_size(), pygame.SRCALPHA)
            water_layer.fill((0, 0, 0, 0))

        fog_start = 110.0
        fog_end = 380.0

        def apply_fog(col: tuple[int, int, int], z: float) -> tuple[int, int, int]:
            t = clamp01((z - fog_start) / (fog_end - fog_start))
            return lerp_col(col, fog_col, t)

        # Project all vertices once.
        for i in range(g * g):
            s0 = frame.project(verts[i])
            if s0 is None:
                proj_ok[i] = False
                continue
            proj_ok[i] = True
            proj[i] = s0

        # Water surface (y=0): tessellated so it renders even when corners are off-screen.
        half = terrain.params.size * 0.5
        water_steps = 18
        wstep = (half * 2.0) / (water_steps - 1)

        # Brighter ocean so it reads clearly in surface view.
        water_base = (24, 118, 190)
        water_alpha = 160

        for wr in range(water_steps - 1):
            z0 = -half + wr * wstep
            z1 = z0 + wstep
            for wc in range(water_steps - 1):
                x0 = -half + wc * wstep
                x1 = x0 + wstep

                p00 = (x0, 0.0, z0)
                p10 = (x1, 0.0, z0)
                p01 = (x0, 0.0, z1)
                p11 = (x1, 0.0, z1)

                tris = ((p00, p11, p10), (p00, p01, p11))
                for a, b, c0 in tris:
                    sa = frame.project(a)
                    sb = frame.project(b)
                    sc = frame.project(c0)
                    if sa is None or sb is None or sc is None:
                        continue
                    zavgw = (sa[2] + sb[2] + sc[2]) / 3.0
                    col = apply_fog(water_base, zavgw)
                    pts = [(sa[0], sa[1]), (sb[0], sb[1]), (sc[0], sc[1])]
                    if show_underwater and water_layer is not None:
                        pygame.draw.polygon(water_layer, (*col, water_alpha), pts)
                    else:
                        pygame.draw.polygon(draw_surf, col, pts)

        # Draw terrain in a stable back-to-front order (no global sort).
        cam_pos = frame.cam_pos
        row_range = range(0, g - 1) if cam_pos[2] >= 0.0 else range(g - 2, -1, -1)
        col_range = range(0, g - 1) if cam_pos[0] >= 0.0 else range(g - 2, -1, -1)

        def seabed_color(avg_y: float) -> tuple[int, int, int]:
            depth = clamp01((-avg_y) / max(1e-6, terrain.params.amplitude * 1.1))
            shallow = (26, 72, 76)
            deep = (8, 22, 28)
            return lerp_col(shallow, deep, depth)

        def push_tri(
            tri3: tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            base_col: tuple[int, int, int],
            out: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]],
        ) -> None:
            pts: list[tuple[float, float]] = []
            zsum = 0.0
            for vv in tri3:
                s = frame.project(vv)
                if s is None:
                    return
                pts.append((s[0], s[1]))
                zsum += s[2]
            zavg = zsum / 3.0
            out.append((zavg, pts, apply_fog(base_col, zavg)))

        # Volumetric look: skirts are only useful in underwater view.
        skirt_polys: list[
            tuple[float, list[tuple[float, float]], tuple[int, int, int]]
        ] = []
        if show_underwater:
            base_y = terrain.base_y
            skirt_base = (36, 34, 30)
            if base_y < -1.0:
                for c in range(g - 1):
                    # North (r=0)
                    a = verts[0 * g + c]
                    b = verts[0 * g + (c + 1)]
                    qa = (a[0], base_y, a[2])
                    qb = (b[0], base_y, b[2])
                    push_tri((a, b, qb), skirt_base, skirt_polys)
                    push_tri((a, qb, qa), skirt_base, skirt_polys)

                    # South (r=g-1)
                    a = verts[(g - 1) * g + c]
                    b = verts[(g - 1) * g + (c + 1)]
                    qa = (a[0], base_y, a[2])
                    qb = (b[0], base_y, b[2])
                    push_tri((a, qb, b), skirt_base, skirt_polys)
                    push_tri((a, qa, qb), skirt_base, skirt_polys)

                for r in range(g - 1):
                    # West (c=0)
                    a = verts[r * g + 0]
                    b = verts[(r + 1) * g + 0]
                    qa = (a[0], base_y, a[2])
                    qb = (b[0], base_y, b[2])
                    push_tri((a, qb, b), skirt_base, skirt_polys)
                    push_tri((a, qa, qb), skirt_base, skirt_polys)

                    # East (c=g-1)
                    a = verts[r * g + (g - 1)]
                    b = verts[(r + 1) * g + (g - 1)]
                    qa = (a[0], base_y, a[2])
                    qb = (b[0], base_y, b[2])
                    push_tri((a, b, qb), skirt_base, skirt_polys)
                    push_tri((a, qb, qa), skirt_base, skirt_polys)

        if show_underwater:
            # Underwater seabed pass (drawn under the water surface).
            for r in row_range:
                base_cell = r * (g - 1)
                for c in col_range:
                    cell = base_cell + c
                    t0 = cell * 2
                    t1 = t0 + 1

                    for t in (t0, t1):
                        miny, maxy = tri_mm[t]
                        if miny >= 0.0:
                            continue

                        ia, ib, ic = tri_indices[t]
                        if maxy <= 0.0:
                            if not (proj_ok[ia] and proj_ok[ib] and proj_ok[ic]):
                                continue
                            sa = proj[ia]
                            sb = proj[ib]
                            sc = proj[ic]
                            pts = [(sa[0], sa[1]), (sb[0], sb[1]), (sc[0], sc[1])]
                            zavg = (sa[2] + sb[2] + sc[2]) / 3.0
                        else:
                            pa = verts[ia]
                            pb = verts[ib]
                            pc = verts[ic]
                            poly3 = clip_poly_y([pa, pb, pc], keep_above=False)
                            if len(poly3) < 3:
                                continue
                            pts = []
                            zsum = 0.0
                            ok = True
                            for vv in poly3:
                                s = frame.project(vv)
                                if s is None:
                                    ok = False
                                    break
                                pts.append((s[0], s[1]))
                                zsum += s[2]
                            if not ok:
                                continue
                            zavg = zsum / len(pts)

                        nrm = tri_normals[t]
                        diff = max(0.0, dot(nrm, light_dir))
                        intensity = 0.22 + 0.55 * diff
                        base = seabed_color(tri_avg_y[t])
                        col = apply_fog(col_mul(base, intensity), zavg)
                        pygame.draw.polygon(draw_surf, col, pts)

        # Draw skirts after seabed (and before land).
        if skirt_polys:
            skirt_polys.sort(key=lambda it: it[0], reverse=True)
            for _, pts, col in skirt_polys:
                pygame.draw.polygon(draw_surf, col, pts)

        # Composite translucent water on top of seabed/skirts.
        if show_underwater and water_layer is not None:
            draw_surf.blit(water_layer, (0, 0))

        for r in row_range:
            base_cell = r * (g - 1)
            for c in col_range:
                cell = base_cell + c
                t0 = cell * 2
                t1 = t0 + 1

                polys: list[
                    tuple[float, list[tuple[float, float]], tuple[int, int, int]]
                ] = []

                for t in (t0, t1):
                    miny, maxy = tri_mm[t]
                    if maxy <= 0.0:
                        continue

                    ia, ib, ic = tri_indices[t]
                    if miny >= 0.0:
                        if not (proj_ok[ia] and proj_ok[ib] and proj_ok[ic]):
                            continue
                        sa = proj[ia]
                        sb = proj[ib]
                        sc = proj[ic]
                        pts = [(sa[0], sa[1]), (sb[0], sb[1]), (sc[0], sc[1])]
                        zavg = (sa[2] + sb[2] + sc[2]) / 3.0
                    else:
                        # Only shore triangles need clipping; most triangles use the fast path.
                        pa = verts[ia]
                        pb = verts[ib]
                        pc = verts[ic]
                        poly3 = clip_poly_y([pa, pb, pc], keep_above=True)
                        if len(poly3) < 3:
                            continue
                        pts = []
                        zsum = 0.0
                        ok = True
                        for vv in poly3:
                            s = frame.project(vv)
                            if s is None:
                                ok = False
                                break
                            pts.append((s[0], s[1]))
                            zsum += s[2]
                        if not ok:
                            continue
                        zavg = zsum / len(pts)

                    base = tri_base[t]
                    nrm = tri_normals[t]
                    diff = max(0.0, dot(nrm, light_dir))
                    intensity = 0.42 + 0.72 * diff
                    col = apply_fog(col_mul(base, intensity), zavg)
                    polys.append((zavg, pts, col))

                if len(polys) == 2 and polys[0][0] < polys[1][0]:
                    polys[0], polys[1] = polys[1], polys[0]

                for _, pts, col in polys:
                    pygame.draw.polygon(draw_surf, col, pts)

        # Rivers: draw on top of terrain.
        river_col = (40, 140, 170)
        for i0, i1, flow in terrain.river_edges:
            p0 = verts[i0]
            p1 = verts[i1]
            if p0[1] <= 0.05:
                continue
            # Lift slightly so it doesn't Z-fight with the land.
            p0 = (p0[0], p0[1] + 0.18, p0[2])
            p1 = (p1[0], p1[1] + 0.18, p1[2])
            s0 = frame.project(p0)
            s1 = frame.project(p1)
            if s0 is None or s1 is None:
                continue
            zavg = (s0[2] + s1[2]) * 0.5
            width = 1
            if flow > terrain.params.river_threshold * 2:
                width = 2
            if flow > terrain.params.river_threshold * 4:
                width = 3
            colr = apply_fog(river_col, zavg)
            pygame.draw.line(draw_surf, colr, (s0[0], s0[1]), (s1[0], s1[1]), width)

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
                            pygame.draw.aaline(draw_surf, col, (x0, y0), (s1[0], s1[1]))
                    if r + stride < g:
                        i2 = (r + stride) * g + c
                        if proj_ok[i2]:
                            s2 = proj[i2]
                            pygame.draw.aaline(draw_surf, col, (x0, y0), (s2[0], s2[1]))

        if draw_surf is not screen:
            # Scale the low-res render up to the window.
            pygame.transform.scale(draw_surf, (w, h), screen)

        # HUD
        fps = clock.get_fps()
        hud_lines = [
            f"seed: {seed}",
            f"grid: {terrain.params.grid} x {terrain.params.grid}",
            f"render: {'wireframe overlay' if show_wireframe else 'solid'} | scale: {render_scale:0.2f}",
            f"water view: {'underwater' if show_underwater else 'surface'}",
            "controls: drag LMB, wheel zoom, R regenerate, G wireframe, U underwater, P perf scale, arrows orbit",
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
