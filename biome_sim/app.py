from __future__ import annotations

import time

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

    wireframe = True
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
                    wireframe = not wireframe
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

        screen.fill((12, 18, 16))
        w, h = screen.get_size()

        g = terrain.params.grid
        line_color_base = (160, 232, 200)

        stride = 1 if wireframe else 2

        # Wireframe draw: connect each vertex to its right/bottom neighbor.
        # This is a simple software 3D renderer: we project 3D points into screen space.
        for r in range(0, g, stride):
            for c in range(0, g, stride):
                p0 = terrain.vertex(r, c)
                pr = terrain.vertex(r, c + stride) if c + stride < g else None
                pd = terrain.vertex(r + stride, c) if r + stride < g else None

                s0 = cam.project(p0, (w, h))
                if s0 is None:
                    continue

                x0, y0, z0 = s0
                fog = max(0.0, min(1.0, 1.0 - (z0 / 420.0)))
                col = (
                    int(line_color_base[0] * fog),
                    int(line_color_base[1] * fog),
                    int(line_color_base[2] * fog),
                )

                if pr is not None:
                    sr = cam.project(pr, (w, h))
                    if sr is not None:
                        pygame.draw.aaline(screen, col, (x0, y0), (sr[0], sr[1]))
                if pd is not None:
                    sd = cam.project(pd, (w, h))
                    if sd is not None:
                        pygame.draw.aaline(screen, col, (x0, y0), (sd[0], sd[1]))

        # HUD
        fps = clock.get_fps()
        hud_lines = [
            f"seed: {seed}",
            f"grid: {terrain.params.grid} x {terrain.params.grid}",
            f"render: {'wireframe' if wireframe else 'sparse wireframe'}",
            "controls: drag LMB, wheel zoom, R regenerate, G toggle, arrows orbit",
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
