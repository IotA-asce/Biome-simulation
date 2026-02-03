import { describe, expect, it } from 'vitest'
import { Perlin2D } from './Perlin2D'

describe('Perlin2D', () => {
  it('is deterministic for a given seed', () => {
    const a = new Perlin2D(12345)
    const b = new Perlin2D(12345)
    const pts: Array<[number, number]> = [
      [0, 0],
      [0.1, 0.2],
      [2.345, -1.75],
      [10.01, 10.02],
    ]
    for (const [x, y] of pts) {
      expect(a.noise(x, y)).toBe(b.noise(x, y))
      expect(a.fbm(x, y, { octaves: 5, lacunarity: 2, persistence: 0.5 })).toBe(
        b.fbm(x, y, { octaves: 5, lacunarity: 2, persistence: 0.5 }),
      )
    }
  })

  it('stays within a sane range', () => {
    const p = new Perlin2D(7)
    for (let i = 0; i < 200; i++) {
      const x = i * 0.173
      const y = i * 0.091
      const n = p.noise(x, y)
      expect(n).toBeGreaterThanOrEqual(-1.01)
      expect(n).toBeLessThanOrEqual(1.01)
      const f = p.fbm(x, y, { octaves: 6, lacunarity: 2.1, persistence: 0.52 })
      expect(f).toBeGreaterThanOrEqual(-1.01)
      expect(f).toBeLessThanOrEqual(1.01)
    }
  })
})
