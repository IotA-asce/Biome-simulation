import { mulberry32 } from '../core/SeededRandom'

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10)
}

function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a)
}

function grad(hash: number, x: number, y: number): number {
  // 8 gradient directions based on lower bits.
  const h = hash & 7
  const u = h < 4 ? x : y
  const v = h < 4 ? y : x
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v)
}

export type FbmOptions = {
  octaves: number
  lacunarity: number
  persistence: number
}

export class Perlin2D {
  private perm: Uint8Array

  constructor(seed: number) {
    const rng = mulberry32(seed)
    const p = new Uint8Array(256)
    for (let i = 0; i < 256; i++) p[i] = i
    for (let i = 255; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      const tmp = p[i]
      p[i] = p[j]
      p[j] = tmp
    }
    this.perm = new Uint8Array(512)
    for (let i = 0; i < 512; i++) this.perm[i] = p[i & 255]
  }

  noise(x: number, y: number): number {
    const xi = Math.floor(x) & 255
    const yi = Math.floor(y) & 255
    const xf = x - Math.floor(x)
    const yf = y - Math.floor(y)

    const u = fade(xf)
    const v = fade(yf)

    const aa = this.perm[xi + this.perm[yi]]
    const ab = this.perm[xi + this.perm[yi + 1]]
    const ba = this.perm[xi + 1 + this.perm[yi]]
    const bb = this.perm[xi + 1 + this.perm[yi + 1]]

    const x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
    const x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)

    // Roughly in [-1, 1]
    return lerp(x1, x2, v) * 0.7071067811865475 // 1/sqrt(2) for gradient normalization
  }

  fbm(x: number, y: number, opts: FbmOptions): number {
    const octaves = Math.max(1, Math.floor(opts.octaves))
    let frequency = 1
    let amplitude = 1
    let sum = 0
    let ampSum = 0

    for (let i = 0; i < octaves; i++) {
      sum += this.noise(x * frequency, y * frequency) * amplitude
      ampSum += amplitude
      frequency *= opts.lacunarity
      amplitude *= opts.persistence
    }

    return ampSum === 0 ? 0 : sum / ampSum
  }
}
