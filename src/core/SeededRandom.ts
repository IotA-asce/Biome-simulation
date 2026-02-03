export type Rng = () => number

// Fast, deterministic PRNG for reproducible worlds.
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function seedFromString(input: string): number {
  // FNV-1a 32-bit
  let h = 0x811c9dc5
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

export function randomSeed(): number {
  // Prefer crypto if available; fall back to time.
  const g = globalThis as unknown as { crypto?: Crypto }
  if (g.crypto?.getRandomValues) {
    const buf = new Uint32Array(1)
    g.crypto.getRandomValues(buf)
    return buf[0] >>> 0
  }
  return (Date.now() & 0xffffffff) >>> 0
}
