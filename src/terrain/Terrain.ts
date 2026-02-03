import * as THREE from 'three'
import { Perlin2D } from './Perlin2D'

export type TerrainParams = {
  seed: number
  size: number
  segments: number
  amplitude: number
  frequency: number
  octaves: number
  lacunarity: number
  persistence: number
  seaLevel: number
}

export class Terrain {
  readonly group = new THREE.Group()
  readonly mesh: THREE.Mesh
  readonly sea: THREE.Mesh
  params: TerrainParams

  private geometry: THREE.PlaneGeometry
  private material: THREE.MeshStandardMaterial
  private seaMaterial: THREE.MeshStandardMaterial

  constructor(params: TerrainParams) {
    this.params = { ...params }

    this.geometry = new THREE.PlaneGeometry(params.size, params.size, params.segments, params.segments)
    this.geometry.rotateX(-Math.PI / 2)

    this.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.95,
      metalness: 0.0,
    })

    this.mesh = new THREE.Mesh(this.geometry, this.material)
    this.mesh.castShadow = false
    this.mesh.receiveShadow = true

    const seaGeo = new THREE.PlaneGeometry(params.size * 1.1, params.size * 1.1, 1, 1)
    seaGeo.rotateX(-Math.PI / 2)
    this.seaMaterial = new THREE.MeshStandardMaterial({
      color: 0x173647,
      roughness: 0.25,
      metalness: 0.0,
      transparent: true,
      opacity: 0.85,
    })
    this.sea = new THREE.Mesh(seaGeo, this.seaMaterial)
    this.sea.position.y = params.seaLevel
    this.sea.receiveShadow = true

    this.group.add(this.mesh)
    this.group.add(this.sea)

    this.regenerate(params.seed)
  }

  regenerate(seed: number): void {
    this.params.seed = seed >>> 0
    const perlin = new Perlin2D(this.params.seed)

    const pos = this.geometry.attributes.position
    const n = pos.count

    let minH = Number.POSITIVE_INFINITY
    let maxH = Number.NEGATIVE_INFINITY

    for (let i = 0; i < n; i++) {
      const x = pos.getX(i)
      const z = pos.getZ(i)

      const nx = (x / this.params.size) * this.params.frequency
      const nz = (z / this.params.size) * this.params.frequency

      const h01 = perlin.fbm(nx + 1000.0, nz - 1000.0, {
        octaves: this.params.octaves,
        lacunarity: this.params.lacunarity,
        persistence: this.params.persistence,
      })

      const h = h01 * this.params.amplitude
      pos.setY(i, h)
      if (h < minH) minH = h
      if (h > maxH) maxH = h
    }

    pos.needsUpdate = true
    this.geometry.computeVertexNormals()

    const colors = new Float32Array(n * 3)
    const normals = this.geometry.attributes.normal

    for (let i = 0; i < n; i++) {
      const h = pos.getY(i)
      const ny = normals.getY(i)
      const slope = 1 - THREE.MathUtils.clamp(ny, 0, 1)
      const t = maxH === minH ? 0.5 : (h - minH) / (maxH - minH)

      // Color bands: deep water -> shore -> grass -> rock -> snow
      const c = new THREE.Color()
      if (h < this.params.seaLevel - 1.5) {
        c.setRGB(0.05, 0.16, 0.22)
      } else if (h < this.params.seaLevel + 0.6) {
        c.setRGB(0.24, 0.38, 0.42)
      } else if (t < 0.56) {
        // grass, darken with slope
        c.setRGB(0.16, 0.34, 0.20).lerp(new THREE.Color(0.10, 0.22, 0.12), slope * 0.75)
      } else if (t < 0.78) {
        // rocky highlands
        c.setRGB(0.38, 0.34, 0.28).lerp(new THREE.Color(0.20, 0.18, 0.15), slope * 0.55)
      } else {
        // snowcaps
        c.setRGB(0.86, 0.88, 0.90).lerp(new THREE.Color(0.55, 0.57, 0.60), slope * 0.25)
      }

      colors[i * 3 + 0] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
    }

    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  }

  setWireframe(enabled: boolean): void {
    this.material.wireframe = enabled
  }

  dispose(): void {
    this.geometry.dispose()
    this.material.dispose()
    this.sea.geometry.dispose()
    this.seaMaterial.dispose()
  }
}
