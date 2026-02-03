declare module 'three/examples/jsm/controls/OrbitControls.js' {
  import type { Camera, EventDispatcher, Vector3 } from 'three'

  export class OrbitControls extends EventDispatcher {
    constructor(object: Camera, domElement?: HTMLElement)
    enabled: boolean
    target: Vector3
    enableDamping: boolean
    dampingFactor: number
    maxPolarAngle: number
    minDistance: number
    maxDistance: number
    update(): void
  }
}
