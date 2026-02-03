import '@fontsource/space-grotesk/400.css'
import '@fontsource/space-grotesk/600.css'
import '@fontsource/space-grotesk/700.css'
import './style.css'

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { randomSeed } from './core/SeededRandom'
import { Terrain } from './terrain/Terrain'

const rootQuery = document.querySelector<HTMLDivElement>('#app')
if (!rootQuery) throw new Error('Missing #app element')
const root = rootQuery

const hud = document.createElement('div')
hud.className = 'hud'
hud.innerHTML = `
  <h1>Biome Simulation</h1>
  <div class="row"><span>World seed</span><b id="seed">—</b></div>
  <div class="row"><span>Terrain</span><b>Perlin fBm</b></div>
  <div class="keys">
    <span class="key"><kbd>R</kbd> regenerate</span>
    <span class="key"><kbd>G</kbd> wireframe</span>
  </div>
`
root.appendChild(hud)

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
renderer.setSize(root.clientWidth, root.clientHeight)
renderer.setClearColor(0x0f1412, 1)
root.appendChild(renderer.domElement)

const scene = new THREE.Scene()
scene.fog = new THREE.Fog(0x0f1412, 60, 220)

const camera = new THREE.PerspectiveCamera(55, root.clientWidth / root.clientHeight, 0.1, 600)
camera.position.set(40, 34, 52)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true
controls.dampingFactor = 0.06
controls.target.set(0, 6, 0)
controls.maxPolarAngle = Math.PI * 0.495
controls.minDistance = 18
controls.maxDistance = 220

const hemi = new THREE.HemisphereLight(0xb6e0ff, 0x10130f, 0.55)
scene.add(hemi)

const sun = new THREE.DirectionalLight(0xfff1d8, 1.25)
sun.position.set(70, 90, 30)
scene.add(sun)

scene.add(new THREE.GridHelper(160, 32, 0x2c3a33, 0x1b231f))

const params = {
  size: 160,
  segments: 220,
  amplitude: 18,
  frequency: 2.4,
  octaves: 5,
  lacunarity: 2.0,
  persistence: 0.5,
  seaLevel: 0,
}

let seed = randomSeed()
const seedEl = hud.querySelector<HTMLSpanElement>('#seed')
if (!seedEl) throw new Error('Missing #seed element')
seedEl.textContent = String(seed)

const terrain = new Terrain({ seed, ...params })
scene.add(terrain.group)

let wireframe = false

function resize(): void {
  const w = root.clientWidth
  const h = root.clientHeight
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  renderer.setSize(w, h)
}

window.addEventListener('resize', resize)

window.addEventListener('keydown', (e) => {
  if (e.key === 'r' || e.key === 'R') {
    seed = randomSeed()
    seedEl.textContent = String(seed)
    terrain.regenerate(seed)
  }
  if (e.key === 'g' || e.key === 'G') {
    wireframe = !wireframe
    terrain.setWireframe(wireframe)
  }
})

const clock = new THREE.Clock()

function tick(): void {
  const dt = clock.getDelta()
  controls.update()
  renderer.render(scene, camera)
  // dt reserved for upcoming simulation steps
  void dt
  requestAnimationFrame(tick)
}

tick()
