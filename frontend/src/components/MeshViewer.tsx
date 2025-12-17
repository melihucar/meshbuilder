import { Suspense, useRef, useEffect } from 'react'
import { Canvas, useFrame, useLoader } from '@react-three/fiber'
import { OrbitControls, Environment, Center, Html } from '@react-three/drei'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import * as THREE from 'three'

interface MeshViewerProps {
  url: string
  autoRotate?: boolean
  wireframe?: boolean
}

function LoadingSpinner() {
  return (
    <Html center>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
        color: 'white'
      }}>
        <div style={{
          width: '40px',
          height: '40px',
          border: '3px solid rgba(255,255,255,0.2)',
          borderTopColor: '#6366f1',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }} />
        <span>Loading model...</span>
      </div>
    </Html>
  )
}

function Model({ url, autoRotate = true, wireframe = false }: { url: string; autoRotate?: boolean; wireframe?: boolean }) {
  const meshRef = useRef<THREE.Group>(null)

  // Determine file type from URL
  const isGlb = url.endsWith('.glb') || url.endsWith('.gltf')
  const isObj = url.endsWith('.obj')
  const isPly = url.endsWith('.ply')

  // Load the appropriate model type
  let scene: THREE.Object3D | null = null

  if (isGlb) {
    const gltf = useLoader(GLTFLoader, url)
    scene = gltf.scene
  } else if (isObj) {
    scene = useLoader(OBJLoader, url)
  } else if (isPly) {
    const geometry = useLoader(PLYLoader, url)
    geometry.computeVertexNormals()
    const material = new THREE.MeshStandardMaterial({
      vertexColors: geometry.hasAttribute('color'),
      side: THREE.DoubleSide
    })
    const mesh = new THREE.Mesh(geometry, material)
    scene = mesh
  }

  // Apply wireframe mode to all meshes
  useEffect(() => {
    if (scene) {
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh && child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(mat => {
              if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshBasicMaterial) {
                mat.wireframe = wireframe
              }
            })
          } else if (child.material instanceof THREE.MeshStandardMaterial || child.material instanceof THREE.MeshBasicMaterial) {
            child.material.wireframe = wireframe
          }
        }
      })
    }
  }, [scene, wireframe])

  // Auto-rotate
  useFrame((_, delta) => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.y += delta * 0.3
    }
  })

  if (!scene) return null

  return (
    <group ref={meshRef}>
      <Center>
        <primitive object={scene} scale={1} />
      </Center>
    </group>
  )
}

export default function MeshViewer({ url, autoRotate = true, wireframe = false }: MeshViewerProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, 3], fov: 50 }}
      style={{ background: 'linear-gradient(180deg, #1a1a2e 0%, #0a0a0f 100%)' }}
    >
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, -10, -5]} intensity={0.5} />

      <Suspense fallback={<LoadingSpinner />}>
        <Model url={url} autoRotate={autoRotate} wireframe={wireframe} />
        <Environment preset="city" />
      </Suspense>

      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={false}
        minDistance={0.5}
        maxDistance={10}
      />

      {/* Grid helper */}
      <gridHelper args={[10, 10, '#333', '#222']} position={[0, -1, 0]} />
    </Canvas>
  )
}
