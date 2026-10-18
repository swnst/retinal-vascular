"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { LandmarkPoint } from "@/types/biometrics";
import { Compass, Pause, Play, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

interface RetinalGlobeProps {
  fundusImageBase64?: string;
  probMapBase64?: string;
  skeletonBase64?: string;
  landmarks?: LandmarkPoint[];
  opticDisc?: { x: number; y: number } | null;
  fovea?: { x: number; y: number } | null;
  showLandmarks?: boolean;
  glowThreshold?: number;
  reliefScale?: number;
  explodedLayers?: boolean;
  imageWidth?: number;
  imageHeight?: number;
}

// --- AAA 3D Concave Ocular Fundus Cup with Dynamic Camera Control Dock ---
export default function RetinalGlobe({
  fundusImageBase64,
  probMapBase64,
  skeletonBase64,
  landmarks = [],
  opticDisc,
  fovea,
  showLandmarks = true,
  glowThreshold = 0.60,
  reliefScale = 0.16,
  explodedLayers = false,
  imageWidth = 565,
  imageHeight = 584,
}: RetinalGlobeProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const ocularGroupRef = useRef<THREE.Group | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const dishMeshRef = useRef<THREE.Mesh | null>(null);
  const skeletonMeshRef = useRef<THREE.Mesh | null>(null);
  const beaconGroupRef = useRef<THREE.Group | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const skelMaterialRef = useRef<THREE.ShaderMaterial | null>(null);

  // 3D Viewport Navigation State
  const [isAutoRotating, setIsAutoRotating] = useState<boolean>(true);
  const [rotationMode, setRotationMode] = useState<"sway" | "turntable">("sway");
  const [zoomLevel, setZoomLevel] = useState<number>(4.4);
  const isAutoRotatingRef = useRef<boolean>(true);
  const rotationModeRef = useRef<"sway" | "turntable">("sway");
  const zoomLevelRef = useRef<number>(4.4);
  const resetRequestedRef = useRef<boolean>(false);

  useEffect(() => {
    isAutoRotatingRef.current = isAutoRotating;
  }, [isAutoRotating]);

  useEffect(() => {
    rotationModeRef.current = rotationMode;
  }, [rotationMode]);

  useEffect(() => {
    zoomLevelRef.current = zoomLevel;
    if (cameraRef.current) {
      cameraRef.current.position.z = zoomLevel;
    }
  }, [zoomLevel]);

  const handleResetCamera = () => {
    resetRequestedRef.current = true;
    setZoomLevel(4.4);
  };

  const handleZoom = (delta: number) => {
    setZoomLevel((prev) => Math.min(7.0, Math.max(2.4, prev + delta)));
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    // 1. Scene & Camera Setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 0, zoomLevelRef.current);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    const ocularGroup = new THREE.Group();
    scene.add(ocularGroup);
    ocularGroupRef.current = ocularGroup;

    // 2. Anatomical Concave Paraboloid Mesh
    const gridRes = 160;
    const planeGeo = new THREE.PlaneGeometry(3.2, 3.2, gridRes, gridRes);
    const posAttr = planeGeo.attributes.position;
    const uvAttr = planeGeo.attributes.uv;

    for (let i = 0; i < posAttr.count; i++) {
      const u = uvAttr.getX(i);
      const v = uvAttr.getY(i);
      const dx = (u - 0.5) * 2.0;
      const dy = (v - 0.5) * 2.0;
      const r = Math.sqrt(dx * dx + dy * dy);
      const cupDepth = -0.42 * Math.max(0.0, 1.0 - r * r);
      posAttr.setZ(i, cupDepth);
    }
    planeGeo.computeVertexNormals();

    // 3. Custom GLSL Shader for Continuous 60FPS Probability Thresholding & Subsurface Glow
    const customMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tFundus: { value: null },
        tProbMap: { value: null },
        hasFundus: { value: 0.0 },
        hasProbMap: { value: 0.0 },
        uGlowThreshold: { value: 0.60 },
        uReliefScale: { value: 0.22 },
        uTime: { value: 0.0 },
        uExplodedZ: { value: 0.0 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        uniform sampler2D tProbMap;
        uniform float hasProbMap;
        uniform float uGlowThreshold;
        uniform float uReliefScale;
        uniform float uExplodedZ;

        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          
          float prob = 0.0;
          if (hasProbMap > 0.5) {
            prob = texture2D(tProbMap, uv).r;
          }
          
          float vesselFactor = smoothstep(uGlowThreshold - 0.05, uGlowThreshold + 0.15, prob);
          vec3 displacedPos = position + normal * (vesselFactor * uReliefScale);
          displacedPos.z += uExplodedZ;

          vec4 mvPosition = modelViewMatrix * vec4(displacedPos, 1.0);
          vViewPosition = -mvPosition.xyz;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        uniform sampler2D tFundus;
        uniform sampler2D tProbMap;
        uniform float hasFundus;
        uniform float hasProbMap;
        uniform float uGlowThreshold;
        uniform float uTime;

        void main() {
          vec2 centeredUv = (vUv - 0.5) * 2.0;
          float distFromCenter = length(centeredUv);
          if (distFromCenter > 0.96) {
            discard;
          }

          vec3 baseColor = vec3(0.06, 0.03, 0.02);
          if (hasFundus > 0.5) {
            vec4 fundusSample = texture2D(tFundus, vUv);
            baseColor = fundusSample.rgb;
          }

          if (hasProbMap > 0.5) {
            float prob = texture2D(tProbMap, vUv).r;
            if (prob >= uGlowThreshold) {
              float intensity = (prob - uGlowThreshold) / max(0.001, (1.0 - uGlowThreshold));
              float pulse = 0.88 + 0.12 * sin(uTime * 3.0 + vUv.x * 20.0);
              vec3 bioluminescentCyan = vec3(0.02, 0.88, 0.95) * (1.2 + intensity * 0.8) * pulse;
              baseColor = mix(baseColor, bioluminescentCyan, 0.82);
            }
          }

          vec3 normal = normalize(vNormal);
          vec3 viewDir = normalize(vViewPosition);
          float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
          baseColor += vec3(0.02, 0.55, 0.75) * fresnel * 0.40;

          float edgeAlpha = 1.0 - smoothstep(0.93, 0.96, distFromCenter);
          gl_FragColor = vec4(baseColor, edgeAlpha);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
    });
    materialRef.current = customMaterial;

    const dishMesh = new THREE.Mesh(planeGeo, customMaterial);
    ocularGroup.add(dishMesh);
    dishMeshRef.current = dishMesh;

    // 4. Pure Glowing 3D Skeleton Shader (Zero Black Background Occlusion)
    const skelMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tSkeleton: { value: null },
        hasSkeleton: { value: 0.0 },
        uOpacity: { value: 0.0 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform sampler2D tSkeleton;
        uniform float hasSkeleton;
        uniform float uOpacity;

        void main() {
          vec2 centeredUv = (vUv - 0.5) * 2.0;
          if (length(centeredUv) > 0.96) discard;

          if (hasSkeleton < 0.5) discard;

          vec4 skeletonTexel = texture2D(tSkeleton, vUv);
          float intensity = max(skeletonTexel.r, max(skeletonTexel.g, skeletonTexel.b));
          if (intensity < 0.18) discard;

          vec3 electricCyan = vec3(0.04, 0.82, 0.98) * 1.35;
          gl_FragColor = vec4(electricCyan, uOpacity);
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    skelMaterialRef.current = skelMaterial;

    const skelMesh = new THREE.Mesh(planeGeo.clone(), skelMaterial);
    skelMesh.position.z = 0.0;
    ocularGroup.add(skelMesh);
    skeletonMeshRef.current = skelMesh;

    // 5. 3D Landmark Beacon Group
    const beaconGroup = new THREE.Group();
    ocularGroup.add(beaconGroup);
    beaconGroupRef.current = beaconGroup;

    // 6. Interactive Orbital Drag & Wheel Zoom
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let velocity = { x: 0, y: 0 };

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
      velocity = { x: 0, y: 0 };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      ocularGroup.rotation.y += dx * 0.005;
      ocularGroup.rotation.x += dy * 0.005;
      velocity = { x: dx * 0.005, y: dy * 0.005 };
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY * 0.003;
      setZoomLevel((prev) => Math.min(7.0, Math.max(2.4, prev + delta)));
    };

    const domElement = renderer.domElement;
    domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    domElement.addEventListener("wheel", onWheel, { passive: false });

    // 7. ResizeObserver
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    resizeObserver.observe(container);

    // 8. Controlled 60FPS Render Loop
    let animId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      if (customMaterial) {
        customMaterial.uniforms.uTime.value = elapsed;
      }

      if (resetRequestedRef.current) {
        ocularGroup.rotation.x += (0.0 - ocularGroup.rotation.x) * 0.12;
        ocularGroup.rotation.y += (0.0 - ocularGroup.rotation.y) * 0.12;
        camera.position.z += (4.4 - camera.position.z) * 0.12;
        if (
          Math.abs(ocularGroup.rotation.x) < 0.005 &&
          Math.abs(ocularGroup.rotation.y) < 0.005 &&
          Math.abs(camera.position.z - 4.4) < 0.01
        ) {
          ocularGroup.rotation.set(0, 0, 0);
          camera.position.z = 4.4;
          resetRequestedRef.current = false;
        }
      } else if (!isDragging) {
        ocularGroup.rotation.y += velocity.x * 0.90;
        ocularGroup.rotation.x += velocity.y * 0.90;
        velocity.x *= 0.90;
        velocity.y *= 0.90;

        if (isAutoRotatingRef.current) {
          if (rotationModeRef.current === "sway") {
            const targetY = Math.sin(elapsed * 0.45) * 0.22;
            const targetX = Math.cos(elapsed * 0.30) * 0.07;
            ocularGroup.rotation.y += (targetY - ocularGroup.rotation.y) * 0.04;
            ocularGroup.rotation.x += (targetX - ocularGroup.rotation.x) * 0.04;
          } else {
            ocularGroup.rotation.y += 0.0025;
          }
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      domElement.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      domElement.removeEventListener("wheel", onWheel);
      resizeObserver.disconnect();
      domElement.remove();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Points) {
          obj.geometry?.dispose();
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m.dispose());
          } else if (obj.material) {
            obj.material.dispose();
          }
        }
      });
      renderer.dispose();
    };
  }, []);

  // Update Textures
  useEffect(() => {
    if (!materialRef.current) return;
    const loader = new THREE.TextureLoader();

    if (fundusImageBase64) {
      loader.load(fundusImageBase64, (tex) => {
        if (!materialRef.current) return;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        if (materialRef.current.uniforms.tFundus.value) {
          materialRef.current.uniforms.tFundus.value.dispose();
        }
        materialRef.current.uniforms.tFundus.value = tex;
        materialRef.current.uniforms.hasFundus.value = 1.0;
        materialRef.current.needsUpdate = true;
      });
    } else {
      if (materialRef.current.uniforms.tFundus.value) {
        materialRef.current.uniforms.tFundus.value.dispose();
        materialRef.current.uniforms.tFundus.value = null;
      }
      materialRef.current.uniforms.hasFundus.value = 0.0;
    }

    if (probMapBase64) {
      loader.load(probMapBase64, (tex) => {
        if (!materialRef.current) return;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        if (materialRef.current.uniforms.tProbMap.value) {
          materialRef.current.uniforms.tProbMap.value.dispose();
        }
        materialRef.current.uniforms.tProbMap.value = tex;
        materialRef.current.uniforms.hasProbMap.value = 1.0;
        materialRef.current.needsUpdate = true;
      });
    } else {
      if (materialRef.current.uniforms.tProbMap.value) {
        materialRef.current.uniforms.tProbMap.value.dispose();
        materialRef.current.uniforms.tProbMap.value = null;
      }
      materialRef.current.uniforms.hasProbMap.value = 0.0;
    }

    if (skeletonBase64 && skelMaterialRef.current) {
      loader.load(skeletonBase64, (tex) => {
        if (!skelMaterialRef.current) return;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        if (skelMaterialRef.current.uniforms.tSkeleton.value) {
          skelMaterialRef.current.uniforms.tSkeleton.value.dispose();
        }
        skelMaterialRef.current.uniforms.tSkeleton.value = tex;
        skelMaterialRef.current.uniforms.hasSkeleton.value = 1.0;
        skelMaterialRef.current.needsUpdate = true;
      });
    }
  }, [fundusImageBase64, probMapBase64, skeletonBase64]);

  // Shader Uniform Updates
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uGlowThreshold.value = glowThreshold;
      materialRef.current.uniforms.uReliefScale.value = reliefScale;
      materialRef.current.uniforms.uExplodedZ.value = explodedLayers ? -0.25 : 0.0;
    }
    if (skelMaterialRef.current && skeletonMeshRef.current) {
      skelMaterialRef.current.uniforms.uOpacity.value = explodedLayers ? 0.90 : 0.0;
      skeletonMeshRef.current.position.z = explodedLayers ? 0.45 : 0.0;
    }
  }, [glowThreshold, reliefScale, explodedLayers]);

  // 3D Landmarks & Bifurcation Nodes
  useEffect(() => {
    if (!beaconGroupRef.current) return;
    const group = beaconGroupRef.current;
    while (group.children.length > 0) {
      group.remove(group.children[0]);
    }

    if (!showLandmarks) return;

    const mapPixelToCup = (px: number, py: number) => {
      const u = px / Math.max(1, imageWidth);
      const v = py / Math.max(1, imageHeight);
      const x = (u - 0.5) * 3.2;
      const y = (0.5 - v) * 3.2;
      const dx = (u - 0.5) * 2.0;
      const dy = (v - 0.5) * 2.0;
      const r = Math.sqrt(dx * dx + dy * dy);
      const z = -0.42 * Math.max(0.0, 1.0 - r * r) + 0.06;
      return new THREE.Vector3(x, y, z);
    };

    // 1. Optic Disc Reticle Beacon (Emerald Green)
    if (opticDisc) {
      const odPos = mapPixelToCup(opticDisc.x, opticDisc.y);
      const ringGeo = new THREE.RingGeometry(0.12, 0.14, 32);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981, side: THREE.DoubleSide });
      const odRing = new THREE.Mesh(ringGeo, ringMat);
      odRing.position.copy(odPos);
      group.add(odRing);

      const beaconGeo = new THREE.CylinderGeometry(0.015, 0.005, 0.4, 16);
      const beaconMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
      const odBeacon = new THREE.Mesh(beaconGeo, beaconMat);
      odBeacon.position.copy(odPos.clone().add(new THREE.Vector3(0, 0, 0.2)));
      odBeacon.rotation.x = Math.PI / 2;
      group.add(odBeacon);
    }

    // 2. Fovea Reticle Beacon (Amber Gold)
    if (fovea) {
      const fovPos = mapPixelToCup(fovea.x, fovea.y);
      const ringGeo = new THREE.RingGeometry(0.08, 0.10, 32);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, side: THREE.DoubleSide });
      const fovRing = new THREE.Mesh(ringGeo, ringMat);
      fovRing.position.copy(fovPos);
      group.add(fovRing);

      const beaconGeo = new THREE.CylinderGeometry(0.012, 0.004, 0.35, 16);
      const beaconMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
      const fovBeacon = new THREE.Mesh(beaconGeo, beaconMat);
      fovBeacon.position.copy(fovPos.clone().add(new THREE.Vector3(0, 0, 0.175)));
      fovBeacon.rotation.x = Math.PI / 2;
      group.add(fovBeacon);
    }

    // 3. Geodesic Arc Connecting Optic Disc and Fovea
    if (opticDisc && fovea) {
      const p1 = mapPixelToCup(opticDisc.x, opticDisc.y);
      const p2 = mapPixelToCup(fovea.x, fovea.y);
      const mid = p1.clone().add(p2).multiplyScalar(0.5);
      mid.z += 0.22;

      const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
      const points = curve.getPoints(24);
      const arcGeo = new THREE.BufferGeometry().setFromPoints(points);
      const arcMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 });
      const arcLine = new THREE.Line(arcGeo, arcMat);
      group.add(arcLine);
    }

    // 4. Vascular Bifurcation Starlight Landmarks
    if (landmarks.length > 0) {
      const pts = landmarks.slice(0, 64);
      const starGeo = new THREE.SphereGeometry(0.02, 8, 8);
      const starMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
      for (const pt of pts) {
        const pos = mapPixelToCup(pt.x, pt.y);
        if (explodedLayers) pos.z += 0.40;
        const star = new THREE.Mesh(starGeo, starMat);
        star.position.copy(pos);
        group.add(star);
      }
    }
  }, [landmarks, opticDisc, fovea, showLandmarks, explodedLayers, imageWidth, imageHeight]);

  return (
    <div className="relative w-full h-full min-h-[500px] flex items-center justify-center overflow-hidden select-none">
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating 3D Camera & Rotation Control Dock (Bottom Center) */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20 flex items-center space-x-2 bg-gray-950/85 backdrop-blur-xl px-4 py-2 rounded-2xl border border-gray-800/80 shadow-2xl text-xs font-mono">
        <button
          onClick={() => setIsAutoRotating(!isAutoRotating)}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-all cursor-pointer ${
            isAutoRotating
              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/50 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
              : "bg-gray-900/80 text-gray-400 border-gray-800 hover:text-gray-200"
          }`}
          title="Toggle 3D Auto-Rotation"
        >
          {isAutoRotating ? <Pause className="w-3.5 h-3.5 text-cyan-400" /> : <Play className="w-3.5 h-3.5" />}
          <span>{isAutoRotating ? "Auto-Rotate" : "Paused"}</span>
        </button>

        <button
          onClick={() => setRotationMode(rotationMode === "sway" ? "turntable" : "sway")}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:text-cyan-300 hover:border-gray-700 text-gray-400 transition-all cursor-pointer"
          title="Switch Motion: Gentle Sway vs 360 Spin"
        >
          <Compass className="w-3.5 h-3.5 text-sky-400" />
          <span>{rotationMode === "sway" ? "Sway ±14°" : "360° Spin"}</span>
        </button>

        <button
          onClick={handleResetCamera}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:text-emerald-300 hover:border-gray-700 text-gray-400 transition-all cursor-pointer"
          title="Reset Camera Orientation & Zoom"
        >
          <RotateCcw className="w-3.5 h-3.5 text-emerald-400" />
          <span>Reset View</span>
        </button>

        <div className="h-4 w-[1px] bg-gray-800 mx-1" />

        <div className="flex items-center space-x-1">
          <button
            onClick={() => handleZoom(-0.5)}
            className="p-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:text-cyan-300 hover:border-gray-700 text-gray-400 transition-all cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => handleZoom(0.5)}
            className="p-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:text-cyan-300 hover:border-gray-700 text-gray-400 transition-all cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Floating Status Pill (Top Left) */}
      <div className="absolute top-5 left-5 z-20 hidden lg:flex items-center space-x-3 text-xs font-mono text-gray-400 bg-gray-950/70 backdrop-blur-md px-3.5 py-1.5 rounded-full border border-gray-800/80 pointer-events-none shadow-xl">
        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        <span>3D Concave Ocular Cup: Drag to Orbit | Scroll to Zoom</span>
      </div>
    </div>
  );
}
