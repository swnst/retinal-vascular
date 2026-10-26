"use client";

import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { MatchedTrianglePair, TemplateSummary } from "@/types/biometrics";
import { Compass, Pause, Play, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";

interface DelaunayConstellationProps {
  fundusABase64?: string;
  fundusBBase64?: string;
  perturbedFundusBBase64?: string;
  templateA?: TemplateSummary | null;
  templateB?: TemplateSummary | null;
  matchedPairs?: MatchedTrianglePair[];
  isMatch?: boolean | null;
  rotateDegB?: number;
  scaleFactorB?: number;
  brightnessB?: number;
}

// --- AAA 3D Dual Retinal Constellation & Real-Time Laser Matcher ---
export default function DelaunayConstellation({
  fundusABase64,
  fundusBBase64,
  perturbedFundusBBase64,
  templateA,
  templateB,
  matchedPairs = [],
  isMatch,
  rotateDegB = 0.0,
  scaleFactorB = 1.0,
  brightnessB = 1.0,
}: DelaunayConstellationProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const mainGroupRef = useRef<THREE.Group | null>(null);
  const planeBGroupRef = useRef<THREE.Group | null>(null);
  const matARef = useRef<THREE.ShaderMaterial | null>(null);
  const matBRef = useRef<THREE.ShaderMaterial | null>(null);
  const networkGroupRef = useRef<THREE.Group | null>(null);

  // --- 3D Viewport Controls ---
  const [isAutoRotating, setIsAutoRotating] = useState<boolean>(true);
  const [rotationMode, setRotationMode] = useState<"sway" | "turntable">("sway");
  const [zoomLevel, setZoomLevel] = useState<number>(5.2);
  const isAutoRotatingRef = useRef<boolean>(true);
  const rotationModeRef = useRef<"sway" | "turntable">("sway");
  const zoomLevelRef = useRef<number>(5.2);
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
    setZoomLevel(5.2);
  };

  const handleZoom = (delta: number) => {
    setZoomLevel((prev) => Math.min(8.0, Math.max(3.0, prev + delta)));
  };

  // --- Target Plane B Perturbations ---
  useEffect(() => {
    if (planeBGroupRef.current) {
      if (perturbedFundusBBase64) {
        planeBGroupRef.current.rotation.z = 0.0;
        planeBGroupRef.current.scale.set(1.0, 1.0, 1.0);
      } else {
        planeBGroupRef.current.rotation.z = rotateDegB * (Math.PI / 180.0);
        planeBGroupRef.current.scale.set(scaleFactorB, scaleFactorB, 1.0);
      }
    }
    if (matBRef.current) {
      matBRef.current.uniforms.uBrightness.value = perturbedFundusBBase64 ? 1.0 : brightnessB;
    }
  }, [rotateDegB, scaleFactorB, brightnessB, perturbedFundusBBase64]);

  // --- Core WebGL Scene Setup (Mounted Once) ---
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 600;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0, zoomLevelRef.current);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    const mainGroup = new THREE.Group();
    scene.add(mainGroup);
    mainGroupRef.current = mainGroup;

    // --- Circular Retinal Disc Material ---
    const createRetinalMaterial = (tintColor: THREE.Color, initialBrightness: number = 1.0) => {
      return new THREE.ShaderMaterial({
        uniforms: {
          tFundus: { value: null },
          hasFundus: { value: 0.0 },
          uBrightness: { value: initialBrightness },
          uTint: { value: tintColor },
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
          uniform sampler2D tFundus;
          uniform float hasFundus;
          uniform float uBrightness;
          uniform vec3 uTint;

          void main() {
            vec2 centeredUv = (vUv - 0.5) * 2.0;
            float r = length(centeredUv);
            if (r > 0.96) discard;

            vec3 baseColor = vec3(0.08, 0.05, 0.04);
            if (hasFundus > 0.5) {
              baseColor = texture2D(tFundus, vUv).rgb * uBrightness;
            }

            float rim = smoothstep(0.85, 0.96, r);
            baseColor = mix(baseColor, uTint, rim * 0.45);

            float edgeAlpha = 1.0 - smoothstep(0.92, 0.96, r);
            gl_FragColor = vec4(baseColor, edgeAlpha * 0.92);
          }
        `,
        transparent: true,
        side: THREE.DoubleSide,
      });
    };

    const planeGeo = new THREE.PlaneGeometry(2.7, 2.7);

    // --- Plane A Setup ---
    const matA = createRetinalMaterial(new THREE.Color(0x06b6d4));
    matARef.current = matA;
    const meshA = new THREE.Mesh(planeGeo, matA);
    meshA.position.set(-1.65, 0, -0.04);
    mainGroup.add(meshA);

    // --- Plane B Setup ---
    const planeBGroup = new THREE.Group();
    planeBGroup.position.set(1.65, 0, 0);
    mainGroup.add(planeBGroup);
    planeBGroupRef.current = planeBGroup;

    const matB = createRetinalMaterial(new THREE.Color(0xf43f5e));
    matBRef.current = matB;
    const meshB = new THREE.Mesh(planeGeo, matB);
    meshB.position.set(0, 0, -0.04);
    planeBGroup.add(meshB);

    // --- Common Constellation Network Group ---
    const networkGroup = new THREE.Group();
    mainGroup.add(networkGroup);
    networkGroupRef.current = networkGroup;

    // --- Drag Orbit & Wheel Zoom ---
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
      mainGroup.rotation.y += dx * 0.005;
      mainGroup.rotation.x += dy * 0.005;
      velocity = { x: dx * 0.005, y: dy * 0.005 };
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY * 0.003;
      setZoomLevel((prev) => Math.min(8.0, Math.max(3.0, prev + delta)));
    };

    const domElement = renderer.domElement;
    domElement.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    domElement.addEventListener("wheel", onWheel, { passive: false });

    // --- Viewport Resize Observer ---
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

    let animId: number;
    const startTime = performance.now();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = (performance.now() - startTime) * 0.001;

      if (resetRequestedRef.current) {
        mainGroup.rotation.x += (0.0 - mainGroup.rotation.x) * 0.12;
        mainGroup.rotation.y += (0.0 - mainGroup.rotation.y) * 0.12;
        camera.position.z += (5.2 - camera.position.z) * 0.12;
        if (
          Math.abs(mainGroup.rotation.x) < 0.005 &&
          Math.abs(mainGroup.rotation.y) < 0.005 &&
          Math.abs(camera.position.z - 5.2) < 0.01
        ) {
          mainGroup.rotation.set(0, 0, 0);
          camera.position.z = 5.2;
          resetRequestedRef.current = false;
        }
      } else if (!isDragging) {
        mainGroup.rotation.y += velocity.x * 0.90;
        mainGroup.rotation.x += velocity.y * 0.90;
        velocity.x *= 0.90;
        velocity.y *= 0.90;

        if (isAutoRotatingRef.current) {
          if (rotationModeRef.current === "sway") {
            const targetY = Math.sin(elapsed * 0.40) * 0.18;
            mainGroup.rotation.y += (targetY - mainGroup.rotation.y) * 0.04;
          } else {
            mainGroup.rotation.y += 0.002;
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
      if (matARef.current?.uniforms.tFundus.value) {
        matARef.current.uniforms.tFundus.value.dispose();
      }
      if (matBRef.current?.uniforms.tFundus.value) {
        matBRef.current.uniforms.tFundus.value.dispose();
      }
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.LineSegments || obj instanceof THREE.Points || obj instanceof THREE.Line) {
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

  // --- Dynamic Texture A Updates ---
  useEffect(() => {
    if (!matARef.current) return;
    if (fundusABase64) {
      new THREE.TextureLoader().load(
        fundusABase64,
        (tex) => {
          if (!matARef.current) return;
          tex.minFilter = THREE.LinearFilter;
          tex.magFilter = THREE.LinearFilter;
          if (matARef.current.uniforms.tFundus.value) {
            matARef.current.uniforms.tFundus.value.dispose();
          }
          matARef.current.uniforms.tFundus.value = tex;
          matARef.current.uniforms.hasFundus.value = 1.0;
          matARef.current.needsUpdate = true;
        },
        undefined,
        () => {
          if (!matARef.current) return;
          if (matARef.current.uniforms.tFundus.value) {
            matARef.current.uniforms.tFundus.value.dispose();
            matARef.current.uniforms.tFundus.value = null;
          }
          matARef.current.uniforms.hasFundus.value = 0.0;
          matARef.current.needsUpdate = true;
        }
      );
    } else {
      if (matARef.current.uniforms.tFundus.value) {
        matARef.current.uniforms.tFundus.value.dispose();
        matARef.current.uniforms.tFundus.value = null;
      }
      matARef.current.uniforms.hasFundus.value = 0.0;
      matARef.current.needsUpdate = true;
    }
  }, [fundusABase64]);

  // --- Dynamic Texture B Updates ---
  useEffect(() => {
    if (!matBRef.current) return;
    const textureToLoadB = perturbedFundusBBase64 || fundusBBase64;
    if (textureToLoadB) {
      new THREE.TextureLoader().load(
        textureToLoadB,
        (tex) => {
          if (!matBRef.current) return;
          tex.minFilter = THREE.LinearFilter;
          tex.magFilter = THREE.LinearFilter;
          if (matBRef.current.uniforms.tFundus.value) {
            matBRef.current.uniforms.tFundus.value.dispose();
          }
          matBRef.current.uniforms.tFundus.value = tex;
          matBRef.current.uniforms.hasFundus.value = 1.0;
          matBRef.current.needsUpdate = true;
        },
        undefined,
        () => {
          if (!matBRef.current) return;
          if (matBRef.current.uniforms.tFundus.value) {
            matBRef.current.uniforms.tFundus.value.dispose();
            matBRef.current.uniforms.tFundus.value = null;
          }
          matBRef.current.uniforms.hasFundus.value = 0.0;
          matBRef.current.needsUpdate = true;
        }
      );
    } else {
      if (matBRef.current.uniforms.tFundus.value) {
        matBRef.current.uniforms.tFundus.value.dispose();
        matBRef.current.uniforms.tFundus.value = null;
      }
      matBRef.current.uniforms.hasFundus.value = 0.0;
      matBRef.current.needsUpdate = true;
    }
  }, [fundusBBase64, perturbedFundusBBase64]);

  // --- Dynamic Constellation & Laser Correspondence Beams ---
  useEffect(() => {
    if (!networkGroupRef.current) return;
    const netGroup = networkGroupRef.current;

    while (netGroup.children.length > 0) {
      const child = netGroup.children[0];
      if (child instanceof THREE.Mesh || child instanceof THREE.Points || child instanceof THREE.LineSegments || child instanceof THREE.Line) {
        child.geometry?.dispose();
        if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
        else child.material?.dispose();
      }
      netGroup.remove(child);
    }

    // --- Coordinate Conversion Matching 2.7x2.7 Plane ---
    const mapPixelTo3D = (px: number, py: number, widthImg: number, heightImg: number, xCenterOffset: number) => {
      const u = px / Math.max(1, widthImg);
      const v = py / Math.max(1, heightImg);
      const x = (u - 0.5) * 2.7 + xCenterOffset;
      const y = (0.5 - v) * 2.7;
      return new THREE.Vector3(x, y, 0.0);
    };

    const buildDelaunayNetwork = (
      template: TemplateSummary,
      xOffset: number,
      nodeColor: number,
      wireColor: number,
      matchedTrianglesVerts: [number, number][][],
    ) => {
      const group = new THREE.Group();
      const wImg = template.width || 565;
      const hImg = template.height || 584;

      const ptGeo = new THREE.BufferGeometry();
      const ptPos: number[] = [];
      template.keypoints.forEach((pt) => {
        const v = mapPixelTo3D(pt.x, pt.y, wImg, hImg, xOffset);
        ptPos.push(v.x, v.y, v.z + 0.02);
      });
      ptGeo.setAttribute("position", new THREE.Float32BufferAttribute(ptPos, 3));
      const ptMat = new THREE.PointsMaterial({
        color: nodeColor,
        size: 0.075,
        transparent: true,
        opacity: 0.95,
      });
      group.add(new THREE.Points(ptGeo, ptMat));

      const standardLines: number[] = [];
      const matchedLines: number[] = [];

      template.triangles.forEach((tri) => {
        const v0 = mapPixelTo3D(tri.vertices[0][0], tri.vertices[0][1], wImg, hImg, xOffset);
        const v1 = mapPixelTo3D(tri.vertices[1][0], tri.vertices[1][1], wImg, hImg, xOffset);
        const v2 = mapPixelTo3D(tri.vertices[2][0], tri.vertices[2][1], wImg, hImg, xOffset);

        standardLines.push(v0.x, v0.y, v0.z + 0.02, v1.x, v1.y, v1.z + 0.02);
        standardLines.push(v1.x, v1.y, v1.z + 0.02, v2.x, v2.y, v2.z + 0.02);
        standardLines.push(v2.x, v2.y, v2.z + 0.02, v0.x, v0.y, v0.z + 0.02);
      });

      matchedTrianglesVerts.forEach((verts) => {
        if (verts && verts.length >= 3) {
          const v0 = mapPixelTo3D(verts[0][0], verts[0][1], wImg, hImg, xOffset);
          const v1 = mapPixelTo3D(verts[1][0], verts[1][1], wImg, hImg, xOffset);
          const v2 = mapPixelTo3D(verts[2][0], verts[2][1], wImg, hImg, xOffset);

          matchedLines.push(v0.x, v0.y, v0.z + 0.03, v1.x, v1.y, v1.z + 0.03);
          matchedLines.push(v1.x, v1.y, v1.z + 0.03, v2.x, v2.y, v2.z + 0.03);
          matchedLines.push(v2.x, v2.y, v2.z + 0.03, v0.x, v0.y, v0.z + 0.03);
        }
      });

      if (standardLines.length > 0) {
        const stdGeo = new THREE.BufferGeometry();
        stdGeo.setAttribute("position", new THREE.Float32BufferAttribute(standardLines, 3));
        const stdMat = new THREE.LineBasicMaterial({ color: wireColor, transparent: true, opacity: 0.35 });
        group.add(new THREE.LineSegments(stdGeo, stdMat));
      }

      if (matchedLines.length > 0) {
        const matchGeo = new THREE.BufferGeometry();
        matchGeo.setAttribute("position", new THREE.Float32BufferAttribute(matchedLines, 3));
        const matchMat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.95 });
        group.add(new THREE.LineSegments(matchGeo, matchMat));
      }

      return group;
    };

    if (templateA && templateA.keypoints.length > 0) {
      const meshNetA = buildDelaunayNetwork(
        templateA,
        -1.65,
        0x06b6d4,
        0x0284c7,
        matchedPairs.map((p) => p.vertices_a)
      );
      netGroup.add(meshNetA);
    }

    if (templateB && templateB.keypoints.length > 0) {
      const meshNetB = buildDelaunayNetwork(
        templateB,
        1.65,
        0xf43f5e,
        0xe11d48,
        matchedPairs.map((p) => p.vertices_b)
      );
      netGroup.add(meshNetB);
    }

    if (matchedPairs.length > 0 && templateA && templateB) {
      const wA = templateA.width || 565;
      const hA = templateA.height || 584;
      const wB = templateB.width || 565;
      const hB = templateB.height || 584;

      matchedPairs.forEach((pair) => {
        const pA = mapPixelTo3D(pair.centroid_a[0], pair.centroid_a[1], wA, hA, -1.65);
        const pB = mapPixelTo3D(pair.centroid_b[0], pair.centroid_b[1], wB, hB, 1.65);
        const mid = pA.clone().add(pB).multiplyScalar(0.5);
        mid.z += 0.55;

        const curve = new THREE.QuadraticBezierCurve3(pA, mid, pB);
        const pts = curve.getPoints(24);
        const arcGeo = new THREE.BufferGeometry().setFromPoints(pts);
        const arcMat = new THREE.LineBasicMaterial({
          color: isMatch ? 0x10b981 : 0xf59e0b,
          transparent: true,
          opacity: 0.85,
        });
        netGroup.add(new THREE.Line(arcGeo, arcMat));
      });
    }
  }, [templateA, templateB, matchedPairs, isMatch]);

  return (
    <div className="relative w-full h-full min-h-[500px] flex items-center justify-center overflow-hidden select-none">
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating 3D Camera & Motion Controls (Bottom Center) */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20 flex items-center space-x-2 bg-gray-950/85 backdrop-blur-xl px-4 py-2 rounded-2xl border border-gray-800/80 shadow-2xl text-xs font-mono">
        <button
          onClick={() => setIsAutoRotating(!isAutoRotating)}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border transition-all cursor-pointer ${
            isAutoRotating
              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/50 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
              : "bg-gray-900/80 text-gray-400 border-gray-800 hover:text-gray-200"
          }`}
          title="Toggle 3D Constellation Orbit"
        >
          {isAutoRotating ? <Pause className="w-3.5 h-3.5 text-cyan-400" /> : <Play className="w-3.5 h-3.5" />}
          <span>{isAutoRotating ? "Orbit" : "Paused"}</span>
        </button>

        <button
          onClick={() => setRotationMode(rotationMode === "sway" ? "turntable" : "sway")}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:text-cyan-300 hover:border-gray-700 text-gray-400 transition-all cursor-pointer"
          title="Switch Motion: Horizontal Sway vs 360 Spin"
        >
          <Compass className="w-3.5 h-3.5 text-sky-400" />
          <span>{rotationMode === "sway" ? "Sway ±10°" : "360° Spin"}</span>
        </button>

        <div className="h-4 w-px bg-gray-800" />

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

        <button
          onClick={handleResetCamera}
          className="p-1.5 rounded-lg border border-gray-800 bg-gray-900/60 hover:text-cyan-300 hover:border-gray-700 text-gray-400 transition-all cursor-pointer"
          title="Reset Constellation View"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}
