# VERA 3D WebGL Ophthalmic & Biometric Studio

Clinical retinal visualization and biometric identity authentication frontend built with Next.js 16 (App Router), React 19, Three.js (r183+), and Tailwind CSS.

---

## 1. Architectural Overview

The studio interfaces directly with the FastAPI backend through a three-stage clinical workflow:

1. **Stage 01: 3D Retinal Anatomy & Vessels (`RetinalGlobe.tsx`)**
   - Ray-marched topological depth extrusion of fundus topography on a spherical eye globe.
   - Interactive vessel probability maps, pruned skeleton overlay with exploded 3D depth stack.
   - Optic Disc (OD) and Fovea Centralis landmark beacons with anatomical distance arcs.
   - 2D Clinical Inspector drawer showing RGB source, probability map, refined mask, and pruned skeleton.

2. **Stage 02: Biometric Identity Verification (`DelaunayConstellation.tsx`)**
   - Dual-plane Three.js WebGL viewport comparing reference (Eye A) and target (Eye B) scans.
   - Real-time Delaunay triangulation wireframes and Rutovitz crossing-number bifurcation nodes.
   - Green highlighted matched triangles with animated laser correspondence alignment beams.
   - 11-parameter precision physical degradation suite (Head tilt, scale, blur, cataract haze, glare, ISO noise, ptosis occlusion, gamma, warp, IOP elasticity).

3. **Stage 03: Population Cohort Matrix & Stress Dashboard**
   - 4x4 cross-subject identity matrix across DRIVE subjects (S01, S02, S23, S24).
   - Dual regime toggle: Clean Baseline and 7-Parameter Judge Torture.
   - Interactive drill-down to Stage 02 and real-time live cohort batch verification runner.
   - 8-vector stress resilience telemetry with 0.00% Equal Error Rate (EER).

---

## 2. Configuration & Environment

The frontend connects to the backend API via `NEXT_PUBLIC_API_URL`:

```bash
# Default fallback: http://127.0.0.1:8000
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

## 3. Development & Build Scripts

Execute commands from the `web/` directory:

```bash
# Install dependencies
npm install

# Start Next.js development server (port 3000)
npm run dev

# Run TypeScript strict type-check
npx tsc --noEmit

# Run ESLint validation
npm run lint

# Production build and optimization
npm run build
```

---

## 4. Source Structure

```
web/
├── src/
│   ├── app/
│   │   ├── globals.css         # Clinical color tokens and scanline styles
│   │   ├── layout.tsx          # Root layout and metadata configuration
│   │   └── page.tsx            # Main 3-stage studio dashboard and state machine
│   ├── components/
│   │   ├── DelaunayConstellation.tsx # WebGL dual-plane constellation and laser beams
│   │   └── RetinalGlobe.tsx    # WebGL spherical eye globe and GLSL shader
│   ├── lib/
│   │   └── api.ts              # REST API client and clinical image upload helper
│   └── types/
│       └── biometrics.ts       # TypeScript data contracts matching backend schemas
├── next.config.ts              # Turbopack and build configuration
├── package.json                # Project dependencies
└── tsconfig.json               # Strict TypeScript configuration
```
