"use client";

import React, { useEffect, useState, useRef } from "react";
import RetinalGlobe from "@/components/RetinalGlobe";
import DelaunayConstellation from "@/components/DelaunayConstellation";
import {
  fetchHealth,
  fetchPresets,
  requestLocalization,
  requestMatch,
  requestSegmentation,
  uploadClinicalImage,
} from "@/lib/api";
import {
  LocalizationResult,
  MatchResult,
  PresetItem,
  SegmentationResult,
} from "@/types/biometrics";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Cpu,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Layers,
  Network,
  Play,
  RefreshCw,
  RotateCw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  UploadCloud,
  X,
  Zap,
} from "lucide-react";

// --- Extensible Workspace Module Registry ---
interface WorkspaceModule {
  id: "01" | "02" | "03";
  stage: "ocular" | "matching" | "cohort";
  shortTitle: string;
  fullTitle: string;
  subtitle: string;
  badge: string;
  isAvailable: boolean;
}

const WORKSPACE_MODULES: WorkspaceModule[] = [
  {
    id: "01",
    stage: "ocular",
    shortTitle: "3D Anatomy",
    fullTitle: "3D Retinal Anatomy & Vessels",
    subtitle: "Topological depth extrusion, skeleton layers, and OD-Fovea axis",
    badge: "STEP 01",
    isAvailable: true,
  },
  {
    id: "02",
    stage: "matching",
    shortTitle: "Identity Match",
    fullTitle: "Biometric Identity Verification",
    subtitle: "Dual Delaunay constellation alignment & 11-parameter judge stress tests",
    badge: "STEP 02",
    isAvailable: true,
  },
  {
    id: "03",
    stage: "cohort",
    shortTitle: "Cohort Matrix",
    fullTitle: "Population Cross-Matching & Stress Matrix",
    subtitle: "Cross-subject confusion matrix, 8-vector stress resilience & live cohort verification",
    badge: "STEP 03",
    isAvailable: true,
  },
];

// --- VERA: Vascular Extraction & Retinal Authentication Platform ---
export default function BiometricDashboard() {
  // --- Telemetry & Infrastructure State ---
  const [serverStatus, setServerStatus] = useState<string>("connecting");
  const [computeDevice, setComputeDevice] = useState<string>("CHECKING...");
  const [presets, setPresets] = useState<PresetItem[]>([]);

  // --- Stage-Specific Preset Selections ---
  const [selectedOcularPresetId, setSelectedOcularPresetId] = useState<string>("ocular_healthy_01");
  const [selectedMatchPresetId, setSelectedMatchPresetId] = useState<string>("match_genuine_01");

  // --- Navigation & Workflow State ---
  const [activeStage, setActiveStage] = useState<"ocular" | "matching" | "cohort">("ocular");

  // --- Additional Image Dimensions & Upload State ---
  const [imageDimA, setImageDimA] = useState<{ width: number; height: number }>({ width: 565, height: 584 });
  const [analyzedThresholdA, setAnalyzedThresholdA] = useState<number | null>(null);
  const [uploadingTarget, setUploadingTarget] = useState<"A" | "B" | null>(null);
  const [customScanA, setCustomScanA] = useState<{ image: string; label: string; width: number; height: number } | null>(null);
  const [customScanB, setCustomScanB] = useState<{ image: string; label: string } | null>(null);
  const sidebarScrollRef = useRef<HTMLDivElement | null>(null);

  // --- Stage 1: Ocular Anatomy State ---
  const [fundusA, setFundusA] = useState<string>("");
  const [segResultA, setSegResultA] = useState<SegmentationResult | null>(null);
  const [locResultA, setLocResultA] = useState<LocalizationResult | null>(null);
  const [segThreshold, setSegThreshold] = useState<number>(0.60);
  const [reliefScale, setReliefScale] = useState<number>(0.16);
  const [explodedLayers, setExplodedLayers] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [showAdvancedShader, setShowAdvancedShader] = useState<boolean>(false);
  const [isProcessingA, setIsProcessingA] = useState<boolean>(false);

  // --- 2D Quad-View Inspector State ---
  const [show2DInspector, setShow2DInspector] = useState<boolean>(false);

  // --- Stage 2: Biometric Matching State ---
  const [fundusB, setFundusB] = useState<string>("");
  const [labelA, setLabelA] = useState<string>("Subject 01 (Original)");
  const [labelB, setLabelB] = useState<string>("Subject 01 (Target)");
  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [isMatching, setIsMatching] = useState<boolean>(false);
  const [showAdvancedPerturb, setShowAdvancedPerturb] = useState<boolean>(false);

  // --- 11-Parameter Perturbation Suite ---
  const [rotateDegB, setRotateDegB] = useState<number>(0.0);
  const [scaleFactorB, setScaleFactorB] = useState<number>(1.0);
  const [brightnessB, setBrightnessB] = useState<number>(1.0);
  const [blurSigmaB, setBlurSigmaB] = useState<number>(0.0);
  const [hazeFactorB, setHazeFactorB] = useState<number>(0.0);
  const [glareIntensityB, setGlareIntensityB] = useState<number>(0.0);
  const [noiseSigmaB, setNoiseSigmaB] = useState<number>(0.0);
  const [occlusionFracB, setOcclusionFracB] = useState<number>(0.0);
  const [gammaB, setGammaB] = useState<number>(1.0);
  const [sphericalWarpB, setSphericalWarpB] = useState<number>(0.0);
  const [elasticAlphaB, setElasticAlphaB] = useState<number>(0.0);

  // --- Active Stress Preset Tag ---
  const [activeStressChip, setActiveStressChip] = useState<string>("baseline");

  // --- Stage 3: Cohort Matrix & Benchmark State ---
  const [cohortRegime, setCohortRegime] = useState<"baseline" | "torture">("baseline");
  const [isRunningCohort, setIsRunningCohort] = useState<boolean>(false);
  const [cohortProgress, setCohortProgress] = useState<{ current: number; total: number } | null>(null);
  const [liveCohortOverrides, setLiveCohortOverrides] = useState<
    Record<string, { score: number; isMatch: boolean; pairs: number; latencyMs: number }>
  >({});

  // --- Notification Banner State ---
  const [statusBanner, setStatusBanner] = useState<{ text: string; type: "info" | "success" | "error" } | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showToast = (text: string, type: "info" | "success" | "error" = "info") => {
    setStatusBanner({ text, type });
    if (toastTimerRef.current) {
      clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = setTimeout(() => {
      setStatusBanner(null);
      toastTimerRef.current = null;
    }, 5000);
  };

  // --- Perturbation Helper & Reset Handler ---
  const updatePerturbation = (setter: (v: number) => void, val: number) => {
    setter(val);
    setActiveStressChip("custom");
    setMatchResult(null);
  };

  const resetAllPerturbations = () => {
    setRotateDegB(0.0);
    setScaleFactorB(1.0);
    setBrightnessB(1.0);
    setBlurSigmaB(0.0);
    setHazeFactorB(0.0);
    setGlareIntensityB(0.0);
    setNoiseSigmaB(0.0);
    setOcclusionFracB(0.0);
    setGammaB(1.0);
    setSphericalWarpB(0.0);
    setElasticAlphaB(0.0);
    setMatchResult(null);
  };

  // --- Preset Label Helper ---
  const getMatchPresetLabels = (preset: PresetItem): { labelA: string; labelB: string } => {
    if (preset.scenario === "genuine") {
      return { labelA: "Subject 01 (Reference)", labelB: "Subject 01 (Target)" };
    } else if (preset.scenario === "impostor") {
      return { labelA: "Subject 01 (Baseline)", labelB: "Subject 02 (Impostor)" };
    } else if (preset.scenario === "sparse_clinical") {
      return { labelA: "Subject 23 (Reference)", labelB: "Subject 23 (Target)" };
    } else if (preset.scenario === "dense_clinical") {
      return { labelA: "Subject 24 (Reference)", labelB: "Subject 24 (Target)" };
    }
    return { labelA: "Reference Eye (A)", labelB: "Target Eye (B)" };
  };

  // --- Backend Health & Preset Synchronization ---
  const syncBackendConnection = React.useCallback(async () => {
    try {
      const h = await fetchHealth();
      setServerStatus("online");
      setComputeDevice(h.device.toUpperCase());

      setPresets((prevPresets) => {
        if (prevPresets.length === 0) {
          fetchPresets()
            .then((data) => {
              if (data.length > 0) {
                setPresets(data);
                const ocularInitial = data.find((p) => p.target_stage === "ocular") || data[0];
                const matchInitial = data.find((p) => p.target_stage === "matching") || data[0];

                setSelectedOcularPresetId(ocularInitial.id);
                setSelectedMatchPresetId(matchInitial.id);

                setFundusA((currA) => currA || ocularInitial.image_base64);
                setLabelA((currL) => (currL === "Subject 01 (Original)" ? ocularInitial.title : currL));

                if (matchInitial.image_b_base64) {
                  setFundusB((currB) => currB || matchInitial.image_b_base64 || "");
                  setLabelB((currL) => (currL === "Subject 01 (Target)" ? "Subject 01 (Target)" : currL));
                }
              }
            })
            .catch(() => {});
        }
        return prevPresets;
      });
    } catch {
      setServerStatus("offline");
      setComputeDevice("OFFLINE");
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    syncBackendConnection();
    const interval = setInterval(() => {
      if (serverStatus !== "online" || presets.length === 0) {
        syncBackendConnection();
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [serverStatus, presets.length, syncBackendConnection]);

  // --- Specimen Synchronization Handler ---
  const handleStageSwitch = (newStage: "ocular" | "matching" | "cohort") => {
    setActiveStage(newStage);
    if (sidebarScrollRef.current) {
      sidebarScrollRef.current.scrollTop = 0;
    }

    if (newStage === "ocular") {
      if (selectedOcularPresetId === "custom" && customScanA) {
        setFundusA(customScanA.image);
        setLabelA(customScanA.label);
        setImageDimA({ width: customScanA.width, height: customScanA.height });
      } else {
        const ocPreset =
          presets.find((p) => p.id === selectedOcularPresetId && p.target_stage === "ocular") ||
          presets.find((p) => p.target_stage === "ocular");
        if (ocPreset) {
          setFundusA(ocPreset.image_base64);
          setLabelA(ocPreset.title);
          setImageDimA({ width: 565, height: 584 });
        }
      }
    } else if (newStage === "matching") {
      if (selectedMatchPresetId === "custom" || customScanA) {
        if (customScanA) {
          setFundusA(customScanA.image);
          setLabelA(customScanA.label);
          if (customScanB) {
            setFundusB(customScanB.image);
            setLabelB(customScanB.label);
          } else {
            setFundusB(customScanA.image);
            setLabelB(`${customScanA.label} (Target)`);
          }
        }
      } else {
        const mtPreset =
          presets.find((p) => p.id === selectedMatchPresetId && p.target_stage === "matching") ||
          presets.find((p) => p.target_stage === "matching");
        if (mtPreset) {
          setFundusA(mtPreset.image_base64);
          setFundusB(mtPreset.image_b_base64 || mtPreset.image_base64);
          const labels = getMatchPresetLabels(mtPreset);
          setLabelA(labels.labelA);
          setLabelB(labels.labelB);
        }
      }
    }

    if (newStage !== "ocular") {
      setShow2DInspector(false);
    }
  };

  // --- Cohort Subject & Matrix Definitions ---
  const COHORT_SUBJECTS = [
    { id: "s01", code: "S01", name: "Subject 01", subtitle: "DRIVE 21 Normal", presetId: "match_genuine_01" },
    { id: "s02", code: "S02", name: "Subject 02", subtitle: "DRIVE 22 Impostor", presetId: "match_impostor_01" },
    { id: "s23", code: "S23", name: "Subject 23", subtitle: "DRIVE 23 Sparse", presetId: "match_sparse_23" },
    { id: "s24", code: "S24", name: "Subject 24", subtitle: "DRIVE 24 Dense", presetId: "match_dense_24" },
  ];

  const COHORT_BENCHMARK_MATRIX: Record<
    "baseline" | "torture",
    Record<string, { score: number; isMatch: boolean; pairs: number }>
  > = {
    baseline: {
      "s01-s01": { score: 1.0, isMatch: true, pairs: 101 },
      "s01-s02": { score: 0.0, isMatch: false, pairs: 0 },
      "s01-s23": { score: 0.0, isMatch: false, pairs: 0 },
      "s01-s24": { score: 0.0, isMatch: false, pairs: 0 },
      "s02-s01": { score: 0.0, isMatch: false, pairs: 0 },
      "s02-s02": { score: 1.0, isMatch: true, pairs: 129 },
      "s02-s23": { score: 0.0, isMatch: false, pairs: 0 },
      "s02-s24": { score: 0.0, isMatch: false, pairs: 0 },
      "s23-s01": { score: 0.0, isMatch: false, pairs: 0 },
      "s23-s02": { score: 0.0, isMatch: false, pairs: 0 },
      "s23-s23": { score: 1.0, isMatch: true, pairs: 58 },
      "s23-s24": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s01": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s02": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s23": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s24": { score: 1.0, isMatch: true, pairs: 142 },
    },
    torture: {
      "s01-s01": { score: 0.634, isMatch: true, pairs: 51 },
      "s01-s02": { score: 0.0, isMatch: false, pairs: 0 },
      "s01-s23": { score: 0.0, isMatch: false, pairs: 0 },
      "s01-s24": { score: 0.0, isMatch: false, pairs: 0 },
      "s02-s01": { score: 0.0, isMatch: false, pairs: 0 },
      "s02-s02": { score: 0.452, isMatch: true, pairs: 37 },
      "s02-s23": { score: 0.0, isMatch: false, pairs: 0 },
      "s02-s24": { score: 0.0, isMatch: false, pairs: 0 },
      "s23-s01": { score: 0.0, isMatch: false, pairs: 0 },
      "s23-s02": { score: 0.0, isMatch: false, pairs: 0 },
      "s23-s23": { score: 0.473, isMatch: true, pairs: 18 },
      "s23-s24": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s01": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s02": { score: 0.128, isMatch: false, pairs: 11 },
      "s24-s23": { score: 0.0, isMatch: false, pairs: 0 },
      "s24-s24": { score: 0.437, isMatch: true, pairs: 40 },
    },
  };

  const STRESS_VECTOR_BENCHMARKS = [
    { key: "baseline", label: "Clean Baseline (0°, 1.0x)", score: 1.0, isMatch: true, desc: "Reference Ground-Truth" },
    { key: "pose", label: "Tilt +25° & Zoom 0.85x", score: 0.808, isMatch: true, desc: "Affine Head Movement" },
    { key: "cataract", label: "Cataract Haze 45% + Low Gamma", score: 0.864, isMatch: true, desc: "Media Opacity & Contrast Loss" },
    { key: "glare", label: "Corneal Glare 65% & Dim 0.8x", score: 0.867, isMatch: true, desc: "Specular Reflection & Low Exposure" },
    { key: "blur", label: "Optical Defocus Blur (sigma=2.5px)", score: 0.827, isMatch: true, desc: "Imperfect Focus & Astigmatism" },
    { key: "noise", label: "Low-Light ISO Noise (sigma=16.0)", score: 0.727, isMatch: true, desc: "High Sensor Gain & Electronic Grain" },
    { key: "eyelid", label: "Eyelid Occlusion 22%", score: 0.864, isMatch: true, desc: "Blink & Ptosis Artifacts" },
    { key: "torture", label: "Judge Torture Multi-Factor (All 7)", score: 0.634, isMatch: true, desc: "Worst-Case Multi-Vector Stress" },
  ];

  // --- Normalized Cohort Subject Retrieval ---
  const getCohortSubjectData = (subjId: string): { image: string; label: string } => {
    switch (subjId) {
      case "s01": {
        const p = presets.find((item) => item.id === "match_genuine_01");
        return {
          image: p?.image_base64 || "",
          label: "Subject 01 (DRIVE 21 Normal)",
        };
      }
      case "s02": {
        const p = presets.find((item) => item.id === "match_impostor_01");
        return {
          image: p?.image_b_base64 || p?.image_base64 || "",
          label: "Subject 02 (DRIVE 22 Impostor)",
        };
      }
      case "s23": {
        const p = presets.find((item) => item.id === "match_sparse_23");
        return {
          image: p?.image_base64 || "",
          label: "Subject 23 (DRIVE 23 Sparse)",
        };
      }
      case "s24": {
        const p = presets.find((item) => item.id === "match_dense_24");
        return {
          image: p?.image_base64 || "",
          label: "Subject 24 (DRIVE 24 Dense)",
        };
      }
      default:
        return { image: "", label: subjId.toUpperCase() };
    }
  };

  // --- Cohort Cell Drill-Down Handler ---
  const handleDrillDownFromCohort = (subjAId: string, subjBId: string, regimeOverride?: "baseline" | "torture") => {
    const targetRegime = regimeOverride || cohortRegime;
    const dataA = getCohortSubjectData(subjAId);
    const dataB = getCohortSubjectData(subjBId);

    if (dataA.image) {
      setFundusA(dataA.image);
      setLabelA(dataA.label);
    }
    if (dataB.image) {
      setFundusB(dataB.image);
      setLabelB(dataB.label);
    }

    const pairKey = `${subjAId}-${subjBId}`;
    if (pairKey === "s01-s01") {
      setSelectedMatchPresetId("match_genuine_01");
    } else if (pairKey === "s01-s02") {
      setSelectedMatchPresetId("match_impostor_01");
    } else if (pairKey === "s23-s23") {
      setSelectedMatchPresetId("match_sparse_23");
    } else if (pairKey === "s24-s24") {
      setSelectedMatchPresetId("match_dense_24");
    } else {
      setSelectedMatchPresetId("custom");
    }
    setCustomScanA(null);
    setCustomScanB(null);

    if (targetRegime === "torture") {
      applyJudgeStressPreset("torture");
    } else {
      applyJudgeStressPreset("baseline");
    }

    setMatchResult(null);
    setActiveStage("matching");
    showToast(
      `Drill-Down Loaded: ${subjAId.toUpperCase()} vs ${subjBId.toUpperCase()} under ${targetRegime.toUpperCase()} regime. Click 'Compare & Verify Identity' to inspect Delaunay graph.`,
      "info"
    );
  };

  // --- Live Cohort Cross-Evaluation ---
  const handleRunLiveCohort = async () => {
    setIsRunningCohort(true);
    showToast(`Evaluating Cohort Cross-Matching under ${cohortRegime.toUpperCase()} regime...`, "info");

    const pairs: Array<{ aId: string; bId: string }> =
      cohortRegime === "torture"
        ? [
            { aId: "s01", bId: "s01" },
            { aId: "s01", bId: "s02" },
            { aId: "s01", bId: "s23" },
            { aId: "s01", bId: "s24" },
            { aId: "s02", bId: "s01" },
            { aId: "s02", bId: "s02" },
            { aId: "s02", bId: "s23" },
            { aId: "s02", bId: "s24" },
            { aId: "s23", bId: "s01" },
            { aId: "s23", bId: "s02" },
            { aId: "s23", bId: "s23" },
            { aId: "s23", bId: "s24" },
            { aId: "s24", bId: "s01" },
            { aId: "s24", bId: "s02" },
            { aId: "s24", bId: "s23" },
            { aId: "s24", bId: "s24" },
          ]
        : [
            { aId: "s01", bId: "s01" },
            { aId: "s02", bId: "s02" },
            { aId: "s23", bId: "s23" },
            { aId: "s24", bId: "s24" },
            { aId: "s01", bId: "s02" },
            { aId: "s01", bId: "s23" },
            { aId: "s01", bId: "s24" },
            { aId: "s02", bId: "s23" },
            { aId: "s02", bId: "s24" },
            { aId: "s23", bId: "s24" },
          ];

    setCohortProgress({ current: 0, total: pairs.length });
    const overrides = { ...liveCohortOverrides };

    for (let i = 0; i < pairs.length; i++) {
      const { aId, bId } = pairs[i];
      setCohortProgress({ current: i + 1, total: pairs.length });
      const dataA = getCohortSubjectData(aId);
      const dataB = getCohortSubjectData(bId);

      if (!dataA.image || !dataB.image) continue;

      // eslint-disable-next-line react-hooks/purity
      const t0 = performance.now();
      try {
        const res = await requestMatch(
          dataA.image,
          dataB.image,
          0.05,
          0.08,
          cohortRegime === "torture" ? 15.0 : 0.0,
          cohortRegime === "torture" ? 0.90 : 1.0,
          1.0,
          cohortRegime === "torture"
            ? {
                blur_sigma_b: 1.8,
                haze_factor_b: 0.30,
                glare_intensity_b: 0.35,
                noise_sigma_b: 10.0,
                elastic_alpha_b: 2.0,
              }
            : {}
        );
        // eslint-disable-next-line react-hooks/purity
        const elapsed = Math.round(performance.now() - t0);
        const cellData = {
          score: res.similarity_score,
          isMatch: res.is_match,
          pairs: res.matched_pairs.length,
          latencyMs: elapsed,
        };
        overrides[`${cohortRegime}:${aId}-${bId}`] = cellData;
        if (cohortRegime === "baseline" && aId !== bId) {
          overrides[`${cohortRegime}:${bId}-${aId}`] = cellData;
        }
        setLiveCohortOverrides({ ...overrides });
      } catch (err) {
        console.error("Cohort eval failed for pair:", aId, bId, err);
      }
    }
    setLiveCohortOverrides(overrides);
    setIsRunningCohort(false);
    setCohortProgress(null);
    showToast(`Live Cohort verification completed across matrix cells!`, "success");
  };

  // --- Stage 1: Specimen Preset Selection ---
  const handleSelectOcularPreset = (preset: PresetItem) => {
    setSelectedOcularPresetId(preset.id);
    setCustomScanA(null);
    setImageDimA({ width: 565, height: 584 });
    setFundusA(preset.image_base64);
    setLabelA(preset.title);
    setSegResultA(null);
    setLocResultA(null);
    setAnalyzedThresholdA(null);
    showToast(`Loaded anatomical specimen: ${preset.title}`, "info");
  };

  // --- Stage 2: Scenario Preset Selection ---
  const handleSelectMatchPreset = (preset: PresetItem) => {
    setSelectedMatchPresetId(preset.id);
    setCustomScanA(null);
    setCustomScanB(null);
    setFundusA(preset.image_base64);
    setFundusB(preset.image_b_base64 || preset.image_base64);
    setMatchResult(null);
    resetAllPerturbations();
    setActiveStressChip("baseline");

    const labels = getMatchPresetLabels(preset);
    setLabelA(labels.labelA);
    setLabelB(labels.labelB);

    showToast(`Loaded biometric pair: ${preset.title}`, "info");
  };

  // --- Clinical Stress Presets ---
  const applyJudgeStressPreset = (key: string) => {
    setActiveStressChip(key);
    setMatchResult(null);
    switch (key) {
      case "baseline":
        resetAllPerturbations();
        showToast("Applied: Clean Baseline (0°, 1.0x)", "info");
        break;
      case "pose":
        resetAllPerturbations();
        setRotateDegB(25.0);
        setScaleFactorB(0.85);
        showToast("Applied: Tilt +25° & Zoom 0.85x", "info");
        break;
      case "cataract":
        resetAllPerturbations();
        setHazeFactorB(0.45);
        setGammaB(0.85);
        showToast("Applied: Cataract Haze 45% + Low Gamma", "info");
        break;
      case "glare":
        resetAllPerturbations();
        setGlareIntensityB(0.65);
        setBrightnessB(0.80);
        showToast("Applied: Corneal Glare 65% & Dim 0.8x", "info");
        break;
      case "blur":
        resetAllPerturbations();
        setBlurSigmaB(2.5);
        showToast("Applied: Optical Defocus Blur (sigma=2.5px)", "info");
        break;
      case "noise":
        resetAllPerturbations();
        setNoiseSigmaB(16.0);
        setBrightnessB(0.85);
        showToast("Applied: Low-Light ISO Noise (sigma=16.0)", "info");
        break;
      case "eyelid":
        resetAllPerturbations();
        setOcclusionFracB(0.22);
        showToast("Applied: Eyelid Occlusion 22%", "info");
        break;
      case "torture":
        setRotateDegB(15.0);
        setScaleFactorB(0.90);
        setHazeFactorB(0.30);
        setBlurSigmaB(1.8);
        setNoiseSigmaB(10.0);
        setGlareIntensityB(0.35);
        setElasticAlphaB(2.0);
        setBrightnessB(1.0);
        setOcclusionFracB(0.0);
        setGammaB(1.0);
        setSphericalWarpB(0.0);
        showToast("Applied: Combined Multi-Factor Judge Stress", "info");
        break;
    }
  };

  // --- Custom Image Upload Handler ---
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, target: "A" | "B") => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    setUploadingTarget(target);
    showToast(`Normalizing clinical image format (${file.name})...`, "info");

    try {
      const norm = await uploadClinicalImage(file, 768);
      if (target === "A") {
        setFundusA(norm.image_base64);
        setImageDimA({ width: norm.width, height: norm.height });
        setLabelA(file.name);
        setCustomScanA({ image: norm.image_base64, label: file.name, width: norm.width, height: norm.height });
        setSegResultA(null);
        setLocResultA(null);
        setAnalyzedThresholdA(null);
        setMatchResult(null);

        if (activeStage === "ocular") {
          setSelectedOcularPresetId("custom");
        } else {
          setSelectedMatchPresetId("custom");
        }
        showToast(
          `Loaded ${file.name} (${norm.original_width}x${norm.original_height} -> ${norm.width}x${norm.height}px)`,
          "success"
        );
      } else {
        setFundusB(norm.image_base64);
        setLabelB(file.name);
        setCustomScanB({ image: norm.image_base64, label: file.name });
        setMatchResult(null);
        setSelectedMatchPresetId("custom");
        showToast(
          `Loaded ${file.name} (${norm.original_width}x${norm.original_height} -> ${norm.width}x${norm.height}px)`,
          "success"
        );
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Image upload failed";
      showToast(message, "error");
    } finally {
      setUploadingTarget(null);
    }
  };

  // --- AI Pipeline Execution ---
  const handleExecuteInference = async () => {
    if (!fundusA) return;
    setIsProcessingA(true);
    showToast("Analyzing neural retinal vessel topology...", "info");
    try {
      const seg = await requestSegmentation(fundusA, segThreshold, true, 25, 15, true);
      setSegResultA(seg);
      setAnalyzedThresholdA(segThreshold);

      showToast("Localizing Optic Disc & Fovea anatomical landmarks...", "info");
      const loc = await requestLocalization(fundusA, seg.vessel_mask_base64);
      setLocResultA(loc);
      showToast(
        `Analysis Complete: Vessel Density ${(seg.vessel_density * 100).toFixed(1)}% | Bifurcations: ${seg.num_keypoints || 0} | OD-Fovea: ${loc.disc_to_fovea_distance.toFixed(1)}px`,
        "success"
      );
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Pipeline execution failed";
      showToast(message, "error");
    } finally {
      setIsProcessingA(false);
    }
  };

  // --- Biometric Matching Execution ---
  const handleExecuteVerification = async () => {
    if (!fundusA || !fundusB) return;
    setIsMatching(true);
    showToast("Extracting Delaunay Invariants & Computing Hungarian Assignment...", "info");
    try {
      const res = await requestMatch(
        fundusA,
        fundusB,
        0.05,
        0.08,
        rotateDegB,
        scaleFactorB,
        brightnessB,
        {
          blur_sigma_b: blurSigmaB,
          haze_factor_b: hazeFactorB,
          glare_intensity_b: glareIntensityB,
          noise_sigma_b: noiseSigmaB,
          occlusion_frac_b: occlusionFracB,
          gamma_b: gammaB,
          spherical_warp_b: sphericalWarpB,
          elastic_alpha_b: elasticAlphaB,
        }
      );
      setMatchResult(res);
      showToast(
        res.is_match
          ? `IDENTITY VERIFIED: Genuine Match (${(res.similarity_score * 100).toFixed(1)}% similarity)`
          : `IMPOSTOR REJECTED: Identity Mismatch (${(res.similarity_score * 100).toFixed(1)}% similarity)`,
        res.is_match ? "success" : "error"
      );
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Biometric verification failed";
      showToast(message, "error");
    } finally {
      setIsMatching(false);
    }
  };

  const currentPresets = presets.filter((p) => p.target_stage === activeStage);

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#050914] text-gray-100 flex flex-col font-sans select-none">
      {/* 1. Header: VERA Branding + Connected Pipeline Track + Health */}
      <header className="h-16 border-b border-gray-800/80 bg-gray-950/80 backdrop-blur-xl px-6 flex items-center justify-between z-30 shrink-0">
        {/* Brand Identity: Clinical Inline Typography */}
        <div className="flex items-center space-x-3">
          <span className="text-sm font-bold tracking-[0.18em] text-gray-100">VERA</span>
          <div className="h-4 w-[1px] bg-gray-800" />
          <span className="text-[10px] font-mono tracking-widest text-gray-400 font-medium">
            VASCULAR EXTRACTION & RETINAL AUTHENTICATION
          </span>
        </div>

        {/* Extensible Connected Pipeline Track */}
        <nav aria-label="Pipeline Stages" className="flex items-center bg-gray-900/90 p-1 rounded-xl border border-gray-800/80 shadow-xl">
          {WORKSPACE_MODULES.map((module, idx) => {
            const isActive = activeStage === module.stage;
            const isAvailable = module.isAvailable;

            return (
              <React.Fragment key={module.id}>
                {idx > 0 && (
                  <div className="w-4 h-[1px] bg-gray-800 mx-0.5" />
                )}
                <button
                  disabled={!isAvailable}
                  onClick={() => isAvailable && handleStageSwitch(module.stage as "ocular" | "matching" | "cohort")}
                  className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-mono transition-all cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 ${
                    isActive
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_12px_rgba(6,182,212,0.25)] font-semibold"
                      : isAvailable
                      ? "text-gray-400 hover:text-gray-200 border border-transparent hover:bg-gray-800/40"
                      : "text-gray-600 border border-transparent"
                  }`}
                >
                  <span className={`text-[10px] px-1.5 py-0.2 rounded font-bold ${isActive ? "bg-cyan-500/30 text-cyan-200" : "bg-gray-800 text-gray-500"}`}>
                    {module.id}
                  </span>
                  <span className="font-sans font-medium">{module.shortTitle}</span>
                  {!isAvailable && (
                    <span className="text-[9px] px-1 rounded bg-gray-800/80 text-gray-500 uppercase font-mono">
                      Preview
                    </span>
                  )}
                </button>
              </React.Fragment>
            );
          })}
        </nav>

        {/* System Health Indicators */}
        <div className="flex items-center space-x-4 text-xs font-mono">
          <div className="hidden sm:flex items-center space-x-1.5 text-gray-400">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>ACCEL:</span>
            <span className="text-gray-200 font-bold">{computeDevice}</span>
          </div>
          <button
            onClick={() => syncBackendConnection()}
            title="Click to reconnect / refresh system health"
            className="flex items-center space-x-1.5 cursor-pointer"
          >
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span
              className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                serverStatus === "online"
                  ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                  : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
              }`}
            >
              {serverStatus === "online" ? "SYSTEM ONLINE" : "OFFLINE"}
            </span>
          </button>
        </div>
      </header>

      {/* Main Interactive Stage */}
      <div className="flex-1 relative flex overflow-hidden">
        {/* Left Full-Viewport 3D WebGL Canvas */}
        <div className="flex-1 relative h-full w-full bg-[#030611] overflow-hidden">
          {activeStage === "ocular" ? (
            <RetinalGlobe
              fundusImageBase64={fundusA}
              probMapBase64={segResultA?.prob_map_base64}
              skeletonBase64={segResultA?.skeleton_base64}
              landmarks={segResultA?.keypoints || []}
              opticDisc={locResultA?.optic_disc}
              fovea={locResultA?.fovea}
              glowThreshold={segThreshold}
              reliefScale={reliefScale}
              explodedLayers={explodedLayers}
              showLandmarks={showLandmarks}
              imageWidth={segResultA?.width || locResultA?.width || imageDimA.width}
              imageHeight={segResultA?.height || locResultA?.height || imageDimA.height}
              inspectorOpen={show2DInspector}
            />
          ) : activeStage === "matching" ? (
            <DelaunayConstellation
              fundusABase64={fundusA}
              fundusBBase64={fundusB}
              perturbedFundusBBase64={matchResult?.perturbed_image_b_base64}
              templateA={matchResult?.template_a}
              templateB={matchResult?.template_b}
              matchedPairs={matchResult?.matched_pairs || []}
              isMatch={matchResult?.is_match}
              rotateDegB={rotateDegB}
              scaleFactorB={scaleFactorB}
              brightnessB={brightnessB}
            />
          ) : (
            /* STAGE 03: POPULATION COHORT MATRIX & STRESS DASHBOARD */
            <div className="h-full w-full p-8 overflow-y-auto space-y-6 bg-[#030611] text-gray-100 font-sans">
              {/* Header Bar */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-gray-800/80">
                <div>
                  <div className="flex items-center space-x-2 text-[11px] font-mono text-cyan-400 font-bold uppercase tracking-wider">
                    <span>STEP 03 OF 03</span>
                    <span>•</span>
                    <span>POPULATION COHORT MATRIX</span>
                  </div>
                  <h2 className="text-xl font-bold tracking-tight text-gray-100 mt-1">
                    Cross-Subject Identity Heatmap & Clinical Stress Benchmark
                  </h2>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Benchmarking zero false-acceptance spoof defense against 7-parameter extreme optical degradations.
                  </p>
                </div>

                {/* Regime Selector Chips */}
                <div className="flex items-center space-x-2 bg-gray-950/90 p-1.5 rounded-xl border border-gray-800">
                  <button
                    onClick={() => setCohortRegime("baseline")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
                      cohortRegime === "baseline"
                        ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_12px_rgba(6,182,212,0.2)] font-semibold"
                        : "text-gray-400 hover:text-gray-200 border border-transparent"
                    }`}
                  >
                    Clean Baseline Regime
                  </button>
                  <button
                    onClick={() => setCohortRegime("torture")}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition-all cursor-pointer ${
                      cohortRegime === "torture"
                        ? "bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-[0_0_12px_rgba(244,63,94,0.2)] font-semibold"
                        : "text-gray-400 hover:text-gray-200 border border-transparent"
                    }`}
                  >
                    7-Param Judge Torture Regime
                  </button>
                </div>
              </div>

              {/* Bento Grid: Panel A (Confusion Matrix) + Panel B (Stress Resilience Telemetry) */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Panel A: 4x4 Confusion Matrix */}
                <div className="lg:col-span-7 glass-panel p-6 rounded-2xl border border-gray-800/80 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold tracking-wide uppercase text-gray-200">
                        4x4 Cross-Subject Identity Matrix
                      </h3>
                      <p className="text-[11px] text-gray-400">
                        Click any cell to drill down into Stage 02 Delaunay constellation comparison.
                      </p>
                    </div>
                    <div className="flex items-center space-x-3 text-[10px] font-mono">
                      <span className="flex items-center space-x-1.5 text-emerald-400">
                        <span className="w-2.5 h-2.5 rounded bg-emerald-500/40 border border-emerald-500/60" />
                        <span>Genuine (Diagonal)</span>
                      </span>
                      <span className="flex items-center space-x-1.5 text-gray-400">
                        <span className="w-2.5 h-2.5 rounded bg-gray-900 border border-gray-700" />
                        <span>Impostor (Off-Diag)</span>
                      </span>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse font-mono text-xs">
                      <thead>
                        <tr>
                          <th className="p-2 text-left text-[10px] text-gray-500 uppercase tracking-wider font-semibold border-b border-gray-800/80 w-24">
                            Ref (A) \ Tgt (B)
                          </th>
                          {COHORT_SUBJECTS.map((s) => (
                            <th key={s.id} className="p-2 text-center border-b border-gray-800/80">
                              <div className="text-cyan-400 font-bold">{s.code}</div>
                              <div className="text-[9px] text-gray-500 truncate max-w-[90px] mx-auto font-sans">
                                {s.subtitle.split(" ")[1] || s.name}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-800/40">
                        {COHORT_SUBJECTS.map((rowSubj) => (
                          <tr key={rowSubj.id}>
                            <td className="p-2 text-left font-bold border-r border-gray-800/80">
                              <div className="text-gray-200 text-xs">{rowSubj.code}</div>
                              <div className="text-[9px] text-gray-500 truncate max-w-[80px] font-sans">
                                {rowSubj.name}
                              </div>
                            </td>
                            {COHORT_SUBJECTS.map((colSubj) => {
                              const key = `${rowSubj.id}-${colSubj.id}`;
                              const isDiagonal = rowSubj.id === colSubj.id;
                              const cell = liveCohortOverrides[`${cohortRegime}:${key}`] || COHORT_BENCHMARK_MATRIX[cohortRegime][key] || {
                                score: 0.0,
                                isMatch: false,
                                pairs: 0,
                              };
                              const pct = (cell.score * 100).toFixed(1);

                              return (
                                <td key={colSubj.id} className="p-1.5 text-center">
                                  <button
                                    onClick={() => handleDrillDownFromCohort(rowSubj.id, colSubj.id)}
                                    className={`w-full p-2.5 rounded-xl border text-center transition-all cursor-pointer group flex flex-col items-center justify-between space-y-1 ${
                                      isDiagonal
                                        ? "bg-emerald-950/40 border-emerald-500/50 hover:bg-emerald-900/60 hover:border-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.15)]"
                                        : "bg-gray-900/40 border-gray-800/80 hover:bg-rose-950/30 hover:border-rose-500/40"
                                    }`}
                                    title={`Click to drill down: ${rowSubj.name} vs ${colSubj.name}`}
                                  >
                                    <div className="flex items-center justify-between w-full">
                                      <span
                                        className={`text-[9px] font-bold px-1.5 py-0.2 rounded-full uppercase ${
                                          cell.isMatch
                                            ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                                            : "bg-gray-800 text-gray-500 border border-gray-700"
                                        }`}
                                      >
                                        {cell.isMatch ? "VERIFIED" : "REJECT"}
                                      </span>
                                      <ExternalLink className="w-2.5 h-2.5 text-gray-500 group-hover:text-cyan-400 transition-colors" />
                                    </div>
                                    <div
                                      className={`text-sm font-extrabold tracking-tight ${
                                        cell.isMatch ? "text-emerald-300" : "text-gray-400"
                                      }`}
                                    >
                                      {pct}%
                                    </div>
                                    <div className="text-[9px] text-gray-500 flex items-center justify-between w-full">
                                      <span>{cell.pairs} tris</span>
                                      {cell.latencyMs && (
                                        <span className="text-cyan-400 font-mono text-[8px]">{cell.latencyMs}ms</span>
                                      )}
                                    </div>
                                  </button>
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Panel B: 8-Vector Stress Resilience Benchmark */}
                <div className="lg:col-span-5 glass-panel p-6 rounded-2xl border border-gray-800/80 space-y-5 flex flex-col justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-bold tracking-wide uppercase text-gray-200">
                        8-Vector Stress Resilience (Subject 01)
                      </h3>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                        Kabsch SE(2) Consensus
                      </span>
                    </div>
                    <p className="text-[11px] text-gray-400">
                      Invariance retention against decision threshold (20.0%). Click any vector to test in Step 2.
                    </p>
                  </div>

                  <div className="space-y-2.5">
                    {STRESS_VECTOR_BENCHMARKS.map((item) => {
                      const pct = (item.score * 100).toFixed(1);
                      const isTorture = item.key === "torture";

                      return (
                        <div
                          key={item.key}
                          onClick={() => {
                            const genuinePreset = presets.find((p) => p.id === "match_genuine_01");
                            if (genuinePreset) {
                              setFundusA(genuinePreset.image_base64);
                              setFundusB(genuinePreset.image_b_base64 || genuinePreset.image_base64);
                              setSelectedMatchPresetId("match_genuine_01");
                              setLabelA("Subject 01 (Reference)");
                              setLabelB("Subject 01 (Target)");
                              setCustomScanA(null);
                              setCustomScanB(null);
                            }
                            applyJudgeStressPreset(item.key);
                            setActiveStage("matching");
                            showToast(`Applied preset ${item.label} into Stage 02`, "info");
                          }}
                          className="group p-2 rounded-xl border border-gray-800/60 bg-gray-900/30 hover:bg-gray-900 hover:border-cyan-500/40 cursor-pointer transition-all space-y-1.5"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <span className={`font-medium ${isTorture ? "text-amber-300 font-bold" : "text-gray-200"}`}>
                              {item.label}
                            </span>
                            <span className="font-mono font-bold text-emerald-400">{pct}%</span>
                          </div>

                          <div className="h-2 rounded-full bg-gray-900 border border-gray-800 relative overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isTorture
                                  ? "bg-gradient-to-r from-cyan-500 to-emerald-400"
                                  : "bg-gradient-to-r from-emerald-500 to-teal-400"
                              }`}
                              style={{ width: `${item.score * 100}%` }}
                            />
                            <div
                              className="absolute top-0 bottom-0 w-[2px] bg-rose-500/80 z-10"
                              style={{ left: "20%" }}
                              title="Decision Threshold (20.0%)"
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Summary KPI Cards */}
                  <div className="grid grid-cols-2 gap-2.5 pt-2 border-t border-gray-800/80">
                    <div className="p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/60 space-y-0.5">
                      <div className="text-[10px] font-mono text-gray-500 uppercase">False Accept (FAR)</div>
                      <div className="text-base font-extrabold text-emerald-400 font-mono">0.0%</div>
                      <div className="text-[9px] text-gray-400 font-sans">0/12 Impostor spoof success</div>
                    </div>
                    <div className="p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/60 space-y-0.5">
                      <div className="text-[10px] font-mono text-gray-500 uppercase">False Reject (FRR)</div>
                      <div className="text-base font-extrabold text-emerald-400 font-mono">0.0%</div>
                      <div className="text-[9px] text-gray-400 font-sans">0/4 Genuines falsely rejected</div>
                    </div>
                    <div className="p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/60 space-y-0.5">
                      <div className="text-[10px] font-mono text-gray-500 uppercase">Equal Error Rate</div>
                      <div className="text-base font-extrabold text-cyan-400 font-mono">0.00%</div>
                      <div className="text-[9px] text-gray-400 font-sans">Clinical biometric grade</div>
                    </div>
                    <div className="p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/60 space-y-0.5">
                      <div className="text-[10px] font-mono text-gray-500 uppercase">Separation Margin</div>
                      <div className="text-base font-extrabold text-emerald-400 font-mono">+43.4%</div>
                      <div className="text-[9px] text-gray-400 font-sans">Min Genuine 63.4% vs 20.0%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Floating Context Information Badges (Top Right of Canvas) */}
          {locResultA && activeStage === "ocular" && (
            <div className="absolute top-5 right-5 z-20 glass-panel rounded-2xl p-4 border border-gray-800/80 font-mono text-xs space-y-2 shadow-2xl pointer-events-none">
              <div className="text-[10px] text-gray-400 tracking-wider uppercase font-semibold">
                ANATOMICAL LANDMARKS
              </div>
              <div className="text-emerald-400 flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                <span>Optic Disc: ({locResultA.optic_disc.x.toFixed(1)}, {locResultA.optic_disc.y.toFixed(1)})</span>
              </div>
              <div className="text-amber-400 flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                <span>Fovea Centralis: ({locResultA.fovea.x.toFixed(1)}, {locResultA.fovea.y.toFixed(1)})</span>
              </div>
              <div className="text-gray-300 pt-1.5 border-t border-gray-800 text-[11px] flex justify-between">
                <span className="text-gray-400">OD-Fovea Arc:</span>
                <span className="text-cyan-400 font-bold">{locResultA.disc_to_fovea_distance.toFixed(1)} px</span>
              </div>
            </div>
          )}

          {/* Biometric Verdict Gauge (Step 2: Matching Mode) */}
          {matchResult && activeStage === "matching" && (
            <div className="absolute top-5 right-5 z-20 glass-panel rounded-2xl p-5 border border-gray-800/80 font-mono space-y-3 shadow-2xl min-w-[240px]">
              <div className="flex items-center space-x-2.5">
                {matchResult.is_match ? (
                  <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0" />
                ) : (
                  <ShieldAlert className="w-6 h-6 text-rose-400 shrink-0" />
                )}
                <div>
                  <div
                    className={`text-sm font-bold tracking-wider uppercase ${
                      matchResult.is_match ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {matchResult.is_match ? "IDENTITY VERIFIED" : "IMPOSTOR REJECTED"}
                  </div>
                  <div className="text-[10px] text-gray-400">
                    {matchResult.is_match ? "Genuine Subject Match" : "Biometric Spoofing Defense"}
                  </div>
                </div>
              </div>

              {/* Similarity Progress Bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-300">Similarity Score:</span>
                  <span className={`font-bold ${matchResult.is_match ? "text-emerald-400" : "text-rose-400"}`}>
                    {(matchResult.similarity_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-900 rounded-full h-2 relative overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      matchResult.is_match ? "bg-emerald-400" : "bg-rose-500"
                    }`}
                    style={{ width: `${Math.min(100, matchResult.similarity_score * 100)}%` }}
                  />
                  {/* Threshold Pin at 20% */}
                  <div className="absolute top-0 bottom-0 left-[20%] w-[2px] bg-cyan-400 z-10" />
                </div>
                <div className="flex justify-between text-[10px] text-gray-400 pt-0.5">
                  <span>Threshold: 20% (EER 0.00%)</span>
                  <span>{matchResult.matched_pairs.length} Matched Triangles</span>
                </div>
              </div>
            </div>
          )}

          {/* 2D Quad-View Inspector Drawer */}
          {show2DInspector && activeStage === "ocular" && (
            <div className="absolute bottom-0 left-0 right-0 h-56 bg-gray-950/95 border-t border-gray-800 backdrop-blur-2xl z-30 p-3 flex flex-col animate-in slide-in-from-bottom duration-300">
              <div className="flex items-center justify-between px-2 pb-2 border-b border-gray-800/80">
                <span className="text-[11px] font-mono font-bold tracking-wider uppercase text-cyan-400">
                  2D CLINICAL INSPECTION SUITE
                </span>
                <button
                  onClick={() => setShow2DInspector(false)}
                  className="p-1 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-gray-200 transition-colors cursor-pointer"
                  title="Close Inspector"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="grid grid-cols-4 gap-3 flex-1 pt-2">
                <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black/60 flex flex-col">
                  <span className="absolute top-2 left-2 text-[10px] font-mono bg-black/80 px-2 py-0.5 rounded text-gray-300 z-10">
                    1. RGB SOURCE
                  </span>
                  {fundusA && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={fundusA} alt="RGB" className="w-full h-full object-contain" />
                  )}
                </div>
                <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black/60 flex flex-col">
                  <span className="absolute top-2 left-2 text-[10px] font-mono bg-black/80 px-2 py-0.5 rounded text-cyan-400 z-10">
                    2. PROBABILITY MAP
                  </span>
                  {segResultA?.prob_map_base64 && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={segResultA.prob_map_base64} alt="Prob" className="w-full h-full object-contain" />
                  )}
                </div>
                <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black/60 flex flex-col">
                  <span className="absolute top-2 left-2 text-[10px] font-mono bg-black/80 px-2 py-0.5 rounded text-emerald-400 z-10">
                    3. REFINED MASK
                  </span>
                  {segResultA?.vessel_mask_base64 && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={segResultA.vessel_mask_base64} alt="Mask" className="w-full h-full object-contain" />
                  )}
                </div>
                <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-black/60 flex flex-col">
                  <span className="absolute top-2 left-2 text-[10px] font-mono bg-black/80 px-2 py-0.5 rounded text-sky-400 z-10">
                    4. PRUNED SKELETON
                  </span>
                  {segResultA?.skeleton_base64 && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={segResultA.skeleton_base64} alt="Skeleton" className="w-full h-full object-contain" />
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Precision Telemetry & Control Studio (440px, Breathing Room, Miller Chunking) */}
        <aside className="w-[440px] border-l border-gray-800/80 bg-gray-950/90 backdrop-blur-xl flex flex-col overflow-hidden shrink-0 z-20">
          <div ref={sidebarScrollRef} className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Studio Stage Identity Card */}
            <div className="p-3.5 rounded-2xl border border-gray-800/80 bg-gray-900/60 space-y-1">
              <div className="flex items-center justify-between text-[10px] font-mono text-cyan-400 font-bold uppercase tracking-wider">
                <span>
                  {activeStage === "ocular"
                    ? "STAGE 01 OF 03"
                    : activeStage === "matching"
                    ? "STAGE 02 OF 03"
                    : "STAGE 03 OF 03"}
                </span>
                <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300">
                  {activeStage === "ocular"
                    ? "ANATOMICAL STUDIO"
                    : activeStage === "matching"
                    ? "BIOMETRIC STUDIO"
                    : "COHORT STUDIO"}
                </span>
              </div>
              <div className="text-xs font-bold text-gray-100">
                {activeStage === "ocular"
                  ? "3D Retinal Anatomy & Vessels"
                  : activeStage === "matching"
                  ? "Biometric Identity Verification"
                  : "Population Cross-Matching & Stress Matrix"}
              </div>
              <p className="text-[11px] text-gray-400 leading-snug">
                {activeStage === "ocular"
                  ? "Interactive 3D depth extrusion, vessel segmentation, and OD-Fovea landmark localization."
                  : activeStage === "matching"
                  ? "Dual Delaunay invariant graph matching across 11 clinical and physical stress vectors."
                  : "Cohort confusion matrix, 8-vector stress resilience, and live batch verification."}
              </p>
            </div>

            {activeStage === "cohort" ? (
              /* Step 3: Cohort Studio Controls */
              <div className="space-y-5">
                <div className="pb-2 border-b border-gray-800/80 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-bold uppercase tracking-wider text-gray-200">
                      STAGE 03: COHORT STRESS EVALUATION
                    </span>
                    <p className="text-[11px] text-gray-400">Population cross-matching & live API runner</p>
                  </div>
                  <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-500/30 px-2 py-0.5 rounded-full font-bold">
                    4 Subjects • 16 Pairs
                  </span>
                </div>

                {/* Cohort Regime Selector */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                      EVALUATION REGIME
                    </span>
                    <span className="text-[10px] font-mono text-gray-500">Dual Mode</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                    <button
                      onClick={() => setCohortRegime("baseline")}
                      className={`p-3 rounded-xl border text-left transition-all cursor-pointer ${
                        cohortRegime === "baseline"
                          ? "bg-cyan-500/20 border-cyan-400/60 text-cyan-200 shadow-[0_0_15px_rgba(6,182,212,0.2)] font-semibold"
                          : "bg-gray-900/60 border-gray-800/80 text-gray-400 hover:bg-gray-900"
                      }`}
                    >
                      <div className="font-bold">Clean Baseline</div>
                      <div className="text-[10px] text-gray-400 font-sans mt-0.5">Ground-truth unperturbed</div>
                    </button>
                    <button
                      onClick={() => setCohortRegime("torture")}
                      className={`p-3 rounded-xl border text-left transition-all cursor-pointer ${
                        cohortRegime === "torture"
                          ? "bg-rose-500/20 border-rose-400/60 text-rose-200 shadow-[0_0_15px_rgba(244,63,94,0.2)] font-semibold"
                          : "bg-gray-900/60 border-gray-800/80 text-gray-400 hover:bg-gray-900"
                      }`}
                    >
                      <div className="font-bold">Judge Torture</div>
                      <div className="text-[10px] text-gray-400 font-sans mt-0.5">All 7 multi-factor stress</div>
                    </button>
                  </div>
                </div>

                {/* Big Primary Action CTA: Run Live Cohort Verification */}
                <button
                  onClick={handleRunLiveCohort}
                  disabled={isRunningCohort}
                  className="w-full h-12 rounded-xl text-xs uppercase tracking-wider font-bold bg-gradient-to-r from-cyan-400 to-sky-500 hover:from-cyan-300 hover:to-sky-400 text-black shadow-[0_0_25px_rgba(6,182,212,0.35)] transition-all flex items-center justify-center space-x-2.5 cursor-pointer disabled:opacity-50"
                >
                  {isRunningCohort ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>
                        Verifying Cohort ({cohortProgress ? `${cohortProgress.current}/${cohortProgress.total}` : "Evaluating..."})
                      </span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-black" />
                      <span>Run Live Cohort Verification</span>
                    </>
                  )}
                </button>

                {/* Quick Jump Biometric Scenarios */}
                <div className="space-y-2.5 pt-2 border-t border-gray-800/80">
                  <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">
                    BIOMETRIC SCENARIOS QUICK-JUMP
                  </span>
                  <div className="space-y-2">
                    <button
                      onClick={() => handleDrillDownFromCohort("s01", "s01", "baseline")}
                      className="w-full p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/40 hover:bg-gray-900 hover:border-cyan-500/40 text-left transition-all cursor-pointer flex items-center justify-between group"
                    >
                      <div>
                        <div className="text-xs font-medium text-gray-200 group-hover:text-cyan-300 transition-colors">
                          Subject 01 Genuine Match
                        </div>
                        <div className="text-[10px] text-gray-400">Baseline identity verification ground-truth</div>
                      </div>
                      <ArrowRight className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    </button>
                    <button
                      onClick={() => {
                        setCohortRegime("torture");
                        handleDrillDownFromCohort("s01", "s01", "torture");
                      }}
                      className="w-full p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/40 hover:bg-gray-900 hover:border-amber-500/40 text-left transition-all cursor-pointer flex items-center justify-between group"
                    >
                      <div>
                        <div className="text-xs font-medium text-gray-200 group-hover:text-amber-300 transition-colors">
                          Subject 01 Judge Torture Benchmark
                        </div>
                        <div className="text-[10px] text-amber-400">All 7 parameters simultaneously activated</div>
                      </div>
                      <ArrowRight className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    </button>
                    <button
                      onClick={() => handleDrillDownFromCohort("s01", "s02", cohortRegime)}
                      className="w-full p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/40 hover:bg-gray-900 hover:border-rose-500/40 text-left transition-all cursor-pointer flex items-center justify-between group"
                    >
                      <div>
                        <div className="text-xs font-medium text-gray-200 group-hover:text-rose-300 transition-colors">
                          Subject 01 vs Subject 02 Impostor
                        </div>
                        <div className="text-[10px] text-rose-400">Zero false-acceptance spoof defense</div>
                      </div>
                      <ArrowRight className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    </button>
                    <button
                      onClick={() => handleDrillDownFromCohort("s23", "s24", cohortRegime)}
                      className="w-full p-2.5 rounded-xl border border-gray-800/80 bg-gray-900/40 hover:bg-gray-900 hover:border-purple-500/40 text-left transition-all cursor-pointer flex items-center justify-between group"
                    >
                      <div>
                        <div className="text-xs font-medium text-gray-200 group-hover:text-purple-300 transition-colors">
                          Subject 23 (Sparse) vs 24 (Dense)
                        </div>
                        <div className="text-[10px] text-purple-400">Extreme density mismatch cross-impostor</div>
                      </div>
                      <ArrowRight className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {/* Chunk 1: Stage-Specific Preset Selector */}
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-gray-300">
                      {activeStage === "ocular" ? "SELECT ANATOMICAL SPECIMEN" : "SELECT BIOMETRIC SCENARIO"}
                    </span>
                    <span className="text-[11px] text-gray-500 font-mono">Pareto 80/20</span>
                  </div>
                  <p className="text-[11px] text-gray-400 leading-relaxed">
                    {activeStage === "ocular"
                      ? "Choose a clinical retinal scan to inspect 3D cup topography, vessel segmentation, and OD-Fovea landmarks:"
                      : "Choose an identity verification pair to benchmark genuine matching or impostor rejection:"}
                  </p>

                  <div className="grid grid-cols-2 gap-2.5">
                    {currentPresets.map((preset) => {
                      const isSelected =
                        activeStage === "ocular"
                          ? selectedOcularPresetId === preset.id
                          : selectedMatchPresetId === preset.id;
                      const isGenuine = preset.scenario === "genuine";
                      const isImpostor = preset.scenario === "impostor";

                      return (
                        <button
                          key={preset.id}
                          onClick={() =>
                            activeStage === "ocular"
                              ? handleSelectOcularPreset(preset)
                              : handleSelectMatchPreset(preset)
                          }
                          className={`p-3 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between space-y-2 ${
                            isSelected
                              ? "bg-cyan-500/15 border-cyan-400/60 shadow-[0_0_15px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800/80 hover:bg-gray-900 hover:border-gray-700"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span
                              className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold uppercase ${
                                isGenuine
                                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                                  : isImpostor
                                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                                  : preset.scenario === "sparse_clinical"
                                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                                  : preset.scenario === "dense_clinical"
                                  ? "bg-purple-500/20 text-purple-300 border border-purple-500/30"
                                  : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                              }`}
                            >
                              {preset.badge_label || (isGenuine ? "Same Person" : isImpostor ? "Impostor Test" : "Clinical")}
                            </span>
                            {isSelected && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />}
                          </div>

                          <div className="text-xs font-medium text-gray-200 line-clamp-1">
                            {preset.title}
                          </div>
                          <div className="text-[10px] text-gray-400 line-clamp-2">
                            {preset.description}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Chunk 2: Step-Specific Action Center */}
                {activeStage === "ocular" ? (
                  /* Step 1: Retinal Anatomy & Vessels */
                  <div className="space-y-4">
                    <div className="pb-2 border-b border-gray-800/80 flex items-center justify-between">
                      <div>
                        <span className="text-xs font-bold uppercase tracking-wider text-gray-200">
                          STAGE 01: VESSEL & LANDMARK EXTRACTION
                        </span>
                        <p className="text-[11px] text-gray-400">Deep U-Net segmentation + Optic Disc & Fovea</p>
                      </div>
                      <label className="px-3 py-1.5 rounded-lg border border-cyan-500/30 bg-cyan-950/40 hover:bg-cyan-950/80 text-cyan-300 text-xs font-medium flex items-center space-x-1.5 cursor-pointer transition-colors">
                        {uploadingTarget === "A" ? (
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <UploadCloud className="w-3.5 h-3.5" />
                        )}
                        <span>{uploadingTarget === "A" ? "Uploading..." : "Upload"}</span>
                        <input
                          type="file"
                          accept="image/*,.tif,.tiff,.ppm,.pgm,.bmp,.gif,.webp"
                          className="hidden"
                          disabled={uploadingTarget !== null}
                          onChange={(e) => handleFileUpload(e, "A")}
                        />
                      </label>
                    </div>

                    {/* Stage 01 Active Specimen Preview Strip */}
                    <div className="p-3 rounded-xl border border-gray-800/80 bg-gray-900/40 flex items-center justify-between space-x-3">
                      <div className="flex items-center space-x-3 min-w-0">
                        <div className="w-10 h-10 rounded-full border border-cyan-500/40 overflow-hidden bg-black/60 shrink-0">
                          {fundusA ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={fundusA} alt="Specimen Thumbnail" className="w-full h-full object-cover" />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[8px] text-gray-500">None</div>
                          )}
                        </div>
                        <div className="min-w-0">
                          <div className="text-xs font-medium text-gray-200 truncate">{labelA}</div>
                          <div className="text-[10px] font-mono text-gray-400 flex items-center space-x-1.5">
                            <span>
                              {segResultA
                                ? `${segResultA.width}x${segResultA.height}px`
                                : locResultA
                                ? `${locResultA.width}x${locResultA.height}px`
                                : `${imageDimA.width}x${imageDimA.height}px`}
                            </span>
                            <span>•</span>
                            <span className={segResultA ? "text-emerald-400" : "text-amber-400"}>
                              {segResultA ? "Segmented" : "Ready to Analyze"}
                            </span>
                          </div>
                        </div>
                      </div>
                      <span
                        className={`text-[9px] font-mono px-2 py-0.5 rounded-full uppercase shrink-0 font-bold ${
                          selectedOcularPresetId === "custom"
                            ? "bg-purple-500/20 text-purple-300 border border-purple-500/40"
                            : "bg-cyan-500/10 text-cyan-300 border border-cyan-500/30"
                        }`}
                      >
                        {selectedOcularPresetId === "custom" ? "CUSTOM UPLOAD" : "PRESET"}
                      </span>
                    </div>

                    {/* Big Primary Action CTA */}
                    <button
                      onClick={handleExecuteInference}
                      disabled={isProcessingA || !fundusA}
                      className="w-full h-12 rounded-xl text-xs uppercase tracking-wider font-bold bg-gradient-to-r from-cyan-400 to-sky-500 hover:from-cyan-300 hover:to-sky-400 text-black shadow-[0_0_25px_rgba(6,182,212,0.35)] transition-all flex items-center justify-center space-x-2.5 cursor-pointer disabled:opacity-50"
                    >
                      {isProcessingA ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>INFERRING TOPOLOGY...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4" />
                          <span>ANALYZE RETINAL VESSELS & LANDMARKS</span>
                        </>
                      )}
                    </button>

                    {/* Quick View Customization Card */}
                    <div className="glass-panel rounded-2xl p-4 border border-gray-800/80 space-y-3 text-xs">
                      <div className="text-xs font-semibold text-gray-300 pb-1 border-b border-gray-800/60">
                        3D VIEW CUSTOMIZATION
                      </div>

                      {/* Exploded 3D View Toggle */}
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-gray-200 font-medium">Exploded 3D Layer Stack</div>
                          <div className="text-[11px] text-gray-400">Float pure vessel skeleton in 3D depth</div>
                        </div>
                        <button
                          onClick={() => setExplodedLayers(!explodedLayers)}
                          className={`w-11 h-6 rounded-full p-1 transition-colors cursor-pointer ${
                            explodedLayers ? "bg-cyan-500" : "bg-gray-800"
                          }`}
                        >
                          <div
                            className={`w-4 h-4 rounded-full bg-black transition-transform ${
                              explodedLayers ? "translate-x-5" : "translate-x-0"
                            }`}
                          />
                        </button>
                      </div>

                      {/* Landmark Beacons Toggle */}
                      <div className="flex items-center justify-between pt-1">
                        <div>
                          <div className="text-gray-200 font-medium">OD & Fovea Beacons</div>
                          <div className="text-[11px] text-gray-400">Optic Disc (green) & Fovea (amber) rings</div>
                        </div>
                        <button
                          onClick={() => setShowLandmarks(!showLandmarks)}
                          className={`w-11 h-6 rounded-full p-1 transition-colors cursor-pointer ${
                            showLandmarks ? "bg-cyan-500" : "bg-gray-800"
                          }`}
                        >
                          <div
                            className={`w-4 h-4 rounded-full bg-black transition-transform ${
                              showLandmarks ? "translate-x-5" : "translate-x-0"
                            }`}
                          />
                        </button>
                      </div>

                      {/* 2D Quad-View Toggle */}
                      <div className="pt-2 border-t border-gray-800/60">
                        <button
                          onClick={() => setShow2DInspector(!show2DInspector)}
                          className="w-full py-2.5 rounded-xl text-xs text-gray-300 hover:text-white border border-gray-800 hover:border-gray-700 bg-gray-900/60 flex items-center justify-center space-x-2 cursor-pointer transition-colors"
                        >
                          <Layers className="w-3.5 h-3.5 text-cyan-400" />
                          <span>{show2DInspector ? "Hide 2D Clinical Inspector" : "Show 2D Clinical Inspector"}</span>
                        </button>
                      </div>
                    </div>

                    {/* Collapsible Fine-Tune Drawer */}
                    <div className="glass-panel rounded-2xl border border-gray-800/80 overflow-hidden">
                      <button
                        onClick={() => setShowAdvancedShader(!showAdvancedShader)}
                        className="w-full p-4 flex items-center justify-between text-xs text-gray-300 hover:text-white transition-colors cursor-pointer"
                      >
                        <div className="flex items-center space-x-2 font-medium">
                          <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                          <span>Fine-Tune Shader & Sensitivity</span>
                        </div>
                        {showAdvancedShader ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                      </button>

                      {showAdvancedShader && (
                        <div className="p-4 pt-0 space-y-4 font-mono text-xs border-t border-gray-800/60">
                          <div>
                            <div className="flex justify-between text-gray-400 mb-1.5">
                              <span>Vessel Sensitivity Threshold:</span>
                              <span className="text-cyan-400 font-bold">{segThreshold.toFixed(2)}</span>
                            </div>
                            <input
                              type="range"
                              min="0.10"
                              max="0.90"
                              step="0.02"
                              value={segThreshold}
                              onChange={(e) => setSegThreshold(parseFloat(e.target.value))}
                              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                            />
                            {segResultA && analyzedThresholdA !== null && Math.abs(segThreshold - analyzedThresholdA) > 1e-3 && (
                              <div className="text-[10px] text-cyan-300/80 font-mono mt-1">
                                3D glow updated live — click Analyze to refresh 2D mask & skeleton
                              </div>
                            )}
                          </div>

                          <div>
                            <div className="flex justify-between text-gray-400 mb-1.5">
                              <span>3D Vascular Relief Scale:</span>
                              <span className="text-cyan-400 font-bold">{reliefScale.toFixed(2)}</span>
                            </div>
                            <input
                              type="range"
                              min="0.04"
                              max="0.32"
                              step="0.02"
                              value={reliefScale}
                              onChange={(e) => setReliefScale(parseFloat(e.target.value))}
                              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  /* Step 2: Biometric Comparison */
                  <div className="space-y-4">
                    <div className="pb-2 border-b border-gray-800/80">
                      <span className="text-xs font-bold uppercase tracking-wider text-gray-200">
                        STAGE 02: 1-TO-1 BIOMETRIC COMPARISON
                      </span>
                      <p className="text-[11px] text-gray-400">Delaunay Invariants + Hungarian Bipartite Matching</p>
                    </div>

                    {/* Self-Explanatory Visual A vs B Comparison Card */}
                    <div className="glass-panel rounded-2xl p-4 border border-gray-800/80 space-y-3">
                      <div className="grid grid-cols-2 gap-3 items-center relative">
                        {/* Scan A */}
                        <div className="space-y-2 flex flex-col items-center text-center">
                          <span className="text-[11px] font-bold text-cyan-400">REFERENCE EYE (A)</span>
                          <div className="w-24 h-24 rounded-full border-2 border-cyan-500/50 overflow-hidden bg-gray-900/60 shadow-lg">
                            {fundusA ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img src={fundusA} alt="Scan A" className="w-full h-full object-cover" />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">Empty</div>
                            )}
                          </div>
                          <div className="text-[10px] text-gray-300 font-medium line-clamp-1">{labelA}</div>
                          <label className="text-[10px] font-medium text-cyan-400 hover:text-cyan-300 cursor-pointer flex items-center space-x-1">
                            {uploadingTarget === "A" ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : (
                              <UploadCloud className="w-3 h-3" />
                            )}
                            <span>{uploadingTarget === "A" ? "Loading..." : "Change"}</span>
                            <input
                              type="file"
                              accept="image/*,.tif,.tiff,.ppm,.pgm,.bmp,.gif,.webp"
                              className="hidden"
                              disabled={uploadingTarget !== null}
                              onChange={(e) => handleFileUpload(e, "A")}
                            />
                          </label>
                        </div>

                        {/* Central VS Badge */}
                        <div className="absolute left-1/2 top-10 -translate-x-1/2 w-7 h-7 rounded-full bg-gray-900 border border-gray-700 flex items-center justify-center text-[10px] font-bold text-gray-300 shadow-md z-10">
                          VS
                        </div>

                        {/* Scan B with Live Perturbation Preview */}
                        <div className="space-y-2 flex flex-col items-center text-center">
                          <span className="text-[11px] font-bold text-rose-400">TARGET EYE (B)</span>
                          <div className="w-24 h-24 rounded-full border-2 border-rose-500/50 overflow-hidden bg-gray-900/60 shadow-lg">
                            {fundusB ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img
                                src={fundusB}
                                alt="Scan B"
                                className="w-full h-full object-cover transition-all duration-150"
                                style={{
                                  transform: `rotate(${-rotateDegB}deg) scale(${scaleFactorB})`,
                                  filter: `brightness(${brightnessB}) blur(${Math.min(blurSigmaB * 0.75, 4)}px) contrast(${gammaB > 1 ? 1.0 + (gammaB - 1) * 0.5 : 1.0 / (1.0 + (1 - gammaB))}) sepia(${Math.min(hazeFactorB * 0.8, 0.6)})`,
                                }}
                              />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">Empty</div>
                            )}
                          </div>
                          <div className="text-[10px] text-gray-300 font-medium line-clamp-1">{labelB}</div>
                          <label className="text-[10px] font-medium text-rose-400 hover:text-rose-300 cursor-pointer flex items-center space-x-1">
                            {uploadingTarget === "B" ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : (
                              <UploadCloud className="w-3 h-3" />
                            )}
                            <span>{uploadingTarget === "B" ? "Loading..." : "Change"}</span>
                            <input
                              type="file"
                              accept="image/*,.tif,.tiff,.ppm,.pgm,.bmp,.gif,.webp"
                              className="hidden"
                              disabled={uploadingTarget !== null}
                              onChange={(e) => handleFileUpload(e, "B")}
                            />
                          </label>
                        </div>
                      </div>
                    </div>

                    {/* 8 One-Click Judge Stress-Test Preset Chips */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-[11px] text-gray-400 font-medium">
                        <span>1-Click Judge Stress Vectors:</span>
                        <span className="text-[10px] font-mono text-cyan-400">8 Presets</span>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 text-xs font-mono">
                        <button
                          onClick={() => applyJudgeStressPreset("baseline")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "baseline"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">Clean Baseline</span>
                          <span className="text-[9px] text-gray-500 ml-1">0°</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("pose")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "pose"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">Tilt +25° & 0.85x</span>
                          <span className="text-[9px] text-cyan-400 ml-1">Pose</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("cataract")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "cataract"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">Cataract Haze 45%</span>
                          <span className="text-[9px] text-amber-400 ml-1">Media</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("glare")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "glare"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">Corneal Glare 65%</span>
                          <span className="text-[9px] text-amber-300 ml-1">Flash</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("blur")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "blur"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">Defocus Blur 2.5px</span>
                          <span className="text-[9px] text-indigo-400 ml-1">Optic</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("noise")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "noise"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">ISO Noise 16.0</span>
                          <span className="text-[9px] text-purple-400 ml-1">CMOS</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("eyelid")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "eyelid"
                              ? "bg-cyan-500/20 text-cyan-300 border-cyan-400/60 shadow-[0_0_10px_rgba(6,182,212,0.2)]"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate">Eyelid Occlusion 22%</span>
                          <span className="text-[9px] text-rose-400 ml-1">Ptosis</span>
                        </button>

                        <button
                          onClick={() => applyJudgeStressPreset("torture")}
                          className={`px-2.5 py-1.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between ${
                            activeStressChip === "torture"
                              ? "bg-gradient-to-r from-rose-500/30 to-amber-500/30 text-rose-200 border-rose-400/70 shadow-[0_0_12px_rgba(244,63,94,0.3)] font-bold"
                              : "bg-gray-900/60 border-gray-800 text-gray-400 hover:text-gray-200"
                          }`}
                        >
                          <span className="truncate flex items-center space-x-1">
                            <Zap className="w-3 h-3 text-rose-400 shrink-0 inline mr-0.5" />
                            <span>Judge Torture</span>
                          </span>
                          <span className="text-[9px] text-rose-400 ml-1">Multi</span>
                        </button>
                      </div>
                    </div>

                    {/* Primary Verification Action Button */}
                    <button
                      onClick={handleExecuteVerification}
                      disabled={isMatching || !fundusA || !fundusB}
                      className="w-full h-12 rounded-xl text-xs uppercase tracking-wider font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 hover:from-emerald-300 hover:to-cyan-400 text-black shadow-[0_0_25px_rgba(16,185,129,0.35)] transition-all flex items-center justify-center space-x-2.5 cursor-pointer disabled:opacity-50"
                    >
                      {isMatching ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>COMPUTING DELAUNAY MATCH...</span>
                        </>
                      ) : (
                        <>
                          <Network className="w-4 h-4" />
                          <span>COMPARE & VERIFY IDENTITY</span>
                        </>
                      )}
                    </button>

                    {/* Collapsible 11-Parameter Fine-Tune Slider Drawer */}
                    <div className="glass-panel rounded-2xl border border-gray-800/80 overflow-hidden">
                      <button
                        onClick={() => setShowAdvancedPerturb(!showAdvancedPerturb)}
                        className="w-full p-4 flex items-center justify-between text-xs text-gray-300 hover:text-white transition-colors cursor-pointer"
                      >
                        <div className="flex items-center space-x-2 font-medium">
                          <RotateCw className="w-3.5 h-3.5 text-emerald-400" />
                          <span>11-Parameter Precision Controls</span>
                        </div>
                        {showAdvancedPerturb ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
                      </button>

                      {showAdvancedPerturb && (
                        <div className="p-4 pt-0 space-y-4 font-mono text-xs border-t border-gray-800/60">
                          {/* Group A: Geometric & Pose */}
                          <div className="space-y-2.5 pt-2">
                            <div className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">
                              GROUP A: GEOMETRIC & POSE
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Head Tilt (Rotation):</span>
                                <span className="text-cyan-400 font-bold">{rotateDegB > 0 ? `+${rotateDegB}` : rotateDegB}°</span>
                              </div>
                              <input
                                type="range"
                                min="-45"
                                max="45"
                                step="1"
                                value={rotateDegB}
                                onChange={(e) => updatePerturbation(setRotateDegB, parseFloat(e.target.value))}
                                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Working Distance (Scale):</span>
                                <span className="text-cyan-400 font-bold">{scaleFactorB.toFixed(2)}x</span>
                              </div>
                              <input
                                type="range"
                                min="0.80"
                                max="1.20"
                                step="0.02"
                                value={scaleFactorB}
                                onChange={(e) => updatePerturbation(setScaleFactorB, parseFloat(e.target.value))}
                                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Spherical Gaze Warp:</span>
                                <span className="text-cyan-400 font-bold">{sphericalWarpB > 0 ? `+${sphericalWarpB.toFixed(2)}` : sphericalWarpB.toFixed(2)}</span>
                              </div>
                              <input
                                type="range"
                                min="-0.25"
                                max="0.25"
                                step="0.01"
                                value={sphericalWarpB}
                                onChange={(e) => updatePerturbation(setSphericalWarpB, parseFloat(e.target.value))}
                                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Pulsatile IOP Elastic:</span>
                                <span className="text-cyan-400 font-bold">{elasticAlphaB.toFixed(1)} px</span>
                              </div>
                              <input
                                type="range"
                                min="0.0"
                                max="5.0"
                                step="0.2"
                                value={elasticAlphaB}
                                onChange={(e) => updatePerturbation(setElasticAlphaB, parseFloat(e.target.value))}
                                className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>
                          </div>

                          {/* Group B: Optical & Media Quality */}
                          <div className="space-y-2.5 pt-2 border-t border-gray-800/60">
                            <div className="text-[10px] uppercase font-bold text-amber-400 tracking-wider">
                              GROUP B: OPTICAL & MEDIA QUALITY
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Defocus Blur (Sigma):</span>
                                <span className="text-amber-400 font-bold">{blurSigmaB.toFixed(1)} px</span>
                              </div>
                              <input
                                type="range"
                                min="0.0"
                                max="4.0"
                                step="0.1"
                                value={blurSigmaB}
                                onChange={(e) => updatePerturbation(setBlurSigmaB, parseFloat(e.target.value))}
                                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Cataract Media Haze:</span>
                                <span className="text-amber-400 font-bold">{Math.round(hazeFactorB * 100)}%</span>
                              </div>
                              <input
                                type="range"
                                min="0.0"
                                max="0.60"
                                step="0.02"
                                value={hazeFactorB}
                                onChange={(e) => updatePerturbation(setHazeFactorB, parseFloat(e.target.value))}
                                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Corneal Specular Glare:</span>
                                <span className="text-amber-400 font-bold">{Math.round(glareIntensityB * 100)}%</span>
                              </div>
                              <input
                                type="range"
                                min="0.0"
                                max="0.80"
                                step="0.05"
                                value={glareIntensityB}
                                onChange={(e) => updatePerturbation(setGlareIntensityB, parseFloat(e.target.value))}
                                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Eyelid Ptosis Occlusion:</span>
                                <span className="text-amber-400 font-bold">{Math.round(occlusionFracB * 100)}%</span>
                              </div>
                              <input
                                type="range"
                                min="0.0"
                                max="0.30"
                                step="0.02"
                                value={occlusionFracB}
                                onChange={(e) => updatePerturbation(setOcclusionFracB, parseFloat(e.target.value))}
                                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>
                          </div>

                          {/* Group C: Sensor & Photometric */}
                          <div className="space-y-2.5 pt-2 border-t border-gray-800/60">
                            <div className="text-[10px] uppercase font-bold text-purple-400 tracking-wider">
                              GROUP C: SENSOR & PHOTOMETRIC
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>Flash Discharge Illumination:</span>
                                <span className="text-purple-400 font-bold">{brightnessB.toFixed(2)}x</span>
                              </div>
                              <input
                                type="range"
                                min="0.60"
                                max="1.40"
                                step="0.05"
                                value={brightnessB}
                                onChange={(e) => updatePerturbation(setBrightnessB, parseFloat(e.target.value))}
                                className="w-full accent-purple-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>ISP Tone Gamma:</span>
                                <span className="text-purple-400 font-bold">{gammaB.toFixed(2)}</span>
                              </div>
                              <input
                                type="range"
                                min="0.50"
                                max="1.80"
                                step="0.05"
                                value={gammaB}
                                onChange={(e) => updatePerturbation(setGammaB, parseFloat(e.target.value))}
                                className="w-full accent-purple-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>

                            <div>
                              <div className="flex justify-between text-gray-400 mb-1">
                                <span>CMOS ISO Thermal Noise:</span>
                                <span className="text-purple-400 font-bold">{noiseSigmaB.toFixed(1)}</span>
                              </div>
                              <input
                                type="range"
                                min="0.0"
                                max="25.0"
                                step="1.0"
                                value={noiseSigmaB}
                                onChange={(e) => updatePerturbation(setNoiseSigmaB, parseFloat(e.target.value))}
                                className="w-full accent-purple-400 cursor-pointer h-1.5 bg-gray-800 rounded-lg"
                              />
                            </div>
                          </div>

                          {/* Reset Button */}
                          <div className="pt-2">
                            <button
                              onClick={() => {
                                resetAllPerturbations();
                                setActiveStressChip("baseline");
                                setMatchResult(null);
                                showToast("All perturbations reset to nominal baseline", "info");
                              }}
                              className="w-full py-2 rounded-xl text-xs text-gray-400 hover:text-white border border-gray-800 hover:border-gray-700 bg-gray-900/60 cursor-pointer transition-colors"
                            >
                              Reset All 11 Perturbations
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Bottom Technical Specifications & Sticky Stage Bridge */}
          <div className="shrink-0 px-6 py-4 border-t border-gray-800/80 bg-gray-950/95 space-y-2.5 font-mono text-[11px]">
            {/* Quick Stage Bridge Navigation */}
            <div className="pb-2 flex justify-between items-center text-xs font-mono border-b border-gray-800/60">
              {activeStage === "matching" ? (
                <button
                  onClick={() => handleStageSwitch("ocular")}
                  className="text-gray-400 hover:text-cyan-300 flex items-center space-x-1.5 cursor-pointer transition-colors"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Return to 3D Anatomy</span>
                </button>
              ) : activeStage === "cohort" ? (
                <button
                  onClick={() => handleStageSwitch("matching")}
                  className="text-gray-400 hover:text-cyan-300 flex items-center space-x-1.5 cursor-pointer transition-colors"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Return to Identity Match</span>
                </button>
              ) : (
                <div />
              )}
              {activeStage === "ocular" ? (
                <button
                  onClick={() => handleStageSwitch("matching")}
                  className="text-cyan-400 hover:text-cyan-300 font-bold flex items-center space-x-1.5 cursor-pointer transition-colors"
                >
                  <span>Proceed to Identity Verification</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              ) : activeStage === "matching" ? (
                <button
                  onClick={() => handleStageSwitch("cohort")}
                  className="text-cyan-400 hover:text-cyan-300 font-bold flex items-center space-x-1.5 cursor-pointer transition-colors"
                >
                  <span>Proceed to Cohort Matrix</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              ) : (
                <button
                  onClick={() => handleStageSwitch("ocular")}
                  className="text-cyan-400 hover:text-cyan-300 font-bold flex items-center space-x-1.5 cursor-pointer transition-colors"
                >
                  <span>Return to 3D Anatomy</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {statusBanner && (
              <div
                className={`p-3 rounded-xl border text-xs flex items-center space-x-2.5 ${
                  statusBanner.type === "success"
                    ? "bg-emerald-950/70 border-emerald-500/50 text-emerald-300"
                    : statusBanner.type === "error"
                    ? "bg-rose-950/70 border-rose-500/50 text-rose-300"
                    : "bg-cyan-950/70 border-cyan-500/50 text-cyan-300"
                }`}
              >
                <span className="w-2 h-2 rounded-full bg-current animate-ping shrink-0" />
                <span className="leading-snug">{statusBanner.text}</span>
              </div>
            )}
            <div className="text-gray-400 flex items-center justify-between">
              <span>ALGORITHM:</span>
              <span className="text-gray-300">Kanjanasurat et al. (2020)</span>
            </div>
            <div className="text-gray-400 flex items-center justify-between">
              <span>INVARIANT BASIS:</span>
              <span className="text-cyan-400 font-bold">Area Ratio + Internal Angles</span>
            </div>
            <div className="text-gray-400 flex items-center justify-between">
              <span>SOLVER:</span>
              <span className="text-emerald-400 font-bold">Hungarian Bipartite 1:1</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
