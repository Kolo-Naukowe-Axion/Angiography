import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  getFrameUrl,
  getHealth,
  getLabels,
  getModels,
  getPatients,
  inferFrame,
  prefetchFrames,
  saveLabels,
  selectModel,
} from "./api";
import { computeFrameMetrics } from "./metrics";
import type { MetricValue } from "./metrics";
import type {
  Box,
  GroundTruthBoxInput,
  HealthResponse,
  InferFrameResponse,
  LabelsResponse,
  ModelCard,
  PatientSummary,
} from "./types";

const SPEED_OPTIONS = [0.5, 1, 2] as const;
const PREFETCH_LOOKAHEAD = 12;
const MIN_BOX_SIZE_PX = 3;

type Dimensions = {
  width: number;
  height: number;
};

type FrameLabels = {
  hasLabels: boolean;
  boxes: Box[];
};

type DraftBox = {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
};

type DisplayRect = {
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
  scale: number;
};

function emptyFrameLabels(): FrameLabels {
  return { hasLabels: false, boxes: [] };
}

function formatMetricValue(metric: MetricValue): string {
  if (metric.status === "na") {
    return "N/A";
  }
  return `${(metric.value * 100).toFixed(1)}%`;
}

function toGroundTruthInput(boxes: Box[]): GroundTruthBoxInput[] {
  return boxes.map((box) => ({
    x1: box.x1,
    y1: box.y1,
    x2: box.x2,
    y2: box.y2,
  }));
}

function normalizeDraftToBox(draft: DraftBox): Box | null {
  const x1 = Math.min(draft.startX, draft.currentX);
  const y1 = Math.min(draft.startY, draft.currentY);
  const x2 = Math.max(draft.startX, draft.currentX);
  const y2 = Math.max(draft.startY, draft.currentY);
  if (x2 - x1 < MIN_BOX_SIZE_PX || y2 - y1 < MIN_BOX_SIZE_PX) {
    return null;
  }

  return {
    x1,
    y1,
    x2,
    y2,
    confidence: 1,
    classId: 0,
    className: "stenosis",
  };
}

function App() {
  const [models, setModels] = useState<ModelCard[]>([]);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("");

  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEED_OPTIONS)[number]>(1);

  const [threshold, setThreshold] = useState(0.5);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [isAnnotating, setIsAnnotating] = useState(false);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [inference, setInference] = useState<InferFrameResponse | null>(null);
  const [frameLabels, setFrameLabels] = useState<FrameLabels>(emptyFrameLabels());
  const [editableBoxes, setEditableBoxes] = useState<Box[]>([]);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const [draftBox, setDraftBox] = useState<DraftBox | null>(null);
  const [activePointerId, setActivePointerId] = useState<number | null>(null);

  const [error, setError] = useState<string>("");
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const viewerRef = useRef<HTMLDivElement | null>(null);
  const [renderSize, setRenderSize] = useState<Dimensions>({ width: 1, height: 1 });
  const [naturalSize, setNaturalSize] = useState<Dimensions>({ width: 1, height: 1 });

  const selectedPatient = useMemo(
    () => patients.find((patient) => patient.id === selectedPatientId) ?? null,
    [patients, selectedPatientId],
  );
  const activeModel = useMemo(() => models.find((model) => model.active) ?? null, [models]);

  const thresholdBoxes = useMemo(
    () => (inference?.boxes ?? []).filter((box) => box.confidence >= threshold),
    [inference, threshold],
  );

  const stenosisDetected = inference?.stenosisDetected ?? false;
  const maxFrameIndex = selectedPatient ? Math.max(0, selectedPatient.frameCount - 1) : 0;
  const isAtEnd = selectedPatient !== null && frameIndex >= maxFrameIndex;
  const isInteractionLocked = isDirty || isSaving;

  const displayRect = useMemo<DisplayRect>(() => {
    const safeNaturalWidth = Math.max(1, naturalSize.width);
    const safeNaturalHeight = Math.max(1, naturalSize.height);
    const safeRenderWidth = Math.max(1, renderSize.width);
    const safeRenderHeight = Math.max(1, renderSize.height);

    const scale = Math.min(safeRenderWidth / safeNaturalWidth, safeRenderHeight / safeNaturalHeight);
    const width = safeNaturalWidth * scale;
    const height = safeNaturalHeight * scale;

    return {
      offsetX: (safeRenderWidth - width) / 2,
      offsetY: (safeRenderHeight - height) / 2,
      width,
      height,
      scale,
    };
  }, [naturalSize, renderSize]);

  const groundTruthBoxes = editableBoxes;

  const toNaturalPoint = (clientX: number, clientY: number, clampToImage: boolean): { x: number; y: number } | null => {
    if (!viewerRef.current) {
      return null;
    }

    const stageRect = viewerRef.current.getBoundingClientRect();
    const localX = clientX - stageRect.left;
    const localY = clientY - stageRect.top;

    const minX = displayRect.offsetX;
    const minY = displayRect.offsetY;
    const maxX = displayRect.offsetX + displayRect.width;
    const maxY = displayRect.offsetY + displayRect.height;

    if (!clampToImage && (localX < minX || localX > maxX || localY < minY || localY > maxY)) {
      return null;
    }

    const mappedX = clampToImage ? Math.min(maxX, Math.max(minX, localX)) : localX;
    const mappedY = clampToImage ? Math.min(maxY, Math.max(minY, localY)) : localY;

    return {
      x: (mappedX - displayRect.offsetX) / displayRect.scale,
      y: (mappedY - displayRect.offsetY) / displayRect.scale,
    };
  };

  const frameImageUrl = selectedPatient ? getFrameUrl(selectedPatient.id, frameIndex) : "";

  const frameMetrics = useMemo(
    () =>
      computeFrameMetrics({
        predictions: inference?.boxes ?? [],
        groundTruth: groundTruthBoxes,
        labelsAvailable: frameLabels.hasLabels,
        iouThreshold: 0.5,
      }),
    [groundTruthBoxes, inference, frameLabels.hasLabels],
  );

  const frameStateLabel = stenosisDetected ? "Stenosis detected" : "No stenosis detected";
  const inferenceMsLabel = inference ? `${inference.inferenceMs.toFixed(1)} ms` : "-";
  const sourceLabel = inference ? (inference.cached ? "cache" : health?.device === "mock" ? "mock" : "live") : "-";
  const metricsUnavailableReason = frameMetrics.iou.status === "na" ? frameMetrics.iou.reason : "";
  const shouldShowGroundTruth = isAnnotating || showGroundTruth;

  const draftRenderBox = useMemo(() => {
    if (!draftBox) {
      return null;
    }
    const x1 = Math.min(draftBox.startX, draftBox.currentX);
    const y1 = Math.min(draftBox.startY, draftBox.currentY);
    const x2 = Math.max(draftBox.startX, draftBox.currentX);
    const y2 = Math.max(draftBox.startY, draftBox.currentY);
    return { x1, y1, x2, y2 };
  }, [draftBox]);

  const handlePlayToggle = () => {
    if (isInteractionLocked) {
      return;
    }

    if (isPlaying) {
      setIsPlaying(false);
      return;
    }
    if (isAtEnd) {
      setFrameIndex(0);
    }
    setIsPlaying(true);
  };

  const handleSelectPatient = (patientId: string) => {
    if (isInteractionLocked || patientId === selectedPatientId) {
      return;
    }
    setSelectedPatientId(patientId);
    setFrameIndex(0);
    setInference(null);
    setFrameLabels(emptyFrameLabels());
    setEditableBoxes([]);
    setSelectedBoxIndex(null);
    setDraftBox(null);
    setIsDirty(false);
    setIsPlaying(false);
  };

  const handleSelectModel = async (modelId: ModelCard["id"]) => {
    const selected = models.find((model) => model.id === modelId);
    if (isSwitchingModel || isInteractionLocked || selected?.active || selected?.status !== "ready") {
      return;
    }

    setIsSwitchingModel(true);
    try {
      const updatedModels = await selectModel(modelId);
      const [updatedPatients, nextHealth] = await Promise.all([getPatients(), getHealth()]);
      setModels(updatedModels);
      setPatients(updatedPatients);
      setHealth(nextHealth);
      setInference(null);
      setFrameLabels(emptyFrameLabels());
      setEditableBoxes([]);
      setSelectedBoxIndex(null);
      setDraftBox(null);
      setIsDirty(false);
      setError("");
      setFrameIndex(0);
      setIsPlaying(false);
      setSelectedPatientId((prev) => {
        if (updatedPatients.some((patient) => patient.id === prev)) {
          return prev;
        }
        return updatedPatients[0]?.id ?? "";
      });
    } catch (switchError) {
      setError(switchError instanceof Error ? switchError.message : "Failed to switch model.");
    } finally {
      setIsSwitchingModel(false);
    }
  };

  const handleDeleteSelected = () => {
    if (selectedBoxIndex === null || !selectedPatient) {
      return;
    }
    setEditableBoxes((prev) => prev.filter((_, index) => index !== selectedBoxIndex));
    setSelectedBoxIndex(null);
    setIsDirty(true);
  };

  const handleDiscard = () => {
    setEditableBoxes(frameLabels.boxes);
    setSelectedBoxIndex(null);
    setDraftBox(null);
    setActivePointerId(null);
    setIsDirty(false);
  };

  const handleSave = async () => {
    if (!selectedPatient || isSaving) {
      return;
    }

    setIsSaving(true);
    try {
      const response = await saveLabels(selectedPatient.id, frameIndex, toGroundTruthInput(editableBoxes));
      const nextLabels = {
        hasLabels: response.hasLabels,
        boxes: response.boxes ?? [],
      };
      setFrameLabels(nextLabels);
      setEditableBoxes(nextLabels.boxes);
      setSelectedBoxIndex(null);
      setDraftBox(null);
      setIsDirty(false);
      setError("");
      setPatients((prev) =>
        prev.map((patient) => (patient.id === selectedPatient.id ? { ...patient, hasLabels: true } : patient)),
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Failed to save labels.");
    } finally {
      setIsSaving(false);
    }
  };

  useEffect(() => {
    let canceled = false;

    async function bootstrap() {
      try {
        const [modelsData, patientsData, healthData] = await Promise.all([getModels(), getPatients(), getHealth()]);
        if (canceled) {
          return;
        }
        setModels(modelsData);
        setPatients(patientsData);
        setHealth(healthData);

        if (patientsData.length > 0) {
          setSelectedPatientId(patientsData[0].id);
          setFrameLabels(emptyFrameLabels());
        }
      } catch (loadError) {
        if (!canceled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load app data.");
        }
      }
    }

    void bootstrap();
    return () => {
      canceled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedPatient) {
      return;
    }
    const patient = selectedPatient;

    let canceled = false;

    async function loadFrameContext() {
      try {
        const labelsPromise = patient.hasLabels
          ? getLabels(patient.id, frameIndex)
          : Promise.resolve<LabelsResponse>({
              patientId: patient.id,
              frameIndex,
              hasLabels: false,
              labelType: "bbox",
              boxes: [],
              mask: null,
            });

        const [nextInference, nextLabels] = await Promise.all([inferFrame(patient.id, frameIndex), labelsPromise]);

        if (canceled) {
          return;
        }

        setError("");
        setInference(nextInference);
        const nextFrameLabels = {
          hasLabels: nextLabels.hasLabels,
          boxes: nextLabels.boxes ?? [],
        };
        setFrameLabels(nextFrameLabels);
        setEditableBoxes(nextFrameLabels.boxes);
        setSelectedBoxIndex(null);
        setDraftBox(null);
        setActivePointerId(null);
        setIsDirty(false);

        const start = Math.min(frameIndex + 1, maxFrameIndex);
        const end = Math.min(frameIndex + PREFETCH_LOOKAHEAD, maxFrameIndex);
        if (end >= start) {
          void prefetchFrames(patient.id, start, end);
        }
      } catch (loadError) {
        if (canceled) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Failed to load frame inference.");
      }
    }

    void loadFrameContext();

    return () => {
      canceled = true;
    };
  }, [frameIndex, maxFrameIndex, selectedPatient]);

  useEffect(() => {
    if (!selectedPatient || !isPlaying || isInteractionLocked) {
      return;
    }

    const intervalMs = 1000 / (selectedPatient.defaultFps * speed);
    const timer = window.setTimeout(() => {
      setFrameIndex((prev) => {
        if (prev >= maxFrameIndex) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      window.clearTimeout(timer);
    };
  }, [isInteractionLocked, isPlaying, selectedPatient, speed, maxFrameIndex, frameIndex]);

  useEffect(() => {
    if (!viewerRef.current) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) {
        return;
      }
      setRenderSize({
        width: Math.max(1, entry.contentRect.width),
        height: Math.max(1, entry.contentRect.height),
      });
    });

    observer.observe(viewerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (isInteractionLocked && isPlaying) {
      setIsPlaying(false);
    }
  }, [isInteractionLocked, isPlaying]);

  return (
    <div className="app-shell">
      <div className="ambient-gradient" />
      <div className="scanline-overlay" />

      <header className="top-header">
        <div className="title-wrap">
          <p className="eyebrow">CADICA DEMO</p>
          <h1>Coronary Angiography CADICA Viewer</h1>
        </div>
        <div className="health-chip">
          <span className="chip-label">Device</span>
          <strong>{health?.device ?? "-"}</strong>
          <span className="chip-meta">Cache {health?.cacheEntries ?? 0}/{health?.cacheSize ?? 0}</span>
          <span className="chip-meta">{activeModel?.name ?? "No model active"}</span>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="layout-grid">
        <aside className="left-rail">
          <section className="rail-card">
            <h2>Models</h2>
            <div className="model-list">
              {models.map((model) => {
                const selectable = !model.active && model.status === "ready";
                const canSelect = selectable && !isSwitchingModel && !isInteractionLocked;

                return (
                  <article
                    key={model.id}
                    className={`model-card ${model.active ? "active" : "inactive"} ${selectable ? "selectable" : ""} ${
                      !canSelect ? "disabled" : ""
                    }`}
                    role={selectable ? "button" : undefined}
                    tabIndex={selectable ? 0 : -1}
                    aria-disabled={selectable ? !canSelect : undefined}
                    onClick={() => {
                      if (!canSelect) {
                        return;
                      }
                      void handleSelectModel(model.id);
                    }}
                    onKeyDown={(event) => {
                      if (!canSelect) {
                        return;
                      }
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        void handleSelectModel(model.id);
                      }
                    }}
                  >
                    <div className="model-header-row">
                      <h3>{model.name}</h3>
                      <span className={`state-pill ${model.status}`}>{model.status.replace("_", " ")}</span>
                    </div>
                    <p>{model.notes}</p>
                    <div className="model-meta-row">
                      <span>{model.datasetId}</span>
                      <span>{model.outputType}</span>
                      <span>{model.inferenceMode}</span>
                    </div>
                    {model.active ? <span className="disabled-note">Active model</span> : null}
                    {!model.active && model.status !== "ready" ? <span className="disabled-note">Unavailable</span> : null}
                    {!model.active && model.status === "ready" ? (
                      <span className="disabled-note">
                        {isSwitchingModel
                          ? "Switching..."
                          : isInteractionLocked
                            ? "Save or discard annotation edits first"
                            : "Click card to activate"}
                      </span>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </section>

          <section className="rail-card patients-card">
            <h2>Sequences</h2>
            <div className="patient-list">
              {patients.length === 0 ? <p className="empty-text">No CADICA sequences loaded.</p> : null}
              {patients.map((patient) => (
                <button
                  key={patient.id}
                  className={`patient-button ${selectedPatientId === patient.id ? "selected" : ""}`}
                  onClick={() => handleSelectPatient(patient.id)}
                  type="button"
                  disabled={isInteractionLocked}
                >
                  <span className="patient-name">{patient.displayName}</span>
                  <span className="patient-meta">{patient.frameCount} frames</span>
                  <span className="patient-badges">{patient.datasetId} · labeled {patient.hasLabels ? "yes" : "no"}</span>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <section className="viewer-panel">
          <div className="viewer-header-row">
            <div className={`classification-banner ${stenosisDetected ? "alert" : "clear"}`}>{frameStateLabel}</div>
            <div className="header-metadata">
              <div className="frame-meta">
                <span>
                  Frame <strong>{frameIndex + 1}</strong>/{selectedPatient ? selectedPatient.frameCount : 0}
                </span>
                <span>
                  Inference <strong>{inferenceMsLabel}</strong>
                </span>
                <span>
                  Source <strong>{sourceLabel}</strong>
                </span>
                <span>
                  Threshold <strong>{threshold.toFixed(2)}</strong>
                </span>
              </div>
              <div className="quality-metrics">
                <div className="metric-chip">
                  <span className="metric-label">IoU (frame)</span>
                  <strong>{formatMetricValue(frameMetrics.iou)}</strong>
                </div>
              </div>
              {metricsUnavailableReason ? <span className="metrics-note">{metricsUnavailableReason}</span> : null}
              {isDirty ? <span className="metrics-note dirty-note">Unsaved annotation changes</span> : null}
            </div>
          </div>

          <div className="viewer-stage" ref={viewerRef}>
            {selectedPatient ? (
              <img
                src={frameImageUrl}
                alt={`CADICA sequence ${selectedPatient.displayName} frame ${frameIndex + 1}`}
                onLoad={(event) => {
                  const image = event.currentTarget;
                  setNaturalSize((prev) => {
                    const width = Math.max(1, image.naturalWidth);
                    const height = Math.max(1, image.naturalHeight);
                    if (prev.width === width && prev.height === height) {
                      return prev;
                    }
                    return { width, height };
                  });
                }}
              />
            ) : (
              <div className="empty-stage">Load CADICA sequence data to start the demo.</div>
            )}

            <div
              className={`overlay-layer ${isAnnotating ? "annotation-enabled" : ""}`}
              onPointerDown={(event) => {
                if (!isAnnotating || isSaving || !selectedPatient || event.button !== 0) {
                  return;
                }
                const point = toNaturalPoint(event.clientX, event.clientY, false);
                if (!point) {
                  return;
                }
                event.currentTarget.setPointerCapture(event.pointerId);
                setActivePointerId(event.pointerId);
                setSelectedBoxIndex(null);
                setDraftBox({
                  startX: point.x,
                  startY: point.y,
                  currentX: point.x,
                  currentY: point.y,
                });
              }}
              onPointerMove={(event) => {
                if (activePointerId !== event.pointerId || !draftBox) {
                  return;
                }
                const point = toNaturalPoint(event.clientX, event.clientY, true);
                if (!point) {
                  return;
                }
                setDraftBox((prev) => {
                  if (!prev) {
                    return prev;
                  }
                  return {
                    ...prev,
                    currentX: point.x,
                    currentY: point.y,
                  };
                });
              }}
              onPointerUp={(event) => {
                if (activePointerId !== event.pointerId || !draftBox) {
                  return;
                }
                const point = toNaturalPoint(event.clientX, event.clientY, true);
                const finalDraft = point
                  ? {
                      ...draftBox,
                      currentX: point.x,
                      currentY: point.y,
                    }
                  : draftBox;
                const box = normalizeDraftToBox(finalDraft);

                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }

                if (box) {
                  setEditableBoxes((prev) => [...prev, box]);
                  setSelectedBoxIndex(editableBoxes.length);
                  setIsDirty(true);
                }

                setDraftBox(null);
                setActivePointerId(null);
              }}
              onPointerCancel={(event) => {
                if (activePointerId !== event.pointerId) {
                  return;
                }
                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }
                setDraftBox(null);
                setActivePointerId(null);
              }}
            >
              {thresholdBoxes.map((box, index) => (
                <div
                  key={`pred-${index}-${box.x1}-${box.y1}`}
                  className="bbox prediction"
                  style={{
                    left: `${displayRect.offsetX + box.x1 * displayRect.scale}px`,
                    top: `${displayRect.offsetY + box.y1 * displayRect.scale}px`,
                    width: `${(box.x2 - box.x1) * displayRect.scale}px`,
                    height: `${(box.y2 - box.y1) * displayRect.scale}px`,
                  }}
                >
                  <span>{Math.round(box.confidence * 100)}%</span>
                </div>
              ))}
              {shouldShowGroundTruth
                ? groundTruthBoxes.map((box, index) => (
                    <div
                      key={`gt-${index}-${box.x1}-${box.y1}`}
                      className={`bbox ground-truth ${selectedBoxIndex === index ? "selected" : ""}`}
                      style={{
                        left: `${displayRect.offsetX + box.x1 * displayRect.scale}px`,
                        top: `${displayRect.offsetY + box.y1 * displayRect.scale}px`,
                        width: `${(box.x2 - box.x1) * displayRect.scale}px`,
                        height: `${(box.y2 - box.y1) * displayRect.scale}px`,
                      }}
                      onPointerDown={(event) => {
                        if (!isAnnotating) {
                          return;
                        }
                        event.stopPropagation();
                        setSelectedBoxIndex(index);
                      }}
                    >
                      <span>GT</span>
                    </div>
                  ))
                : null}
              {isAnnotating && draftRenderBox ? (
                <div
                  className="bbox draft-box"
                  style={{
                    left: `${displayRect.offsetX + draftRenderBox.x1 * displayRect.scale}px`,
                    top: `${displayRect.offsetY + draftRenderBox.y1 * displayRect.scale}px`,
                    width: `${(draftRenderBox.x2 - draftRenderBox.x1) * displayRect.scale}px`,
                    height: `${(draftRenderBox.y2 - draftRenderBox.y1) * displayRect.scale}px`,
                  }}
                >
                  <span>Draft</span>
                </div>
              ) : null}
            </div>
          </div>

          <div className="control-deck">
            <div className="annotation-controls">
              <button
                type="button"
                className={showGroundTruth ? "selected" : ""}
                onClick={() => setShowGroundTruth((prev) => !prev)}
                disabled={!selectedPatient || isSaving}
              >
                {showGroundTruth ? "Hide ground truth" : "Show ground truth"}
              </button>
              <button
                type="button"
                className={isAnnotating ? "selected" : ""}
                onClick={() => {
                  const next = !isAnnotating;
                  setIsAnnotating(next);
                  if (next) {
                    setShowGroundTruth(true);
                  } else {
                    setSelectedBoxIndex(null);
                    setDraftBox(null);
                    setActivePointerId(null);
                  }
                }}
                disabled={!selectedPatient || isSaving}
              >
                {isAnnotating ? "Exit annotation" : "Annotate frame"}
              </button>
              <button type="button" onClick={handleDeleteSelected} disabled={!isAnnotating || selectedBoxIndex === null || isSaving}>
                Delete selected
              </button>
              <button type="button" onClick={handleDiscard} disabled={!isDirty || isSaving}>
                Discard edits
              </button>
              <button type="button" onClick={() => void handleSave()} disabled={!isDirty || isSaving || !selectedPatient}>
                {isSaving ? "Saving..." : "Save frame labels"}
              </button>
            </div>

            <div className="transport-controls">
              <button
                type="button"
                onClick={() => setFrameIndex((value) => Math.max(0, value - 1))}
                disabled={isInteractionLocked || !selectedPatient}
              >
                Prev
              </button>
              <button type="button" onClick={handlePlayToggle} disabled={isInteractionLocked || !selectedPatient}>
                {isPlaying ? "Pause" : isAtEnd ? "Play again" : "Play"}
              </button>
              <button
                type="button"
                onClick={() => setFrameIndex((value) => Math.min(maxFrameIndex, value + 1))}
                disabled={isInteractionLocked || !selectedPatient}
              >
                Next
              </button>
            </div>

            <input
              type="range"
              min={0}
              max={maxFrameIndex}
              value={frameIndex}
              onChange={(event) => setFrameIndex(Number(event.target.value))}
              disabled={isInteractionLocked || !selectedPatient}
            />

            <div className="slider-row">
              <label>
                Confidence threshold
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
              </label>
              <span>{threshold.toFixed(2)}</span>
            </div>

            <div className="speed-row">
              {SPEED_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={speed === option ? "selected" : ""}
                  onClick={() => setSpeed(option)}
                  disabled={!selectedPatient}
                >
                  {option}x
                </button>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
