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
} from "./api";
import type { Box, HealthResponse, InferFrameResponse, ModelCard, PatientSummary } from "./types";

const SPEED_OPTIONS = [0.5, 1, 2] as const;
const PREFETCH_LOOKAHEAD = 12;

type Dimensions = {
  width: number;
  height: number;
};

function App() {
  const [models, setModels] = useState<ModelCard[]>([]);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("");

  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEED_OPTIONS)[number]>(1);

  const [threshold, setThreshold] = useState(0.5);
  const [showGroundTruth, setShowGroundTruth] = useState(true);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [inference, setInference] = useState<InferFrameResponse | null>(null);
  const [groundTruth, setGroundTruth] = useState<Box[]>([]);

  const [error, setError] = useState<string>("");

  const viewerRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [renderSize, setRenderSize] = useState<Dimensions>({ width: 1, height: 1 });
  const [naturalSize, setNaturalSize] = useState<Dimensions>({ width: 1, height: 1 });

  const selectedPatient = useMemo(
    () => patients.find((patient) => patient.id === selectedPatientId) ?? null,
    [patients, selectedPatientId],
  );

  const thresholdBoxes = useMemo(
    () => (inference?.boxes ?? []).filter((box) => box.confidence >= threshold),
    [inference, threshold],
  );

  const stenosisDetected = thresholdBoxes.length > 0;
  const maxFrameIndex = selectedPatient ? Math.max(0, selectedPatient.frameCount - 1) : 0;
  const isAtEnd = selectedPatient !== null && frameIndex >= maxFrameIndex;

  const handlePlayToggle = () => {
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
    if (patientId === selectedPatientId) {
      return;
    }
    setSelectedPatientId(patientId);
    setFrameIndex(0);
    setInference(null);
    setGroundTruth([]);
    setIsPlaying(false);
  };

  useEffect(() => {
    let canceled = false;

    async function bootstrap() {
      try {
        const [modelsData, patientsData, healthData] = await Promise.all([
          getModels(),
          getPatients(),
          getHealth(),
        ]);
        if (canceled) {
          return;
        }
        setModels(modelsData);
        setPatients(patientsData);
        setHealth(healthData);

        if (patientsData.length > 0) {
          setSelectedPatientId(patientsData[0].id);
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
        const [nextInference, nextLabels] = await Promise.all([
          inferFrame(patient.id, frameIndex),
          showGroundTruth && patient.hasLabels
            ? getLabels(patient.id, frameIndex)
            : Promise.resolve({ boxes: [] }),
        ]);

        if (canceled) {
          return;
        }

        setError("");
        setInference(nextInference);
        setGroundTruth(nextLabels.boxes ?? []);

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
  }, [frameIndex, maxFrameIndex, selectedPatient, showGroundTruth]);

  useEffect(() => {
    if (!selectedPatient) {
      return;
    }

    if (!isPlaying) {
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
  }, [isPlaying, selectedPatient, speed, maxFrameIndex, frameIndex]);

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

  const xScale = renderSize.width / naturalSize.width;
  const yScale = renderSize.height / naturalSize.height;

  const frameImageUrl = selectedPatient ? getFrameUrl(selectedPatient.id, frameIndex) : "";

  const frameStateLabel = stenosisDetected ? "Stenosis detected" : "No stenosis detected";
  const inferenceMsLabel = inference ? `${inference.inferenceMs.toFixed(1)} ms` : "-";
  const sourceLabel = inference ? (inference.cached ? "cache" : "live") : "-";

  return (
    <div className="app-shell">
      <div className="ambient-gradient" />
      <div className="scanline-overlay" />

      <header className="top-header">
        <div className="title-wrap">
          <p className="eyebrow">AXION RESEARCH DEMO</p>
          <h1>Coronary Angiography Live Classifier</h1>
        </div>
        <div className="health-chip">
          <span className="chip-label">Device</span>
          <strong>{health?.device ?? "-"}</strong>
          <span className="chip-meta">Cache {health?.cacheEntries ?? 0}/{health?.cacheSize ?? 0}</span>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <main className="layout-grid">
        <aside className="left-rail">
          <section className="rail-card">
            <h2>Models</h2>
            <div className="model-list">
              {models.map((model) => (
                <article key={model.id} className={`model-card ${model.active ? "active" : "inactive"}`}>
                  <div className="model-header-row">
                    <h3>{model.name}</h3>
                    <span className={`state-pill ${model.status}`}>{model.status.replace("_", " ")}</span>
                  </div>
                  <p>{model.notes}</p>
                  {!model.active ? <span className="disabled-note">Controls disabled in v1</span> : null}
                </article>
              ))}
            </div>
          </section>

          <section className="rail-card">
            <h2>Patients</h2>
            <div className="patient-list">
              {patients.length === 0 ? <p className="empty-text">No curated data loaded.</p> : null}
              {patients.map((patient) => (
                <button
                  key={patient.id}
                  className={`patient-button ${selectedPatientId === patient.id ? "selected" : ""}`}
                  onClick={() => handleSelectPatient(patient.id)}
                  type="button"
                >
                  <span className="patient-name">{patient.displayName}</span>
                  <span className="patient-meta">{patient.frameCount} frames</span>
                </button>
              ))}
            </div>
          </section>
        </aside>

        <section className="viewer-panel">
          <div className="viewer-header-row">
            <div className={`classification-banner ${stenosisDetected ? "alert" : "clear"}`}>
              {frameStateLabel}
            </div>
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
            </div>
          </div>

          <div className="viewer-stage" ref={viewerRef}>
            {selectedPatient ? (
              <img
                ref={imageRef}
                src={frameImageUrl}
                alt={`Patient ${selectedPatient.displayName} frame ${frameIndex + 1}`}
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
              <div className="empty-stage">Load patient data to start the demo.</div>
            )}

            <div className="overlay-layer">
              {thresholdBoxes.map((box, index) => (
                <div
                  key={`pred-${index}-${box.x1}-${box.y1}`}
                  className="bbox prediction"
                  style={{
                    left: `${box.x1 * xScale}px`,
                    top: `${box.y1 * yScale}px`,
                    width: `${(box.x2 - box.x1) * xScale}px`,
                    height: `${(box.y2 - box.y1) * yScale}px`,
                  }}
                >
                  <span>{Math.round(box.confidence * 100)}%</span>
                </div>
              ))}
              {showGroundTruth
                ? groundTruth.map((box, index) => (
                    <div
                      key={`gt-${index}-${box.x1}-${box.y1}`}
                      className="bbox ground-truth"
                      style={{
                        left: `${box.x1 * xScale}px`,
                        top: `${box.y1 * yScale}px`,
                        width: `${(box.x2 - box.x1) * xScale}px`,
                        height: `${(box.y2 - box.y1) * yScale}px`,
                      }}
                    >
                      <span>GT</span>
                    </div>
                  ))
                : null}
            </div>
          </div>

          <div className="control-deck">
            <div className="transport-controls">
              <button type="button" onClick={() => setFrameIndex((value) => Math.max(0, value - 1))}>
                Prev
              </button>
              <button type="button" onClick={handlePlayToggle}>
                {isPlaying ? "Pause" : isAtEnd ? "Play again" : "Play"}
              </button>
              <button
                type="button"
                onClick={() => setFrameIndex((value) => Math.min(maxFrameIndex, value + 1))}
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
              className="timeline-slider"
            />

            <div className="knob-row">
              <label>
                Threshold
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.01}
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
                <span>{threshold.toFixed(2)}</span>
              </label>

              <div className="speed-group">
                {SPEED_OPTIONS.map((option) => (
                  <button
                    key={option}
                    type="button"
                    className={speed === option ? "selected" : ""}
                    onClick={() => setSpeed(option)}
                  >
                    {option}x
                  </button>
                ))}
              </div>

              <label className="toggle-label">
                <input
                  type="checkbox"
                  checked={showGroundTruth}
                  onChange={(event) => setShowGroundTruth(event.target.checked)}
                  disabled={!selectedPatient?.hasLabels}
                />
                Ground-truth overlay
              </label>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
