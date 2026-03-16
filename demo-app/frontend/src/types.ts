export type DatasetId = "cadica";
export type LabelType = "bbox";
export type OutputType = "bbox";
export type InferenceMode = "live" | "mock";

export type ModelCard = {
  id: "yolo26m_cadica" | "yolo26x_cadica";
  name: string;
  active: boolean;
  status: "ready" | "unavailable";
  notes: string;
  datasetId: DatasetId;
  outputType: OutputType;
  inferenceMode: InferenceMode;
};

export type PatientSummary = {
  id: string;
  displayName: string;
  frameCount: number;
  hasLabels: boolean;
  defaultFps: number;
  datasetId: DatasetId;
  labelType: LabelType;
};

export type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  className: "stenosis";
};

export type InferFrameResponse = {
  patientId: string;
  frameIndex: number;
  outputType: OutputType;
  boxes: Box[];
  mask: null;
  stenosisDetected: boolean;
  cached: boolean;
  inferenceMs: number;
};

export type HealthResponse = {
  status: "ok";
  modelLoaded: boolean;
  modelPath: string;
  device: string;
  cacheSize: number;
  cacheEntries: number;
  prefetchQueueSize: number;
  prefetchQueued: number;
};

export type LabelsResponse = {
  patientId: string;
  frameIndex: number;
  hasLabels: boolean;
  labelType: LabelType;
  boxes: Box[];
  mask: null;
};

export type GroundTruthBoxInput = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};
