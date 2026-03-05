export type DatasetId = "mendeley" | "arcade";
export type LabelType = "bbox" | "mask";
export type OutputType = "bbox" | "mask";
export type InferenceMode = "live" | "precomputed" | "mock";

export type ModelCard = {
  id: "yolo26s" | "yolo26n" | "sam_vmnet_arcade";
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

export type MaskPayload = {
  url: string;
  width: number;
  height: number;
  positivePixelRatio: number;
};

export type InferFrameResponse = {
  patientId: string;
  frameIndex: number;
  outputType: OutputType;
  boxes: Box[];
  mask: MaskPayload | null;
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
  mask: MaskPayload | null;
};

export type GroundTruthBoxInput = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};
