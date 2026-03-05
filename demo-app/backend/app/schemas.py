from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DatasetId = Literal["mendeley", "arcade"]
LabelType = Literal["bbox", "mask"]
OutputType = Literal["bbox", "mask"]
InferenceMode = Literal["live", "precomputed", "mock"]


class ModelCard(BaseModel):
    id: Literal["yolo26s", "yolo26n", "sam_vmnet_arcade"]
    name: str
    active: bool
    status: Literal["ready", "unavailable"]
    notes: str
    datasetId: DatasetId
    outputType: OutputType
    inferenceMode: InferenceMode


class PatientSummary(BaseModel):
    id: str
    displayName: str
    frameCount: int
    hasLabels: bool
    defaultFps: int
    datasetId: DatasetId
    labelType: LabelType


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    classId: int
    className: Literal["stenosis"] = "stenosis"


class GroundTruthBoxInput(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class MaskPayload(BaseModel):
    url: str
    width: int
    height: int
    positivePixelRatio: float


class InferFrameRequest(BaseModel):
    patientId: str
    frameIndex: int = Field(ge=0)


class InferFrameResponse(BaseModel):
    patientId: str
    frameIndex: int
    outputType: OutputType
    boxes: list[Box] = Field(default_factory=list)
    mask: MaskPayload | None = None
    stenosisDetected: bool
    cached: bool
    inferenceMs: float


class PrefetchRequest(BaseModel):
    patientId: str
    startFrame: int = Field(ge=0)
    endFrame: int = Field(ge=0)


class PrefetchResponse(BaseModel):
    patientId: str
    queued: int
    startFrame: int
    endFrame: int


class ModelSelectRequest(BaseModel):
    modelId: Literal["yolo26s", "yolo26n", "sam_vmnet_arcade"]


class SaveLabelsRequest(BaseModel):
    boxes: list[GroundTruthBoxInput]


class LabelsResponse(BaseModel):
    patientId: str
    frameIndex: int
    hasLabels: bool
    labelType: LabelType
    boxes: list[Box] = Field(default_factory=list)
    mask: MaskPayload | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelLoaded: bool
    modelPath: str
    device: str
    cacheSize: int
    cacheEntries: int
    prefetchQueueSize: int
    prefetchQueued: int
