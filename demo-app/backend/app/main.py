from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .classification import has_stenosis
from .config import Settings
from .data import ManifestValidationError, PatientStore
from .inference import MockInferenceService, YOLOInferenceService
from .label_utils import parse_yolo_labels_to_boxes
from .models_registry import get_model_cards
from .schemas import (
    HealthResponse,
    InferFrameRequest,
    InferFrameResponse,
    PrefetchRequest,
    PrefetchResponse,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            patient_store = PatientStore(app_settings.data_dir)
        except ManifestValidationError as error:
            raise RuntimeError(f"Manifest validation failed: {error}") from error

        if app_settings.use_mock_model:
            model_service = await MockInferenceService.create(
                model_path=app_settings.model_path,
                cache_size=app_settings.cache_size,
                min_infer_confidence=app_settings.min_infer_confidence,
                prefetch_queue_size=app_settings.prefetch_queue_size,
            )
        else:
            model_service = await YOLOInferenceService.from_weights(
                model_path=app_settings.model_path,
                cache_size=app_settings.cache_size,
                min_infer_confidence=app_settings.min_infer_confidence,
                prefetch_queue_size=app_settings.prefetch_queue_size,
            )

        await model_service.start()
        app.state.settings = app_settings
        app.state.patient_store = patient_store
        app.state.model_service = model_service
        yield
        await model_service.stop()

    app = FastAPI(title="Angiography Demo API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[app_settings.frontend_origin, "http://localhost:5173", "http://127.0.0.1:5173"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        health_payload = app.state.model_service.health()
        return HealthResponse(status="ok", **health_payload)

    @app.get("/api/models")
    async def models():
        return get_model_cards()

    @app.get("/api/patients")
    async def patients():
        return app.state.patient_store.summaries()

    @app.get("/api/patients/{patient_id}/frames/{frame_index}")
    async def frame(patient_id: str, frame_index: int):
        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}") from error
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        return FileResponse(frame_path)

    @app.post("/api/infer/frame", response_model=InferFrameResponse)
    async def infer_frame(payload: InferFrameRequest) -> InferFrameResponse:
        try:
            frame_path = app.state.patient_store.get_frame_path(payload.patientId, payload.frameIndex)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {payload.patientId}") from error
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {payload.frameIndex}") from error

        return await app.state.model_service.infer_frame(payload.patientId, payload.frameIndex, frame_path)

    @app.post("/api/infer/prefetch", response_model=PrefetchResponse)
    async def prefetch(payload: PrefetchRequest) -> PrefetchResponse:
        try:
            patient = app.state.patient_store.get_patient(payload.patientId)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {payload.patientId}") from error

        start = min(payload.startFrame, payload.endFrame)
        end = max(payload.startFrame, payload.endFrame)
        end = min(end, len(patient.frames) - 1)

        queued = 0
        for frame_index in range(start, end + 1):
            frame_path = patient.frames[frame_index]
            if await app.state.model_service.queue_prefetch(payload.patientId, frame_index, frame_path):
                queued += 1

        return PrefetchResponse(
            patientId=payload.patientId,
            queued=queued,
            startFrame=start,
            endFrame=end,
        )

    @app.get("/api/labels/{patient_id}/{frame_index}")
    async def labels(patient_id: str, frame_index: int):
        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index)
            label_path = app.state.patient_store.get_label_path(patient_id, frame_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}") from error
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        if label_path is None:
            return {"patientId": patient_id, "frameIndex": frame_index, "hasLabels": False, "boxes": []}

        boxes = parse_yolo_labels_to_boxes(label_path, frame_path)
        return {"patientId": patient_id, "frameIndex": frame_index, "hasLabels": True, "boxes": boxes}

    @app.get("/api/classification/{patient_id}/{frame_index}")
    async def classification(patient_id: str, frame_index: int, threshold: float = 0.5):
        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}") from error
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        inference = await app.state.model_service.infer_frame(patient_id, frame_index, frame_path)
        detected = has_stenosis(inference.boxes, threshold)
        return {
            "patientId": patient_id,
            "frameIndex": frame_index,
            "threshold": threshold,
            "stenosisDetected": detected,
            "boxCount": len(inference.boxes),
        }

    return app


app = create_app()
