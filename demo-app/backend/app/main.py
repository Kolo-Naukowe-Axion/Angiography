from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .classification import has_stenosis
from .config import Settings
from .data import ManifestValidationError, PatientStore
from .inference import MockInferenceService, YOLOInferenceService
from .label_utils import parse_yolo_labels_to_boxes
from .models_registry import MODEL_PATHS, get_model_cards, get_model_path
from .schemas import (
    HealthResponse,
    InferFrameRequest,
    InferFrameResponse,
    ModelSelectRequest,
    PrefetchRequest,
    PrefetchResponse,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()

    async def _create_model_service(model_path: Path):
        if app_settings.use_mock_model:
            return await MockInferenceService.create(
                model_path=model_path,
                cache_size=app_settings.cache_size,
                min_infer_confidence=app_settings.min_infer_confidence,
                prefetch_queue_size=app_settings.prefetch_queue_size,
            )
        return await YOLOInferenceService.from_weights(
            model_path=model_path,
            cache_size=app_settings.cache_size,
            min_infer_confidence=app_settings.min_infer_confidence,
            prefetch_queue_size=app_settings.prefetch_queue_size,
        )

    async def _probe_model(model_path: Path) -> tuple[bool, str | None]:
        if not app_settings.use_mock_model and not model_path.exists():
            return False, f"weights not found at {model_path}"

        probe_service = None
        try:
            probe_service = await _create_model_service(model_path)
            await probe_service.start()
            return True, None
        except Exception as error:
            return False, str(error)
        finally:
            if probe_service is not None:
                try:
                    await probe_service.stop()
                except Exception:
                    pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            patient_store = PatientStore(app_settings.data_dir)
        except ManifestValidationError as error:
            raise RuntimeError(f"Manifest validation failed: {error}") from error

        model_service = await _create_model_service(app_settings.model_path)

        await model_service.start()
        app.state.settings = app_settings
        app.state.model_switch_lock = asyncio.Lock()
        app.state.patient_store = patient_store
        app.state.model_service = model_service
        model_availability: dict[str, bool] = {}
        model_availability_reasons: dict[str, str] = {}
        for model_id, model_path in MODEL_PATHS.items():
            if model_path == app_settings.model_path:
                model_availability[model_id] = True
                continue
            is_ready, reason = await _probe_model(model_path)
            model_availability[model_id] = is_ready
            if reason:
                model_availability_reasons[model_id] = reason
        app.state.model_availability = model_availability
        app.state.model_availability_reasons = model_availability_reasons
        yield
        await app.state.model_service.stop()

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
        return get_model_cards(
            app.state.settings.model_path,
            app.state.model_availability,
            app.state.model_availability_reasons,
        )

    @app.post("/api/models/select")
    async def select_model(payload: ModelSelectRequest):
        model_path = get_model_path(payload.modelId)
        if model_path is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model id: {payload.modelId}")

        async with app.state.model_switch_lock:
            current_model_path = app.state.settings.model_path
            if current_model_path == model_path:
                return get_model_cards(
                    current_model_path,
                    app.state.model_availability,
                    app.state.model_availability_reasons,
                )

            previous_service = app.state.model_service
            try:
                next_service = await _create_model_service(model_path)
                await next_service.start()
            except Exception as error:
                app.state.model_availability[payload.modelId] = False
                app.state.model_availability_reasons[payload.modelId] = str(error)
                raise HTTPException(status_code=500, detail=f"Failed to load selected model: {error}") from error

            app.state.model_service = next_service
            app.state.settings = replace(app.state.settings, model_path=model_path)
            app.state.model_availability[payload.modelId] = True
            app.state.model_availability_reasons.pop(payload.modelId, None)
            await previous_service.stop()
            return get_model_cards(
                model_path,
                app.state.model_availability,
                app.state.model_availability_reasons,
            )

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
