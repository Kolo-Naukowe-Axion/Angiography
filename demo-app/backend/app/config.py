from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "yolo26s" / "weights" / "best.pt"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_path: Path
    cache_size: int
    min_infer_confidence: float
    frontend_origin: str
    use_mock_model: bool
    prefetch_queue_size: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("DEMO_DATA_DIR", ROOT_DIR / "demo-app/data/patients")).resolve()
        model_path = Path(os.getenv("DEMO_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
        cache_size = int(os.getenv("DEMO_CACHE_SIZE", "512"))
        min_infer_confidence = float(os.getenv("DEMO_MIN_INFER_CONFIDENCE", "0.10"))
        frontend_origin = os.getenv("DEMO_FRONTEND_ORIGIN", "http://127.0.0.1:5173")
        use_mock_model = os.getenv("DEMO_USE_MOCK_MODEL", "0").lower() in {"1", "true", "yes"}
        prefetch_queue_size = int(os.getenv("DEMO_PREFETCH_QUEUE_SIZE", "256"))
        return cls(
            data_dir=data_dir,
            model_path=model_path,
            cache_size=cache_size,
            min_infer_confidence=min_infer_confidence,
            frontend_origin=frontend_origin,
            use_mock_model=use_mock_model,
            prefetch_queue_size=prefetch_queue_size,
        )
