"""Single-GPU HTTP service for persistent, prompt-keyed T5 precaching."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from scail2_inference.contracts import ProductionProfile
from t5_precache_service.database import (
    T5CacheDatabase,
    T5CacheRecord,
    normalize_prompt,
)
from t5_precache_service.engine import T5PrecacheEngine

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_DIR / "work" / "cache"


class PrecacheRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=65536)


class T5PrecacheService:
    """Combine the resident encoder and persistent file database."""

    def __init__(self, engine: Any, database: T5CacheDatabase):
        self.engine = engine
        self.database = database

    def get_or_create(self, prompt: str) -> T5CacheRecord:
        return self.database.get_or_create(prompt, self.engine.encode)

    def health(self) -> dict[str, object]:
        return {
            "status": "ready",
            "engine": self.engine.health(),
            "cache": {
                "root": str(self.database.root),
                **self.database.statistics(),
            },
        }


def _file_response(record: T5CacheRecord) -> FileResponse:
    return FileResponse(
        record.path,
        media_type="application/octet-stream",
        filename=f"{record.prompt_hash}.safetensors",
        headers={
            "ETag": f'"{record.prompt_hash}"',
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-SCAIL2-Prompt-SHA256": record.prompt_hash,
            "X-SCAIL2-Cache-Hit": "true" if record.cache_hit else "false",
        },
    )


def create_app(service: T5PrecacheService) -> FastAPI:
    app = FastAPI(title="SCAIL-2 T5 Precache Service", version="1")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return service.health()

    @app.post("/v1/t5-cache", response_class=FileResponse)
    async def create_cache(request: PrecacheRequest) -> FileResponse:
        # This intentionally blocks the single Uvicorn event loop. The service
        # contract is serial: a second request waits until this one completes.
        try:
            prompt = normalize_prompt(request.prompt)
            record = service.get_or_create(prompt)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logging.exception("SCAIL2_T5_SERVICE request failed")
            raise HTTPException(status_code=500, detail=str(error)) from error
        logging.info(
            "SCAIL2_T5_SERVICE stage=request status=complete prompt_hash=%s cache_hit=%s",
            record.prompt_hash,
            record.cache_hit,
        )
        return _file_response(record)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="physical GPU index; the service exposes only this device to CUDA",
    )
    parser.add_argument(
        "--host", default=os.getenv("SCAIL2_T5_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("SCAIL2_T5_PORT", "8001"))
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.getenv("SCAIL2_T5_MODEL_DIR", "/models")),
        help="directory containing the T5 checkpoint and tokenizer files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("SCAIL2_T5_CACHE_DIR", str(DEFAULT_CACHE_DIR))),
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("SCAIL2_PROFILE", "scail2-512p-bf16-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu < 0:
        raise ValueError("gpu index must be nonnegative")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    # T5PrecacheEngine imports torch lazily, so CUDA visibility is finalized
    # here before the CUDA runtime can be initialized.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    profile = ProductionProfile.from_name(args.profile)
    engine = T5PrecacheEngine(args.checkpoint_dir, profile)
    engine.load()
    database = T5CacheDatabase(
        args.cache_dir,
        profile=profile.name,
        text_len=engine.text_len,
        t5_checkpoint=engine.t5_checkpoint,
    )
    app = create_app(T5PrecacheService(engine, database))

    import uvicorn

    logging.info(
        "SCAIL2_T5_SERVICE status=ready physical_gpu=%d host=%s port=%d cache_dir=%s",
        args.gpu,
        args.host,
        args.port,
        database.root,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
