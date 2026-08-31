"""Resident T5 encoder for the prompt precache service."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from scail2_inference.conditioning import NEGATIVE_CONTEXT, TEXT_CONTEXT
from scail2_inference.contracts import ProductionProfile


class T5PrecacheEngine:
    """Load T5 once and encode prompts serially on one visible GPU."""

    def __init__(self, t5_dir: Path, profile: ProductionProfile):
        self.t5_dir = Path(t5_dir)
        self.profile = profile
        self.device: Any | None = None
        self.config: Any | None = None
        self.encoder: Any | None = None
        self.negative_context: Any | None = None

    @property
    def t5_checkpoint(self) -> Path:
        if self.config is None:
            raise RuntimeError("T5 engine is not loaded")
        return self.t5_dir / Path(self.config.t5_checkpoint).name

    @property
    def text_len(self) -> int:
        if self.config is None:
            raise RuntimeError("T5 engine is not loaded")
        return int(self.config.text_len)

    def load(self) -> None:
        if self.encoder is not None:
            raise RuntimeError("T5 engine is already loaded")
        started = time.monotonic()
        import torch
        from wan.configs import SCAIL_CONFIGS
        from wan.modules.t5 import T5EncoderModel

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("T5 service requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.config = SCAIL_CONFIGS[self.profile.model.upper()]
        checkpoint = self.t5_checkpoint
        tokenizer = self.t5_dir
        if not checkpoint.is_file() or not tokenizer.exists():
            raise FileNotFoundError(
                "T5 checkpoint or tokenizer installation is incomplete"
            )

        logging.info("SCAIL2_T5_SERVICE stage=model_load status=start")
        self.encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            checkpoint_path=str(checkpoint),
            tokenizer_path=str(tokenizer),
            meta_load=True,
        )
        with torch.inference_mode():
            self.negative_context = (
                self.encoder([""], self.device)[0].detach().cpu().contiguous()
            )
        logging.info(
            "SCAIL2_T5_SERVICE stage=model_load status=complete elapsed_seconds=%.3f",
            time.monotonic() - started,
        )

    def encode(self, prompt: str) -> dict[str, object]:
        if self.encoder is None or self.device is None or self.negative_context is None:
            raise RuntimeError("T5 engine is not loaded")
        started = time.monotonic()
        import torch

        with torch.inference_mode():
            text_context = (
                self.encoder([prompt], self.device)[0].detach().cpu().contiguous()
            )
        logging.info(
            "SCAIL2_T5_SERVICE stage=encode status=complete elapsed_seconds=%.3f tokens=%d",
            time.monotonic() - started,
            text_context.shape[0],
        )
        return {
            TEXT_CONTEXT: text_context,
            NEGATIVE_CONTEXT: self.negative_context,
        }

    def health(self) -> dict[str, object]:
        return {
            "ready": self.encoder is not None,
            "profile": self.profile.name,
            "model": self.profile.model,
            "device": "cuda:0" if self.encoder is not None else None,
            "text_len": None if self.config is None else int(self.config.text_len),
            "checkpoint": None if self.config is None else str(self.t5_checkpoint),
        }
