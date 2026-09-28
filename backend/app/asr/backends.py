"""Local, dedicated-Pod, and RunPod Serverless ASR adapters."""

from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod

import httpx
import numpy as np

from ..audio import encode_wav
from ..config import Settings
from ..schemas import RawAsrResult
from .engine import QwenCommandEngine


class AsrBackend(ABC):
    name: str

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, beams: int) -> RawAsrResult:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError


class LocalAsrBackend(AsrBackend):
    name = "local"

    def __init__(self, settings: Settings) -> None:
        self.engine = QwenCommandEngine(
            settings.foundation,
            settings.adapter,
            settings.inference_config,
            settings.device,
        )
        self._lock = asyncio.Lock()

    @property
    def device(self) -> str:
        return f"{self.engine.device}/{str(self.engine.dtype).replace('torch.', '')}"

    def _release_unused_buffers(self):
        if self.engine.device == 'mps':
            import torch
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except RuntimeError:
                logging.getLogger(__name__).warning('Could not release unused MPS buffers')

    async def _locked_inference(self, operation):
        async with self._lock:
            worker = asyncio.create_task(asyncio.to_thread(operation))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Cancelling an await cannot stop PyTorch's worker thread. Keep
                # the model lock until it finishes so a new recording cannot
                # race that thread or its allocator cleanup.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not worker.cancelled():
                    worker.exception()
                raise

    async def transcribe(self, audio: np.ndarray, beams: int) -> RawAsrResult:
        def run():
            try:
                return self.engine.transcribe(audio, beams, "local")
            finally:
                self._release_unused_buffers()
        return await self._locked_inference(run)

    async def transcribe_verified(self, audio, beams, verifier):
        """Keep decoding and frozen-feature scoring under the same model lock."""
        def run():
            try:
                raw = self.engine.transcribe(audio, beams, "local")
                scores = None
                if verifier.available:
                    try:
                        features = self.engine.extract_features(audio)
                        scores = verifier.score(features, raw.hypotheses)
                    except Exception:
                        # Keep genuine evidence when verification is unavailable.
                        scores = None
                return raw, scores
            finally:
                self._release_unused_buffers()
        return await self._locked_inference(run)


class _RemoteBackend(AsrBackend):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def device(self) -> str:
        return "remote-gpu"

    @staticmethod
    def payload(audio: np.ndarray, beams: int) -> dict:
        return {
            "audio_base64": base64.b64encode(encode_wav(audio)).decode("ascii"),
            "filename": "normalized.wav",
            "beams": beams,
        }


class PodAsrBackend(_RemoteBackend):
    name = "pod"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.pod_url or not settings.pod_token:
            raise RuntimeError("ECHORA_POD_URL and ECHORA_POD_TOKEN are required")

    async def transcribe(self, audio: np.ndarray, beams: int) -> RawAsrResult:
        headers = {"Authorization": f"Bearer {self.settings.pod_token}"}
        async with httpx.AsyncClient(timeout=self.settings.remote_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.pod_url.rstrip('/')}/v1/transcribe",
                headers=headers,
                json=self.payload(audio, beams),
            )
            response.raise_for_status()
            return RawAsrResult.model_validate(response.json())


class RunpodAsrBackend(_RemoteBackend):
    name = "runpod"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.runpod_endpoint_id or not settings.runpod_api_key:
            raise RuntimeError("RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY are required")

    async def transcribe(self, audio: np.ndarray, beams: int) -> RawAsrResult:
        endpoint = self.settings.runpod_endpoint_id
        key = self.settings.runpod_api_key
        base = f"https://api.runpod.ai/v2/{endpoint}"
        headers = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=30) as client:
            submitted = await client.post(
                f"{base}/run", headers=headers, json={"input": self.payload(audio, beams)}
            )
            submitted.raise_for_status()
            job_id = submitted.json()["id"]
            deadline = asyncio.get_running_loop().time() + self.settings.remote_timeout_seconds
            pause = 1.0
            while asyncio.get_running_loop().time() < deadline:
                status = await client.get(f"{base}/status/{job_id}", headers=headers)
                status.raise_for_status()
                body = status.json()
                if body.get("status") == "COMPLETED":
                    return RawAsrResult.model_validate(body["output"])
                if body.get("status") in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                    raise RuntimeError(f"RunPod job {body.get('status')}: {body.get('error', 'unknown error')}")
                await asyncio.sleep(pause)
                pause = min(pause * 1.35, 5.0)
        raise TimeoutError("RunPod did not complete before ECHORA_REMOTE_TIMEOUT_SECONDS")


def create_backend(settings: Settings) -> AsrBackend:
    if settings.backend == "local":
        return LocalAsrBackend(settings)
    if settings.backend == "pod":
        return PodAsrBackend(settings)
    return RunpodAsrBackend(settings)
