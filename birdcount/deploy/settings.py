"""Settings read by the live runtime."""

from __future__ import annotations

from pydantic import Field

from birdcount.core.settings import CoreSettings


class DeploySettings(CoreSettings):
    # Inference rate per camera. Frames are decoded at the camera's own rate,
    # but only this many per second are handed to the model.
    fps: int = Field(10, gt=0)
    # Frame slots per camera in the shared-memory ring.
    num_buffers: int = Field(4, ge=3)
    # Inference processes; camera i goes to worker i % num_workers_per_gpu.
    num_workers_per_gpu: int = Field(1, ge=1)
    cuda_device: str = "0"
    # Decode video on the GPU's video engine (NVDEC via D3D11 on Windows),
    # falling back to software per stream when unavailable.
    hw_decode: bool = True
