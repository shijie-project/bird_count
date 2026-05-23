from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time

import cv2
import numpy as np
import torch

from datasets.transforms import IMAGENET_MEAN_BGR, IMAGENET_STD_BGR
from models.shufflenet import get_shufflenet_density_model
from runtime.config import Config
from runtime.shared_memory import BufferState, SharedMemory, SharedMemoryConfig
from utils import setup_cuda, setup_logging

from ._utils import BatchInferenceResult, InferenceResult, get_optimal_memory_format, permute_first_conv_for_bgr


logger = logging.getLogger(__name__)


# Idle-poll: how long to sleep when no streams have a READY frame this tick.
# Short enough to keep latency low; long enough to avoid 100%-CPU busy-spin.
_IDLE_POLL_SEC = 0.001

# Max time `result_queue.put` will block before we declare the consumer
# back-pressured and drop the batch (releasing its SHM buffers to FREE).
_DISPATCH_TIMEOUT_SEC = 0.01

# Back-off after an unexpected loop-level exception so we don't fault-spin.
_ERROR_BACKOFF_SEC = 0.01


class InferencerProcess(mp.Process):
    """
    High-Performance Inference Engine.
    Processes assigned video streams using GPU acceleration and Shared Memory.
    """

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        result_queue: mp.Queue,
        worker_id: int,
        total_workers: int,
        warmup_event: mp.synchronize.Event | None = None,
    ):
        # Unique name for tracing
        super().__init__(name=f"InferenceWorker-{worker_id}")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.total_workers = total_workers
        self._stop_event = mp.Event()
        # Set after _init_resource() finishes (incl. GPU warmup). ResultProcess
        # blocks on this so its GUI only appears once inference is ready —
        # otherwise the first user click sits behind 1-2s of warmup with no
        # log feedback visible to the operator.
        self.warmup_event = warmup_event

        # --- Stream Partitioning ---
        # Logic: stream_id % total_workers == worker_id
        self.assigned_streams = [s for s in range(self.config.num_streams) if s % total_workers == worker_id]

        # Placeholders for resources initialized in run()
        self.device: torch.device | None = None
        self.model: torch.nn.Module | None = None
        self.shm_client: SharedMemory | None = None

        # Optimization Constants
        self._fused_scale: torch.Tensor | None = None
        self._fused_bias: torch.Tensor | None = None
        self._memory_format: torch.memory_format | None = None
        self._interval = config.frame_interval

        # Pinned host buffers — allocated in _init_resource() once we know
        # we are running with CUDA. `_pinned_input` enables truly-async H2D
        # for the frame batch; `_pinned_density` does the same for the
        # density-map D2H on the way back.
        self._pinned_input: torch.Tensor | None = None
        self._pinned_input_np: np.ndarray | None = None
        self._pinned_density: torch.Tensor | None = None

    def _init_resource(self):
        """Initializes GPU and SHM resources within the child process context."""
        try:
            setup_cuda()
            self.device = torch.device(self.config.device)
            # The hot path (pinned H2D, AMP autocast, channels-last) is CUDA-only.
            # Fail loudly here rather than crashing on a NoneType subscript in
            # `_preprocess_batch_on_gpu` once a batch arrives.
            if self.device.type != "cuda":
                raise RuntimeError(
                    f"InferencerProcess requires a CUDA device; got {self.device}. "
                    "Set config.device to a 'cuda' device or run inference outside this process."
                )

            # Detect optimal memory layout (NHWC for Tensor Cores)
            self._memory_format = get_optimal_memory_format(self.device)

            # Pre-calculate Fused Normalization: x * scale + bias = (x/255 - mean)/std.
            # mean/std are in BGR order because the first conv layer is permuted
            # below to accept BGR directly — this lets us skip the BGR→RGB flip
            # entirely in the hot path.
            mean = torch.tensor(IMAGENET_MEAN_BGR).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor(IMAGENET_STD_BGR).view(1, 3, 1, 1).to(self.device)
            self._fused_scale = 1.0 / (255.0 * std)
            self._fused_bias = -(mean / std)

            # Load Model with Structural Fusion
            self.model = get_shufflenet_density_model(self.config.model.path, device=self.device, fuse=True)
            self.model.eval()

            # Re-key the first Conv2d so the model consumes BGR input directly
            # (equivalent to feeding it RGB; the math is symmetric — see helper).
            permute_first_conv_for_bgr(self.model, name=self.name)

            # Connect to Shared Memory
            self.shm_client = SharedMemory(self.shm_config, name=self.name)
            self.shm_client.connect()

            # Pinned host buffers for true async DMA in both directions.
            max_b = max(1, len(self.assigned_streams))
            H, W, C = self.shm_config.frames.shape[2:]
            self._pinned_input = torch.empty((max_b, H, W, C), dtype=torch.uint8, pin_memory=True)
            # Numpy view shares memory with the pinned tensor — we stage frames
            # here so torch's .to(device, non_blocking=True) is genuinely async.
            self._pinned_input_np = self._pinned_input.numpy()
            logger.info(
                "[%s] Allocated pinned input H2D buffer: (%d, %d, %d, %d) uint8 (%.1f MB)",
                self.name,
                max_b,
                H,
                W,
                C,
                self._pinned_input.numel() / (1024 * 1024),
            )
            if self.shm_config.density is not None:
                _, _, H_d, W_d = self.shm_config.density.shape
                self._pinned_density = torch.empty((max_b, H_d, W_d), dtype=torch.float16, pin_memory=True)
                logger.info(
                    "[%s] Allocated pinned density D2H buffer: (%d, %d, %d) float16 (%.1f KB)",
                    self.name,
                    max_b,
                    H_d,
                    W_d,
                    self._pinned_density.numel() * 2 / 1024,
                )

            self._warmup_gpu()

            logger.info("[%s] Initialization complete. Ready for streams %s", self.name, self.assigned_streams)

        except Exception as e:
            logger.critical("[%s] Startup failed: %s", self.name, e, exc_info=True)
            raise

    def _warmup_gpu(self):
        """
        Warmup every batch size from 1 .. max_assigned so cudnn.benchmark autotune
        does not stall the first runtime batch of any partial-batch size.
        """
        max_bsz = len(self.assigned_streams)
        if max_bsz <= 0:
            return
        warmup_sizes = list(range(1, max_bsz + 1))
        logger.info("[%s] Starting GPU Warmup for batch sizes %s...", self.name, warmup_sizes)

        with torch.inference_mode():
            for bsz in warmup_sizes:
                if self._pinned_input_np is not None:
                    self._pinned_input_np[:bsz].fill(0)
                input_t, _ = self._preprocess_batch_on_gpu(bsz)
                with torch.amp.autocast("cuda"):
                    _ = self.model(input_t)

                logger.debug("[%s] Warmup for Batch Size %d complete.", self.name, bsz)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _preprocess_batch_on_gpu(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Issue H2D from the pinned input buffer and apply fused normalization.

        Frames must already be staged into `self._pinned_input_np[:n]`. Because
        that buffer is pinned, the .to(device, non_blocking=True) is genuinely
        async — the next `_collect_batch_from_shm` can populate while the GPU
        runs the forward pass. The model accepts BGR directly (conv1 permuted
        at startup), so no BGR→RGB flip is needed.
        """
        raw_tensor = self._pinned_input[:n].to(self.device, non_blocking=True)
        input_tensor = raw_tensor.permute(0, 3, 1, 2).to(memory_format=self._memory_format).float()
        input_tensor.mul_(self._fused_scale).add_(self._fused_bias)
        return input_tensor, raw_tensor

    def _collect_batch_from_shm(self) -> tuple[int, np.ndarray | None, list, list]:
        """Pick the freshest READY frame per assigned stream and stage them into
        the pinned host input buffer for async H2D.

        Returns (batch_size, batch_meta, sids, b_idxs). batch_size==0 means
        nothing was ready this tick — batch_meta is None and the two lists empty.
        """
        ready_pairs = [
            (s_id, b_idx) for s_id in self.assigned_streams if (b_idx := self._get_latest_frame(s_id)) != -1
        ]

        if not ready_pairs:
            return 0, None, [], []

        sids, b_idxs = map(list, zip(*ready_pairs))

        # Bulk update metadata state to READING (minimizes memory barriers)
        self.shm_client.metadata[sids, b_idxs]["state"] = BufferState.READING

        n = len(sids)
        frames = self.shm_client.frames
        # Stage into pinned host buffer. Per-stream copies via np.copyto are
        # ~as fast as the previous fancy-indexing single-shot copy, and land
        # the data in pinned memory so the H2D below is truly async.
        if self._pinned_input_np is not None:
            pinned_np = self._pinned_input_np
            for i in range(n):
                np.copyto(pinned_np[i], frames[sids[i], b_idxs[i]])

        # Populate the downscaled preview block for the dashboard renderer.
        # Done here (rather than in MonitorHandler) so the resize cost is paid
        # once per frame and amortized across every display tick that uses it.
        # INTER_AREA gives the best visual result for downsampling.
        preview = self.shm_client.preview
        if preview is not None:
            ph, pw = preview.shape[2], preview.shape[3]
            for i in range(n):
                cv2.resize(
                    frames[sids[i], b_idxs[i]],
                    (pw, ph),
                    dst=preview[sids[i], b_idxs[i]],
                    interpolation=cv2.INTER_AREA,
                )

        batch_meta = self.shm_client.metadata[sids, b_idxs]

        return n, batch_meta, sids, b_idxs

    def _get_latest_frame(self, stream_id: int) -> int:
        """Selects the newest READY frame and clears stale buffers in bulk."""
        stream_meta = self.shm_client.stream_metadata[stream_id]
        ready_mask = stream_meta["state"] == BufferState.READY

        if not ready_mask.any():
            return -1

        ready_indices = np.where(ready_mask)[0]
        ready_f_indices = stream_meta["frame_idx"][ready_mask]

        # Pick max frame_idx
        best_local_idx = np.argmax(ready_f_indices)
        best_idx = ready_indices[best_local_idx]

        # Bulk free stale frames
        stale_mask = ready_mask.copy()
        stale_mask[best_idx] = False
        if stale_mask.any():
            stream_meta["state"][stale_mask] = BufferState.FREE

        return best_idx

    def run(self):
        """Main Inference Loop."""
        setup_logging(self.config.envs.debug)
        self._init_resource()
        if self.warmup_event is not None:
            self.warmup_event.set()

        write_density = self.shm_client.density is not None

        while not self._stop_event.is_set():
            try:
                t0 = time.perf_counter()

                # --- 1. Collection ---
                n, batch_meta, sids, b_idxs = self._collect_batch_from_shm()
                if not sids:
                    time.sleep(_IDLE_POLL_SEC)
                    continue

                # --- 2. GPU Preprocessing ---
                input_tensor, raw_tensor = self._preprocess_batch_on_gpu(n)
                t1 = time.perf_counter()

                # --- 3. Inference ---
                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    density_map = self.model(input_tensor)  # (B, 1, H/8, W/8)
                    counts_tensor = torch.sum(density_map, dim=(1, 2, 3))

                # --- 4. Data Transfer (D2H) ---
                # Density to dedicated SHM block (FP16, ~50KB/stream/frame). Raw
                # frames SHM is intentionally NOT written so the recorder always
                # reads pristine camera frames; the heatmap is composed lazily
                # in MonitorHandler at display time.
                density_fp16 = None
                if write_density:
                    density_fp16 = density_map.squeeze(1).to(torch.float16)
                    if self._pinned_density is not None:
                        pinned_view = self._pinned_density[:n]
                        pinned_view.copy_(density_fp16, non_blocking=True)

                # Move counts to CPU (tiny transfer; this also serializes the prior
                # non_blocking copy because both use the same default stream).
                counts_cpu = counts_tensor.cpu().numpy()

                if write_density:
                    if self._pinned_density is not None:
                        self.shm_client.density[sids, b_idxs] = self._pinned_density[:n].numpy()
                    else:
                        self.shm_client.density[sids, b_idxs] = density_fp16.cpu().numpy()

                t2 = time.perf_counter()
                process_time = time.time()

                # --- 5. Packaging (Optimized List Comprehension) ---
                batch_packet = BatchInferenceResult(
                    results=[
                        InferenceResult(
                            stream_id=int(batch_meta[i]["stream_id"]),
                            buffer_idx=int(batch_meta[i]["buffer_idx"]),
                            frame_idx=int(batch_meta[i]["frame_idx"]),
                            timestamp=float(batch_meta[i]["timestamp"]),
                            count=float(counts_cpu[i]),
                            latency=process_time - float(batch_meta[i]["timestamp"]),
                        )
                        for i in range(n)
                    ]
                )

                # --- 6. Dispatch ---
                try:
                    self.result_queue.put(batch_packet, timeout=_DISPATCH_TIMEOUT_SEC)
                except queue.Full:
                    # Critical: Release SHM buffers if consumer is blocked to prevent deadlock
                    self.shm_client.metadata[sids, b_idxs]["state"] = BufferState.FREE
                    logger.warning("[%s] Output Queue Full. Dropping %d frames.", self.name, n)

                self._log_performance(batch_meta, t0, t1, t2)

                # FPS Control
                elapsed = time.perf_counter() - t0
                if elapsed < self._interval:
                    time.sleep(self._interval - elapsed)

            except Exception as e:
                logger.error("[%s] Loop Error: %s", self.name, e, exc_info=True)
                time.sleep(_ERROR_BACKOFF_SEC)

        self.shm_client.disconnect()
        logger.info("[%s] Stopped.", self.name)

    def stop(self):
        self._stop_event.set()

    def _log_performance(self, batch_meta: np.ndarray, t0: float, t1: float, t2: float):
        """Calculates and logs throughput metrics. No-op unless verbose_debug=True."""
        if not self.config.envs.verbose_debug:
            return
        t3 = time.perf_counter()
        now = time.time()
        batch_size = len(batch_meta)

        avg_age = sum([(now - m["timestamp"]) for m in batch_meta]) / batch_size * 1000
        prep_ms = (t1 - t0) * 1000
        infer_ms = (t2 - t1) * 1000
        total_ms = (t3 - t0) * 1000

        logger.debug(
            "[%s] Batch=%d | Age=%.1fms | Prep=%.1fms | Infer=%.1fms | Total=%.1fms",
            self.name,
            batch_size,
            avg_age,
            prep_ms,
            infer_ms,
            total_ms,
        )
