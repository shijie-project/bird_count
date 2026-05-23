"""Latency benchmarks for the ShuffleNet density model.

Sweeps a configurable matrix of {backend × batch_size × resolution} and
reports median / mean / p95 latency, throughput (FPS), peak GPU memory
(PyTorch backends only), and per-config speedup vs a baseline backend.

Backends:
  pytorch-fp32   — PyTorch FP32, Conv+BN NOT fused (reference baseline)
  pytorch-fused  — PyTorch FP32, Conv+BN fused
  pytorch-amp    — PyTorch fused + torch.amp.autocast (mixed FP16/FP32)
  pytorch-fp16   — PyTorch fused, `.half()`-cast model + FP16 input
  onnx-cpu       — ONNX Runtime CPU EP, FP32
  onnx-cuda      — ONNX Runtime CUDA EP, FP32
  onnx-trt-fp32  — ONNX Runtime TensorRT EP, FP32
  onnx-trt-fp16  — ONNX Runtime TensorRT EP, FP16
  onnx-trt-int8  — ONNX Runtime TensorRT EP, INT8 (+ FP16 fallback); needs --int8-cache

Usage:
    # Default sweep — every backend at B=1,4,16,64 @ 720x1080
    python -m tools.test_inference_time --onnx ../ckpts/best.onnx

    # Custom matrix + CSV output
    python -m tools.test_inference_time \\
        --backends pytorch-amp,onnx-trt-fp16,onnx-trt-int8 \\
        --batch-sizes 1,8,32 \\
        --resolutions 720x1080,1080x1920 \\
        --onnx ../ckpts/best.onnx \\
        --int8-cache ../ckpts/best.int8.cache \\
        --csv-out bench.csv

    # Tighter CIs for a single config
    python -m tools.test_inference_time --backends pytorch-amp \\
        --batch-sizes 1 --resolutions 720x1080 --iterations 500
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import onnxruntime as ort
import torch
from torch.amp import autocast

from models.shufflenet import get_shufflenet_density_model


# ----------------------------------------------------------------------
# Backend registry — each factory returns (step_fn, human_label).
# step_fn is a zero-arg callable that performs ONE forward pass.
# ----------------------------------------------------------------------

BackendFactory = Callable[[argparse.Namespace, int, int, int, str], tuple[Callable[[], None], str]]
_BACKENDS: dict[str, BackendFactory] = {}


def _register(name: str) -> Callable[[BackendFactory], BackendFactory]:
    def deco(fn: BackendFactory) -> BackendFactory:
        _BACKENDS[name] = fn
        return fn

    return deco


# ----- PyTorch backends -----


def _pytorch_step(model: torch.nn.Module, dummy: torch.Tensor, use_amp: bool) -> Callable[[], None]:
    if use_amp:

        @torch.no_grad()
        def step() -> None:
            with autocast("cuda"):
                model(dummy)
    else:

        @torch.no_grad()
        def step() -> None:
            model(dummy)

    return step


@_register("pytorch-fp32")
def _f_pytorch_fp32(args, h, w, batch_size, device):
    model = get_shufflenet_density_model(model_path=args.ckpt, device=device, fuse=False).eval()
    dummy = torch.randn(batch_size, 3, h, w, device=device)
    return _pytorch_step(model, dummy, use_amp=False), "PyTorch FP32 (unfused)"


@_register("pytorch-fused")
def _f_pytorch_fused(args, h, w, batch_size, device):
    model = get_shufflenet_density_model(model_path=args.ckpt, device=device, fuse=True).eval()
    dummy = torch.randn(batch_size, 3, h, w, device=device)
    return _pytorch_step(model, dummy, use_amp=False), "PyTorch FP32 (fused)"


@_register("pytorch-amp")
def _f_pytorch_amp(args, h, w, batch_size, device):
    model = get_shufflenet_density_model(model_path=args.ckpt, device=device, fuse=True).eval()
    dummy = torch.randn(batch_size, 3, h, w, device=device)
    return _pytorch_step(model, dummy, use_amp=True), "PyTorch AMP (fused)"


@_register("pytorch-fp16")
def _f_pytorch_fp16(args, h, w, batch_size, device):
    model = get_shufflenet_density_model(model_path=args.ckpt, device=device, fuse=True).eval().half()
    dummy = torch.randn(batch_size, 3, h, w, device=device, dtype=torch.float16)
    return _pytorch_step(model, dummy, use_amp=False), "PyTorch FP16 (fused, .half())"


# ----- ONNX Runtime backends -----


def _require_onnx(args: argparse.Namespace) -> None:
    if not args.onnx:
        raise FileNotFoundError("--onnx not provided; required for onnx-* backends.")
    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")


def _ort_step(session: ort.InferenceSession, dummy_np: np.ndarray) -> Callable[[], None]:
    input_name = session.get_inputs()[0].name

    def step() -> None:
        session.run(None, {input_name: dummy_np})

    return step


def _build_ort_session(args, *, providers) -> ort.InferenceSession:
    _require_onnx(args)
    so = ort.SessionOptions()
    so.log_severity_level = 3  # quiet
    return ort.InferenceSession(args.onnx, sess_options=so, providers=providers)


def _trt_providers(args, *, fp16: bool, int8: bool) -> list:
    trt_opts = {
        "device_id": 0,
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
        "trt_fp16_enable": fp16,
    }
    if int8:
        if not args.int8_cache or not os.path.exists(args.int8_cache):
            raise FileNotFoundError(
                f"INT8 calibration cache not found: {args.int8_cache!r}. Run tools/calibrate_trt_int8.py first."
            )
        trt_opts["trt_int8_enable"] = True
        trt_opts["trt_int8_calibration_table_name"] = args.int8_cache
        trt_opts["trt_int8_use_native_calibration_table"] = False
    return [
        ("TensorrtExecutionProvider", trt_opts),
        ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}),
        "CPUExecutionProvider",
    ]


@_register("onnx-cpu")
def _f_onnx_cpu(args, h, w, batch_size, device):
    session = _build_ort_session(args, providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    return _ort_step(session, dummy_np), "ONNX-RT CPU EP (FP32)"


@_register("onnx-cuda")
def _f_onnx_cuda(args, h, w, batch_size, device):
    session = _build_ort_session(
        args,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    )
    dummy_np = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    return _ort_step(session, dummy_np), "ONNX-RT CUDA EP (FP32)"


@_register("onnx-trt-fp32")
def _f_onnx_trt_fp32(args, h, w, batch_size, device):
    session = _build_ort_session(args, providers=_trt_providers(args, fp16=False, int8=False))
    dummy_np = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    return _ort_step(session, dummy_np), "ONNX-RT TRT FP32"


@_register("onnx-trt-fp16")
def _f_onnx_trt_fp16(args, h, w, batch_size, device):
    session = _build_ort_session(args, providers=_trt_providers(args, fp16=True, int8=False))
    dummy_np = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    return _ort_step(session, dummy_np), "ONNX-RT TRT FP16"


@_register("onnx-trt-int8")
def _f_onnx_trt_int8(args, h, w, batch_size, device):
    session = _build_ort_session(args, providers=_trt_providers(args, fp16=True, int8=True))
    dummy_np = np.random.randn(batch_size, 3, h, w).astype(np.float32)
    return _ort_step(session, dummy_np), "ONNX-RT TRT INT8+FP16"


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------


@dataclass
class Measurement:
    backend: str
    backend_label: str
    batch_size: int
    h: int
    w: int
    iterations: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    std_ms: float
    fps: float
    peak_mem_mb: float  # PyTorch backends only; 0.0 for ORT (ORT allocates outside torch's tracker)


def _measure(
    step: Callable[[], None],
    iterations: int,
    warmups: int,
    sync_cuda: bool,
    track_torch_mem: bool,
) -> tuple[list[float], float]:
    for _ in range(warmups):
        step()
    if sync_cuda:
        torch.cuda.synchronize()
    if track_torch_mem:
        torch.cuda.reset_peak_memory_stats()

    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        step()
        if sync_cuda:
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = 0.0
    if track_torch_mem:
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return samples_ms, peak_mb


def run_one(backend_name: str, args: argparse.Namespace, h: int, w: int, batch_size: int, device: str) -> Measurement:
    if backend_name not in _BACKENDS:
        raise ValueError(f"unknown backend {backend_name!r}; pick from {sorted(_BACKENDS)}")
    step, label = _BACKENDS[backend_name](args, h, w, batch_size, device)

    is_pytorch = backend_name.startswith("pytorch-")
    sync_cuda = (device == "cuda") and is_pytorch  # ORT manages its own sync internally
    track_torch_mem = (device == "cuda") and is_pytorch

    samples_ms, peak_mb = _measure(step, args.iterations, args.warmups, sync_cuda, track_torch_mem)

    median_ms = statistics.median(samples_ms)
    p95 = statistics.quantiles(samples_ms, n=20)[18] if len(samples_ms) >= 20 else max(samples_ms)
    std = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return Measurement(
        backend=backend_name,
        backend_label=label,
        batch_size=batch_size,
        h=h,
        w=w,
        iterations=args.iterations,
        median_ms=median_ms,
        mean_ms=statistics.mean(samples_ms),
        p95_ms=p95,
        std_ms=std,
        fps=batch_size / median_ms * 1000.0,
        peak_mem_mb=peak_mb,
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_table(measurements: list[Measurement], baseline_backend: str | None) -> None:
    if not measurements:
        print("\n(no successful measurements)")
        return

    # Speedup lookup: per (batch, h, w) → baseline median.
    base: dict[tuple[int, int, int], float] = {}
    if baseline_backend:
        for m in measurements:
            if m.backend == baseline_backend:
                base[(m.batch_size, m.h, m.w)] = m.median_ms

    width = 116
    print("\n" + "=" * width)
    header = (
        f"{'Backend':<30} {'B':>4} {'Resolution':>11} "
        f"{'Median (ms)':>12} {'Mean (ms)':>10} {'p95 (ms)':>10} "
        f"{'Std':>7} {'FPS':>8} {'Mem MB':>8} {'Speedup':>9}"
    )
    print(header)
    print("-" * width)

    # Stable ordering: by (h, w, batch, backend) for easy visual comparison
    sorted_ms = sorted(measurements, key=lambda m: (m.h, m.w, m.batch_size, m.backend))
    for m in sorted_ms:
        sp = base.get((m.batch_size, m.h, m.w))
        sp_str = f"{sp / m.median_ms:.2f}x" if sp else "—"
        mem_str = f"{m.peak_mem_mb:.1f}" if m.peak_mem_mb > 0 else "—"
        print(
            f"{m.backend_label:<30} {m.batch_size:>4} {m.h:>5}x{m.w:<5} "
            f"{m.median_ms:>12.2f} {m.mean_ms:>10.2f} {m.p95_ms:>10.2f} "
            f"{m.std_ms:>7.2f} {m.fps:>8.1f} {mem_str:>8} {sp_str:>9}"
        )
    print("=" * width)
    if baseline_backend:
        print(f"Speedup = baseline_median / this_median, where baseline = {baseline_backend!r}")
    print("Mem column: peak torch.cuda allocator usage (PyTorch backends only; ORT allocates outside torch).")


def write_csv(measurements: list[Measurement], path: str) -> None:
    field_names = [f.name for f in fields(Measurement)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_names)
        w.writeheader()
        for m in measurements:
            w.writerow({k: getattr(m, k) for k in field_names})
    print(f"\nWrote {len(measurements)} rows to {path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_resolutions(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for token in s.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(f"resolution {token!r} must look like HxW (e.g. 720x1080)")
        h_str, w_str = token.split("x", 1)
        h, w = int(h_str), int(w_str)
        if h % 8 or w % 8:
            raise ValueError(f"resolution {h}x{w} must have both dims divisible by 8 (U-Net concat requirement)")
        out.append((h, w))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latency benchmark sweep across runtimes, batch sizes, and input resolutions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--backends",
        default="pytorch-fp32,pytorch-fused,pytorch-amp,onnx-trt-fp16,onnx-trt-int8",
        help=f"Comma-separated backends. Available: {','.join(sorted(_BACKENDS))}",
    )
    p.add_argument("--batch-sizes", default="1,4,16,64", help="Comma-separated batch sizes to sweep")
    p.add_argument(
        "--resolutions",
        default="720x1080",
        help="Comma-separated HxW pairs (e.g. 360x640,720x1080,1080x1920). Both dims must be multiples of 8.",
    )
    p.add_argument("--iterations", type=int, default=100, help="Timed iterations per config (bigger = tighter CIs)")
    p.add_argument("--warmups", type=int, default=20, help="Warmup iterations per config (not measured)")
    p.add_argument(
        "--baseline",
        default="pytorch-fp32",
        help="Backend used as the reference for the Speedup column. Set to '' to disable.",
    )
    p.add_argument("--ckpt", default=None, help="Optional checkpoint for PyTorch backends (random init otherwise)")
    p.add_argument("--onnx", default=None, help="ONNX file path (required for onnx-* backends)")
    p.add_argument(
        "--int8-cache",
        default=None,
        help="TensorRT INT8 calibration cache path (required for onnx-trt-int8)",
    )
    p.add_argument("--csv-out", default=None, help="Optional CSV path to dump every measurement")
    p.add_argument("--list-backends", action="store_true", help="Print available backends and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_backends:
        for name in sorted(_BACKENDS):
            print(f"  {name}")
        return

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    resolutions = parse_resolutions(args.resolutions)
    baseline = args.baseline.strip() or None

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    if cuda_available:
        torch.backends.cudnn.benchmark = True

    print(f"Device           : {device}" + (f" — {torch.cuda.get_device_name(0)}" if cuda_available else ""))
    print(f"Backends         : {backends}")
    print(f"Batch sizes      : {batch_sizes}")
    print(f"Resolutions      : {[f'{h}x{w}' for h, w in resolutions]}")
    print(f"Iter / warmups   : {args.iterations} / {args.warmups}")
    if baseline:
        print(f"Baseline         : {baseline}")
    print()

    measurements: list[Measurement] = []
    n_configs = len(backends) * len(batch_sizes) * len(resolutions)
    i = 0
    for backend in backends:
        for h, w in resolutions:
            for batch_size in batch_sizes:
                i += 1
                tag = f"[{i}/{n_configs}] {backend} @ B={batch_size} {h}x{w}"
                print(f">> {tag}")
                try:
                    m = run_one(backend, args, h, w, batch_size, device)
                except Exception as exc:
                    print(f"   FAILED: {type(exc).__name__}: {exc}")
                    continue
                measurements.append(m)
                print(
                    f"   median {m.median_ms:7.2f} ms | p95 {m.p95_ms:7.2f} ms "
                    f"| fps {m.fps:7.1f} | mem {m.peak_mem_mb:6.1f} MB"
                )
                if cuda_available:
                    torch.cuda.empty_cache()

    print_table(measurements, baseline_backend=baseline)
    if args.csv_out:
        write_csv(measurements, args.csv_out)


if __name__ == "__main__":
    main()
