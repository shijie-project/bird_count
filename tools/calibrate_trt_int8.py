"""Build a TensorRT INT8 calibration cache for the ShuffleNet density ONNX model.

Uses ONNX Runtime's TensorRT execution provider with `trt_int8_enable=True`.
On the first inference run the EP populates the calibration table from the
data we feed it; subsequent runs (in benchmarks or production) reuse the
cache and skip recalibration.

Usage:
    python -m tools.calibrate_trt_int8 \\
        --onnx ../ckpts/best.onnx \\
        --cache ../ckpts/best.int8.cache \\
        --data-dir ../data --split val \\
        --batch-size 64 --max-batches 200
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import collate
from datasets.bird import Bird


def build_calibration_cache(
    onnx_path: str,
    cache_path: str,
    data_dir: str,
    split: str,
    batch_size: int,
    max_batches: int,
    fp16_fallback: bool,
):
    import onnxruntime as ort

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if os.path.exists(cache_path):
        print(f"Calibration cache already exists at {cache_path}; remove it to rebuild.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)

    # 384 is a common training crop_size; the dataset only needs it to refuse
    # un-divisible val sizes. Adjust if your val frames differ.
    dataset = Bird(data_dir, crop_size=384, downsample_ratio=8, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate,
    )

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GB
                "trt_fp16_enable": fp16_fallback,
                "trt_int8_enable": True,
                "trt_int8_calibration_table_name": cache_path,
                "trt_int8_use_native_calibration_table": False,
            },
        ),
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    print(f"Building INT8 cache: onnx={onnx_path}, cache={cache_path}")
    print(f"Calibration data: {split} split @ batch={batch_size}, up to {max_batches} batches")

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    n_seen = 0
    t0 = time.perf_counter()
    for sample in loader:
        if n_seen >= max_batches:
            break
        x = sample["image"].numpy().astype(np.float32)
        # Val batches have native sizes — TensorRT EP will allocate per shape.
        session.run(None, {input_name: x})
        n_seen += 1
        if n_seen % 10 == 0:
            print(f"  calibrated on {n_seen} batches")
    dt = time.perf_counter() - t0
    print(f"Done: {n_seen} batches in {dt:.1f}s. Cache written to {cache_path}.")


def main():
    p = argparse.ArgumentParser(description="Build TRT INT8 calibration cache")
    p.add_argument("--onnx", required=True, help="path to ONNX model")
    p.add_argument("--cache", required=True, help="output calibration cache path")
    p.add_argument("--data-dir", default="../data", help="dataset root")
    p.add_argument("--split", default="val", help="dataset split to calibrate on")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=200, help="number of calibration batches; 100-500 is typical")
    p.add_argument(
        "--no-fp16-fallback", action="store_true", help="disable FP16 fallback for layers TRT can't quantize"
    )
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    build_calibration_cache(
        onnx_path=args.onnx,
        cache_path=args.cache,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        fp16_fallback=not args.no_fp16_fallback,
    )


if __name__ == "__main__":
    main()
