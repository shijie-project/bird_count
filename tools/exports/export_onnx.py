"""Export the ShuffleNet density model to ONNX with dynamic batch / spatial axes.

Usage:
    python -m tools.export_onnx --ckpt ../ckpts/best.pth --out ../ckpts/best.onnx
"""

import argparse
import logging

import torch

from models.shufflenet import get_shufflenet_density_model


logger = logging.getLogger(__name__)


def export(model_path: str, onnx_path: str, h: int = 720, w: int = 1080, opset: int = 11):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_shufflenet_density_model(model_path=model_path, device=device, fuse=True)
    model.eval()
    dummy = torch.randn(1, 3, h, w, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "out_height", 3: "out_width"},
        },
    )
    logger.info(f"Wrote {onnx_path}")


def main():
    p = argparse.ArgumentParser(description="Export ShuffleNetDensityNet to ONNX")
    p.add_argument("--ckpt", required=True, help="path to .pth or .tar checkpoint")
    p.add_argument("--out", required=True, help="output .onnx path")
    p.add_argument("--h", type=int, default=720, help="dummy input height")
    p.add_argument("--w", type=int, default=1080, help="dummy input width")
    p.add_argument("--opset", type=int, default=11)
    p.add_argument("--check", action="store_true", help="run onnx.checker after export")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export(args.ckpt, args.out, args.h, args.w, args.opset)

    if args.check:
        import onnx

        m = onnx.load(args.out)
        onnx.checker.check_model(m)
        print("ONNX model check passed.")


if __name__ == "__main__":
    main()
