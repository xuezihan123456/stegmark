from __future__ import annotations

import argparse
from pathlib import Path

from stegmark.training.export import export_hidden_onnx


def main() -> int:
    parser = argparse.ArgumentParser(description="Export hidden encoder/decoder ONNX graphs.")
    parser.add_argument("--message-bits", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder_checkpoint", type=Path, default=None)
    parser.add_argument("--decoder_checkpoint", type=Path, default=None)
    args = parser.parse_args()

    export_hidden_onnx(
        message_bits=args.message_bits,
        image_size=args.image_size,
        encoder_output=args.output_dir / "encoder.onnx",
        decoder_output=args.output_dir / "decoder.onnx",
        encoder_ckpt=args.encoder_checkpoint,
        decoder_ckpt=args.decoder_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
