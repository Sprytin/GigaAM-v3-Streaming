"""Export a GigaAM RNN-T checkpoint to ONNX with GigaAM's own exporter, quantize its encoder.

Writes ``<name>_encoder.onnx`` (fp32), ``<name>_encoder_int8.onnx`` (dynamic INT8 MatMuls, as
gigastt ships it), ``<name>_decoder.onnx``, ``<name>_joint.onnx``, ``<name>.yaml`` into ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-channel", action="store_true", help="per-channel INT8 weights")
    parser.add_argument(
        "--suffix", default="int8", help="name of the quantized encoder: <name>_encoder_<suffix>"
    )
    parser.add_argument(
        "--skip-export", action="store_true", help="only quantize an existing fp32 encoder"
    )
    args = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    args.out.mkdir(parents=True, exist_ok=True)
    fp32 = args.out / f"{args.model}_encoder.onnx"
    if not args.skip_export:
        import gigaam

        model = gigaam.load_model(
            args.model, fp16_encoder=False, device="cpu", download_root=args.download_root
        )
        model.to_onnx(str(args.out))
    quantize_dynamic(
        str(fp32),
        str(args.out / f"{args.model}_encoder_{args.suffix}.onnx"),
        op_types_to_quantize=["MatMul"],
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    for path in sorted(args.out.iterdir()):
        print(f"{path.name}\t{path.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
