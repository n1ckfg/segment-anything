#!/usr/bin/env python3
"""
Export MobileSAM (TinyViT-5M image encoder + SAM mask decoder) to FP16 ONNX
for fast, compact in-browser inference with onnxruntime-web (WebGPU / WASM).
"""

import argparse
import inspect
import os
import tempfile
import warnings

import onnx
import torch
import torch.nn as nn
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel
from onnxruntime.transformers import float16 as ort_fp16
from onnxruntime.transformers.onnx_model import OnnxModel


class MobileSamEncoderOnnx(nn.Module):
    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(input_image)


class FusedLayerNorm2d(nn.Module):
    """Replaces manual pow(2) LayerNorm2d with a single ONNX LayerNormalization op."""

    def __init__(self, ln2d: nn.Module) -> None:
        super().__init__()
        num_channels = ln2d.weight.shape[0]
        self.ln = nn.LayerNorm(num_channels, eps=ln2d.eps)
        with torch.no_grad():
            self.ln.weight.copy_(ln2d.weight)
            self.ln.bias.copy_(ln2d.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def fuse_layernorm2d(module: nn.Module) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ == "LayerNorm2d":
            setattr(module, name, FusedLayerNorm2d(child))
            count += 1
        else:
            count += fuse_layernorm2d(child)
    return count


def set_gelu_tanh(module: nn.Module) -> None:
    for _, m in module.named_modules():
        if isinstance(m, nn.GELU):
            m.approximate = "tanh"


def convert_to_fp16(fp32_path: str, fp16_path: str, fp32_node_prefixes=()) -> None:
    model = onnx.load(fp32_path)
    einsums = [n.name for n in model.graph.node if n.op_type == "Einsum"]
    print(f"  FP32 graph: nodes={len(model.graph.node)}, einsum_nodes={len(einsums)}")
    # Keep LayerNorm2d blocks (which compute (x - u).pow(2), overflowing FP16 when |x| > 256,
    # e.g. neck.2 outputs reach -608 -> (-608)^2 = 370k > 65504) in FP32.
    node_block_list = [
        n.name
        for n in model.graph.node
        if any(p in n.name for p in fp32_node_prefixes)
    ]
    print(f"  Keeping {len(node_block_list)} LayerNorm2d nodes in FP32")
    model_fp16 = ort_fp16.convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=node_block_list,
    )
    om = OnnxModel(model_fp16)
    om.topological_sort()
    om.save_model_to_file(fp16_path)
    onnx.checker.check_model(fp16_path)
    size_mb = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"  Saved & verified FP16: {fp16_path} ({size_mb:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MobileSAM encoder & decoder to FP16 ONNX.")
    parser.add_argument("--checkpoint", type=str, default="/Users/nick/Downloads/mobile_sam.pt")
    parser.add_argument("--out-dir", type=str, default="demo2")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    print(f"Loading MobileSAM (vit_t) from {args.checkpoint}...")
    sam = sam_model_registry["vit_t"](checkpoint=args.checkpoint)
    sam.eval()
    print(
        "  pixel_mean:", sam.pixel_mean.flatten().tolist(),
        "pixel_std:", sam.pixel_std.flatten().tolist(),
        "img_size:", sam.image_encoder.img_size,
    )

    fused_count = fuse_layernorm2d(sam)
    print(f"  Replaced {fused_count} LayerNorm2d modules with fused LayerNormalization")
    set_gelu_tanh(sam)

    base_export_kwargs = dict(
        export_params=True,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        base_export_kwargs["dynamo"] = False

    os.makedirs(args.out_dir, exist_ok=True)
    enc_out = os.path.join(args.out_dir, "mobile_sam_encoder_fp16.onnx")
    dec_out = os.path.join(args.out_dir, "mobile_sam_decoder_fp16.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Export TinyViT-5M Image Encoder
        enc_fp32 = os.path.join(tmpdir, "mobile_sam_encoder_fp32.onnx")
        encoder = MobileSamEncoderOnnx(sam.image_encoder)
        dummy_img = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

        print("Exporting MobileSAM encoder...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with torch.no_grad():
                torch.onnx.export(
                    encoder,
                    (dummy_img,),
                    enc_fp32,
                    input_names=["input_image"],
                    output_names=["image_embeddings"],
                    **base_export_kwargs,
                )
        convert_to_fp16(enc_fp32, enc_out, fp32_node_prefixes=("neck/neck.1/", "neck/neck.3/"))

        # 2. Export Prompt Encoder + Mask Decoder
        dec_fp32 = os.path.join(tmpdir, "mobile_sam_decoder_fp32.onnx")
        onnx_decoder = SamOnnxModel(model=sam, return_single_mask=True)
        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_dec_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float32),
            "has_mask_input": torch.tensor([1], dtype=torch.float32),
            "orig_im_size": torch.tensor([540, 960], dtype=torch.float32),
        }
        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }

        print("Exporting MobileSAM decoder...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with torch.no_grad():
                torch.onnx.export(
                    onnx_decoder,
                    tuple(dummy_dec_inputs.values()),
                    dec_fp32,
                    input_names=list(dummy_dec_inputs.keys()),
                    output_names=["masks", "iou_predictions", "low_res_masks"],
                    dynamic_axes=dynamic_axes,
                    **base_export_kwargs,
                )
        convert_to_fp16(
            dec_fp32,
            dec_out,
            fp32_node_prefixes=("mask_downscaling.1/", "mask_downscaling.4/", "output_upscaling.1/"),
        )

    print("Done exporting MobileSAM FP16 ONNX models!")


if __name__ == "__main__":
    main()
