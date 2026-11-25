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
import torch.nn.functional as F
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel
from onnxruntime.transformers import float16 as ort_fp16
from onnxruntime.transformers.onnx_model import OnnxModel


class MobileSamEncoderOnnx(nn.Module):
    """
    Dynamic-resolution wrapper for MobileSAM's TinyViT-5M image encoder.
    Accepts any (1, 3, H, W) tensor where H and W are multiples of 16 (e.g. 384, 512,
    640, 768, 1024) and always outputs (1, 256, 64, 64) embeddings for SAM's decoder.
    """

    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        enc = self.image_encoder
        x = enc.patch_embed(input_image)  # (B, C, H/4, W/4)

        # Stage 0: MBConv blocks + PatchMerging (kept in 4D NCHW)
        l0 = enc.layers[0]
        for blk in l0.blocks:
            x = blk(x)
        pm0 = l0.downsample
        x = pm0.conv3(pm0.act(pm0.conv2(pm0.act(pm0.conv1(x)))))  # (B, C, H/8, W/8)

        # Stages 1..3: TinyViTBlock (local windowed attention) + optional PatchMerging
        for layer in enc.layers[1:]:
            B, C, H, W = x.shape
            x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            for blk in layer.blocks:
                ws = blk.window_size
                res_x = x_seq
                x4 = x_seq.view(B, H, W, C)
                pad_b = (ws - (H % ws)) % ws
                pad_r = (ws - (W % ws)) % ws
                x4 = F.pad(x4, (0, 0, 0, pad_r, 0, pad_b))
                pH = H + pad_b
                pW = W + pad_r
                nH = pH // ws
                nW = pW // ws
                x4 = (
                    x4.view(B, nH, ws, nW, ws, C)
                    .transpose(2, 3)
                    .reshape(B * nH * nW, ws * ws, C)
                )
                x4 = blk.attn(x4)
                x4 = (
                    x4.view(B, nH, nW, ws, ws, C)
                    .transpose(2, 3)
                    .reshape(B, pH, pW, C)
                )
                x4 = x4[:, :H, :W, :].contiguous()
                x_seq = res_x + x4.view(B, H * W, C)
                x_conv = blk.local_conv(x_seq.transpose(1, 2).reshape(B, C, H, W))
                x_seq = x_conv.flatten(2).transpose(1, 2)
                x_seq = x_seq + blk.mlp(x_seq)
            x = x_seq.transpose(1, 2).reshape(B, C, H, W)
            if layer.downsample is not None:
                pm = layer.downsample
                x = pm.conv3(pm.act(pm.conv2(pm.act(pm.conv1(x)))))

        x = enc.neck(x)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return x


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
        # 1. Export dynamic-resolution TinyViT-5M Image Encoder
        enc_fp32 = os.path.join(tmpdir, "mobile_sam_encoder_fp32.onnx")
        encoder = MobileSamEncoderOnnx(sam.image_encoder)
        # Use 640x640 dummy (where 640/8=80 and 640/16=40 are not multiples of 7 or 14)
        # so TorchScript traces the non-zero pad and slice branches dynamically.
        dummy_img = torch.randn(1, 3, 640, 640, dtype=torch.float32)

        print("Exporting dynamic-resolution MobileSAM encoder...")
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
                    dynamic_axes={"input_image": {2: "height", 3: "width"}},
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
