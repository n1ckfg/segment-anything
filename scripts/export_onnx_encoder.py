# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Export SAM's ImageEncoderViT to ONNX (FP32 and optional FP16) for in-browser
inference with onnxruntime-web (WebGPU / WASM).

Replaces torch.einsum in decomposed relative positional attention with standard
torch.matmul so the exported graph contains zero Einsum operators (which are
not natively accelerated on WebGPU).
"""

import argparse
import inspect
import os
import warnings

import onnx
import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from segment_anything.modeling import image_encoder as ie_module


def add_decomposed_rel_pos_matmul(
    attn: torch.Tensor,
    q: torch.Tensor,
    Rh: torch.Tensor,
    Rw: torch.Tensor,
    q_size,
    k_size,
) -> torch.Tensor:
    """MatMul-equivalent of add_decomposed_rel_pos (avoids ONNX Einsum ops)."""
    q_h, q_w = q_size
    k_h, k_w = k_size

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    # einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_h = torch.matmul(r_q, Rh.transpose(-2, -1))
    # einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_w = torch.matmul(r_q.transpose(1, 2), Rw.transpose(-2, -1)).transpose(1, 2)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


def webgpu_friendly_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in forward for ImageEncoderViT Attention that:
    1. Replaces Einsum with standard MatMul (natively accelerated on WebGPU).
    2. Loops per head on global attention (H*W == 4096) so the attention matrix is
       (1, 4096, 4096) (~33.5 MB FP16) instead of (num_heads, 4096, 4096) (~402 MB FP16),
       staying well below WebGPU's 128 MB maxStorageBufferBindingSize limit.
    """
    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

    Rh = ie_module.get_rel_pos(H, H, self.rel_pos_h) if self.use_rel_pos else None
    Rw = ie_module.get_rel_pos(W, W, self.rel_pos_w) if self.use_rel_pos else None

    if H * W > 1024:
        # Global attention: compute one head at a time to keep peak buffer <= 33.5 MB
        out_heads = []
        for h_idx in range(B * self.num_heads):
            qi = q[h_idx : h_idx + 1]
            ki = k[h_idx : h_idx + 1]
            vi = v[h_idx : h_idx + 1]
            attn_i = (qi * self.scale) @ ki.transpose(-2, -1)
            if self.use_rel_pos:
                attn_i = add_decomposed_rel_pos_matmul(attn_i, qi, Rh, Rw, (H, W), (H, W))
            attn_i = attn_i.softmax(dim=-1)
            out_heads.append(attn_i @ vi)
        out = torch.cat(out_heads, dim=0)
    else:
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos_matmul(attn, q, Rh, Rw, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        out = attn @ v

    x = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    x = self.proj(x)
    return x


class SamImageEncoderOnnx(nn.Module):
    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(input_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SAM ImageEncoderViT to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["default", "vit_h", "vit_l", "vit_b"],
    )
    parser.add_argument("--output", type=str, required=True, help="Output FP32 ONNX path.")
    parser.add_argument("--fp16-output", type=str, default=None, help="Optional FP16 ONNX path.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--gelu-approximate", action="store_true")
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep the intermediate FP32 file when --fp16-output is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading SAM {args.model_type} from {args.checkpoint}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.eval()

    encoder = SamImageEncoderOnnx(sam.image_encoder)
    if args.gelu_approximate:
        for _, m in encoder.named_modules():
            if isinstance(m, nn.GELU):
                m.approximate = "tanh"

    # Verify WebGPU-friendly attention replacement matches original numerically before exporting
    orig_attn_forward = ie_module.Attention.forward
    dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        ref_out = encoder(dummy)
        ie_module.Attention.forward = webgpu_friendly_attention_forward
        new_out = encoder(dummy)
    max_diff = (ref_out - new_out).abs().max().item()
    print(f"WebGPU-friendly attention max abs diff vs original: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Unexpected attention mismatch: {max_diff}"

    export_kwargs = dict(
        export_params=True,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["image_embeddings"],
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    print(f"Exporting FP32 encoder to {args.output}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.no_grad():
            torch.onnx.export(encoder, (dummy,), args.output, **export_kwargs)

    ie_module.Attention.forward = orig_attn_forward

    model = onnx.load(args.output)
    einsums = [n.name for n in model.graph.node if n.op_type == "Einsum"]
    print(f"Exported FP32 encoder: nodes={len(model.graph.node)}, einsum_nodes={len(einsums)}")
    assert len(einsums) == 0, f"Unexpected Einsum nodes remain: {einsums}"

    if args.fp16_output:
        from onnxruntime.transformers import float16 as ort_fp16
        from onnxruntime.transformers.onnx_model import OnnxModel

        print(f"Converting encoder to FP16 -> {args.fp16_output}...")
        model_fp16 = ort_fp16.convert_float_to_float16(model, keep_io_types=True)
        om = OnnxModel(model_fp16)
        om.topological_sort()
        om.save_model_to_file(args.fp16_output)
        onnx.checker.check_model(args.fp16_output)
        print(f"Verified FP16 encoder: {args.fp16_output}")

        if not args.keep_fp32 and os.path.abspath(args.output) != os.path.abspath(args.fp16_output):
            os.remove(args.output)
            print(f"Removed intermediate FP32 file: {args.output}")


if __name__ == "__main__":
    main()
