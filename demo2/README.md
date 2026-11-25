# MobileSAM Realtime Webcam Demo (`demo2`)

100% browser-only realtime webcam segmentation using **MobileSAM** (TinyViT-5M FP16 encoder + FP16 mask decoder) via `onnxruntime-web` (WebGPU with automatic WASM fallback).

## Models in `demo2/`
1. `mobile_sam_encoder_fp16.onnx` — TinyViT-5M image encoder (FP16, `float32` I/O)
2. `mobile_sam_decoder_fp16.onnx` — MobileSAM prompt encoder & mask decoder (FP16, `float32` I/O)

## Run

Run `./run.command` (or `python3 -m http.server 8081 --directory demo2`), open the browser page, and click **Start Webcam**.
