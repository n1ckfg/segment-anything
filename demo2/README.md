# MobileSAM Realtime Webcam Demo (`demo2`)

100% browser-only realtime webcam segmentation using **MobileSAM** (dynamic-resolution TinyViT-5M FP16 encoder + FP16 mask decoder) via `onnxruntime-web` (WebGPU with automatic WASM fallback).

## Models in `demo2/`
1. `mobile_sam_encoder_fp16.onnx` (`13.6 MB`) — Dynamic-resolution TinyViT-5M image encoder (FP16, `float32` I/O, fused FP32 `LayerNormalization` in neck; supports `384`, `512`, `640`, `768`, and `1024` px inputs)
2. `mobile_sam_decoder_fp16.onnx` (`8.1 MB`) — MobileSAM prompt encoder & mask decoder (FP16, `float32` I/O)

## Run

Run `./run.command` (or `python3 -m http.server 8081 --directory demo2`), open the browser page, click **Start Webcam**, and use the **Resolution** dropdown (`384 px` – `1024 px`) to trade mask sharpness for faster FPS.
