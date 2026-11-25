---
name: onnx-model-export
description: >-
  Exports a Segment Anything (SAM) prompt encoder and mask decoder to ONNX using
  scripts/export_onnx_model.py, validates it against an example image and prompt,
  and optionally optimizes the model for web deployment (including QUInt8 dynamic
  quantization, GELU tanh approximation, and FP16 conversion). Use this skill when
  the user asks to export, quantize, or optimize a SAM checkpoint to ONNX.
---

# SAM ONNX Model Export

Export SAM's lightweight prompt encoder, mask decoder, and mask postprocessing to ONNX, validate the exported model on a real image and prompt, and optionally optimize it for web runtimes (`onnxruntime-web`).

## Step 1: Gather Required Inputs

Before running any export commands, ask the user for any missing inputs:

1. **Model Checkpoint** — Ask the user to choose one of the three checkpoints:
   - `vit_h` (ViT-H, default / highest accuracy): `/Users/nick/Downloads/sam_vit_h_4b8939.pth`
   - `vit_l` (ViT-L, balanced): `/Users/nick/Downloads/sam_vit_l_0b3195.pth`
   - `vit_b` (ViT-B, smallest / fastest): `/Users/nick/Downloads/sam_vit_b_01ec64.pth`
2. **Example Image** — Path to an example image to generate the `.npy` image embedding and validate ONNX inference (e.g., `notebooks/images/truck.jpg`, `notebooks/images/dog.jpg`, or `demo/src/assets/data/dogs.jpg`).
3. **Prompt** — Prompt to run on the example image for validation:
   - **Point prompt(s)**: pixel coordinates `[[x, y], ...]` and labels (`1` = foreground, `0` = background), and/or
   - **Box prompt**: `[x0, y0, x1, y1]` in XYXY pixel coordinates.
4. **Output Path & Web Optimization Preferences**:
   - Output `.onnx` path (default: `./sam_<model_type>.onnx`).
   - Whether the model targets **web deployment** (`onnxruntime-web`) and which optimizations to apply:
     - `--return-single-mask` (recommended for web / high-res images to avoid upscaling unused masks)
     - `--gelu-approximate` (replaces `erf`-based GELU with `tanh` approximation for faster web execution)
     - **FP16 conversion** (halves model size; keep `float32` I/O types for `onnxruntime-web` compatibility)
     - **QUInt8 dynamic quantization** (`--quantize-out`, recommended for CPU/WASM web runtimes)

Verify that the chosen checkpoint file and example image exist before proceeding.

## Step 2: Export the ONNX Model

Run `scripts/export_onnx_model.py` from the repository root (`/Users/nick/GitHub/segment-anything`):

```bash
PYTHONPATH=. python scripts/export_onnx_model.py \
  --checkpoint <CHECKPOINT_PATH> \
  --model-type <vit_h|vit_l|vit_b> \
  --output <OUTPUT_ONNX_PATH> \
  --opset 17 \
  --return-single-mask
```

Add optional flags when targeting web deployment:
- `--gelu-approximate` — faster GELU in runtimes with slow `erf`.
- `--quantize-out <QUANTIZED_ONNX_PATH>` — produces a dynamically quantized `QUInt8` model via `onnxruntime.quantization.quantize_dynamic`.

`scripts/export_onnx_model.py` automatically runs a CPU smoke test with `onnxruntime` if `onnxruntime` is installed. Confirm the log prints `Model has successfully been run with ONNXRuntime.`.

## Step 3: Web Optimization & FP16 Conversion (If Requested)

If optimizing for web deployment:

1. **FP16 Conversion**: Convert the exported FP32 ONNX graph to FP16 using `onnxruntime.transformers.float16` while preserving `float32` model inputs and outputs (`keep_io_types=True`) so `onnxruntime-web` callers can continue passing `Float32Array` tensors (and avoid precision loss in coordinate/image-size inputs), then topologically sort the graph:

   ```bash
   python -c "
   import onnx
   from onnxruntime.transformers import float16 as ort_fp16
   from onnxruntime.transformers.onnx_model import OnnxModel

   model = onnx.load('<OUTPUT_ONNX_PATH>')
   model_fp16 = ort_fp16.convert_float_to_float16(model, keep_io_types=True)
   om = OnnxModel(model_fp16)
   om.topological_sort()
   om.save_model_to_file('<OUTPUT_FP16_ONNX_PATH>')
   onnx.checker.check_model('<OUTPUT_FP16_ONNX_PATH>')
   print('Saved and verified FP16 model:', '<OUTPUT_FP16_ONNX_PATH>')
   "
   ```

   - Note: FP16 is best suited for WebGPU / FP16-capable backends; for WASM CPU execution in `demo/`, `QUInt8` dynamic quantization (`--quantize-out`) is typically faster and avoids FP16-to-FP32 cast overhead. Do not stack `quantize_dynamic` on top of an FP16 model—generate them as separate artifacts from the FP32 export if both are needed.

2. **Web Demo Compatibility Checklist**:
   - Exported with `--return-single-mask` (matches `demo/src/App.tsx`).
   - Exported with `--gelu-approximate` if targeting WASM/WebGL.
   - Precomputed image embedding saved as `.npy` (`float32`, shape `(1, 256, 64, 64)`) using the **same** checkpoint.

## Step 4: Generate Image Embedding & Validate with Example Image and Prompt

Compute the image embedding from the example image using `SamPredictor`, save it as `<IMAGE_STEM>_embedding.npy`, and run inference on every exported ONNX artifact (FP32, and FP16 / QUInt8 if created) using the user's prompt:

```python
import cv2
import numpy as np
import onnxruntime
import torch
from segment_anything import SamPredictor, sam_model_registry

# 1. Compute & save image embedding
sam = sam_model_registry["<MODEL_TYPE>"](checkpoint="<CHECKPOINT_PATH>")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)
predictor = SamPredictor(sam)

image = cv2.cvtColor(cv2.imread("<EXAMPLE_IMAGE_PATH>"), cv2.COLOR_BGR2RGB)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("<EMBEDDING_NPY_PATH>", image_embedding)

# 2. Format prompt inputs (all float32)
# Points-only: append padding point (0.0, 0.0) with label -1.
# Box + optional points: append box corners [x0, y0], [x1, y1] with labels [2, 3] (no padding point).
onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

ort_inputs = {
    "image_embeddings": image_embedding,
    "point_coords": onnx_coord,
    "point_labels": onnx_label.astype(np.float32),
    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
    "has_mask_input": np.zeros(1, dtype=np.float32),
    "orig_im_size": np.array(image.shape[:2], dtype=np.float32),
}

# 3. Run ONNX Runtime & verify outputs
ort_session = onnxruntime.InferenceSession("<TARGET_ONNX_PATH>", providers=["CPUExecutionProvider"])
masks, iou_predictions, low_res_logits = ort_session.run(None, ort_inputs)
binary_mask = masks > predictor.model.mask_threshold
print("Mask shape:", binary_mask.shape, "Foreground pixels:", int(binary_mask.sum()), "IoU:", iou_predictions)
```

Verify that:
- `binary_mask.shape` ends with `(H, W)` matching the original example image dimensions.
- `binary_mask.sum() > 0` and `iou_predictions` are finite and reasonable (no `NaN`/`Inf`, especially after FP16 conversion).
- If both FP32 and FP16/QUInt8 models were created, compare their binary mask IoU (e.g., `(m1 & m2).sum() / (m1 | m2).sum()`) to confirm negligible accuracy degradation.

## Step 5: Report Results

Summarize for the user:
- Selected checkpoint and `model_type`.
- Generated files and sizes (FP32 `.onnx`, optional FP16 `.onnx`, optional QUInt8 `.onnx`, and `.npy` image embedding).
- Validation results on the example image and prompt (output mask shape, foreground pixel count, predicted IoU, and FP32 vs. optimized mask agreement if applicable).
