# Real-Time Streaming Adaptation of SAM3

![sam3-real-time](https://github.com/user-attachments/assets/f2a01527-c1c8-486f-9a10-cbed1ca6a2b4)

This repository transforms [SAM3](https://github.com/facebookresearch/sam3)’s offline video inference into a live, real-time streaming pipeline. Instead of preloading and processing an entire video sequence offline, it ingests frames incrementally and performs per-frame inference on the fly. This allows SAM3 to work with any live video source (e.g. webcams, RTSP streams, [YARP](https://www.yarp.it/latest/index.html) ports), enabling online operation and expanding use cases to robotics, teleoperation, live surveillance, AR/VR and real-time content creation.

## Installation
This fork uses **Pixi** for environment management to provide fully reproducible setups with fast and stable dependency resolution.

### 1. Install Pixi (if needed)
As of December 2025, you can install Pixi on Linux and MacOS via:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```
Or if you prefer using `wget`:
```bash
wget -qO- https://pixi.sh/install.sh | sh
```
For Windows or more recent instructions, visit the [Pixi installation guide](https://pixi.sh/dev/installation/).

### 2. Create the environment
From the repository root:

```bash
pixi install
```

### 3. Enter the environment
```bash
pixi shell
```

### 4. Verify SAM3 loads from this fork
```bash
python -c "import sam3; print(sam3.__file__)"
```

You should see a path pointing to this repository, confirming the editable install.

## Quick Start


⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)


### Demo Notebook
Mirroring the original demo notebooks, `examples/sam3_stream_predictor_example.ipynb` demonstrates how to run SAM3 in real-time on a video stream. The notebook loads a video file, starts a streaming session, adds a text prompt on frame 0 and pushes frames incrementally, running per-frame inference with optional visualization and FPS reporting.

### CLI Script
For a command-line run, use `scripts/inference/video_stream.py`. It mirrors the notebook flow: opens a live source (webcam/video/YARP), starts a streaming session, adds a text prompt on the first frame and performs per-frame inference (with optional visualization and saving). Note that you can also potentially add more textual prompts in later frames.

- Basic webcam example:
  - `python scripts/inference/video_stream.py --stream_type webcam --webcam_index 0 --viz_results --save_video`
- Flags of interest:
  - `--stream_type {webcam|video|yarp}`: choose the input source
  - `--video_path PATH`: path to a video file when `--stream_type video`
  - `--viz_results`: display live overlays
  - `--save_images` / `--save_video`: store outputs under `outputs/<run_id>/`
  - `--run_output_name NAME`: set a custom run id (else datetime is used)
  - `--compile`: `torch.compile` the model for ~10–15% higher steady-state FPS (one-time warm-up on start)
  - `--fast_preprocess`: resize/normalize frames on the GPU (~4× faster ingest; slightly different resampling than the CPU/PIL default)

## Fixed Issues
- **Memory leak / long-run OOM** — **FIXED.** The tracker inherited SAM2/SAM3's offline memory bank, which stores per-frame mask-memory tensors and grows `O(frames × objects)` — fine for finite videos, but it OOMs on an open-ended stream (the originally reported OOM after ~5 min at 480p on an RTX 3090). Both memory banks are now bounded: the non-conditioning bank is trimmed to a fixed horizon ([`8112931`](https://github.com/matteo-tafuro/sam3-realtime/commit/8112931ff8f0d9346b5cc4f66f05d191953a770d), [PR #4](https://github.com/matteo-tafuro/sam3-realtime/pull/4)), and the reconditioning-created conditioning frames that a forward-only stream can never attend again are evicted ([`73c9acf`](https://github.com/matteo-tafuro/sam3-realtime/commit/73c9acfd50fdcaba39f9923fa840104b053f0aa5)). Both fixes are output-preserving (bit-identical masks). GPU memory now plateaus instead of climbing.
- **Redundant frame-0 inference** — **FIXED.** Adding a prompt already runs inference for that frame and returns its outputs; the CLI then ran a second full forward pass on the same frame. It now reuses the prompt's outputs ([`8112931`](https://github.com/matteo-tafuro/sam3-realtime/commit/8112931ff8f0d9346b5cc4f66f05d191953a770d), [PR #4](https://github.com/matteo-tafuro/sam3-realtime/pull/4)).
- **`torch.compile` unavailable** — **FIXED.** The streaming model can now be compiled for ~10–15% higher steady-state FPS via `--compile`, with the one-time compilation cost front-loaded through a warm-up so the first live frame isn't stalled ([`3670295`](https://github.com/matteo-tafuro/sam3-realtime/commit/36702956defc520ecf41f6e09bdea80d793ad269), [PR #4](https://github.com/matteo-tafuro/sam3-realtime/pull/4)).
- **CPU-bound preprocessing** — **IMPROVED.** Optional GPU-side resize/normalize (`--fast_preprocess`) cuts per-frame ingest from ~13 ms to ~3.5 ms ([`5e17d43`](https://github.com/matteo-tafuro/sam3-realtime/commit/5e17d4347d69b37d6df8569951b59b88ba3716d1), [PR #4](https://github.com/matteo-tafuro/sam3-realtime/pull/4)).
- **Early-frame tracker mismatch** — **FIXED.** In a stream the tracker's object-pointer memory budget and temporal-position-encoding normalizer were capped by `min(num_frames, max_obj_ptrs_in_encoder)`, silently shrinking for the first ~15 frames after a prompt (a train/test mismatch the offline model never hits). The tracker is now given a training-consistent frame-count hint ([`186a274`](https://github.com/matteo-tafuro/sam3-realtime/commit/186a274bd8d931a71f5681c7b2a1acb5b5c3da4b)).
- **Minor upstream-inherited bugs** — **FIXED.** Several small correctness bugs flagged in review (e.g. a re-raised `JSONDecodeError` that itself raised `TypeError`, `NestedTensor.pin_memory` returning `None`, a `LOCAL_RANK` parse that bypassed its own assert) ([`08064a7`](https://github.com/matteo-tafuro/sam3-realtime/commit/08064a71c71d7630d3c57a6aaf2deac1cb86723b), [PR #4](https://github.com/matteo-tafuro/sam3-realtime/pull/4)).

## Current Limitations and Known Issues
- **Single-GPU streaming:** The provided streaming predictor targets one GPU. Multi-GPU support exists in the base model but isn’t integrated into the streaming predictor yet.
- **Throughput is ViT-bound:** Steady-state FPS is dominated by the image backbone (~75% of per-frame compute). `--compile` helps; on Hopper GPUs FlashAttention-3 (`use_fa3`) and/or a lower input resolution are the larger levers.
