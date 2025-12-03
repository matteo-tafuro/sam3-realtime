# Real-Time Streaming Adaptation of SAM3

![SAM3 Livestream Demo](assets/livestream.gif)

This repository transforms [SAM3](https://github.com/facebookresearch/sam3)’s offline video inference into a live, real-time streaming pipeline. Instead of preloading and processing an entire video sequence offline, it ingests frames incrementally and performs per-frame inference on the fly. This allows SAM3 to work with any live video source (e.g. webcams, RTSP streams, [YARP](https://www.yarp.it/latest/index.html) ports), enabling online operation and expanding use cases to robotics, teleoperation, live surveillance, AR/VR and real-time content creation.

## Installation
Follow the instructions in `README-ORIGINAL.md` for environment setup, dependencies and checkpoint authentication.

## Quick Start

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

## Current Limitations
- **Single-GPU streaming:** The provided streaming predictor targets one GPU. Multi-GPU support exists in the base model but isn’t integrated into the streaming predictor yet.
