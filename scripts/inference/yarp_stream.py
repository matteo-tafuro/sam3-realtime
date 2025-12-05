import argparse
import ctypes
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import yarp
from PIL import Image
from saving_utils import save_video

from sam3.model_builder import build_sam3_stream_predictor
from sam3.visualization_utils import render_masklet_frame

INPUT_RGB_PORT = "/sam3/rgbImage:i"
OUTPUT_RGB_PORT = "/sam3/rgbImage:o"

DEFAULT_TEXT_PROMPT = "hand"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with a specified checkpoint directory on a specified video."
    )

    parser.add_argument(
        "--run_output_name",
        type=str,
        default=None,
        help="Name of the run's output directory and video file (without extension). "
        "If not specified, uses datetime.",
    )

    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="outputs",
        help="Root output directory",
    )

    # === Input source options ===
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help=f"Text prompt for segmentation (default: '{DEFAULT_TEXT_PROMPT}')",
    )

    # === Saving options ===
    save_images_group = parser.add_mutually_exclusive_group()
    save_images_group.add_argument(
        "--save_images",
        dest="save_images",
        action="store_true",
        help="Save generated images.",
    )
    save_images_group.add_argument(
        "--no_save_images",
        dest="save_images",
        action="store_false",
        help="Do not save generated images.",
    )
    parser.set_defaults(save_images=True)

    save_video_group = parser.add_mutually_exclusive_group()
    save_video_group.add_argument(
        "--save_video",
        dest="save_video",
        action="store_true",
        help="Save output video.",
    )
    save_video_group.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
        help="Do not save output video.",
    )
    parser.set_defaults(save_video=True)

    # ===============================

    args = parser.parse_args()

    # Log args
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"Starting realtime inference using text prompt: '{args.text_prompt}'")

    # Use datetime as video ID if run_output_name not specified
    if args.run_output_name is not None:
        video_id = args.run_output_name
    else:
        video_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # If we need to save results, create output dir
    output_dir = os.path.join(
        args.output_root_dir,
        video_id,
    )
    if args.save_images or args.save_video:
        os.makedirs(output_dir, exist_ok=True)

    images_dir = None
    if args.save_images:
        images_dir = os.path.join(output_dir, "frames")
        os.makedirs(images_dir, exist_ok=True)

    # Initialize predictor (single-GPU streaming)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam3_stream_predictor(device=device)

    resp = predictor.handle_request({"type": "start_session"})
    session_id = resp["session_id"]

    print("Opening yarp ports:")
    print(f"- Input at: {INPUT_RGB_PORT}")
    print(f"- Output at: {OUTPUT_RGB_PORT}")

    # Initialize Network
    yarp.Network.init()
    # Open YARP ports
    rgb_input_port = yarp.BufferedPortImageRgb()
    rgb_input_port.open(INPUT_RGB_PORT)
    rgb_output_port = yarp.BufferedPortImageRgb()
    rgb_output_port.open(OUTPUT_RGB_PORT)

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save

    stop_processing = False
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while stop_processing is not True:
                # Read frame (RGB) via input port =============================
                img_rgb = rgb_input_port.read(False)  # non-blocking
                if not img_rgb:
                    # No new frame yet; try again.
                    continue
                width = img_rgb.width()
                height = img_rgb.height()
                char_array_ptr = ctypes.cast(
                    int(img_rgb.getRawImage()), ctypes.POINTER(ctypes.c_char)
                )
                bytes_data = ctypes.string_at(char_array_ptr, img_rgb.getRawImageSize())
                image_array = np.frombuffer(bytes_data, dtype=np.uint8)
                frame_rgb = image_array.reshape((height, width, 3))
                # ============================================================

                # Push frame
                predictor.handle_request(
                    {"type": "add_frame", "session_id": session_id, "frame": frame_rgb}
                )

                # Add text prompt only on first frame
                if frame_idx == 0:
                    predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": args.text_prompt,
                        }
                    )

                # Run per-frame inference
                resp = predictor.handle_request(
                    {
                        "type": "run_inference",
                        "session_id": session_id,
                        "frame_index": frame_idx,
                    }
                )
                outputs = resp.get("outputs")
                if outputs is not None:
                    overlay_rgb = render_masklet_frame(
                        frame_rgb, outputs, frame_idx=frame_idx, alpha=0.5
                    )
                else:
                    overlay_rgb = frame_rgb

                # Save image per frame with 4-digit index
                if args.save_images:
                    img_path = os.path.join(images_dir, f"{frame_idx:04d}.png")
                    Image.fromarray(overlay_rgb).save(img_path)

                # Collect video frames for final save
                if args.save_video:
                    # Keep numpy RGB arrays to pass directly to save_video
                    video_frames.append(overlay_rgb)

                # Send output frame via YARP port =========================
                output_img = rgb_output_port.prepare()
                output_img.resize(width, height)
                output_img.setExternal(overlay_rgb.data, overlay_rgb.shape[1], overlay_rgb.shape[0])
                rgb_output_port.write()
                # ==========================================================

                # Update running statistics
                frame_idx += 1
                frame_timestamps.append(time.time())
                current_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                peak_memory = max(peak_memory, current_peak_memory)
                print(
                    f"Processed frame {frame_idx}. "
                    f"Current peak memory: {current_peak_memory:.2f} GB, "
                    f"Overall peak memory: {peak_memory:.2f} GB.",
                    end="\r",
                )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping processing gracefully...")

    finally:
        # Clean up YARP ports
        rgb_input_port.close()
        rgb_output_port.close()
        yarp.Network.fini()

        if args.save_video:
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                # Use average FPS over the whole run
                effective_fps = (len(frame_timestamps) - 1) / elapsed
            else:
                effective_fps = 30.0

            save_video(
                frames=video_frames,
                output_name=video_id,
                output_dir=output_dir,
                fps=effective_fps,
                overlay_text=f"Text prompt: {args.text_prompt}",
            )
            print(
                f"\nSaved video to {os.path.join(output_dir, video_id + '.mp4')} at {effective_fps:.2f} FPS."
            )

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
