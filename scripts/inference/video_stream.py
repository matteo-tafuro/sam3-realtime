import argparse
import os
import time
from datetime import datetime

import cv2
import torch
from PIL import Image
from saving_utils import save_video
from stream_handler import FrameStatus, InputStreamHandler

from sam3.model_builder import build_sam3_stream_predictor
from sam3.visualization_utils import render_masklet_frame

YARP_IMAGE_PORT = "/depthCamera/rgbImage:i"
TEXT_PROMPT = "hand"

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
        "--stream_type",
        type=str,
        choices=["yarp", "video", "webcam"],
        default="yarp",
        help="Input source kind: yarp | video | webcam",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to input video when --source=video",
    )
    parser.add_argument(
        "--webcam_index",
        type=int,
        default=0,
        help="Webcam index for --source=webcam",
    )
    parser.add_argument(
        "--yarp_port",
        type=str,
        default=YARP_IMAGE_PORT,
        help="YARP port name for --source=yarp",
    )

    # === Visualization options ===
    viz_results_group = parser.add_mutually_exclusive_group()
    viz_results_group.add_argument(
        "--viz_results",
        dest="viz_results",
        action="store_true",
        help="Visualize results on the fly.",
    )
    viz_results_group.add_argument(
        "--no_viz_results",
        dest="viz_results",
        action="store_false",
        help="Do not visualize results on the fly.",
    )
    parser.set_defaults(viz_results=False)

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
    print(f"Starting realtime inference using text prompt: '{TEXT_PROMPT}'")

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

    # Initialize input source
    src = InputStreamHandler(
        kind=args.stream_type,
        video_path=args.video_path,
        webcam_index=args.webcam_index,
        yarp_port_name=args.yarp_port,
    )
    print(f"Opening source: {args.stream_type}")
    src.open()

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save

    stop_processing = False
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while stop_processing is not True:
                # Read frame (RGB)
                stream_buffer = src.read()
                if stream_buffer.status == FrameStatus.NO_FRAME:
                    # YARP: no new frame yet; try again.
                    continue
                if stream_buffer.status == FrameStatus.EOS:
                    # End of stream for video/webcam or closed YARP port.
                    break
                frame_rgb = stream_buffer.frame

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
                            "text": TEXT_PROMPT,
                        }
                    )

                # # You can potentially add more prompts on later frames too
                # if frame_idx == 30:
                #     predictor.handle_request(
                #         {
                #             "type": "add_prompt",
                #             "session_id": session_id,
                #             "frame_index": 30,
                #             "text": "bottle",
                #         }
                #     )

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

                if args.viz_results:
                    cv2.imshow(
                        "SAM3 Livestream", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_processing = True

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
        # Source cleanup
        src.close()

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
                overlay_text=f"Text prompt: {TEXT_PROMPT}",
            )
            print(
                f"\nSaved video to {os.path.join(output_dir, video_id + '.mp4')} at {effective_fps:.2f} FPS."
            )

        # Close any OpenCV windows
        if args.viz_results:
            cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
