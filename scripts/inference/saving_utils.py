import os
from typing import Iterable, Optional

import cv2
import numpy as np


def save_video(
    frames: Iterable[np.ndarray],
    output_name: str,
    output_dir: str,
    fps: float = 30.0,
    overlay_text: Optional[str] = None,
) -> None:
    """
    Save a collection of RGB numpy arrays as a video to disk.

    Args:
        frames: Iterable of RGB numpy arrays with shape (H, W, 3), dtype uint8.
        output_name: Name of the output video file (without extension).
        output_dir: Base output directory.
        fps: Frames per second for the output video.
        overlay_text: Optional text to overlay on each video frame.
    """

    os.makedirs(output_dir, exist_ok=True)

    video_filename = os.path.join(output_dir, f"{output_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    first_frame = True
    video_writer = None

    for frame_rgb in frames:
        # Ensure array is RGB uint8
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8, copy=False)
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if overlay_text:
            # Place white text at bottom-left with padding
            (text_w, text_h), baseline = cv2.getTextSize(
                overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            pad = 12
            h = frame_bgr.shape[0]
            x = pad
            y = h - pad - baseline
            cv2.putText(
                frame_bgr,
                overlay_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if first_frame:
            height, width, _ = frame_bgr.shape
            video_writer = cv2.VideoWriter(
                video_filename,
                cv2.CAP_FFMPEG,
                fourcc,
                fps,
                (width, height),
            )
            first_frame = False

        video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.release()
