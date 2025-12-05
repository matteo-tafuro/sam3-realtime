import ctypes
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import cv2
import numpy as np


class FrameStatus(Enum):
    """Enumeration of frame read outcomes.

    OK: A frame was successfully retrieved.
    NO_FRAME: Source is live/non-blocking and no new frame is currently available (try again later).
    EOS: End-of-stream; no more frames will become available.
    """

    OK = "ok"
    NO_FRAME = "no_frame"
    EOS = "end_of_stream"


@dataclass
class FrameRead:
    """Container for a frame read attempt.

    Attributes:
        status: FrameStatus indicating outcome.
        frame: RGB ndarray (H,W,3) when status == OK, else None.
    """

    status: FrameStatus
    frame: Optional[np.ndarray] = None


class InputStreamHandler:
    """Unified frame source abstraction with explicit read status.

    Supported kinds:
      - video: frames from a video file (blocking); EOS when finished.
      - webcam: frames from a camera device (blocking until failure); EOS if capture ends.
      - yarp: frames from a YARP port (non-blocking); NO_FRAME when no new frame yet.

    read() returns a FrameRead object instead of simply returning raw frame / None.
    This removes ambiguity: None could mean "no frame yet" (YARP) *or* end-of-stream (video/webcam).

    Typical usage:
        src = InputStreamHandler(kind="video", video_path="/path/to/video.mp4")
        src.open()
        while True:
            fr = src.read()
            if fr.status == FrameStatus.NO_FRAME:
                continue  # (Only applies to YARP)
            if fr.status == FrameStatus.EOS:
                break
            frame = fr.frame  # RGB ndarray (H,W,3)
        src.close()
    """

    def __init__(
        self,
        kind: str,
        video_path: Optional[str] = None,
        webcam_index: int = 0,
        yarp_port_name: str = "/depthCamera/rgbImage:i",
    ) -> None:
        self.kind = kind
        self.video_path = video_path
        self.webcam_index = webcam_index
        self.yarp_port_name = yarp_port_name

        self._cap: Optional[cv2.VideoCapture] = None
        self._yarp_port = None
        self._yarp_initialized = False

    def open(self) -> None:
        kind = self.kind.lower()
        if kind == "video":
            if not self.video_path or not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Video path not found: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            self._cap = cap
        elif kind == "webcam":
            cap = cv2.VideoCapture(self.webcam_index)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam index {self.webcam_index}")
            self._cap = cap
        elif kind == "yarp":
            try:
                import yarp
            except ImportError:
                raise ImportError(
                    "YARP library is not installed or not found. Please install it to use YARP input source."
                )

            yarp.Network.init()
            self._yarp_initialized = True
            port = yarp.BufferedPortImageRgb()
            port.open(self.yarp_port_name)
            print(
                f"Opened YARP image port at {self.yarp_port_name}. Connect your image source to it"
            )
            self._yarp_port = port
        else:
            raise ValueError(f"Unknown source kind: {self.kind}")

    def read(self) -> FrameRead:
        """Attempt to read next RGB frame.

        Returns:
            FrameRead: (status, frame) where frame is present only if status == FrameStatus.OK.
        """
        kind = self.kind.lower()
        if kind in ("video", "webcam"):
            if self._cap is None:
                return FrameRead(status=FrameStatus.EOS, frame=None)
            ok, bgr = self._cap.read()
            if not ok:
                return FrameRead(status=FrameStatus.EOS, frame=None)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return FrameRead(status=FrameStatus.OK, frame=rgb)
        elif kind == "yarp":
            if self._yarp_port is None:
                return FrameRead(status=FrameStatus.EOS, frame=None)
            img_rgb = self._yarp_port.read(False)  # non-blocking
            if not img_rgb:
                return FrameRead(status=FrameStatus.NO_FRAME, frame=None)
            width = img_rgb.width()
            height = img_rgb.height()
            char_array_ptr = ctypes.cast(
                int(img_rgb.getRawImage()), ctypes.POINTER(ctypes.c_char)
            )
            bytes_data = ctypes.string_at(char_array_ptr, img_rgb.getRawImageSize())
            image_array = np.frombuffer(bytes_data, dtype=np.uint8)
            frame_rgb = image_array.reshape((height, width, 3))
            return FrameRead(status=FrameStatus.OK, frame=frame_rgb)
        else:
            return FrameRead(status=FrameStatus.EOS, frame=None)

    def close(self) -> None:
        kind = self.kind.lower()
        if kind in ("video", "webcam"):
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        elif kind == "yarp":
            if self._yarp_port is not None:
                try:
                    self._yarp_port.close()
                except Exception:
                    pass
                self._yarp_port = None
            if self._yarp_initialized:
                try:
                    yarp.Network.fini()
                except Exception:
                    pass
                self._yarp_initialized = False

    # Optional convenience: iterator protocol
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def frames(self) -> Iterator[np.ndarray]:
        """Iterator over available frames.

        For YARP sources this will skip NO_FRAME cycles and only yield real frames.
        Terminates on EOS.
        """
        while True:
            fr = self.read()
            if fr.status == FrameStatus.NO_FRAME:
                continue
            if fr.status == FrameStatus.EOS:
                break
            yield fr.frame
