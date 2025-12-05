# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import gc
import time
import uuid
from typing import Optional

import torch

from sam3.logger import get_logger


logger = get_logger(__name__)


class Sam3StreamPredictor:
    """Single-GPU streaming predictor that mirrors Sam3VideoPredictor API.

    This wraps the real-time Sam3StreamInference, providing session management
    and a request-based interface tailored for frame-by-frame streaming.

    Exposed request types:
      - {"type": "start_session", "session_id": Optional[str]}
      - {"type": "add_frame", "session_id": str, "frame": raw_image}
      - {"type": "add_prompt", "session_id": str, "frame_index": int, "text": Optional[str],
         "bounding_boxes": Optional[List[List[float]]], "bounding_box_labels": Optional[List[int]]}
      - {"type": "run_inference", "session_id": str, "frame_index": Optional[int]}
      - {"type": "get_cached_output", "session_id": str, "frame_index": int}
      - {"type": "reset_session", "session_id": str}
      - {"type": "close_session", "session_id": str}
      - {"type": "warm_up_compilation"}
    """

    _ALL_INFERENCE_STATES = {}

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        bpe_path: Optional[str] = None,
        has_presence_token: bool = True,
        geo_encoder_use_img_cross_attn: bool = True,  # kept for parity; not used directly here
        strict_state_dict_loading: bool = True,
        apply_temporal_disambiguation: bool = True,
        device: Optional[str] = None,
        compile: bool = False,
    ) -> None:

        from sam3.model_builder import build_sam3_stream_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = (
            build_sam3_stream_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=True if checkpoint_path is None else False,
                bpe_path=bpe_path,
                has_presence_token=has_presence_token,
                geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn,
                strict_state_dict_loading=strict_state_dict_loading,
                apply_temporal_disambiguation=apply_temporal_disambiguation,
                device=device,
                compile=compile,
            )
            .to(device=device)
            .eval()
        )

    @torch.inference_mode()
    def handle_request(self, request):
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(session_id=request.get("session_id"))
        elif request_type == "add_frame":
            return self.add_frame(
                session_id=request["session_id"],
                frame=request["frame"],
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text"),
                bounding_boxes=request.get("bounding_boxes"),
                bounding_box_labels=request.get("bounding_box_labels"),
            )
        elif request_type == "run_inference":
            return self.run_inference(
                session_id=request["session_id"],
                frame_idx=request.get("frame_index"),
            )
        elif request_type == "get_cached_output":
            return self.get_cached_output(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(session_id=request["session_id"])
        elif request_type == "warm_up_compilation":
            return self.warm_up_compilation()
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    def start_session(self, session_id: Optional[str] = None):
        inference_state = self.model.init_stream_state()
        if not session_id:
            session_id = str(uuid.uuid4())
        self._ALL_INFERENCE_STATES[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
        }
        logger.debug(
            f"started new stream session {session_id}; {self._get_session_stats()}; "
            f"{self._get_torch_and_gpu_properties()}"
        )
        return {"session_id": session_id}

    def add_frame(self, session_id: str, frame):
        session = self._get_session(session_id)
        inference_state = session["state"]
        frame_idx = self.model.add_frame(inference_state=inference_state, raw_image=frame)
        logger.debug(f"added frame -> session={session_id}, frame_index={frame_idx}")
        return {"frame_index": frame_idx}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        bounding_boxes: Optional[list] = None,
        bounding_box_labels: Optional[list] = None,
    ):
        session = self._get_session(session_id)
        inference_state = session["state"]

        logger.debug(
            f"add prompt on frame {frame_idx} in session {session_id}: "
            f"text={text}, boxes={bounding_boxes}, box_labels={bounding_box_labels}"
        )
        frame_idx, outputs = self.model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
        )
        return {"frame_index": frame_idx, "outputs": outputs}

    def run_inference(self, session_id: str, frame_idx: Optional[int] = None):
        session = self._get_session(session_id)
        inference_state = session["state"]
        outputs = self.model.run_single_frame_inference(
            inference_state=inference_state, frame_idx=frame_idx
        )
        return {"frame_index": inference_state["curr_frame_idx"] if frame_idx is None else frame_idx, "outputs": outputs}

    def get_cached_output(self, session_id: str, frame_idx: int):
        session = self._get_session(session_id)
        inference_state = session["state"]
        cached = self.model.get_cached_output_for_frame(inference_state, frame_idx)
        return {"frame_index": frame_idx, "cached": cached}

    def reset_session(self, session_id: str):
        logger.debug(f"reset stream session {session_id}")
        session = self._get_session(session_id)
        self.model.reset_stream(session["state"])
        return {"is_success": True}

    def close_session(self, session_id: str):
        session = self._ALL_INFERENCE_STATES.pop(session_id, None)
        if session is None:
            logger.warning(
                f"cannot close session {session_id} as it does not exist (it might have expired); "
                f"{self._get_session_stats()}"
            )
        else:
            del session
            gc.collect()
            logger.info(f"removed session {session_id}; {self._get_session_stats()}")
        return {"is_success": True}

    def warm_up_compilation(self, num_frames: int = 8):
        self.model.warm_up_compilation(num_frames=num_frames)
        return {"is_success": True}

    def _get_session(self, session_id: str):
        session = self._ALL_INFERENCE_STATES.get(session_id, None)
        if session is None:
            raise RuntimeError(f"Cannot find session {session_id}; it might have expired")
        return session

    def _get_session_stats(self):
        # print both the session ids and their frame numbers
        live_session_strs = [
            f"'{session_id}' ({session['state'].get('num_frames', 0)} frames)"
            for session_id, session in self._ALL_INFERENCE_STATES.items()
        ]
        session_stats_str = (
            f"live sessions: [{', '.join(live_session_strs)}], GPU memory: "
            f"{torch.cuda.memory_allocated() // 1024**2} MiB used and "
            f"{torch.cuda.memory_reserved() // 1024**2} MiB reserved"
            f" (max over time: {torch.cuda.max_memory_allocated() // 1024**2} MiB used "
            f"and {torch.cuda.max_memory_reserved() // 1024**2} MiB reserved)"
        )
        return session_stats_str

    def _get_torch_and_gpu_properties(self):
        torch_and_gpu_str = (
            f"torch: {torch.__version__} with CUDA arch {torch.cuda.get_arch_list()}, "
            f"GPU device: {torch.cuda.get_device_properties(torch.cuda.current_device())}"
        )
        return torch_and_gpu_str

    def shutdown(self):
        self._ALL_INFERENCE_STATES.clear()
