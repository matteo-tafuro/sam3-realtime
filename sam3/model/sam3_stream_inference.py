# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import torchvision.transforms.functional as TF

from sam3 import perflib
from sam3.logger import get_logger
from sam3.model.act_ckpt_utils import clone_output_wrapper
from sam3.model.box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh
from sam3.model.data_misc import BatchedDatapoint, convert_my_tensors, FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.io_utils import IMAGE_EXTS, load_resource_as_video_frames
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.sam3_video_base import MaskletConfirmationStatus, Sam3VideoBase
from sam3.model.utils.misc import copy_data_to_device
from sam3.perflib.compile import compile_wrapper, shape_logging_wrapper
from sam3.perflib.masks_ops import masks_to_boxes as perf_masks_to_boxes
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm

logger = get_logger(__name__)


class Sam3StreamInference(Sam3VideoBase):
    """Real-time streaming inference for SAM3.

    Frames are pushed incrementally; per-frame inference runs immediately without
    any offline propagation assumptions.
    """

    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1

    def __init__(
        self,
        image_size: int = 1008,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        compile_model: bool = False,
        # memory bounding knobs (CPU ingress + bounded caches like offline)
        max_cached_frames: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model
        self.max_cached_frames = max_cached_frames

    @torch.inference_mode()
    def init_stream_state(self) -> Dict[str, Any]:
        device = self.device
        bs = 1
        return {
            "image_size": self.image_size,
            "orig_height": None,
            "orig_width": None,
            "curr_frame_idx": -1,
            "constants": {
                "empty_geometric_prompt": Prompt(
                    box_embeddings=torch.zeros(0, bs, 4, device=device),
                    box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
                    box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
                    point_embeddings=torch.zeros(0, bs, 2, device=device),
                    point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
                    point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
                )
            },
            "input_batch": None,
            "previous_stages_out": [],
            "per_frame_raw_point_input": [],
            "per_frame_raw_box_input": [],
            "per_frame_visual_prompt": [],
            "per_frame_geometric_prompt": [],
            "per_frame_cur_step": [],
            "text_prompt": None,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "cached_frame_outputs": {},
            "action_history": [],
            "visual_prompt_embed": None,
            "visual_prompt_mask": None,
            "is_image_only": False,
        }

    @torch.inference_mode()
    def reset_stream(self, inference_state: Dict[str, Any]) -> None:
        new_state = self.init_stream_state()
        inference_state.clear()
        inference_state.update(new_state)

    def _preprocess_raw_image(self, raw_image: Any) -> Tuple[torch.Tensor, int, int]:
        if isinstance(raw_image, Image.Image):
            orig_w, orig_h = raw_image.width, raw_image.height
            img = TF.resize(raw_image.convert("RGB"), size=(self.image_size, self.image_size))
            img_t = TF.to_tensor(img)
        elif isinstance(raw_image, np.ndarray):
            arr = raw_image
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            assert arr.ndim == 3 and arr.shape[2] == 3, "Expected HxWx3 numpy array"
            orig_h, orig_w = arr.shape[0], arr.shape[1]
            img = Image.fromarray(arr, mode="RGB")
            img = TF.resize(img, size=(self.image_size, self.image_size))
            img_t = TF.to_tensor(img)
        elif isinstance(raw_image, torch.Tensor):
            t = raw_image
            if t.ndim == 3 and t.shape[0] in (1, 3):
                orig_h, orig_w = t.shape[1], t.shape[2]
            elif t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1).contiguous()
                orig_h, orig_w = t.shape[1], t.shape[2]
            else:
                raise ValueError("Expected CxHxW or HxWxC torch tensor for raw_image")
            img_t = TF.resize(t.float() / (255.0 if t.dtype in (torch.uint8, torch.int8) else 1.0), size=(self.image_size, self.image_size))
        else:
            raise TypeError("Unsupported raw_image type; expected PIL, numpy, or torch.Tensor")

        mean = torch.tensor(self.image_mean, dtype=torch.float16, device="cpu").view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=torch.float16, device="cpu").view(3, 1, 1)
        img_t = img_t.to(dtype=torch.float16, device="cpu")
        img_t = (img_t - mean) / std
        return img_t, orig_h, orig_w

    @torch.inference_mode()
    def add_frame(self, inference_state: Dict[str, Any], raw_image: Any) -> int:
        img_t, orig_h, orig_w = self._preprocess_raw_image(raw_image)
        if inference_state["orig_height"] is None:
            inference_state["orig_height"] = orig_h
            inference_state["orig_width"] = orig_w

        if inference_state["input_batch"] is None:
            find_text_batch = ["<text placeholder>", "visual"]
            input_box_embedding_dim = 258
            input_points_embedding_dim = 257
            stage = FindStage(
                img_ids=[0],
                text_ids=[self.TEXT_ID_FOR_VISUAL],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            stage = convert_my_tensors(stage)
            # Keep images as a CPU list of tensors to match offline CPU offload path
            img_batch = [img_t]
            input_batch = BatchedDatapoint(
                img_batch=img_batch,  # keep on CPU
                find_text_batch=find_text_batch,
                find_inputs=[copy_data_to_device(stage, self.device, non_blocking=True)],  # stages on GPU
                find_targets=[None],
                find_metadatas=[None],
            )
            inference_state["input_batch"] = input_batch
            inference_state["curr_frame_idx"] = 0
        else:
            input_batch = inference_state["input_batch"]
            # Determine previous length and append
            if isinstance(input_batch.img_batch, torch.Tensor):
                T_prev = input_batch.img_batch.shape[0]
                input_batch.img_batch = torch.cat([input_batch.img_batch, img_t.unsqueeze(0)], dim=0)
            else:
                T_prev = len(input_batch.img_batch)
                input_batch.img_batch.append(img_t)

            input_box_embedding_dim = 258
            input_points_embedding_dim = 257
            text_id = self.TEXT_ID_FOR_TEXT if inference_state.get("text_prompt") else self.TEXT_ID_FOR_VISUAL
            stage = FindStage(
                img_ids=[T_prev],
                text_ids=[text_id],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            stage = convert_my_tensors(stage)
            stage = copy_data_to_device(stage, self.device, non_blocking=True)
            input_batch.find_inputs.append(stage)
            input_batch.find_targets.append(None)
            input_batch.find_metadatas.append(None)
            inference_state["curr_frame_idx"] = T_prev

        # Grow per-frame placeholders
        inference_state["previous_stages_out"].append(None)
        inference_state["per_frame_raw_point_input"].append(None)
        inference_state["per_frame_raw_box_input"].append(None)
        inference_state["per_frame_visual_prompt"].append(None)
        inference_state["per_frame_geometric_prompt"].append(None)
        inference_state["per_frame_cur_step"].append(0)

        # Update total frames and keep tracker states in sync with growing stream length
        img_batch_ref = inference_state["input_batch"].img_batch
        inference_state["num_frames"] = img_batch_ref.shape[0] if isinstance(img_batch_ref, torch.Tensor) else len(img_batch_ref)
        if inference_state["tracker_inference_states"]:
            for trk_state in inference_state["tracker_inference_states"]:
                # Extend tracker-visible video length so it can propagate to this frame
                trk_state["num_frames"] = inference_state["num_frames"]
                # Ensure original video resolution is set for resizing masks
                if trk_state.get("video_height", None) is None:
                    trk_state["video_height"] = inference_state["orig_height"]
                    trk_state["video_width"] = inference_state["orig_width"]

        return inference_state["curr_frame_idx"]

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        is_new_visual_prompt = (
            inference_state["per_frame_visual_prompt"][frame_idx] is None
            and inference_state["previous_stages_out"][frame_idx] is None
        )
        if is_new_visual_prompt:
            if boxes_cxcywh.size(0) != 1:
                raise RuntimeError("visual prompt should contain exactly one box")
            if not box_labels.item():
                logging.warning("A negative box is added as a visual prompt.")
            device = self.device
            new_visual_prompt = Prompt(
                box_embeddings=boxes_cxcywh[None, 0:1, :].to(device),
                box_mask=None,
                box_labels=box_labels[None, 0:1].to(device),
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )
            inference_state["per_frame_visual_prompt"][frame_idx] = new_visual_prompt
        else:
            new_visual_prompt = None
        if inference_state["per_frame_visual_prompt"][frame_idx] is not None:
            boxes_cxcywh = boxes_cxcywh[1:]
            box_labels = box_labels[1:]
        return boxes_cxcywh, box_labels, new_visual_prompt

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state: Dict[str, Any],
        frame_idx: int,
        text_str: Optional[str] = None,
        boxes_xywh: Optional[torch.Tensor] = None,
        box_labels: Optional[torch.Tensor] = None,
    ):
        assert inference_state.get("input_batch") is not None, "No frames added yet. Call add_frame first."
        assert 0 <= frame_idx <= inference_state["curr_frame_idx"], "frame_idx must exist"

        if text_str is not None and text_str != "visual":
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            text_id = self.TEXT_ID_FOR_TEXT
        else:
            inference_state["text_prompt"] = None
            inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
            text_id = self.TEXT_ID_FOR_VISUAL
        for st in inference_state["input_batch"].find_inputs:
            st.text_ids[...] = text_id

        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            assert boxes_xywh.dim() == 2 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)

            inference_state["per_frame_raw_box_input"][frame_idx] = (boxes_xywh.clone(), box_labels.clone())
            boxes_cxcywh, box_labels, geometric_prompt_visual = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )
            geometric_prompt = None
            if boxes_cxcywh.numel() > 0:
                device = self.device
                geometric_prompt = Prompt(
                    box_embeddings=boxes_cxcywh.unsqueeze(0).to(device),
                    box_mask=None,
                    box_labels=box_labels.unsqueeze(0).to(device),
                    point_embeddings=None,
                    point_mask=None,
                    point_labels=None,
                )
            inference_state["per_frame_geometric_prompt"][frame_idx] = (
                geometric_prompt_visual if geometric_prompt_visual is not None else geometric_prompt
            )

        return frame_idx, self.run_single_frame_inference(inference_state, frame_idx)

    @torch.inference_mode()
    def run_single_frame_inference(self, inference_state: Dict[str, Any], frame_idx: Optional[int] = None):
        assert inference_state.get("input_batch") is not None, "No frames added yet. Call add_frame first."
        # Lazily compile on first real inference when enabled
        self._compile_model()
        if frame_idx is None:
            frame_idx = inference_state["curr_frame_idx"]
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        num_frames_dynamic = input_batch.img_batch.shape[0] if isinstance(input_batch.img_batch, torch.Tensor) else len(input_batch.img_batch)

        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=num_frames_dynamic,
            reverse=False,
            input_batch=input_batch,
            geometric_prompt=(
                inference_state["constants"]["empty_geometric_prompt"]
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=False,
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )

        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        # Do not cache yet; caching happens after postprocess below (with suppression filtering)

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
            "frame_stats": frame_stats,
        }

        if self.rank == 0:
            rank0_metadata = tracker_metadata_new.get("rank0_metadata", {})
            removed_obj_ids = rank0_metadata.get("removed_obj_ids", set())
            suppressed_obj_ids = rank0_metadata.get("suppressed_obj_ids", defaultdict(set)).get(frame_idx, set())
            unconfirmed = []
            if self.masklet_confirmation_enable and "masklet_confirmation" in rank0_metadata:
                status = rank0_metadata["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                unconfirmed = tracker_metadata_new["obj_ids_all_gpu"][is_unconfirmed].tolist()

            post = self._postprocess_output(
                inference_state,
                out,
                removed_obj_ids=removed_obj_ids,
                suppressed_obj_ids=suppressed_obj_ids,
                unconfirmed_obj_ids=unconfirmed,
            )
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
                removed_obj_ids=removed_obj_ids,
                unconfirmed_obj_ids=unconfirmed,
            )
            return post
        else:
            # Mirroring original behavior for non-rank0 processes
            return None

    def _postprocess_output(
        self,
        inference_state,
        out,
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        obj_id_to_mask = out["obj_id_to_mask"]
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        H_video, W_video = inference_state["orig_height"], inference_state["orig_width"]
        if len(curr_obj_ids) == 0:
            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_probs = torch.zeros(0, dtype=torch.float32)
            out_binary_masks = torch.zeros(0, H_video, W_video, dtype=torch.bool)
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
        else:
            out_obj_ids = torch.tensor(curr_obj_ids, dtype=torch.int64)
            out_probs = torch.tensor([out["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids])
            out_tracker_probs = torch.tensor([
                (
                    out["obj_id_to_tracker_score"].get(obj_id, 0.0)
                    if isinstance(out["obj_id_to_tracker_score"], dict)
                    else (out["obj_id_to_tracker_score"][obj_id] if obj_id in out["obj_id_to_tracker_score"] else 0.0)
                )
                for obj_id in curr_obj_ids
            ])
            out_binary_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)

            assert out_binary_masks.dtype == torch.bool
            keep = out_binary_masks.any(dim=(1, 2)).cpu()
            obj_ids_to_hide = []
            if suppressed_obj_ids is not None:
                obj_ids_to_hide.extend(list(suppressed_obj_ids))
            if removed_obj_ids is not None:
                obj_ids_to_hide.extend(list(removed_obj_ids))
            if unconfirmed_obj_ids is not None:
                obj_ids_to_hide.extend(list(unconfirmed_obj_ids))
            if len(obj_ids_to_hide) > 0:
                obj_ids_to_hide_t = torch.tensor(obj_ids_to_hide, dtype=torch.int64)
                keep &= ~torch.isin(out_obj_ids, obj_ids_to_hide_t)

            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            keep_idx_gpu = keep_idx.pin_memory().to(device=out_binary_masks.device, non_blocking=True)

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_tracker_probs = torch.index_select(out_tracker_probs, 0, keep_idx)
            out_binary_masks = torch.index_select(out_binary_masks, 0, keep_idx_gpu)

            if perflib.is_enabled:
                out_boxes_xyxy = perf_masks_to_boxes(out_binary_masks, out_obj_ids.tolist())
            else:
                out_boxes_xyxy = masks_to_boxes(out_binary_masks)

            out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)
            out_boxes_xywh[..., 0] /= W_video
            out_boxes_xywh[..., 1] /= H_video
            out_boxes_xywh[..., 2] /= W_video
            out_boxes_xywh[..., 3] /= H_video

        if out_binary_masks.shape[0] > 1:
            assert len(out_binary_masks) == len(out_tracker_probs)
            out_binary_masks = (
                self.tracker._apply_object_wise_non_overlapping_constraints(
                    out_binary_masks.unsqueeze(1),
                    out_tracker_probs.unsqueeze(1).to(out_binary_masks.device),
                    background_value=0,
                ).squeeze(1)
            ) > 0

        outputs = {
            "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_probs": out_probs.cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
            "frame_stats": out.get("frame_stats", None),
        }
        return outputs

    def _cache_frame_outputs(
        self,
        inference_state,
        frame_idx,
        obj_id_to_mask,
        suppressed_obj_ids=None,
        removed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        filtered_obj_id_to_mask = dict(obj_id_to_mask)
        objects_to_exclude = set()
        if suppressed_obj_ids is not None:
            objects_to_exclude.update(suppressed_obj_ids)
        if removed_obj_ids is not None:
            objects_to_exclude.update(removed_obj_ids)
        if unconfirmed_obj_ids is not None:
            objects_to_exclude.update(unconfirmed_obj_ids)
        for k in list(filtered_obj_id_to_mask.keys()):
            if k in objects_to_exclude:
                filtered_obj_id_to_mask.pop(k, None)
        inference_state["cached_frame_outputs"][frame_idx] = filtered_obj_id_to_mask
        # Prune cached frames if exceeding limit
        max_cached = self.max_cached_frames
        if max_cached > 0:
            cached_keys = sorted(
                [k for k in inference_state["cached_frame_outputs"].keys() if isinstance(k, int)]
            )
            if len(cached_keys) > max_cached:
                to_remove = cached_keys[: len(cached_keys) - max_cached]
                for old_k in to_remove:
                    inference_state["cached_frame_outputs"].pop(old_k, None)

    def _build_tracker_output(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
        assert (
            "cached_frame_outputs" in inference_state and frame_idx in inference_state["cached_frame_outputs"]
        ), "No cached outputs found. Ensure at least one inference has run to populate the cache."
        cached_outputs = inference_state["cached_frame_outputs"][frame_idx]
        obj_id_to_mask = dict(cached_outputs)
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                assert refined_mask is not None, f"Refined mask data must be provided for obj_id {obj_id}"
                obj_id_to_mask[obj_id] = refined_mask
        return obj_id_to_mask

    def get_cached_output_for_frame(self, inference_state: Dict[str, Any], frame_idx: int):
        return inference_state.get("cached_frame_outputs", {}).get(frame_idx, {})

    def _compile_model(self):
        is_compiled = getattr(self, "_model_is_compiled", False)
        if is_compiled or not self.compile_model:
            return
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        self.detector.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(self.detector.backbone.vision_backbone.forward, fullgraph=True, mode="max-autotune")
        )
        self.detector.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(self.detector.transformer.encoder.forward, fullgraph=True, mode="max-autotune")
        )
        self.detector.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(self.detector.transformer.decoder.forward, fullgraph=True, mode="max-autotune", dynamic=False)
        )
        self.detector.segmentation_head.forward = clone_output_wrapper(
            torch.compile(self.detector.segmentation_head.forward, fullgraph=True, mode="max-autotune")
        )
        self.tracker.maskmem_backbone.forward = compile_wrapper(
            self.tracker.maskmem_backbone.forward, mode="max-autotune", fullgraph=True, dynamic=False
        )
        self.tracker.transformer.encoder.forward = shape_logging_wrapper(
            compile_wrapper(
                self.tracker.transformer.encoder.forward,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            ),
            keep_kwargs=["src", "src_pos", "prompt", "prompt_pos"],
        )
        self.tracker.sam_mask_decoder.forward = compile_wrapper(
            self.tracker.sam_mask_decoder.forward, mode="max-autotune", fullgraph=True, dynamic=False
        )
        self._model_is_compiled = True

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def warm_up_compilation(self, num_frames: int = 8):
        if not self.compile_model:
            return
        if self.device.type != "cuda":
            raise RuntimeError(f"The model must be on CUDA for warm-up compilation, got {self.device=}")
        orig_rank, orig_world = self.rank, self.world_size
        self.rank = self.detector.rank = 0
        self.world_size = self.detector.world_size = 1
        self._compile_model()
        state = self.init_stream_state()
        dummy_text = "object"
        for i in range(num_frames):
            arr = (np.random.rand(self.image_size, self.image_size, 3) * 255).astype(np.uint8)
            self.add_frame(state, arr)
            if i == 0:
                self.add_prompt(state, frame_idx=0, text_str=dummy_text)
            _ = self.run_single_frame_inference(state, frame_idx=i)
        self.rank = self.detector.rank = orig_rank
        self.world_size = self.detector.world_size = orig_world
        logger.info("Warm-up compilation completed (streaming mode).")
