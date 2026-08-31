"""Media preparation and encoding adapter for the SCAIL-2 engine."""

import logging
import os
import sys

from einops import rearrange
from PIL import Image

import wan
from wan.utils.scail_utils import (
    load_image_to_tensor_chw_normalized,
    load_video_for_pose_sample,
    resize_for_rectangle_crop,
)
from wan.utils.utils import cache_video


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _check_input_path(path, name):
    if path is None:
        raise ValueError(f"Please specify {name}.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} is not a file: {path}")


def generate_video(
    pipeline: wan.SCAIL2Pipeline,
    image_path: str,
    image_mask_path: str,
    pose_path: str,
    driving_mask_path: str,
    args,
    device,
    cfg,
    replace_flag,
    additional_task_input=None,
    output_fps=None,
    output_fps_fraction=None,
    conditioning=None,
):
    _check_input_path(image_path, "input image")
    _check_input_path(image_mask_path, "input mask image")
    _check_input_path(pose_path, "input pose video")
    _check_input_path(driving_mask_path, "input mask video")

    additional_task_input = additional_task_input or {}
    additional_input = {}
    image = Image.open(image_path).convert("RGB")
    image_tensor = load_image_to_tensor_chw_normalized(image).to(device)
    _, _, height, width = image_tensor.shape
    target_h, target_w = args.target_h, args.target_w
    if target_h is None or target_w is None:
        target_h, target_w = height, width
    if (height < width and target_h > target_w) or (height > width and target_h < target_w):
        target_h, target_w = target_w, target_h

    mask_image = Image.open(image_mask_path).convert("RGB")
    mask_tensor = load_image_to_tensor_chw_normalized(mask_image).to(device)

    image_paths = additional_task_input.get("additional_ref_image_paths")
    if image_paths is not None:
        mask_paths = additional_task_input.get("additional_ref_mask_image_paths")
        if mask_paths is None or len(image_paths) != len(mask_paths):
            raise ValueError("Additional reference image/mask paths must match")
        additional_imgs = []
        additional_masks = []
        for index, (extra_image_path, extra_mask_path) in enumerate(zip(image_paths, mask_paths)):
            _check_input_path(extra_image_path, f"additional ref image {index}")
            _check_input_path(extra_mask_path, f"additional ref mask image {index}")
            extra_image = load_image_to_tensor_chw_normalized(
                Image.open(extra_image_path).convert("RGB")
            ).to(device)
            extra_mask = load_image_to_tensor_chw_normalized(
                Image.open(extra_mask_path).convert("RGB")
            ).to(device)
            additional_imgs.append(
                resize_for_rectangle_crop(extra_image, (target_h, target_w), reshape_mode="center").squeeze(0)
            )
            additional_masks.append(
                resize_for_rectangle_crop(extra_mask, (target_h, target_w), reshape_mode="center").squeeze(0)
            )
        additional_input["additional_ref_imgs"] = additional_imgs
        additional_input["additional_ref_mask_imgs"] = additional_masks

    pose_video = load_video_for_pose_sample(pose_path, image_size=(target_h, target_w))
    driving_mask_video = load_video_for_pose_sample(
        driving_mask_path, image_size=(target_h, target_w)
    )
    driving_mask_video = rearrange(driving_mask_video, "t c h w -> c t h w")
    image_tensor = resize_for_rectangle_crop(
        image_tensor, (target_h, target_w), reshape_mode="center"
    ).squeeze(0)
    mask_tensor = resize_for_rectangle_crop(
        mask_tensor, (target_h, target_w), reshape_mode="center"
    ).squeeze(0)

    video = pipeline.generate(
        image_tensor,
        ref_mask_img=mask_tensor,
        pose_video=pose_video,
        driving_mask_video=driving_mask_video,
        replace_flag=replace_flag,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        segment_len=args.segment_len,
        segment_overlap=args.segment_overlap,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        conditioning=conditioning,
        **additional_input,
    )
    logging.info("Saving generated video to %s", args.save_file)
    written_video = cache_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps if output_fps is None else output_fps,
        fps_fraction=output_fps_fraction,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    if written_video is None:
        raise RuntimeError(f"Failed to encode generated video: {args.save_file}")
