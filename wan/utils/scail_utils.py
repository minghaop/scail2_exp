import os
import decord
import numpy as np
from decord import VideoReader
import torch
import torch.nn.functional as F
import logging

from PIL import Image
import torchvision.transforms as TT

from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import center_crop, resize

def load_image_to_tensor_chw_normalized(image: Image.Image):
    # Open image using PIL
    # image = Image.open(image_data).convert('RGB')  # Convert to RGB in case it's a grayscale image or has an alpha channel
    # Define a transform to convert image to tensor
    transform = TT.Compose([TT.ToTensor()])
    # Apply the transform
    image_tensor = transform(image)
    # Scale the tensor back to [0, 255] and convert to uint8 (decord does this too)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)  # 1 C H W, -1-1
    return image_tensor

def load_video_for_pose_sample(video_data, image_size=None, batch_size=None):
    """Decode a video, optionally resizing it in bounded-memory frame batches.

    The legacy path decoded every source-resolution frame at once and only then
    resized the full tensor. A long 4K video can require tens of gigabytes per
    FSDP rank before the resize. Supplying ``image_size`` preserves the same
    per-frame bicubic center crop while retaining only a small source-resolution
    batch and one uint8 target-resolution tensor in memory.
    """
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    frame_count = len(vr)
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {video_data}")
    if image_size is not None:
        target_h, target_w = (int(value) for value in image_size)
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid target video size: {image_size}")
        if batch_size is None:
            batch_size = int(os.getenv("SCAIL_VIDEO_DECODE_BATCH_SIZE", "8"))
        if batch_size <= 0:
            raise ValueError("Video decode batch size must be positive")

        resized_frames = torch.empty(
            (frame_count, 3, target_h, target_w), dtype=torch.uint8
        )
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            frame_batch = vr.get_batch(np.arange(start, end))
            if not isinstance(frame_batch, torch.Tensor):
                frame_batch = torch.from_numpy(frame_batch)
            if frame_batch.ndim != 4 or frame_batch.shape[-1] != 3:
                raise ValueError(
                    f"Expected THWC RGB video frames, got {tuple(frame_batch.shape)}: "
                    f"{video_data}"
                )
            frame_batch = frame_batch.permute(0, 3, 1, 2)
            frame_batch = resize_for_rectangle_crop(
                frame_batch, (target_h, target_w), reshape_mode="center"
            )
            if frame_batch.dtype != torch.uint8:
                raise TypeError(
                    f"Expected uint8 resized video frames, got {frame_batch.dtype}: "
                    f"{video_data}"
                )
            resized_frames[start:end].copy_(frame_batch)
        return resized_frames

    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    return tensor_frms


def normalize_condition_segment(segment, device):
    """Move one condition segment to ``device`` and return normalized FP32.

    Float tensors are backward-compatible API inputs that are already expected
    to be in [-1, 1]. The bounded video loader returns uint8 [0, 255], which is
    normalized only after slicing the active segment. Reject other integer
    types instead of silently treating arbitrary integer data as pixels.
    """
    if segment.is_floating_point():
        return segment.to(device=device, dtype=torch.float32)
    if segment.dtype != torch.uint8:
        raise TypeError(
            f"Condition segment must be floating point or uint8, got {segment.dtype}"
        )
    return (
        segment.to(device=device, dtype=torch.float32)
        .sub_(127.5)
        .div_(127.5)
    )


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr

def find_file_with_patterns(directory, patterns):
    """Find file matching any of the given patterns in the directory"""
    for pattern in patterns:
        file_path = os.path.join(directory, pattern)
        if os.path.exists(file_path):
            return file_path
    return None

def get_tasks_from_txt(path):
    tasks = []
    idx = 0
    with open(path, "r") as f:
        for line in f:
            text = line.strip()
            text_parts = text.split('@@')
            text = text_parts[0]
            input_dir = text_parts[1]
            
            # Find reference image with multiple possible names
            ref_image_patterns = ['ref.jpg', 'ref.png', 'ref_image.jpg', 'ref_image.png']
            image_path = find_file_with_patterns(input_dir, ref_image_patterns)
            if image_path is None:
                raise FileNotFoundError(f"Reference image not found in {input_dir}. Tried: {ref_image_patterns}")
            
            # Find pose video with multiple possible names
            pose_patterns = ['rendered.mp4', 'smpl_aligned.mp4', 'smpl_render.mp4']
            pose_path = find_file_with_patterns(input_dir, pose_patterns)
            if pose_path is None:
                raise FileNotFoundError(f"Pose video not found in {input_dir}. Tried: {pose_patterns}")
            
            if text == "None":
                text = ""
            else:
                text = text

            tasks.append((text, image_path, pose_path, idx))
            idx += 1
    return tasks


def extract_and_compress_mask_to_latent(mask_cthw, additional_spatial_downsample=1, temporal_compression_stride=4):
    """将 3通道 RGB 分割mask 转换为 28通道二值 latent，不经过 VAE。
    输入: (3, T, H, W)，值域 [-1, 1]
    输出: (28, T_latent, H_latent, W_latent)，值域 {0, 1}
    """
    C, T, H, W = mask_cthw.shape
    _ON_THRESH = (225.0 - 127.5) / 127.5  # ≈ 0.765，原始像素值 ≥ 225 才算"亮"
    mask = mask_cthw.permute(1, 0, 2, 3).float()  # (T, 3, H, W)
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch.cat([
        R * G * B, R * nG * nB, nR * G * nB, nR * nG * B,
        R * G * nB, R * nG * B, nR * G * B,
    ], dim=1)  # (T, 7, H, W)
    _color_names = ['white', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    _total = H * W * T
    for _i, _name in enumerate(_color_names):
        _ratio = binary_7ch[:, _i].sum().item() / _total
        if _ratio > 0.001:
            logging.info(f"  [mask debug] ch{_i} {_name}: {_ratio:.4f} ({_ratio*100:.2f}%)")
    H_lat, W_lat = H, W
    if additional_spatial_downsample > 1:
        H_lat = H_lat // additional_spatial_downsample
        W_lat = W_lat // additional_spatial_downsample
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = F.interpolate(binary_7ch, size=(H_lat, W_lat), mode='area')  # area=均值下采样，完整保留覆盖比例
    T_latent = (T - 1) // temporal_compression_stride + 1
    padded = torch.cat([binary_7ch[:1].repeat(temporal_compression_stride, 1, 1, 1), binary_7ch[1:]], dim=0)
    out = padded.view(T_latent, temporal_compression_stride * 7, H_lat, W_lat).permute(1, 0, 2, 3)
    return out  # (28, T_latent, H_lat, W_lat)
