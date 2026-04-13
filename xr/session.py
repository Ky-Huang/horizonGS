import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from xr.frame_sources import SocketFrameSource, load_xr_frames
from xr.openxr_bridge import build_minicam_from_openxr_view, load_xr_session_config


def _read_rgb_frame(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _pad_frame_to_block(frame, block_size):
    block_size = max(int(block_size), 1)
    if block_size == 1:
        return frame
    height, width = frame.shape[:2]
    padded_h = ((height + block_size - 1) // block_size) * block_size
    padded_w = ((width + block_size - 1) // block_size) * block_size
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _write_video_from_render_dir(render_dir, output_path, fps):
    frame_paths = sorted(
        os.path.join(render_dir, name)
        for name in os.listdir(render_dir)
        if name.lower().endswith(".png")
    )
    if not frame_paths:
        return False

    writer = imageio.get_writer(
        output_path,
        fps=int(fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for frame_path in frame_paths:
            frame = _pad_frame_to_block(_read_rgb_frame(frame_path), 16)
            writer.append_data(frame)
    finally:
        writer.close()
    return True


def _iter_frames(xr_mode, xr_input, xr_socket_host, xr_socket_port):
    if xr_mode == "openxr_replay":
        for frame in load_xr_frames(xr_input):
            yield frame
        return

    if xr_mode == "openxr_socket":
        with SocketFrameSource(xr_socket_host, xr_socket_port) as source:
            for frame in source:
                yield frame
        return

    raise ValueError(f"Unsupported XR mode: {xr_mode}")


def run_openxr_render_session(
    model_path,
    iteration,
    gaussians,
    pipe,
    background,
    render_fn,
    xr_mode,
    xr_input="",
    xr_config_path="",
    xr_output_name="openxr",
    xr_output_layout="both",
    xr_save_video=False,
    xr_video_fps=30,
    xr_socket_host="127.0.0.1",
    xr_socket_port=6110,
    xr_max_frames=-1,
):
    config = load_xr_session_config(xr_config_path)
    output_root = os.path.join(model_path, xr_output_name, f"ours_{iteration}")
    left_dir = os.path.join(output_root, "left_eye")
    right_dir = os.path.join(output_root, "right_eye")
    sbs_dir = os.path.join(output_root, "side_by_side")
    for path in [left_dir, right_dir]:
        os.makedirs(path, exist_ok=True)
    if xr_output_layout in {"side_by_side", "both"}:
        os.makedirs(sbs_dir, exist_ok=True)

    frame_records = []
    iterator = _iter_frames(xr_mode, xr_input, xr_socket_host, xr_socket_port)
    total = None if xr_mode == "openxr_socket" else max(int(xr_max_frames), 0) if xr_max_frames > 0 else None
    progress = tqdm(iterator, total=total, desc="OpenXR rendering")
    for frame_idx, frame in enumerate(progress):
        if xr_max_frames > 0 and frame_idx >= xr_max_frames:
            break

        frame_id = int(frame.get("frame_id", frame_idx))
        left_cam = build_minicam_from_openxr_view(frame, "left", config)
        right_cam = build_minicam_from_openxr_view(frame, "right", config)

        left_pkg = render_fn(left_cam, gaussians, pipe, background)
        right_pkg = render_fn(right_cam, gaussians, pipe, background)
        left_rgb = torch.clamp(left_pkg["render"], 0.0, 1.0)
        right_rgb = torch.clamp(right_pkg["render"], 0.0, 1.0)

        left_path = os.path.join(left_dir, f"{frame_id:05d}.png")
        right_path = os.path.join(right_dir, f"{frame_id:05d}.png")
        torchvision.utils.save_image(left_rgb, left_path)
        torchvision.utils.save_image(right_rgb, right_path)

        if xr_output_layout in {"side_by_side", "both"}:
            sbs_path = os.path.join(sbs_dir, f"{frame_id:05d}.png")
            torchvision.utils.save_image(torch.cat([left_rgb, right_rgb], dim=2), sbs_path)

        frame_records.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": frame.get("timestamp_ns"),
                "left_path": os.path.relpath(left_path, output_root),
                "right_path": os.path.relpath(right_path, output_root),
                "left_camera": {
                    "center": [float(x) for x in left_cam.camera_center.detach().cpu().tolist()],
                    "fx": float(left_cam.fx),
                    "fy": float(left_cam.fy),
                    "cx": float(left_cam.cx),
                    "cy": float(left_cam.cy),
                    "width": int(left_cam.image_width),
                    "height": int(left_cam.image_height),
                },
                "right_camera": {
                    "center": [float(x) for x in right_cam.camera_center.detach().cpu().tolist()],
                    "fx": float(right_cam.fx),
                    "fy": float(right_cam.fy),
                    "cx": float(right_cam.cx),
                    "cy": float(right_cam.cy),
                    "width": int(right_cam.image_width),
                    "height": int(right_cam.image_height),
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "xr_mode": xr_mode,
        "xr_input": xr_input,
        "xr_config_path": xr_config_path,
        "output_layout": xr_output_layout,
        "frame_count": len(frame_records),
        "frames": frame_records,
    }
    with open(os.path.join(output_root, "xr_session_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if xr_save_video:
        if _write_video_from_render_dir(left_dir, os.path.join(output_root, "left_eye.mp4"), xr_video_fps):
            print(f"[openxr] wrote video: {os.path.join(output_root, 'left_eye.mp4')}")
        if _write_video_from_render_dir(right_dir, os.path.join(output_root, "right_eye.mp4"), xr_video_fps):
            print(f"[openxr] wrote video: {os.path.join(output_root, 'right_eye.mp4')}")
        if xr_output_layout in {"side_by_side", "both"}:
            if _write_video_from_render_dir(sbs_dir, os.path.join(output_root, "side_by_side.mp4"), xr_video_fps):
                print(f"[openxr] wrote video: {os.path.join(output_root, 'side_by_side.mp4')}")

    print(f"[openxr] rendered {len(frame_records)} stereo frames into {output_root}")
    return True
