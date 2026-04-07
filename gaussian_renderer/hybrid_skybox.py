import os
from typing import Dict, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SKYBOX_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


def _read_skybox_array(skybox_path: str):
    ext = os.path.splitext(skybox_path)[1].lower()

    if ext in {".hdr", ".exr"}:
        try:
            array = imageio.imread(skybox_path)
        except Exception:
            array = None
            try:
                import cv2

                array = cv2.imread(skybox_path, cv2.IMREAD_UNCHANGED)
                if array is not None and array.ndim >= 3:
                    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            except Exception:
                array = None
        if array is None:
            raise ValueError(f"Failed to load HDR skybox from '{skybox_path}'.")
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=2)
        if array.shape[-1] > 3:
            array = array[..., :3]
        return array

    image = Image.open(skybox_path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_skybox_tensor(skybox_path: str, device):
    cache_key = (os.path.abspath(skybox_path), str(device))
    cached = _SKYBOX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    array = _read_skybox_array(skybox_path)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    _SKYBOX_CACHE[cache_key] = tensor
    return tensor


def _fallback_background(viewpoint_camera, bg_color):
    return bg_color[:, None, None].expand(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))


def render_skybox(viewpoint_camera, skybox_path: str, bg_color: torch.Tensor):
    if not skybox_path:
        return {"sky_rgb": _fallback_background(viewpoint_camera, bg_color)}

    device = viewpoint_camera.world_view_transform.device
    rays_world = viewpoint_camera.generate_camera_rays(device=device)
    rays_world = torch.nn.functional.normalize(rays_world, dim=-1)

    x = rays_world[..., 0]
    y = rays_world[..., 1]
    z = rays_world[..., 2]

    u = torch.atan2(x, z) / (2.0 * torch.pi) + 0.5
    v = torch.acos(torch.clamp(y, -1.0, 1.0)) / torch.pi

    grid_x = (torch.remainder(u, 1.0) * 2.0) - 1.0
    grid_y = (v.clamp(0.0, 1.0) * 2.0) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    skybox = _load_skybox_tensor(skybox_path, device)
    sky_rgb = F.grid_sample(skybox, grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    return {"sky_rgb": sky_rgb}
