import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from gaussian_renderer.render import render as render_gaussians
from gaussian_renderer.hybrid_skybox import render_skybox
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


def _normalize(vectors, eps=1e-8):
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)


def _cube_face_bases(device, dtype):
    faces = [
        {"name": "px", "forward": [1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
        {"name": "nx", "forward": [-1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
        {"name": "py", "forward": [0.0, 1.0, 0.0], "up": [0.0, 0.0, -1.0]},
        {"name": "ny", "forward": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0]},
        {"name": "pz", "forward": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0]},
        {"name": "nz", "forward": [0.0, 0.0, -1.0], "up": [0.0, 1.0, 0.0]},
    ]
    for face in faces:
        forward = torch.tensor(face["forward"], device=device, dtype=dtype)
        up_hint = torch.tensor(face["up"], device=device, dtype=dtype)
        right = _normalize(torch.cross(up_hint, forward, dim=0))
        up = _normalize(torch.cross(forward, right, dim=0))
        face["forward"] = forward
        face["right"] = right
        face["up"] = up
    return faces


def _make_minicam(center_world, right, up, forward, resolution, znear, zfar):
    device = center_world.device
    dtype = center_world.dtype

    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, 0] = right
    # Match HorizonGS internal camera convention: x-right, y-down, z-forward.
    c2w[:3, 1] = -up
    c2w[:3, 2] = forward
    c2w[:3, 3] = center_world

    world_view_transform = torch.inverse(c2w).transpose(0, 1).contiguous()
    projection = getProjectionMatrix(
        znear=znear,
        zfar=zfar,
        fovX=math.pi * 0.5,
        fovY=math.pi * 0.5,
    ).transpose(0, 1).to(device=device, dtype=dtype)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0))).squeeze(0)

    return MiniCam(
        width=resolution,
        height=resolution,
        fovy=math.pi * 0.5,
        fovx=math.pi * 0.5,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def build_gs_env_cubemap(center_world, pc, pipe, bg_color):
    resolution = int(getattr(pipe, "hybrid_mesh_env_resolution", 64))
    resolution = max(resolution, 8)
    device = center_world.device
    dtype = center_world.dtype
    bases = _cube_face_bases(device, dtype)
    gs_bg = torch.zeros_like(bg_color, device=device)
    skybox_path = getattr(pipe, "hybrid_skybox_path", "")
    cubemap_faces: List[torch.Tensor] = []

    for face in bases:
        env_cam = _make_minicam(
            center_world=center_world,
            right=face["right"],
            up=face["up"],
            forward=face["forward"],
            resolution=resolution,
            znear=0.01,
            zfar=100.0,
        )
        gs_pkg = render_gaussians(env_cam, pc, pipe, gs_bg)
        face_rgb = gs_pkg["render"]
        if skybox_path:
            sky_rgb = render_skybox(env_cam, skybox_path, bg_color)["sky_rgb"]
            face_rgb = face_rgb + (1.0 - gs_pkg["render_alphas"]) * sky_rgb
        cubemap_faces.append(face_rgb.clamp(0.0, 1.0))

    cubemap = torch.stack(cubemap_faces, dim=0)

    if bool(getattr(pipe, "hybrid_verbose", False)):
        center = [round(v, 3) for v in center_world.detach().cpu().tolist()]
        print(f"[hybrid-env] center={center} resolution={resolution}")

    return {
        "cubemap": cubemap,
        "faces": bases,
        "center_world": center_world,
    }


def sample_env_cubemap(cubemap, faces, directions_world):
    dirs = _normalize(directions_world)
    flat_dirs = dirs.reshape(-1, 3)
    forward_stack = torch.stack([face["forward"] for face in faces], dim=0)
    scores = flat_dirs @ forward_stack.transpose(0, 1)
    face_indices = torch.argmax(scores, dim=1)

    out = torch.zeros((flat_dirs.shape[0], 3), device=flat_dirs.device, dtype=cubemap.dtype)
    for face_idx, face in enumerate(faces):
        mask = face_indices == face_idx
        if not torch.any(mask):
            continue

        dirs_face = flat_dirs[mask]
        x = (dirs_face * face["right"].view(1, 3)).sum(dim=-1)
        y = (dirs_face * face["up"].view(1, 3)).sum(dim=-1)
        z = (dirs_face * face["forward"].view(1, 3)).sum(dim=-1).clamp_min(1e-6)
        grid = torch.stack((x / z, y / z), dim=-1).view(1, -1, 1, 2).clamp(-1.0, 1.0)
        sampled = F.grid_sample(
            cubemap[face_idx : face_idx + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, :, :, 0].transpose(0, 1)
        out[mask] = sampled

    return out.view(*dirs.shape[:-1], 3)


def apply_env_lighting(albedo_rgb, normal_world, alpha, env_ctx, pipe):
    diffuse_strength = float(getattr(pipe, "hybrid_mesh_env_diffuse_strength", 1.0))
    ambient_strength = float(getattr(pipe, "hybrid_mesh_env_ambient_strength", 0.25))

    env_rgb = sample_env_cubemap(env_ctx["cubemap"], env_ctx["faces"], normal_world.permute(1, 2, 0))
    env_rgb = env_rgb.permute(2, 0, 1).contiguous()
    lit_rgb = albedo_rgb * (ambient_strength + diffuse_strength * env_rgb)
    return torch.where(alpha > 1e-4, lit_rgb.clamp(0.0, 1.0), albedo_rgb)
