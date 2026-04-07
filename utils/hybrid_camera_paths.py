import copy
import math
import os
from typing import List, Optional

import numpy as np
import torch


def _normalize(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def _camera_name_key(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0].lower()


def find_camera_by_name(cameras, target_name: str):
    target_key = _camera_name_key(target_name)
    for cam in cameras:
        if _camera_name_key(getattr(cam, "image_name", "")) == target_key:
            return cam
    raise ValueError(f"Camera '{target_name}' was not found in the loaded camera pool.")


def _look_at_c2w(position, target, up_hint):
    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up_hint = _normalize(np.asarray(up_hint, dtype=np.float32))

    forward = _normalize(target - position)
    right = np.cross(up_hint, forward)
    if np.linalg.norm(right) < 1e-6:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(alt_up, forward)) > 0.95:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(alt_up, forward)
    right = _normalize(right)
    up = _normalize(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    # HorizonGS cameras follow the internal convention used by Camera /
    # world_view_transform: x-right, y-down, z-forward. Keep orbit cameras in
    # the same basis so synthetic paths match the loaded COLMAP cameras.
    c2w[:3, 1] = -up
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def _project_to_plane(vec, normal):
    normal = _normalize(normal)
    return vec - normal * np.dot(vec, normal)


def _stable_up_hint(reference_up, forward, fallback):
    up_hint = _project_to_plane(reference_up, forward)
    if np.linalg.norm(up_hint) < 1e-6:
        up_hint = _project_to_plane(fallback, forward)
    if np.linalg.norm(up_hint) < 1e-6:
        up_hint = _project_to_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), forward)
    if np.linalg.norm(up_hint) < 1e-6:
        up_hint = _project_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), forward)
    return _normalize(up_hint)


def _basis_from_camera(camera):
    c2w = camera.get_camera_to_world().detach().cpu().numpy()
    right = c2w[:3, 0]
    down = c2w[:3, 1]
    forward = c2w[:3, 2]
    up = -down
    visual_c2w = np.eye(4, dtype=np.float32)
    visual_c2w[:3, 0] = right
    visual_c2w[:3, 1] = up
    visual_c2w[:3, 2] = forward
    visual_c2w[:3, 3] = c2w[:3, 3]
    return {
        "c2w": c2w,
        "visual_c2w": visual_c2w,
        "position": c2w[:3, 3],
        "right": right,
        "down": down,
        "up": up,
        "forward": forward,
    }


def estimate_local_scene_up(cameras, reference_camera, k_neighbors: int = 32):
    ref = _basis_from_camera(reference_camera)
    ref_pos = ref["position"]
    ref_type = getattr(reference_camera, "image_type", None)

    bases = []
    for cam in cameras:
        if ref_type is not None and getattr(cam, "image_type", None) != ref_type:
            continue
        basis = _basis_from_camera(cam)
        dist = np.linalg.norm(basis["position"] - ref_pos)
        bases.append((dist, basis["up"]))

    if not bases:
        return ref["up"]

    bases.sort(key=lambda item: item[0])
    ups = np.stack([up for _, up in bases[: max(1, int(k_neighbors))]], axis=0)
    avg_up = _normalize(ups.mean(axis=0))
    if np.dot(avg_up, ref["up"]) < 0.0:
        avg_up = -avg_up
    return avg_up


def resolve_orbit_axis(reference_camera, cameras, axis=None, axis_mode: str = "world"):
    axis_mode = (axis_mode or "world").lower()
    ref = _basis_from_camera(reference_camera)

    if axis_mode == "auto_scene_up":
        return estimate_local_scene_up(cameras, reference_camera)

    if axis_mode == "reference_local":
        local_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32) if axis is None else np.asarray(axis, dtype=np.float32)
        local_axis = _normalize(local_axis)
        return _normalize(ref["visual_c2w"][:3, :3] @ local_axis)

    if axis is None:
        return estimate_local_scene_up(cameras, reference_camera)
    return _normalize(np.asarray(axis, dtype=np.float32))


def make_camera_from_c2w(template_camera, c2w, image_name: str, image_type: Optional[str] = None):
    cam = copy.copy(template_camera)
    device = template_camera.world_view_transform.device
    dtype = template_camera.world_view_transform.dtype

    c2w_tensor = torch.tensor(c2w, device=device, dtype=dtype)
    w2c = torch.inverse(c2w_tensor).transpose(0, 1).contiguous()

    cam.world_view_transform = w2c
    cam.c2w = c2w_tensor
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    cam.image_name = image_name
    cam.image_path = image_name
    cam.image_type = image_type or getattr(template_camera, "image_type", "street")
    cam.render_only = True
    cam.original_image = None
    cam.alpha_mask = None
    return cam


def build_interpolated_path(start_camera, end_camera, num_frames: int):
    num_frames = max(int(num_frames), 2)
    start = _basis_from_camera(start_camera)
    end = _basis_from_camera(end_camera)
    image_type = getattr(start_camera, "image_type", "street")

    cameras = []
    for idx, t in enumerate(np.linspace(0.0, 1.0, num_frames, dtype=np.float32)):
        position = (1.0 - t) * start["position"] + t * end["position"]
        forward = _normalize((1.0 - t) * start["forward"] + t * end["forward"])
        up_hint = _normalize((1.0 - t) * start["up"] + t * end["up"])
        c2w = _look_at_c2w(position, position + forward, up_hint)
        cameras.append(
            make_camera_from_c2w(
                start_camera,
                c2w,
                image_name=f"path_interp_{idx:05d}.png",
                image_type=image_type,
            )
        )
    return cameras


def build_orbit_path(
    reference_camera,
    cameras,
    center,
    axis,
    radius: float,
    polar_deg: float,
    num_frames: int,
    start_azimuth_deg: float = 0.0,
    sweep_deg: float = 360.0,
    axis_mode: str = "world",
    orbit_style: str = "sphere",
):
    num_frames = max(int(num_frames), 2)
    center = np.asarray(center, dtype=np.float32)
    axis = resolve_orbit_axis(reference_camera, cameras, axis=axis, axis_mode=axis_mode)
    radius = float(radius)
    polar_rad = math.radians(float(polar_deg))
    start_azimuth = math.radians(float(start_azimuth_deg))
    sweep_rad = math.radians(float(sweep_deg))
    orbit_style = (orbit_style or "sphere").lower()

    ref = _basis_from_camera(reference_camera)
    ref_offset = ref["position"] - center
    tangent_u = ref_offset - axis * np.dot(ref_offset, axis)
    if np.linalg.norm(tangent_u) < 1e-6:
        tangent_u = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        if np.linalg.norm(tangent_u) < 1e-6:
            tangent_u = np.cross(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    tangent_u = _normalize(tangent_u)
    tangent_v = _normalize(np.cross(axis, tangent_u))

    cameras = []
    for idx, theta in enumerate(np.linspace(start_azimuth, start_azimuth + sweep_rad, num_frames, endpoint=False)):
        if orbit_style == "reference_plane_circle":
            # Orbit around the reference camera center on the plane orthogonal
            # to the reference view axis, while always looking at the target.
            view_axis = _normalize(ref["position"] - center)
            top_dir = _project_to_plane(ref["up"], view_axis)
            if np.linalg.norm(top_dir) < 1e-6:
                top_dir = _project_to_plane(axis, view_axis)
            if np.linalg.norm(top_dir) < 1e-6:
                top_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                top_dir = _project_to_plane(top_dir, view_axis)
            top_dir = _normalize(top_dir)

            right_dir = _project_to_plane(ref["right"], view_axis)
            if np.linalg.norm(right_dir) < 1e-6:
                right_dir = np.cross(view_axis, top_dir)
            right_dir = _normalize(right_dir)
            left_dir = -right_dir

            ring_dir = math.cos(theta) * top_dir + math.sin(theta) * left_dir
            position = ref["position"] + radius * ring_dir
            forward = _normalize(center - position)
            up_hint = _stable_up_hint(ref["up"], forward, top_dir)
        else:
            radial = math.cos(theta) * tangent_u + math.sin(theta) * tangent_v
            if orbit_style == "level_circle":
                height = float(np.dot(ref_offset, axis))
                horizontal_radius = float(np.linalg.norm(ref_offset - axis * height))
                if radius > 0.0:
                    horizontal_radius = radius
                position = center + axis * height + radial * horizontal_radius
            else:
                orbit_dir = math.sin(polar_rad) * radial + math.cos(polar_rad) * axis
                position = center + radius * orbit_dir
            forward = _normalize(center - position)
            up_hint = _stable_up_hint(ref["up"], forward, axis)
        c2w = _look_at_c2w(position, center, up_hint)
        cameras.append(
            make_camera_from_c2w(
                reference_camera,
                c2w,
                image_name=f"path_orbit_{idx:05d}.png",
                image_type=getattr(reference_camera, "image_type", "street"),
            )
        )
    return cameras


def collect_camera_pool(scene) -> List[object]:
    return scene.getTrainCameras() + scene.getTestCameras()
