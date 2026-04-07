import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gaussian_renderer.hybrid_lighting import apply_env_lighting, build_gs_env_cubemap

_MESH_CACHE: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}
_NVDIFFRAST_CONTEXTS: Dict[str, object] = {}


def _as_tensor(array, device, dtype):
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    if isinstance(array, np.ndarray):
        array = np.array(array, copy=True)
    return torch.as_tensor(array, device=device, dtype=dtype)


def _normalize(vectors, eps=1e-8):
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)


def _load_mesh_with_trimesh(mesh_path):
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "Hybrid mesh rendering requires the optional 'trimesh' dependency when enabled."
        ) from exc

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh at '{mesh_path}' does not contain triangle faces.")
    return mesh


def _extract_mesh_buffers(mesh, device):
    vertices = _as_tensor(np.asarray(mesh.vertices), device, torch.float32)
    faces = _as_tensor(np.asarray(mesh.faces), device, torch.long)

    if getattr(mesh.visual, "vertex_colors", None) is not None and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
        vertex_colors = _as_tensor(colors, device, torch.float32)
    else:
        vertex_colors = None

    vertex_uv = None
    texture_image = None
    material_color = None
    visual = getattr(mesh, "visual", None)
    if getattr(visual, "uv", None) is not None and len(visual.uv) == len(mesh.vertices):
        vertex_uv = _as_tensor(np.asarray(visual.uv), device, torch.float32)

    material = getattr(visual, "material", None)
    if material is not None:
        image = getattr(material, "image", None)
        if image is not None:
            if hasattr(image, "convert"):
                image = image.convert("RGBA")
            texture_np = np.asarray(image, dtype=np.float32) / 255.0
            texture_image = _as_tensor(texture_np, device, torch.float32).permute(2, 0, 1).contiguous()
        color_value = None
        for attr in ["main_color", "diffuse", "ambient"]:
            value = getattr(material, attr, None)
            if value is not None:
                color_value = np.asarray(value, dtype=np.float32)
                break
        if color_value is not None:
            if color_value.max() > 1.0:
                color_value = color_value / 255.0
            material_color = _as_tensor(color_value[:3], device, torch.float32).clamp(0.0, 1.0)

    vertex_normals = None
    if getattr(mesh, "vertex_normals", None) is not None and len(mesh.vertex_normals) == len(mesh.vertices):
        vertex_normals = _normalize(_as_tensor(np.asarray(mesh.vertex_normals), device, torch.float32))

    return {
        "vertices": vertices,
        "faces": faces,
        "vertex_colors": vertex_colors,
        "vertex_uv": vertex_uv,
        "texture_image": texture_image,
        "material_color": material_color,
        "vertex_normals": vertex_normals,
    }


def _build_face_colored_cube(device, size=1.0):
    half = 0.5 * float(size)
    face_specs = [
        ([1.0, 0.0, 0.0], [[half, -half, -half], [half, half, -half], [half, half, half], [half, -half, half]]),      # +X
        ([0.0, 1.0, 0.0], [[-half, -half, half], [-half, half, half], [-half, half, -half], [-half, -half, -half]]),  # -X
        ([0.0, 0.0, 1.0], [[-half, half, -half], [-half, half, half], [half, half, half], [half, half, -half]]),      # +Y
        ([1.0, 1.0, 0.0], [[-half, -half, half], [-half, -half, -half], [half, -half, -half], [half, -half, half]]),  # -Y
        ([1.0, 0.0, 1.0], [[-half, -half, half], [half, -half, half], [half, half, half], [-half, half, half]]),      # +Z
        ([0.0, 1.0, 1.0], [[half, -half, -half], [-half, -half, -half], [-half, half, -half], [half, half, -half]]),  # -Z
    ]

    vertices = []
    colors = []
    faces = []
    normals = []
    for face_idx, (color, quad) in enumerate(face_specs):
        base = 4 * face_idx
        vertices.extend(quad)
        colors.extend([color] * 4)
        faces.extend([[base + 0, base + 1, base + 2], [base + 0, base + 2, base + 3]])
        quad_tensor = torch.tensor(quad, dtype=torch.float32)
        face_normal = torch.cross(quad_tensor[1] - quad_tensor[0], quad_tensor[2] - quad_tensor[0], dim=0)
        face_normal = face_normal / torch.linalg.norm(face_normal).clamp_min(1e-8)
        normals.extend([face_normal.tolist()] * 4)

    return {
        "vertices": torch.tensor(vertices, dtype=torch.float32, device=device),
        "faces": torch.tensor(faces, dtype=torch.long, device=device),
        "vertex_colors": torch.tensor(colors, dtype=torch.float32, device=device),
        "vertex_normals": torch.tensor(normals, dtype=torch.float32, device=device),
    }


def _expand_vec3(value, device, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype).flatten()
    elif isinstance(value, (list, tuple, np.ndarray)):
        tensor = torch.tensor(value, device=device, dtype=dtype).flatten()
    else:
        tensor = torch.tensor([value], device=device, dtype=dtype)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    if tensor.numel() != 3:
        raise ValueError(f"Expected scalar or length-3 value, got {value}.")
    return tensor


def _euler_rotation_matrix_deg(degrees_xyz, device, dtype=torch.float32):
    rx, ry, rz = torch.deg2rad(_expand_vec3(degrees_xyz, device, dtype=dtype))
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    rot_x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        device=device,
        dtype=dtype,
    )
    rot_y = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        device=device,
        dtype=dtype,
    )
    rot_z = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return rot_z @ rot_y @ rot_x


def _translation_matrix(translation, device, dtype):
    mat = torch.eye(4, device=device, dtype=dtype)
    mat[:3, 3] = _expand_vec3(translation, device, dtype=dtype)
    return mat


def _apply_transform_matrix(vertices, matrix):
    ones = torch.ones((vertices.shape[0], 1), device=vertices.device, dtype=vertices.dtype)
    hom = torch.cat([vertices, ones], dim=1)
    transformed = (matrix @ hom.transpose(0, 1)).transpose(0, 1)
    return transformed[:, :3] / transformed[:, 3:4].clamp_min(1e-8)


def _apply_normal_matrix(normals, matrix):
    if normals is None:
        return None
    linear = matrix[:3, :3]
    normal_matrix = torch.inverse(linear).transpose(0, 1)
    transformed = normals @ normal_matrix.transpose(0, 1)
    return _normalize(transformed)


def _compose_local_transform(vertices, pipe):
    device = vertices.device
    scale = _expand_vec3(getattr(pipe, "hybrid_mesh_scale", 1.0), device, dtype=vertices.dtype)
    rotation = _euler_rotation_matrix_deg(
        getattr(pipe, "hybrid_mesh_rotation_deg", [0.0, 0.0, 0.0]),
        device,
        dtype=vertices.dtype,
    )

    center = 0.5 * (vertices.min(dim=0).values + vertices.max(dim=0).values)
    scale_matrix = torch.diag(torch.cat([scale, torch.ones(1, device=device, dtype=vertices.dtype)]))
    rotation_matrix = torch.eye(4, device=device, dtype=vertices.dtype)
    rotation_matrix[:3, :3] = rotation
    local_transform = (
        _translation_matrix(center, device, vertices.dtype)
        @ rotation_matrix
        @ scale_matrix
        @ _translation_matrix(-center, device, vertices.dtype)
    )
    return local_transform, center


def _compose_instance_transform(vertices, pipe, viewpoint_camera=None, pc=None):
    device = vertices.device
    dtype = vertices.dtype
    user_matrix = getattr(pipe, "hybrid_mesh_transform_matrix", None)
    if user_matrix is not None:
        user_matrix = torch.tensor(user_matrix, device=device, dtype=dtype).reshape(4, 4)
        return user_matrix

    local_transform, local_center = _compose_local_transform(vertices, pipe)
    current_center = _apply_transform_matrix(local_center.view(1, 3), local_transform)[0]
    translation = torch.zeros(3, device=device, dtype=dtype)

    center_world = getattr(pipe, "hybrid_mesh_center_world", None)
    if center_world is not None:
        translation = translation + _expand_vec3(center_world, device, dtype=dtype) - current_center
    elif bool(getattr(pipe, "hybrid_auto_place_mesh", False)) or bool(getattr(pipe, "hybrid_debug_cube", False)):
        target_center = _compute_target_center(viewpoint_camera, vertices, pipe, pc=pc)
        translation = translation + target_center - current_center

    if getattr(pipe, "hybrid_mesh_translation", None) is not None:
        translation = translation + _expand_vec3(getattr(pipe, "hybrid_mesh_translation"), device, dtype=dtype)

    return _translation_matrix(translation, device, dtype) @ local_transform


def _compute_scene_bounds(pc, device):
    if pc is None:
        return None
    anchors = getattr(pc, "get_anchor", None)
    if anchors is None or len(anchors) == 0:
        return None
    anchors = anchors.detach().to(device=device, dtype=torch.float32)
    bbox_min = anchors.min(dim=0).values
    bbox_max = anchors.max(dim=0).values
    center = 0.5 * (bbox_min + bbox_max)
    radius = 0.5 * torch.linalg.norm(bbox_max - bbox_min)
    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "center": center,
        "radius": radius,
    }


def _get_camera_basis(viewpoint_camera, device):
    rays = viewpoint_camera.generate_camera_rays(device=device)
    forward = torch.nn.functional.normalize(
        rays[int(viewpoint_camera.image_height) // 2, int(viewpoint_camera.image_width) // 2], dim=0
    )
    cam_to_world = viewpoint_camera.get_camera_to_world()[:3, :3]
    right = torch.nn.functional.normalize(cam_to_world[:, 0], dim=0)
    # HorizonGS stores camera basis as x-right, y-down, z-forward.
    up = torch.nn.functional.normalize(-cam_to_world[:, 1], dim=0)
    return right, up, forward


def _compute_target_center(viewpoint_camera, vertices_local, pipe, pc=None):
    device = vertices_local.device
    local_min = vertices_local.min(dim=0).values
    local_max = vertices_local.max(dim=0).values
    local_radius = 0.5 * torch.linalg.norm(local_max - local_min).clamp_min(1e-6)

    placement_mode = getattr(pipe, "hybrid_debug_placement", "camera_locked")
    offset = torch.tensor(getattr(pipe, "hybrid_debug_cube_offset", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
    right, up, forward = _get_camera_basis(viewpoint_camera, device)
    cam_pos = viewpoint_camera.camera_center.to(device=device, dtype=torch.float32)
    scene_bounds = _compute_scene_bounds(pc, device)

    if placement_mode == "scene_centered" and scene_bounds is not None:
        scene_depth = torch.dot(scene_bounds["center"] - cam_pos, forward)
        scene_radius = scene_bounds["radius"]
        margin = float(getattr(pipe, "hybrid_debug_scene_margin", 0.05)) * scene_radius.clamp_min(1.0)
        target_depth = scene_depth - scene_radius - local_radius - margin
        min_depth = max(float(getattr(pipe, "hybrid_debug_min_distance", 2.0)), float(local_radius.item() * 2.5))
        target_depth = torch.clamp(target_depth, min=min_depth)
    else:
        target_depth = torch.tensor(float(getattr(pipe, "hybrid_debug_cube_distance", 4.0)), dtype=torch.float32, device=device)

    return cam_pos + forward * (target_depth + offset[2]) + right * offset[0] + up * offset[1]


def load_mesh_buffers(mesh_path: str, device) -> Dict[str, torch.Tensor]:
    if not mesh_path:
        raise ValueError("mesh_path must be set when hybrid rendering is enabled.")
    cache_key = (os.path.abspath(mesh_path), str(device))
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mesh = _load_mesh_with_trimesh(mesh_path)
    buffers = _extract_mesh_buffers(mesh, device)
    _MESH_CACHE[cache_key] = buffers
    return buffers


def _project_vertices(viewpoint_camera, vertices: torch.Tensor):
    ones = torch.ones((vertices.shape[0], 1), device=vertices.device, dtype=vertices.dtype)
    vertices_h = torch.cat([vertices, ones], dim=1)
    cam = vertices_h @ viewpoint_camera.world_view_transform
    z = cam[:, 2]
    x = viewpoint_camera.fx * (cam[:, 0] / z.clamp_min(1e-8)) + viewpoint_camera.cx
    y = viewpoint_camera.fy * (cam[:, 1] / z.clamp_min(1e-8)) + viewpoint_camera.cy
    return torch.stack([x, y, z], dim=1), cam[:, :3]


def _face_normals(vertices: torch.Tensor, faces: torch.Tensor):
    tris = vertices[faces]
    normals = torch.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0], dim=1)
    return _normalize(normals)


def _resolve_base_color(pipe, device):
    value = getattr(pipe, "hybrid_mesh_color", [1.0, 1.0, 1.0])
    color = torch.tensor(value, dtype=torch.float32, device=device)
    return color.clamp(0.0, 1.0)


def _get_scene_mesh_specs(pipe) -> List[dict]:
    scene_data = getattr(pipe, "hybrid_scene_data", None)
    if isinstance(scene_data, dict) and scene_data.get("meshes"):
        specs = []
        for idx, spec in enumerate(scene_data.get("meshes", [])):
            if not spec or not spec.get("enabled", True):
                continue
            spec = dict(spec)
            spec.setdefault("name", f"mesh_{idx:02d}")
            specs.append(spec)
        return specs

    mesh_path = getattr(pipe, "hybrid_mesh_path", "")
    use_debug_cube = bool(getattr(pipe, "hybrid_debug_cube", False))
    if not mesh_path and not use_debug_cube:
        return []
    return [{"name": "default", "path": mesh_path, "debug_cube": use_debug_cube}]


def _make_pipe_with_mesh_spec(pipe, spec):
    base = dict(getattr(pipe, "__dict__", {}))
    overrides = {
        "hybrid_mesh_name": spec.get("name", "mesh"),
        "hybrid_mesh_path": spec.get("path", base.get("hybrid_mesh_path", "")),
        "hybrid_debug_cube": bool(spec.get("debug_cube", False)),
        "hybrid_debug_cube_size": spec.get("debug_cube_size", base.get("hybrid_debug_cube_size", 1.0)),
        # Scene-config meshes are world-anchored by default; they must not inherit debug auto-placement.
        "hybrid_mesh_scale": spec.get("scale", 1.0),
        "hybrid_mesh_rotation_deg": spec.get("rotation_deg", [0.0, 0.0, 0.0]),
        "hybrid_mesh_translation": spec.get("translation", [0.0, 0.0, 0.0]),
        "hybrid_mesh_center_world": spec.get("center_world", None),
        "hybrid_mesh_transform_matrix": spec.get("transform_matrix", None),
        "hybrid_auto_place_mesh": bool(spec.get("auto_place", False)),
        "hybrid_debug_placement": spec.get("placement", "camera_locked"),
        "hybrid_debug_scene_margin": spec.get("scene_margin", 0.05),
        "hybrid_debug_min_distance": spec.get("min_distance", 2.0),
        "hybrid_mesh_color": spec.get("color", base.get("hybrid_mesh_color", [1.0, 1.0, 1.0])),
        "hybrid_mesh_lighting": spec.get("lighting", base.get("hybrid_mesh_lighting", "unlit")),
        "hybrid_use_gs_env_light": bool(spec.get("use_gs_env_light", base.get("hybrid_use_gs_env_light", False))),
        "hybrid_verbose": bool(base.get("hybrid_verbose", False)),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _resolve_mesh_data(viewpoint_camera, pipe, pc=None):
    mesh_path = getattr(pipe, "hybrid_mesh_path", "")
    use_debug_cube = bool(getattr(pipe, "hybrid_debug_cube", False))
    if not mesh_path and not use_debug_cube:
        return None

    device = viewpoint_camera.world_view_transform.device
    if use_debug_cube:
        cube_local = _build_face_colored_cube(device, getattr(pipe, "hybrid_debug_cube_size", 1.0))
        mesh = dict(cube_local)
        mesh["vertex_uv"] = None
        mesh["texture_image"] = None
        mesh["material_color"] = None
    else:
        mesh = dict(load_mesh_buffers(mesh_path, device))

    world_transform = _compose_instance_transform(mesh["vertices"], pipe, viewpoint_camera=viewpoint_camera, pc=pc)
    mesh["vertices"] = _apply_transform_matrix(mesh["vertices"], world_transform)
    mesh["vertex_normals"] = _apply_normal_matrix(mesh["vertex_normals"], world_transform)
    mesh["world_transform"] = world_transform
    mesh["instance_name"] = getattr(pipe, "hybrid_mesh_name", "default")
    return mesh


def _shade_mesh(pipe, colors, normals):
    lighting_mode = getattr(pipe, "hybrid_mesh_lighting", "unlit")
    if lighting_mode != "lambert":
        return colors
    light_dir = torch.tensor(
        getattr(pipe, "hybrid_light_direction", [0.3, 0.5, 1.0]),
        dtype=colors.dtype,
        device=colors.device,
    )
    light_dir = _normalize(light_dir.unsqueeze(0)).squeeze(0)
    lambert = (normals * light_dir.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    return colors * (0.2 + 0.8 * lambert)


def _screen_bbox_from_alpha(alpha):
    valid = alpha[0] > 1e-4
    if not torch.any(valid):
        return None
    ys, xs = torch.where(valid)
    return (
        int(xs.min().item()),
        int(ys.min().item()),
        int(xs.max().item()),
        int(ys.max().item()),
    )


def _maybe_log_mesh_debug(viewpoint_camera, mesh, alpha, backend, pipe, pc=None):
    if not bool(getattr(pipe, "hybrid_verbose", False)):
        return

    vertices = mesh["vertices"]
    projected, _ = _project_vertices(viewpoint_camera, vertices)
    vertex_depth = projected[:, 2]
    bbox_min = vertices.min(dim=0).values.detach().cpu().tolist()
    bbox_max = vertices.max(dim=0).values.detach().cpu().tolist()
    bbox_center = (0.5 * (vertices.min(dim=0).values + vertices.max(dim=0).values)).detach().cpu().tolist()
    proj_min = projected[:, :2].min(dim=0).values.detach().cpu().tolist()
    proj_max = projected[:, :2].max(dim=0).values.detach().cpu().tolist()
    depth_min = float(vertex_depth.min().item())
    depth_max = float(vertex_depth.max().item())
    alpha_ratio = float((alpha > 1e-4).float().mean().item())
    screen_bbox = _screen_bbox_from_alpha(alpha)
    scene_bounds = _compute_scene_bounds(pc, vertices.device)
    scene_msg = ""
    if scene_bounds is not None:
        scene_center = scene_bounds["center"].detach().cpu().tolist()
        scene_bbox_min = scene_bounds["bbox_min"].detach().cpu().tolist()
        scene_bbox_max = scene_bounds["bbox_max"].detach().cpu().tolist()
        center_delta = (0.5 * (vertices.min(dim=0).values + vertices.max(dim=0).values) - scene_bounds["center"]).detach().cpu().tolist()
        scene_msg = (
            f" scene_bbox_min={np.round(np.asarray(scene_bbox_min), 3).tolist()}"
            f" scene_bbox_max={np.round(np.asarray(scene_bbox_max), 3).tolist()}"
            f" scene_center={np.round(np.asarray(scene_center), 3).tolist()}"
            f" mesh_center_delta={np.round(np.asarray(center_delta), 3).tolist()}"
        )

    print(
        "[hybrid-mesh] "
        f"name={mesh.get('instance_name', 'default')} "
        f"backend={backend} "
        f"depth=[{depth_min:.3f}, {depth_max:.3f}] "
        f"world_bbox_min={np.round(np.asarray(bbox_min), 3).tolist()} "
        f"world_bbox_max={np.round(np.asarray(bbox_max), 3).tolist()} "
        f"world_center={np.round(np.asarray(bbox_center), 3).tolist()} "
        f"proj_min={np.round(np.asarray(proj_min), 1).tolist()} "
        f"proj_max={np.round(np.asarray(proj_max), 1).tolist()} "
        f"alpha_ratio={alpha_ratio:.6f} "
        f"screen_bbox={screen_bbox}"
        f"{scene_msg}"
    )


def _sample_texture_numpy(texture_image, uv):
    if texture_image is None or uv is None:
        return None
    texture = np.transpose(texture_image.detach().cpu().numpy(), (1, 2, 0))
    tex_h, tex_w = texture.shape[:2]
    u = np.mod(uv[..., 0], 1.0)
    v = 1.0 - np.clip(uv[..., 1], 0.0, 1.0)
    x = np.clip(u * (tex_w - 1), 0.0, tex_w - 1)
    y = np.clip(v * (tex_h - 1), 0.0, tex_h - 1)

    x0 = np.floor(x).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tex_w - 1)
    y0 = np.floor(y).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, tex_h - 1)

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    c00 = texture[y0, x0]
    c01 = texture[y0, x1]
    c10 = texture[y1, x0]
    c11 = texture[y1, x1]
    c0 = c00 * (1.0 - wx) + c01 * wx
    c1 = c10 * (1.0 - wx) + c11 * wx
    return c0 * (1.0 - wy) + c1 * wy


def _sample_texture_torch(texture_image, uv):
    if texture_image is None or uv is None:
        return None
    grid_x = torch.remainder(uv[..., 0], 1.0) * 2.0 - 1.0
    grid_y = (1.0 - uv[..., 1].clamp(0.0, 1.0)) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    sampled = torch.nn.functional.grid_sample(
        texture_image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0].permute(1, 2, 0)


def _maybe_build_env_context(mesh, pc, pipe, bg_color):
    if pc is None or bg_color is None:
        return None
    if not bool(getattr(pipe, "hybrid_use_gs_env_light", False)):
        return None
    center_world = 0.5 * (mesh["vertices"].min(dim=0).values + mesh["vertices"].max(dim=0).values)
    return build_gs_env_cubemap(center_world=center_world, pc=pc, pipe=pipe, bg_color=bg_color)


def _apply_optional_env_lighting(result, env_ctx, pipe):
    if env_ctx is None:
        return result
    result["mesh_rgb"] = apply_env_lighting(
        albedo_rgb=result["mesh_rgb"],
        normal_world=result["mesh_normal"],
        alpha=result["mesh_alpha"],
        env_ctx=env_ctx,
        pipe=pipe,
    )
    return result


def _render_mesh_cpu(viewpoint_camera, pipe, pc=None, bg_color=None):
    mesh = _resolve_mesh_data(viewpoint_camera, pipe, pc=pc)
    if mesh is None:
        return empty_mesh_buffers(viewpoint_camera)
    env_ctx = _maybe_build_env_context(mesh, pc, pipe, bg_color)

    device = viewpoint_camera.world_view_transform.device
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    vertex_colors = mesh["vertex_colors"]
    vertex_uv = mesh.get("vertex_uv")
    texture_image = mesh.get("texture_image")
    material_color = mesh.get("material_color")
    vertex_normals = mesh["vertex_normals"]

    image_h = int(viewpoint_camera.image_height)
    image_w = int(viewpoint_camera.image_width)
    rgb = torch.zeros((3, image_h, image_w), dtype=torch.float32, device=device)
    depth = torch.full((1, image_h, image_w), float("inf"), dtype=torch.float32, device=device)
    alpha = torch.zeros((1, image_h, image_w), dtype=torch.float32, device=device)
    normal = torch.zeros((3, image_h, image_w), dtype=torch.float32, device=device)

    screen_vertices, _ = _project_vertices(viewpoint_camera, vertices)
    face_normals = _face_normals(vertices, faces)
    base_color = material_color if material_color is not None else _resolve_base_color(pipe, device)
    screen_vertices_np = screen_vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    face_normals_np = face_normals.detach().cpu().numpy()
    vertex_colors_np = vertex_colors.detach().cpu().numpy() if vertex_colors is not None else None
    vertex_uv_np = vertex_uv.detach().cpu().numpy() if vertex_uv is not None else None
    vertex_normals_np = vertex_normals.detach().cpu().numpy() if vertex_normals is not None else None
    base_color_np = base_color.detach().cpu().numpy()
    lighting_mode = getattr(pipe, "hybrid_mesh_lighting", "unlit")
    light_dir = np.asarray(getattr(pipe, "hybrid_light_direction", [0.3, 0.5, 1.0]), dtype=np.float32)
    light_dir = light_dir / max(np.linalg.norm(light_dir), 1e-8)

    rgb_np = rgb.detach().cpu().numpy()
    depth_np = depth.detach().cpu().numpy()
    alpha_np = alpha.detach().cpu().numpy()
    normal_np = normal.detach().cpu().numpy()

    for face_idx, face in enumerate(faces_np):
        tri_screen = screen_vertices_np[face]
        tri_depth = tri_screen[:, 2]
        if np.any(tri_depth <= viewpoint_camera.znear):
            continue

        min_x = max(int(np.floor(tri_screen[:, 0].min())), 0)
        max_x = min(int(np.ceil(tri_screen[:, 0].max())), image_w - 1)
        min_y = max(int(np.floor(tri_screen[:, 1].min())), 0)
        max_y = min(int(np.ceil(tri_screen[:, 1].max())), image_h - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area = (
            (tri_screen[1, 0] - tri_screen[0, 0]) * (tri_screen[2, 1] - tri_screen[0, 1])
            - (tri_screen[2, 0] - tri_screen[0, 0]) * (tri_screen[1, 1] - tri_screen[0, 1])
        )
        if abs(area) < 1e-8:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
        ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
        sample_x, sample_y = np.meshgrid(xs, ys)

        w0 = (
            (tri_screen[1, 0] - sample_x) * (tri_screen[2, 1] - sample_y)
            - (tri_screen[2, 0] - sample_x) * (tri_screen[1, 1] - sample_y)
        ) / area
        w1 = (
            (tri_screen[2, 0] - sample_x) * (tri_screen[0, 1] - sample_y)
            - (tri_screen[0, 0] - sample_x) * (tri_screen[2, 1] - sample_y)
        ) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(inside):
            continue

        pixel_depth = w0 * tri_depth[0] + w1 * tri_depth[1] + w2 * tri_depth[2]
        depth_patch = depth_np[0, min_y:max_y + 1, min_x:max_x + 1]
        update_mask = inside & (pixel_depth < depth_patch)
        if not np.any(update_mask):
            continue

        texture_rgba = None
        if vertex_uv_np is not None and texture_image is not None:
            tri_uv = vertex_uv_np[face]
            uv = (
                w0[..., None] * tri_uv[0][None, None, :]
                + w1[..., None] * tri_uv[1][None, None, :]
                + w2[..., None] * tri_uv[2][None, None, :]
            )
            texture_rgba = _sample_texture_numpy(texture_image, uv)
            color = texture_rgba[..., :3]
        elif vertex_colors_np is not None:
            tri_color = vertex_colors_np[face]
            color = (
                w0[..., None] * tri_color[0][None, None, :]
                + w1[..., None] * tri_color[1][None, None, :]
                + w2[..., None] * tri_color[2][None, None, :]
            )
        else:
            color = np.broadcast_to(base_color_np[None, None, :], update_mask.shape + (3,))

        if vertex_normals_np is not None:
            tri_normal = vertex_normals_np[face]
            shading_normal = (
                w0[..., None] * tri_normal[0][None, None, :]
                + w1[..., None] * tri_normal[1][None, None, :]
                + w2[..., None] * tri_normal[2][None, None, :]
            )
            shading_normal = shading_normal / np.clip(np.linalg.norm(shading_normal, axis=-1, keepdims=True), 1e-8, None)
        else:
            shading_normal = np.broadcast_to(face_normals_np[face_idx][None, None, :], update_mask.shape + (3,))

        if lighting_mode == "lambert":
            lambert = np.clip((shading_normal * light_dir[None, None, :]).sum(axis=-1, keepdims=True), 0.0, 1.0)
            color = color * (0.2 + 0.8 * lambert)

        if texture_rgba is not None and texture_rgba.shape[-1] > 3:
            update_mask = update_mask & (texture_rgba[..., 3] > 1e-4)
            if not np.any(update_mask):
                continue

        depth_patch[update_mask] = pixel_depth[update_mask]
        alpha_patch = alpha_np[0, min_y:max_y + 1, min_x:max_x + 1]
        if texture_rgba is not None and texture_rgba.shape[-1] > 3:
            alpha_patch[update_mask] = texture_rgba[..., 3][update_mask]
        else:
            alpha_patch[update_mask] = 1.0

        rgb_patch = rgb_np[:, min_y:max_y + 1, min_x:max_x + 1]
        normal_patch = normal_np[:, min_y:max_y + 1, min_x:max_x + 1]
        for ch in range(3):
            rgb_channel = rgb_patch[ch]
            rgb_channel[update_mask] = color[..., ch][update_mask]
            rgb_patch[ch] = rgb_channel

            normal_channel = normal_patch[ch]
            normal_channel[update_mask] = shading_normal[..., ch][update_mask]
            normal_patch[ch] = normal_channel

    rgb = torch.from_numpy(rgb_np).to(device=device, dtype=torch.float32)
    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    alpha = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32)
    normal = torch.from_numpy(normal_np).to(device=device, dtype=torch.float32)

    depth = torch.where(alpha > 0.0, depth, torch.zeros_like(depth))

    result = {
        "mesh_rgb": rgb,
        "mesh_depth": depth,
        "mesh_alpha": alpha,
        "mesh_normal": normal,
        "mesh_instance": {
            "name": mesh.get("instance_name", "default"),
            "path": getattr(pipe, "hybrid_mesh_path", ""),
            "world_transform": mesh["world_transform"].detach().cpu().tolist() if mesh.get("world_transform") is not None else None,
            "world_center": (0.5 * (mesh["vertices"].min(dim=0).values + mesh["vertices"].max(dim=0).values)).detach().cpu().tolist(),
        },
    }
    result = _apply_optional_env_lighting(result, env_ctx, pipe)
    _maybe_log_mesh_debug(viewpoint_camera, mesh, alpha, "cpu", pipe, pc=pc)
    return result


def _render_mesh_nvdiffrast(viewpoint_camera, pipe, pc=None, bg_color=None):
    try:
        import nvdiffrast.torch as dr
    except ImportError as exc:
        raise ImportError(
            "hybrid_mesh_backend='nvdiffrast' was requested, but nvdiffrast is not installed in the current environment."
        ) from exc

    mesh = _resolve_mesh_data(viewpoint_camera, pipe, pc=pc)
    if mesh is None:
        return empty_mesh_buffers(viewpoint_camera)
    env_ctx = _maybe_build_env_context(mesh, pc, pipe, bg_color)

    device = viewpoint_camera.world_view_transform.device
    ctx = _NVDIFFRAST_CONTEXTS.get(str(device))
    if ctx is None:
        ctx = dr.RasterizeCudaContext(device=device)
        _NVDIFFRAST_CONTEXTS[str(device)] = ctx

    vertices = mesh["vertices"]
    faces = mesh["faces"].to(dtype=torch.int32).contiguous()
    vertex_colors = mesh["vertex_colors"]
    vertex_uv = mesh.get("vertex_uv")
    texture_image = mesh.get("texture_image")
    material_color = mesh.get("material_color")
    vertex_normals = mesh["vertex_normals"]

    if vertex_colors is None:
        base_color = material_color if material_color is not None else _resolve_base_color(pipe, device)
        vertex_colors = base_color.expand(vertices.shape[0], 3).contiguous()
    if vertex_normals is None:
        face_normals = _face_normals(vertices, faces.long())
        accum_normals = torch.zeros_like(vertices)
        for corner in range(3):
            accum_normals.index_add_(0, faces[:, corner].long(), face_normals)
        vertex_normals = _normalize(accum_normals)

    ones = torch.ones((vertices.shape[0], 1), device=device, dtype=vertices.dtype)
    vertices_h = torch.cat([vertices, ones], dim=1)
    cam_vertices = vertices_h @ viewpoint_camera.world_view_transform
    cam_depth = cam_vertices[:, 2:3].contiguous()
    z = cam_depth[:, 0].clamp_min(1e-6)
    x_pix = viewpoint_camera.fx * (cam_vertices[:, 0] / z) + viewpoint_camera.cx
    y_pix = viewpoint_camera.fy * (cam_vertices[:, 1] / z) + viewpoint_camera.cy
    x_ndc = 2.0 * (x_pix / max(float(viewpoint_camera.image_width - 1), 1.0)) - 1.0
    # Match the image-space convention used by the current camera path/render
    # stack so mesh motion aligns with the GS render in vertical direction.
    y_ndc = 2.0 * (y_pix / max(float(viewpoint_camera.image_height - 1), 1.0)) - 1.0
    z_ndc = 2.0 * ((z - viewpoint_camera.znear) / max(viewpoint_camera.zfar - viewpoint_camera.znear, 1e-6)) - 1.0
    clip_vertices = torch.stack((x_ndc * z, y_ndc * z, z_ndc * z, z), dim=1).contiguous()

    resolution = [int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)]
    rast, _ = dr.rasterize(ctx, clip_vertices.unsqueeze(0), faces, resolution=resolution)
    alpha = (rast[..., 3:4] > 0).to(dtype=torch.float32).contiguous()

    interp_color, _ = dr.interpolate(vertex_colors.unsqueeze(0), rast, faces)
    interp_normal, _ = dr.interpolate(vertex_normals.unsqueeze(0), rast, faces)
    interp_normal = torch.nn.functional.normalize(interp_normal, dim=-1)
    interp_depth, _ = dr.interpolate(cam_depth.unsqueeze(0), rast, faces)

    texture_rgba = None
    if vertex_uv is not None and texture_image is not None:
        interp_uv, _ = dr.interpolate(vertex_uv.unsqueeze(0), rast, faces)
        texture_rgba = _sample_texture_torch(texture_image, interp_uv[0].unsqueeze(0))
        interp_color = texture_rgba[..., :3].unsqueeze(0)

    shaded_color = _shade_mesh(pipe, interp_color, interp_normal)
    shaded_color = dr.antialias(shaded_color.contiguous(), rast.contiguous(), clip_vertices.unsqueeze(0).contiguous(), faces)
    alpha = dr.antialias(alpha, rast, clip_vertices.unsqueeze(0), faces).clamp(0.0, 1.0)
    if texture_rgba is not None and texture_rgba.shape[-1] > 3:
        tex_alpha = texture_rgba[..., 3:4].unsqueeze(0)
        alpha = alpha * tex_alpha

    rgb = shaded_color[0].permute(2, 0, 1)
    normal = interp_normal[0].permute(2, 0, 1)
    depth = interp_depth[0].permute(2, 0, 1)
    depth = torch.where(alpha[0].permute(2, 0, 1) > 1e-4, depth, torch.zeros_like(depth))
    alpha = alpha[0].permute(2, 0, 1)

    result = {
        "mesh_rgb": rgb.clamp(0.0, 1.0),
        "mesh_depth": depth,
        "mesh_alpha": alpha,
        "mesh_normal": normal,
        "mesh_instance": {
            "name": mesh.get("instance_name", "default"),
            "path": getattr(pipe, "hybrid_mesh_path", ""),
            "world_transform": mesh["world_transform"].detach().cpu().tolist() if mesh.get("world_transform") is not None else None,
            "world_center": (0.5 * (mesh["vertices"].min(dim=0).values + mesh["vertices"].max(dim=0).values)).detach().cpu().tolist(),
        },
    }
    result = _apply_optional_env_lighting(result, env_ctx, pipe)
    _maybe_log_mesh_debug(viewpoint_camera, mesh, alpha, "nvdiffrast", pipe, pc=pc)
    return result


def _composite_mesh_layers(base, layer):
    layer_valid = (layer["mesh_alpha"] > 1e-4) & (layer["mesh_depth"] > 0.0)
    base_valid = (base["mesh_alpha"] > 1e-4) & (base["mesh_depth"] > 0.0)
    take_layer = layer_valid & ((~base_valid) | (layer["mesh_depth"] < base["mesh_depth"]))

    for key in ["mesh_rgb", "mesh_depth", "mesh_alpha", "mesh_normal"]:
        base[key] = torch.where(take_layer.expand_as(base[key]), layer[key], base[key])
    return base


def render_mesh(viewpoint_camera, pipe, pc=None, bg_color=None):
    backend = getattr(pipe, "hybrid_mesh_backend", "cpu").lower()
    mesh_specs = _get_scene_mesh_specs(pipe)
    if not mesh_specs:
        result = empty_mesh_buffers(viewpoint_camera)
        result["mesh_instances"] = []
        return result

    combined = empty_mesh_buffers(viewpoint_camera)
    instance_infos = []
    for spec in mesh_specs:
        instance_pipe = _make_pipe_with_mesh_spec(pipe, spec)
        setattr(instance_pipe, "hybrid_mesh_name", spec.get("name", "mesh"))
        if backend == "cpu":
            layer = _render_mesh_cpu(viewpoint_camera, instance_pipe, pc=pc, bg_color=bg_color)
        elif backend == "nvdiffrast":
            layer = _render_mesh_nvdiffrast(viewpoint_camera, instance_pipe, pc=pc, bg_color=bg_color)
        else:
            raise ValueError(f"Unknown hybrid mesh backend: {backend}")

        _composite_mesh_layers(combined, layer)
        mesh_meta = layer.get("mesh_instance")
        if mesh_meta is not None:
            instance_infos.append(mesh_meta)

    combined["mesh_instances"] = instance_infos
    return combined


def empty_mesh_buffers(viewpoint_camera, device: Optional[torch.device] = None):
    device = device or viewpoint_camera.world_view_transform.device
    image_h = int(viewpoint_camera.image_height)
    image_w = int(viewpoint_camera.image_width)
    return {
        "mesh_rgb": torch.zeros((3, image_h, image_w), dtype=torch.float32, device=device),
        "mesh_depth": torch.zeros((1, image_h, image_w), dtype=torch.float32, device=device),
        "mesh_alpha": torch.zeros((1, image_h, image_w), dtype=torch.float32, device=device),
        "mesh_normal": torch.zeros((3, image_h, image_w), dtype=torch.float32, device=device),
    }
