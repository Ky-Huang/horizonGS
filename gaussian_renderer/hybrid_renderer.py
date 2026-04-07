import torch

from gaussian_renderer.render import render as render_gaussians
from gaussian_renderer.hybrid_mesh import render_mesh, empty_mesh_buffers
from gaussian_renderer.hybrid_skybox import render_skybox


def _resolve_depth_band(mesh_depth, gs_depth, pipe):
    seam_rel = float(getattr(pipe, "hybrid_seam_depth_rel", 0.01))
    seam_width = float(getattr(pipe, "hybrid_seam_width_px", 1.5))
    base = torch.minimum(
        torch.where(mesh_depth > 0.0, mesh_depth, torch.full_like(mesh_depth, float("inf"))),
        torch.where(gs_depth > 0.0, gs_depth, torch.full_like(gs_depth, float("inf"))),
    )
    base = torch.where(torch.isfinite(base), base, torch.ones_like(base))
    return (base * seam_rel * max(seam_width, 1e-4)).clamp_min(1e-4)


def composite_hybrid(gs_rgb, gs_alpha, gs_depth, mesh_rgb, mesh_alpha, mesh_depth, background_rgb, pipe):
    device = gs_rgb.device
    if gs_depth is None:
        gs_depth = torch.zeros_like(gs_alpha, device=device)

    gs_valid = (gs_alpha > 1e-4) & (gs_depth > 0.0)
    mesh_valid = (mesh_alpha > 1e-4) & (mesh_depth > 0.0)

    background = background_rgb
    gs_over_bg = gs_rgb + (1.0 - gs_alpha) * background
    gs_over_mesh = gs_rgb + (1.0 - gs_alpha) * torch.where(mesh_valid.expand_as(mesh_rgb), mesh_rgb, background)

    final_rgb = background.clone()
    final_depth = torch.zeros_like(gs_alpha)

    mesh_bias = float(getattr(pipe, "hybrid_mesh_depth_bias", 0.0))
    adjusted_mesh_depth = torch.where(mesh_valid, mesh_depth + mesh_bias, mesh_depth)
    seam_band = _resolve_depth_band(adjusted_mesh_depth, gs_depth, pipe)
    depth_delta = gs_depth - adjusted_mesh_depth
    mesh_weight = torch.clamp(0.5 + depth_delta / (2.0 * seam_band), 0.0, 1.0)

    both_valid = gs_valid & mesh_valid
    mesh_only = mesh_valid & (~gs_valid)
    gs_only = gs_valid & (~mesh_valid)

    mixed_both = mesh_weight * mesh_rgb + (1.0 - mesh_weight) * gs_over_mesh

    final_rgb = torch.where(mesh_only.expand_as(final_rgb), mesh_rgb, final_rgb)
    final_rgb = torch.where(gs_only.expand_as(final_rgb), gs_over_bg, final_rgb)
    final_rgb = torch.where(both_valid.expand_as(final_rgb), mixed_both, final_rgb)

    mesh_front = mesh_weight >= 0.5
    final_depth = torch.where(mesh_only, mesh_depth, final_depth)
    final_depth = torch.where(gs_only, gs_depth, final_depth)
    final_depth = torch.where(both_valid & mesh_front, mesh_depth, final_depth)
    final_depth = torch.where(both_valid & (~mesh_front), gs_depth, final_depth)

    return {
        "render": final_rgb.clamp(0.0, 1.0),
        "render_depth": final_depth,
        "render_alpha": torch.where(mesh_valid, torch.ones_like(gs_alpha), gs_alpha).clamp(0.0, 1.0),
        "mesh_front_weight": mesh_weight,
    }


def hybrid_render(viewpoint_camera, pc, pipe, bg_color):
    device = bg_color.device
    gs_bg = torch.zeros_like(bg_color, device=device)
    gs_pkg = render_gaussians(viewpoint_camera, pc, pipe, gs_bg)

    scene_data = getattr(pipe, "hybrid_scene_data", None)
    mesh_enabled = (
        bool(getattr(pipe, "hybrid_mesh_path", ""))
        or bool(getattr(pipe, "hybrid_debug_cube", False))
        or bool(isinstance(scene_data, dict) and scene_data.get("meshes"))
    )
    mesh_pkg = render_mesh(viewpoint_camera, pipe, pc=pc, bg_color=bg_color) if mesh_enabled else empty_mesh_buffers(viewpoint_camera)
    skybox_path = getattr(pipe, "hybrid_skybox_path", "")
    if (not skybox_path) and isinstance(scene_data, dict):
        skybox_path = scene_data.get("skybox_path", "")
    sky_pkg = render_skybox(viewpoint_camera, skybox_path, bg_color)

    composite_pkg = composite_hybrid(
        gs_rgb=gs_pkg["render"],
        gs_alpha=gs_pkg["render_alphas"],
        gs_depth=gs_pkg.get("render_depth"),
        mesh_rgb=mesh_pkg["mesh_rgb"],
        mesh_alpha=mesh_pkg["mesh_alpha"],
        mesh_depth=mesh_pkg["mesh_depth"],
        background_rgb=sky_pkg["sky_rgb"],
        pipe=pipe,
    )

    return {
        **gs_pkg,
        **mesh_pkg,
        **sky_pkg,
        **composite_pkg,
        "gs_rgb": gs_pkg["render"],
        "gs_alpha": gs_pkg["render_alphas"],
        "gs_depth": gs_pkg.get("render_depth"),
        "mesh_instances": mesh_pkg.get("mesh_instances", []),
    }
