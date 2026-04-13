#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from torch import nn
from utils.general_utils import PILtoTorch
from utils.graphics_utils import (
    getProjectionMatrix,
    getProjectionMatrixFromIntrinsics,
    getWorld2View2,
)
import cv2
# import kornia

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, Cx, Cy, FoVx, FoVy, image, alpha_mask, 
                 image_type, image_name, image_path, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 data_format='matrixcity', gt_depth=None, depth_params=None,
                 orig_w=None, orig_h=None, lazy_load=False, render_only=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.image_type = image_type
        self.resolution_scale = resolution_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.lazy_load = lazy_load
        self.render_only = render_only
        self._image_src = image
        self._alpha_src = alpha_mask
        self._resolution = resolution

        self.invdepthmap = None
        self.original_image = None
        self.alpha_mask = None

        self.image_width = int(resolution[0])
        self.image_height = int(resolution[1])

        if not self.lazy_load and not self.render_only:
            self.ensure_image_tensors()
        
        if gt_depth is not None and not self.render_only:
            if data_format == 'colmap':
                invdepthmapScaled = gt_depth * depth_params["scale"] + depth_params["offset"]
                invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
                invdepthmapScaled[invdepthmapScaled < 0] = 0
                if invdepthmapScaled.ndim != 2:
                    invdepthmapScaled = invdepthmapScaled[..., 0]
                self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)
            elif data_format == 'blender' or data_format == 'city':
                gt_depth = torch.from_numpy(cv2.resize(gt_depth, resolution)[None])
                invdepthmap = 1. / gt_depth
                self.invdepthmap = invdepthmap.to(self.data_device)

        if self.alpha_mask is not None:
            self.depth_mask = self.alpha_mask.clone()
        elif self.invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.invdepthmap > 0)
        else:
            self.depth_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        base_w = orig_w if orig_w is not None else image.size[0]
        base_h = orig_h if orig_h is not None else image.size[1]
        self.cx = Cx * resolution[0] / base_w
        self.cy = Cy * resolution[1] / base_h
        self.fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        self.fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()

    def get_intrinsics(self, device=None, dtype=torch.float32):
        device = device or self.world_view_transform.device
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        )

    def get_world_to_camera(self):
        return self.world_view_transform.transpose(0, 1)

    def get_camera_to_world(self):
        return self.c2w

    def generate_camera_rays(self, device=None, dtype=torch.float32):
        device = device or self.world_view_transform.device
        ys, xs = torch.meshgrid(
            torch.arange(self.image_height, device=device, dtype=dtype),
            torch.arange(self.image_width, device=device, dtype=dtype),
            indexing="ij",
        )
        x = (xs + 0.5 - self.cx) / self.fx
        y = (ys + 0.5 - self.cy) / self.fy
        dirs_cam = torch.stack((x, y, torch.ones_like(x)), dim=-1)
        dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)
        dirs_world = dirs_cam @ self.get_camera_to_world()[:3, :3].transpose(0, 1)
        return torch.nn.functional.normalize(dirs_world, dim=-1)

    def ensure_image_tensors(self):
        if self.render_only:
            return
        if self.original_image is not None and self.alpha_mask is not None:
            return
        resized_image_rgb = PILtoTorch(self._image_src, self._resolution).to(self.data_device)
        gt_image = resized_image_rgb[:3, ...]
        if self._alpha_src is not None:
            self.alpha_mask = PILtoTorch(self._alpha_src, self._resolution).to(self.data_device)
        elif resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else:
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...]).to(self.data_device)
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)

    def release_image_tensors(self):
        if self.lazy_load:
            self.original_image = None
            self.alpha_mask = None

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform=None,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        projection_matrix=None,
        resolution_scale=1.0,
        image_name="",
        image_path="",
        image_type="street",
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.resolution_scale = resolution_scale
        self.image_name = image_name
        self.image_path = image_path or image_name
        self.image_type = image_type
        self.render_only = True
        self.original_image = None
        self.alpha_mask = None
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

        default_fx = self.image_width / (2 * np.tan(self.FoVx * 0.5))
        default_fy = self.image_height / (2 * np.tan(self.FoVy * 0.5))
        self.fx = float(default_fx if fx is None else fx)
        self.fy = float(default_fy if fy is None else fy)
        self.cx = float(self.image_width * 0.5 if cx is None else cx)
        self.cy = float(self.image_height * 0.5 if cy is None else cy)

        if projection_matrix is None:
            if fx is None and fy is None and cx is None and cy is None:
                projection_matrix = getProjectionMatrix(
                    znear=self.znear,
                    zfar=self.zfar,
                    fovX=self.FoVx,
                    fovY=self.FoVy,
                ).transpose(0, 1)
            else:
                projection_matrix = getProjectionMatrixFromIntrinsics(
                    width=self.image_width,
                    height=self.image_height,
                    fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy,
                    znear=self.znear,
                    zfar=self.zfar,
                )
        self.projection_matrix = projection_matrix.to(
            device=self.world_view_transform.device,
            dtype=self.world_view_transform.dtype,
        )
        if full_proj_transform is None:
            full_proj_transform = (
                self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
            ).squeeze(0)
        self.full_proj_transform = full_proj_transform
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()

    def get_intrinsics(self, device=None, dtype=torch.float32):
        device = device or self.world_view_transform.device
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        )

    def get_world_to_camera(self):
        return self.world_view_transform.transpose(0, 1)

    def get_camera_to_world(self):
        return self.c2w

    def generate_camera_rays(self, device=None, dtype=torch.float32):
        device = device or self.world_view_transform.device
        ys, xs = torch.meshgrid(
            torch.arange(self.image_height, device=device, dtype=dtype),
            torch.arange(self.image_width, device=device, dtype=dtype),
            indexing="ij",
        )
        x = (xs + 0.5 - self.cx) / self.fx
        y = (ys + 0.5 - self.cy) / self.fy
        dirs_cam = torch.stack((x, y, torch.ones_like(x)), dim=-1)
        dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)
        dirs_world = dirs_cam @ self.get_camera_to_world()[:3, :3].transpose(0, 1)
        return torch.nn.functional.normalize(dirs_world, dim=-1)
