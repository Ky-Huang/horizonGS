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
import os
import sys
import imageio
import yaml
from os import makedirs
import torch
import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state, parse_cfg, visualize_depth, visualize_normal
from utils.image_utils import save_rgba
from argparse import ArgumentParser

def render_set(model_path, name, iteration, views, gaussians, pipe, background, add_aerial, add_street, render_only=False):
    vis_normal = False
    vis_depth = False
    if gaussians.gs_attr == "2D" and not render_only:
        vis_normal = True
        vis_depth = True
        
    if add_aerial:
        aerial_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "renders")
        makedirs(aerial_render_path, exist_ok=True)
        if not render_only:
            aerial_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "errors")
            aerial_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "gt")
            makedirs(aerial_error_path, exist_ok=True)
            makedirs(aerial_gts_path, exist_ok=True)
        
        if vis_normal:
            aerial_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "normal")
            makedirs(aerial_normal_path, exist_ok=True)
        if vis_depth:
            aerial_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "depth")
            makedirs(aerial_depth_path, exist_ok=True)
    if add_street:
        street_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "renders")
        makedirs(street_render_path, exist_ok=True)
        if not render_only:
            street_error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "errors")
            street_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "gt")
            makedirs(street_error_path, exist_ok=True)
            makedirs(street_gts_path, exist_ok=True)
        
        if vis_normal:
            street_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "normal")
            makedirs(street_normal_path, exist_ok=True)
        if vis_depth:
            street_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "street", "depth")
            makedirs(street_depth_path, exist_ok=True)

    modules = __import__('gaussian_renderer')

    street_t_list = []
    street_visible_count_list = []
    street_per_view_dict = {}
    street_views = [view for view in views if view.image_type=="street"]
    for idx, view in enumerate(tqdm(street_views, desc="Street rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()
        
        street_t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        if not render_only:
            view.ensure_image_tensors()
            gt = view.original_image.cuda()
            alpha_mask = view.alpha_mask.cuda()
            rendering = torch.cat([rendering, alpha_mask], dim=0)
            gt = torch.cat([gt, alpha_mask], dim=0)
            if gt.device != rendering.device:
                rendering = rendering.to(gt.device)
            errormap = (rendering - gt).abs()
        
        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0].detach()
            vis_normal_map = visualize_normal(normal_map, view)
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            imageio.imwrite(os.path.join(street_normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
        
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            vis_depth_map = visualize_depth(depth_map) 
            vis_depth_map = torch.concat([vis_depth_map,alpha_mask],dim=0)
            torchvision.utils.save_image(vis_depth_map, os.path.join(street_depth_path, '{0:05d}'.format(idx) + ".png"))

        if render_only:
            torchvision.utils.save_image(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
        else:
            save_rgba(rendering, os.path.join(street_render_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(errormap, os.path.join(street_error_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(gt, os.path.join(street_gts_path, '{0:05d}'.format(idx) + ".png"))
            view.release_image_tensors()
        street_visible_count_list.append(int(visible_count.item()))
        street_per_view_dict['{0:05d}'.format(idx) + ".png"] = int(visible_count.item())
    
    if len(street_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "street", "per_view_count.json"), 'w') as fp:
            json.dump(street_per_view_dict, fp, indent=True)
    
    aerial_t_list = []
    aerial_visible_count_list = []
    aerial_per_view_dict = {}
    aerial_views = [view for view in views if view.image_type=="aerial"]
    for idx, view in enumerate(tqdm(aerial_views, desc="Aerial rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        render_pkg = getattr(modules, 'render')(view, gaussians, pipe, background)
        torch.cuda.synchronize();t_end = time.time()

        aerial_t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        if not render_only:
            view.ensure_image_tensors()
            gt = view.original_image.cuda()
            alpha_mask = view.alpha_mask.cuda()
            rendering = torch.cat([rendering, alpha_mask], dim=0)
            gt = torch.cat([gt, alpha_mask], dim=0)
            if gt.device != rendering.device:
                rendering = rendering.to(gt.device)
            errormap = (rendering - gt).abs()
        
        if vis_normal == True:
            normal_map = render_pkg['render_normals'][0] 
            vis_normal_map = visualize_normal(normal_map, view)
            vis_alpha_mask = ((alpha_mask * 255).byte()).permute(1, 2, 0).cpu().numpy()
            vis_normal_map = np.concatenate((vis_normal_map,vis_alpha_mask),axis=2)
            imageio.imwrite(os.path.join(aerial_normal_path, '{0:05d}'.format(idx) + ".png"), vis_normal_map)
        
        if vis_depth == True:
            depth_map = render_pkg["render_depth"]
            vis_depth_map = visualize_depth(depth_map) 
            vis_depth_map = torch.concat([vis_depth_map,alpha_mask],dim=0)
            torchvision.utils.save_image(vis_depth_map, os.path.join(aerial_depth_path, '{0:05d}'.format(idx) + ".png"))

        if render_only:
            torchvision.utils.save_image(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
        else:
            save_rgba(rendering, os.path.join(aerial_render_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(errormap, os.path.join(aerial_error_path, '{0:05d}'.format(idx) + ".png"))
            save_rgba(gt, os.path.join(aerial_gts_path, '{0:05d}'.format(idx) + ".png"))
            view.release_image_tensors()
        aerial_visible_count_list.append(int(visible_count.item()))
        aerial_per_view_dict['{0:05d}'.format(idx) + ".png"] = int(visible_count.item())

    if len(aerial_views) > 0:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "aerial", "per_view_count.json"), 'w') as fp:
            json.dump(aerial_per_view_dict, fp, indent=True)

    aerial_valid_count = max(len(aerial_t_list) - 5, 0)
    street_valid_count = max(len(street_t_list) - 5, 0)
    aerial_time = sum(aerial_t_list[5:])
    street_time = sum(street_t_list[5:])

    aerial_fps = aerial_valid_count / aerial_time if aerial_time > 0 else 0.0
    street_fps = street_valid_count / street_time if street_time > 0 else 0.0

    total_valid_count = aerial_valid_count + street_valid_count
    total_time = aerial_time + street_time
    total_fps = total_valid_count / total_time if total_time > 0 else 0.0

    print("Aerial帧率: {}".format(aerial_fps))
    print("Street帧率: {}".format(street_fps))
    print("总帧率: {}".format(total_fps))

    
def render_sets(dataset, opt, pipe, iteration, skip_train, skip_test, ape_code, explicit):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
        modules = __import__('scene')
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(modules, model_config['name'])(**model_config['kwargs'])
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit, skip_train=skip_train, skip_test=skip_test)
        gaussians.eval()

        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipe, scene.background, dataset.add_aerial, dataset.add_street, render_only=getattr(dataset, "render_only", False))

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipe, scene.background, dataset.add_aerial, dataset.add_street, render_only=getattr(dataset, "render_only", False))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lazy_load_images", action="store_true")
    parser.add_argument("--render_only", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.num_workers > 0:
        os.environ["HGS_NUM_WORKERS"] = str(args.num_workers)

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path
        lp.lazy_load_images = args.lazy_load_images
        lp.render_only = args.render_only
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(lp, op, pp, args.iteration, args.skip_train, args.skip_test, args.ape, args.explicit)
