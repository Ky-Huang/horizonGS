import numpy as np
import torch
import cv2
import os
import math
import open3d as o3d
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def depth2pc(frames, ds=5, depth_cut=5, ratio=10):    
    c2ws = []
    for frame in frames:
        cx = frames[0]["cx"]/ds
        cy = frames[0]["cy"]/ds
        w = int(frames[0]["w"]/ds)
        h = int(frames[0]["h"]/ds)
        fx = frames[0]["fl_x"]/ds
        fy = frames[0]["fl_y"]/ds
        
        c2w=np.array(frame["transform_matrix"])
        c2w[3,3]=1
        # c2w[:3,3]*=10
        c2ws.append(c2w.tolist()) 
    c2ws=np.stack(c2ws) #[B,4,4]
    print(f"xmin:{c2ws[:,0,3].min()}, xmax:{c2ws[:,0,3].max()}, ymin:{c2ws[:,1,3].min()}, ymax:{c2ws[:,1,3].max()}")
    # assume all images share the same intrinsic
    intrinsic = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    
    depths=[]
    rgbs=[]
    for i, frame in enumerate(frames):
        file_path = frame['file_path']
        rgb = cv2.imread(os.path.join(file_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w,h))
        # import pdb;pdb.set_trace()
        depth = cv2.imread(frame['depth_path'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] / 10000. # cm -> 100m
        # depth_path = file_path.split('/')[-2]+"_depth"
        # depth = cv2.imread(os.path.join(args.depth_path,depth_path, file_path.split('/')[-1].replace("png","exr")), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] / 10000. # cm -> 100m
        # depth = cv2.imread(frame['depth_path'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] / 10000. # cm -> 100m
        # depth = cv2.imread(frame['depth_path'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] / 1000. # cm -> 10m
        depth = cv2.resize(depth, (w,h))
        rgbs.append(rgb)
        depths.append(depth)
        
    rgbs = np.stack(rgbs) # [B,H,W,3]
    depths = np.stack(depths) # [B,H,W,1]
    
    # convert to torch
    rgbs = torch.from_numpy(rgbs).float()
    depths = torch.from_numpy(depths).float()
    intrinsic = torch.from_numpy(intrinsic).float()
    c2ws = torch.from_numpy(c2ws).float()
    
    # project to world
    all_points = []
    all_colors = []
    # Compute the pixel coordinates of each point in the depth image
    for i in range(depths.shape[0]):
        y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=depths.device),
                            torch.arange(0, w, dtype=torch.float32, device=depths.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h * w), x.view(h * w)
        xyz = torch.stack((x, y, torch.ones_like(x)))
        
        # if depth > thre, mask
        if depth_cut != -1:
            depth_mask = depths[i] < depth_cut
        else:
            depth_mask = torch.ones(depths[i].shape,dtype=torch.bool)
        
        # Convert pixel coordinates to camera coordinates
        inv_K = torch.inverse(intrinsic)
        cam_coords1 = inv_K.clone() @ (xyz.clone() * depths[i].reshape(-1))
        cam_coords1[1,:] = -cam_coords1[1,:]
        cam_coords1[2,:] = -cam_coords1[2,:]
        world_coords = (c2ws[i] @ torch.cat([cam_coords1, torch.ones((1, cam_coords1.shape[1]))], dim=0)).T
        world_coords = world_coords[:,:3]
        
        world_coords = world_coords[depth_mask.reshape(-1)]
        color = rgbs[i].reshape(-1,3)/255.
        color = color[depth_mask.reshape(-1)]
        
        all_points.append(world_coords)
        all_colors.append(color)
        
    merged_points = np.vstack(all_points)[::ratio,:]
    merged_colors = np.vstack(all_colors)[::ratio,:]

    # save the final point cloud
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(merged_points)
    final_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    return final_pcd

def depth2pc_partition(cameras, ds=10, depth_cut=5, ratio=100):
    if len(cameras) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    c2ws = []
    for camera in cameras:
        cx = camera.cx / ds
        cy = camera.cy / ds
        fx = camera.fx / ds
        fy = camera.fy / ds
        w = int(camera.image_width // ds)
        h = int(camera.image_height // ds)

        c2w = camera.c2w.cpu()
        c2w[:3, 1:3] *= -1
        c2ws.append(c2w)
    c2ws = torch.stack(c2ws)

    intrinsic = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    inv_K = torch.inverse(intrinsic).float()

    y, x = torch.meshgrid([
        torch.arange(0, h, dtype=torch.float32, device=inv_K.device),
        torch.arange(0, w, dtype=torch.float32, device=inv_K.device),
    ])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(h * w), x.view(h * w)
    xyz = torch.stack((x, y, torch.ones_like(x)))

    all_points = []
    all_colors = []
    for i, camera in enumerate(cameras):
        rgb = camera.original_image
        depth = 1. / camera.invdepthmap

        rgb = F.interpolate(rgb.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
        depth = F.interpolate(depth.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)

        if depth_cut != -1:
            depth_mask = depth < depth_cut
        else:
            depth_mask = torch.ones(depth.shape, dtype=torch.bool, device=depth.device)

        cam_coords1 = inv_K.clone() @ (xyz.clone() * depth.reshape(-1))
        cam_coords1[1, :] = -cam_coords1[1, :]
        cam_coords1[2, :] = -cam_coords1[2, :]
        world_coords = (c2ws[i] @ torch.cat([cam_coords1, torch.ones((1, cam_coords1.shape[1]), dtype=cam_coords1.dtype)], dim=0)).T
        world_coords = world_coords[:, :3]

        world_coords = world_coords[depth_mask.reshape(-1)].numpy()
        color = rgb.reshape(-1, 3)
        color = color[depth_mask.reshape(-1)].numpy()

        all_points.append(world_coords)
        all_colors.append(color)

    merged_points = np.vstack(all_points)[::ratio, :]
    merged_colors = np.vstack(all_colors)[::ratio, :]

    return merged_points, merged_colors