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
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import torch
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic_mask: np.array = None
    depthmap: np.array = None
    normalmap: np.array = None
    confidence_map: np.array = None
    depthloss: float = 0.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def optimize_depth(source, target, mask, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda().float()
    target = torch.from_numpy(target).cuda().float()
    mask = torch.from_numpy(mask).cuda()

    # Filter out any nan or inf values in source and target
    valid_source_mask = ~(torch.isinf(source) | torch.isnan(source))
    valid_target_mask = ~(torch.isinf(target) | torch.isnan(target))
    mask = torch.logical_and(mask, torch.logical_and(valid_source_mask, valid_target_mask))

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        valid_targets = target[torch.logical_and(mask, target > 1e-7)]
        if valid_targets.numel() > 0:
            target_depth_sorted = valid_targets.sort().values
            min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
            max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

            mask2 = target > min_prune_threshold
            mask3 = target < max_prune_threshold
            mask = torch.logical_and(torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0

    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2))

        # penalize negative depths
        loss_hinge = 0.0
        if (source_hat<=0.0).any():
            loss_hinge = 2.0*((source_hat[source_hat<=0.0])**2).mean()

        loss = loss + loss_hinge

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
        refined_source[torch.isinf(refined_source)] = 0.0
        refined_source[torch.isnan(refined_source)] = 0.0

    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss, scale.item(), shift.item()


def calculate_confidence(depthmap, image, window_size=5):
    """
    Combines edge, texture, gradient features to compute depth confidence.
    """
    
    # Ensure the image is a numpy array (convert if it's a PIL Image)
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convert from PIL to numpy array

    # Clean depthmap of any inf or nan values to prevent Sobel/Laplacian NaN propagation
    depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Convert the image to grayscale (required for edge and texture detection)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Edge confidence using Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)
    edge_confidence = 1 - edges / 255.0  # Invert edges to get confidence (edge regions have low confidence)

    # 2. Texture confidence using Laplacian (sharp texture regions = high confidence)
    laplacian = cv2.Laplacian(depthmap, cv2.CV_32F)
    max_lap = np.max(np.abs(laplacian))
    texture_confidence = 1 - np.abs(laplacian) / (max_lap + 1e-6)

    # 3. Gradient confidence from the depth map
    grad_x = cv2.Sobel(depthmap, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depthmap, cv2.CV_32F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_confidence = 1 / (grad_magnitude + 1e-6)  # Inverse gradient magnitude for confidence
    
    # Normalize each confidence feature to the range [0, 1]
    edge_confidence = (edge_confidence - edge_confidence.min()) / (edge_confidence.max() - edge_confidence.min() + 1e-6)
    texture_confidence = (texture_confidence - texture_confidence.min()) / (texture_confidence.max() - texture_confidence.min() + 1e-6)
    gradient_confidence = (gradient_confidence - gradient_confidence.min()) / (gradient_confidence.max() - gradient_confidence.min() + 1e-6)

    # Combine the four confidence features (equal weights for simplicity)
    final_confidence_map = 0.2 * edge_confidence + 0.5 * texture_confidence + 0.3 * gradient_confidence

    # Normalize the final confidence map to [0, 1]
    final_confidence_map = (final_confidence_map - final_confidence_map.min()) / (final_confidence_map.max() - final_confidence_map.min() + 1e-6)

    return final_confidence_map.astype(np.float16)


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, instance_folder=None, depth_folder=None, normal_folder=None, pcd=None, dataset_path=None, model_path=None, disable_confidence=True):
    cam_infos = []

    scale_shift_cache_path = None
    scale_shift_cache = None
    cache_dirty = False
    # if dataset_path is not None and depth_folder is not None:
    #     scale_shift_cache_path = os.path.join(dataset_path, "depth_scale_shift.json")
    #     if os.path.exists(scale_shift_cache_path):
    #         try:
    #             with open(scale_shift_cache_path, 'r') as f:
    #                 scale_shift_cache = json.load(f)
    #             print(f"\n[INFO] Loaded optimized depth scale/shift parameters from {scale_shift_cache_path}")
    #         except Exception as e:
    #             print(f"\n[WARNING] Failed to load cache file {scale_shift_cache_path}: {e}")
    #             scale_shift_cache = {}
    #     else:
    #         scale_shift_cache = {}

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        semantic_mask = None
        if instance_folder is not None:
            mask_path = os.path.join(instance_folder, image_name + ".npy")
            if os.path.exists(mask_path):
                semantic_mask = np.load(mask_path)
            else:
                # Optional: try .png if .npy doesn't exist?
                png_path = os.path.join(instance_folder, image_name + ".png")
                if os.path.exists(png_path):
                    semantic_mask = np.array(Image.open(png_path))
                else:
                    # Only print once to avoid spamming
                    if idx == 0:
                        print(f"[WARNING] Semantic masks not found in {os.path.abspath(instance_folder)}. Checked for .npy and .png.")

        monodepth = None
        if depth_folder is not None:
            mask_path = os.path.join(depth_folder, image_name + ".npy")
            if os.path.exists(mask_path):
                monodepth = np.load(mask_path)
            else:
                # Optional: try .png if .npy doesn't exist?
                png_path = os.path.join(depth_folder, image_name + ".png")
                if os.path.exists(png_path):
                    monodepth = np.array(Image.open(png_path))
                else:
                    # Only print once to avoid spamming
                    if idx == 0:
                        print(f"[WARNING] Monodepth not found in {os.path.abspath(depth_folder)}. Checked for .npy and .png.")
        
        mononormal = None
        if normal_folder is not None:
            mask_path = os.path.join(normal_folder, image_name + ".png")
            if os.path.exists(mask_path):
                mononormal = np.array(Image.open(mask_path))
            else:
                # Only print once to avoid spamming
                if idx == 0:
                    print(f"[WARNING] Mononormal not found in {os.path.abspath(normal_folder)}. Checked for .png.")

        depthmap, confidence_map = None, None
        depthloss = 1e8
        if monodepth is not None:
            # Set to True to use loaded dense colmap depths directly. Set to False to optimize monodepth using camera PCD.
            if True:
                depthmap = monodepth
                depthmap[np.isinf(depthmap)] = 0.0
                depthmap[np.isnan(depthmap)] = 0.0
            elif pcd is not None:
                depthmap = np.zeros((height, width), dtype=np.float32)
                K = np.array([
                    [focal_length_x, 0, width/2],
                    [0, focal_length_y, height/2],
                    [0, 0, 1]
                ])
                cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1))
                valid_idx = np.where(np.logical_and.reduce((
                    cam_coord[2]>0, 
                    cam_coord[0]/cam_coord[2]>=0, 
                    cam_coord[0]/cam_coord[2]<=width-1, 
                    cam_coord[1]/cam_coord[2]>=0, 
                    cam_coord[1]/cam_coord[2]<=height-1
                )))[0]
                pts_depths = cam_coord[-1:, valid_idx]
                cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]

                y_indices = np.round(cam_coord[1]).astype(np.int32).clip(0, height-1)
                x_indices = np.round(cam_coord[0]).astype(np.int32).clip(0, width-1)
                
                depthmap[y_indices, x_indices] = pts_depths

                # Check if scale/shift cache contains this image
                if scale_shift_cache is not None and image_name in scale_shift_cache:
                    cache_entry = scale_shift_cache[image_name]
                    scale_val = cache_entry["scale"]
                    shift_val = cache_entry["shift"]
                    depthloss = cache_entry["loss"]
                    # Reconstruct depthmap from monodepth using cached scale and shift
                    refined_source = scale_val * monodepth + shift_val
                    refined_source[np.isinf(refined_source)] = 0.0
                    refined_source[np.isnan(refined_source)] = 0.0
                    depthmap = refined_source
                else:
                    refined_source, depthloss, scale_val, shift_val = optimize_depth(source=monodepth, target=depthmap, mask=depthmap>0.0)
                    depthmap = refined_source
                    if scale_shift_cache is not None:
                        scale_shift_cache[image_name] = {"scale": scale_val, "shift": shift_val, "loss": depthloss}
                        cache_dirty = True

            if depthmap is not None:
                if False:
                    confidence_map = calculate_confidence(depthmap, image)

                    # Save the confidence map as a human-readable color image in the visualizations folder
                    visual_dir_base = model_path if model_path is not None else dataset_path
                    if visual_dir_base is not None:
                        visualizations_dir = os.path.join(visual_dir_base, "visualizations")
                        os.makedirs(visualizations_dir, exist_ok=True)
                        # Convert 0-1 float to 0-255 uint8 BGR using JET colormap
                        conf_u8 = (confidence_map * 255.0).astype(np.uint8)
                        conf_color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_JET)
                        conf_save_path = os.path.join(visualizations_dir, image_name + "_confidence.png")
                        cv2.imwrite(conf_save_path, conf_color)

                    confidence_map = confidence_map.astype(np.float16)
                # Cast to float16 to optimize CPU memory footprint
                depthmap = depthmap.astype(np.float16)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              semantic_mask=semantic_mask, depthmap=depthmap, normalmap=mononormal, 
                              confidence_map=confidence_map, depthloss=depthloss)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')

    # Save cache if dirty
    if cache_dirty and scale_shift_cache_path is not None:
        try:
            with open(scale_shift_cache_path, 'w') as f:
                json.dump(scale_shift_cache, f, indent=4)
            print(f"[INFO] Saved optimized depth scale/shift parameters to {scale_shift_cache_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save cache file {scale_shift_cache_path}: {e}")

    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, no_depth_prior=False, model_path=None, disable_confidence=True, dense_init=False):
    if os.path.exists(os.path.join(path, "sparse/0", "images.bin")) or os.path.exists(os.path.join(path, "sparse/0", "images.txt")):
        colmap_path = os.path.join(path, "sparse/0")
    else:
        colmap_path = os.path.join(path, "sparse")

    ply_path = os.path.join(colmap_path, "points3D.ply")
    bin_path = os.path.join(colmap_path, "points3D.bin")
    txt_path = os.path.join(colmap_path, "points3D.txt")
    if not os.path.exists(ply_path) and not dense_init:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    elif dense_init:
        ply_path = os.path.join(path, "fused.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        # pcd = None
        try:
            pcd = trimesh.load(ply_path)
            point_id = np.random.choice(np.arange(len(pcd.vertices)), 1200000)
            pcd = BasicPointCloud(points=pcd.vertices[point_id], colors=pcd.colors[point_id][:,:3].astype(np.float32)/255, normals=None)
        except:
            pcd = None

    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    instance_dir = "instance"
    instance_folder = os.path.join(path, instance_dir)
    if not os.path.exists(instance_folder):
        fallback_folder = os.path.join(path, "..", instance_dir)
        if os.path.exists(fallback_folder):
            instance_folder = fallback_folder
            
    depth_folder = None
    normal_folder = None
    if not no_depth_prior:
        depth_dir = "depth"
        depth_folder = os.path.join(path, depth_dir)
        if not os.path.exists(depth_folder):
            fallback_folder = os.path.join(path, "..", depth_dir)
            if os.path.exists(fallback_folder):
                depth_folder = fallback_folder
        normal_dir = "normals"
        normal_folder = os.path.join(path, normal_dir)
        if not os.path.exists(normal_folder):
            fallback_folder = os.path.join(path, "..", normal_dir)
            if os.path.exists(fallback_folder):
                normal_folder = fallback_folder
            
    print(f"[INFO] Looking for semantic masks in: {os.path.abspath(instance_folder)}")
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir),
                                           instance_folder=instance_folder,
                                           depth_folder=depth_folder,
                                           normal_folder=normal_folder,
                                           pcd=pcd,
                                           dataset_path=path,
                                           model_path=model_path,
                                           disable_confidence=disable_confidence)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    print(f'cameras extent: {nerf_normalization["radius"]}')

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}