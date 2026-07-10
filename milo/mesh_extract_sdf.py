import yaml
import torch
import torch.nn as nn
from functools import partial
from scene import Scene
import os
from os import makedirs
import random
import copy
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, GraphCutParams, get_combined_args
from gaussian_renderer import GaussianModel, render_simp
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from utils.camera_utils import get_cameras_spatial_extent

from scene.mesh import MeshRasterizer, MeshRenderer, ScalableMeshRenderer, Meshes

from regularization.sdf.integration import evaluate_cull_sdf_values as compute_sdf_with_integration
from regularization.sdf.depth_fusion import evaluate_sdf_values as compute_sdf_with_depth_fusion
from regularization.sdf.depth_fusion import evaluate_mesh_occupancy, evaluate_mesh_colors_all_vertices
from regularization.sdf.learnable import compute_initial_sdf_with_binary_search

import gc
from utils.geometry_utils import depth_to_normal as depth_double_to_normal
from torchvision.utils import save_image
from regularization.sdf.learnable import convert_occupancy_to_sdf, convert_sdf_to_occupancy
from utils.geometry_utils import (
    flatten_voronoi_features, 
    unflatten_voronoi_features, 
    is_in_view_frustum, 
    identify_out_of_field_points,
)

from scene.gaussian_model import SparseGaussianAdam
from graphcut_segmentation import graphcut_segmentation, load_gc_args

from tetranerf.utils.extension import cpp
import time

import matplotlib.pyplot as plt
from random import randint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')


def extract_mesh_with_sdf_refinement(
    dataset, 
    iteration, 
    pipe, 
    graphcutparams,
    n_delaunay_sites, 
    mtet_on_cpu,
    refine_iter,
    refine_lr,
    initialization_method,
    args,
    mesh_config,
):
    torch.cuda.set_device("cuda:0")
    device = torch.device(torch.cuda.current_device())
    
    # Load Gaussian model
    print("[DEBUG] Initializing Gaussian Model...")
    gaussians = GaussianModel(sh_degree=dataset.sh_degree, num_classes=dataset.num_classes)
    print("[DEBUG] Initializing Scene (Camera Loading)...")
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    print("[DEBUG] Loading PLY (Gaussian Loading)...")
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
    print("[DEBUG] Setting occupancy mode...")
    gaussians.set_occupancy_mode(mesh_config["occupancy_mode"])

    # if args.target_class != -1:
    #     print(f"[INFO] Hard-pruning Gaussians for target class ID: {args.target_class}")
    #     with torch.no_grad():
    #         # # Create a boolean mask of points we want to KEEP
    #         # keep_mask = (gaussians.get_semantic.argmax(dim=-1) == args.target_class)
            
    #         # # Overwrite all tensors with only the kept points
    #         # gaussians._xyz = nn.Parameter(gaussians._xyz[keep_mask.to(gaussians._xyz.device)].requires_grad_(True))
    #         # gaussians._features_dc = nn.Parameter(gaussians._features_dc[keep_mask.to(gaussians._features_dc.device)].requires_grad_(True))
    #         # gaussians._features_rest = nn.Parameter(gaussians._features_rest[keep_mask.to(gaussians._features_rest.device)].requires_grad_(True))
    #         # gaussians._opacity = nn.Parameter(gaussians._opacity[keep_mask.to(gaussians._opacity.device)].requires_grad_(True))
    #         # gaussians._scaling = nn.Parameter(gaussians._scaling[keep_mask.to(gaussians._scaling.device)].requires_grad_(True))
    #         # gaussians._rotation = nn.Parameter(gaussians._rotation[keep_mask.to(gaussians._rotation.device)].requires_grad_(True))
    #         # gaussians._semantic = nn.Parameter(gaussians._semantic[keep_mask.to(gaussians._semantic.device)].requires_grad_(True))
            
    #         # if gaussians.learn_occupancy:
    #         #     gaussians._base_occupancy = nn.Parameter(gaussians._base_occupancy[keep_mask.to(gaussians._base_occupancy.device)].requires_grad_(False))
    #         #     gaussians._occupancy_shift = nn.Parameter(gaussians._occupancy_shift[keep_mask.to(gaussians._occupancy_shift.device)].requires_grad_(True))
            
    #         # if hasattr(gaussians, 'max_radii2D') and gaussians.max_radii2D.shape[0] > 0:
    #         #      gaussians.max_radii2D = gaussians.max_radii2D[keep_mask.to(gaussians.max_radii2D.device)]
            
    #         # if hasattr(gaussians, 'filter_3D'):
    #         #      gaussians.filter_3D = gaussians.filter_3D[keep_mask.to(gaussians.filter_3D.device)]

    #         # if hasattr(gaussians, '_culling'):
    #         #      gaussians._culling = gaussians._culling[keep_mask.to(gaussians._culling.device)]
                 
    #         # print(f"Kept {keep_mask.sum().item()} Gaussians.")

    #         weights = torch.zeros_like(gaussians._opacity)
    #         weights_cnt = torch.zeros_like(gaussians._opacity)
    #         gaussians_for_sink = copy.deepcopy(gaussians)
    #         gaussians_for_source = copy.deepcopy(gaussians)
    #         weights_sink = torch.zeros_like(gaussians._opacity)
    #         weights_cnt_sink = torch.zeros_like(gaussians._opacity)
    #         view_points = scene.getTrainCameras().copy()
            
    #         # Ensure debug directory exists
    #         debug_dir = os.path.join(dataset.model_path, "graphcut")
    #         os.makedirs(debug_dir, exist_ok=True)

    #         for index, _ in enumerate(tqdm(view_points,  desc="Coarse rasterization progress")):
    #             # Create a single-channel mask (1.0 for target class, 0.0 otherwise)
    #             mask = (view_points[index].semantic_mask == args.target_class).float().cuda()
    #             if mask.dim() == 2:
    #                 mask = mask.unsqueeze(0) # (1, H, W)
                
    #             # Use the camera object as the first argument
    #             gaussians_for_source.apply_weights(view_points[index], weights, weights_cnt, mask)
    #             gaussians_for_sink.apply_weights(view_points[index], weights_sink, weights_cnt_sink, 1.0 - mask)
            
    #         foreground_threshold = 0.9
    #         weights = torch.where(weights_cnt == 0, torch.zeros_like(weights),
    #                           weights / weights_cnt)
    #         selected_mask = weights >= foreground_threshold
            
    #         print("Number of gaussians removed: ", torch.sum(selected_mask.int() == 0).item())
    #         print("Number of gaussians kept: ", torch.sum(selected_mask.int() == 1).item())
    #         print("Number of gaussians before: ", len(gaussians._opacity))

    #         gaussians_for_source.remove_low_score_gaussians(selected_mask.bool().squeeze(1))
    #         gaussians_for_source.save_ply(os.path.join(debug_dir, "gaussians_source.ply"))

    #         weights_sink = torch.where(weights_cnt_sink == 0, torch.ones_like(weights_sink), weights_sink / weights_cnt_sink)
    #         selected_mask_sink = weights < foreground_threshold
    #         print("Number of gaussians removed sink: ", torch.sum(selected_mask_sink.int() == 0).item())
    #         print("Number of gaussians kept sink: ", torch.sum(selected_mask_sink.int() == 1).item())
    #         gaussians_for_sink.remove_low_score_gaussians(selected_mask_sink.bool().squeeze(1))
    #         # gaussians_for_sink.save_ply(os.path.join(debug_dir, "gaussians_sink.ply"))

    #         foreground, background_index = graphcut_segmentation(args, dataset.model_path, graphcutparams, weights, weights_sink, gaussians, gaussians_for_source, gaussians_for_sink)
    #         gaussians.remove_low_score_gaussians((background_index == 0))
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    # Get camera parameters
    train_cameras = scene.getTrainCameras()
    gaussians.spatial_lr_scale = get_cameras_spatial_extent(train_cameras)["radius"]
    
    # Compute Delaunay sites
    n_max_points_in_delaunay = n_delaunay_sites if n_delaunay_sites > 0 else int(mesh_config["n_max_points_in_delaunay"])
    print(f"[INFO] Maximum number of Gaussian pivots: {n_max_points_in_delaunay}")

    with torch.no_grad():
        n_gaussians_to_sample_from = gaussians._xyz.shape[0]
        n_max_gaussians_for_delaunay = int(n_max_points_in_delaunay / 9.)
        downsample_gaussians_for_delaunay = n_max_gaussians_for_delaunay < n_gaussians_to_sample_from

        if downsample_gaussians_for_delaunay:            
            print(f"[INFO] Downsampling Delaunay Gaussians from {n_gaussians_to_sample_from} to {n_max_gaussians_for_delaunay}.")
            n_nonzero = (gaussians._base_occupancy != 0.).any(dim=-1).sum().item()
            if n_nonzero == 0:
                if args.imp_metric == "none":
                    raise ValueError(f"imp_metric should not be 'none' in this context.")
                print(f"[INFO] Computing Delaunay Gaussians with importance sampling.")
                delaunay_xyz_idx = gaussians.sample_surface_gaussians(
                    scene=scene,
                    render_simp=render_simp,
                    iteration=iteration,
                    args=args,
                    pipe=pipe,
                    background=background,
                    n_samples=n_max_gaussians_for_delaunay,
                )
            else:
                print(f"[INFO] Using training Delaunay Gaussians for downsampling.")
                delaunay_xyz_idx = (gaussians._base_occupancy != 0.).any(dim=-1).nonzero().squeeze()
            print(f"[INFO] Downsampled Delaunay Gaussians from {n_gaussians_to_sample_from} to {len(delaunay_xyz_idx)}.")
            
        else:
            delaunay_xyz_idx = None
            print(f"[INFO] No need to downsample the {n_gaussians_to_sample_from} Delaunay Gaussians.")

        voronoi_points, voronoi_scales = gaussians.get_tetra_points(
            downsample_ratio=None,
            let_gradients_flow=False,
            xyz_idx=delaunay_xyz_idx, # Pass the computed indices
            verbose=True
        )
                
    # Compute Delaunay triangulation
    start_time = time.time()
    delaunay_tets = cpp.triangulate(voronoi_points.detach()).cuda().long()
    end_time = time.time()
    print(f"Delaunay triangulation time: {end_time - start_time} seconds")
    
    # Get Mesh renderer
    mesh_rasterizer = MeshRasterizer(cameras=scene.getTrainCameras().copy())
    if mesh_config["use_scalable_renderer"]:
        print("[INFO] Using scalable mesh renderer.")
        mesh_renderer = ScalableMeshRenderer(mesh_rasterizer)
    else:
        mesh_renderer = MeshRenderer(mesh_rasterizer)
    
    # Define the optimizer
    l = [{'params': [gaussians._occupancy_shift], 'lr': refine_lr, "name": "occupancy_shift"}]
    refine_optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
    for param_group in refine_optimizer.param_groups:
        print(param_group["name"], param_group["lr"], param_group["params"][0].shape)
    
    # --------Initialization--------
    viewpoint_stack = None
    sdf_reset_linearization_enforce_std = True
    
    with torch.no_grad():
        if initialization_method == "learnable":
            print(f"[INFO] Initializing the SDF with SDF values learned during training.")
        elif initialization_method in ["integration", "depth_fusion"]:
            print(f"[INFO] Initializing the SDF using {initialization_method}.")
            if initialization_method == "integration":
                sdf_function = partial(
                    compute_sdf_with_integration,
                    views=scene.getTrainCameras().copy(), 
                    masks=None, 
                    gaussians=gaussians, 
                    pipeline=pipe, 
                    background=background, 
                    kernel_size=kernel_size, 
                    return_colors=False, 
                    isosurface_value=args.sdf_default_isosurface, 
                    transform_sdf_to_linear_space=args.transform_initial_sdf_to_linear_space, 
                    min_occupancy_value=args.min_occupancy_value,
                    integrate_func=integrate,
                )
            elif initialization_method == 'depth_fusion':                        
                sdf_function = partial(
                    compute_sdf_with_depth_fusion,
                    views=scene.getTrainCameras().copy(), 
                    masks=None, 
                    gaussians=gaussians, 
                    pipeline=pipe, 
                    background=background, 
                    kernel_size=kernel_size, 
                    return_colors=False,
                    trunc_margin=None, 
                    render_func=render,
                )
            # Compute and linearize initial occupancy values with binary search if needed
            base_occupancy = compute_initial_sdf_with_binary_search(
                voronoi_points=voronoi_points,
                voronoi_scales=voronoi_scales,
                delaunay_tets=delaunay_tets,
                sdf_function=sdf_function,
                n_binary_steps=args.n_binary_steps_to_reset_sdf,
                n_linearization_steps=args.sdf_reset_linearization_n_steps,
                enforce_std=sdf_reset_linearization_enforce_std if args.n_binary_steps_to_reset_sdf > 0 else None,
            )  # Between -1 and 1
            base_occupancy = convert_sdf_to_occupancy(base_occupancy)  # Between 0.005 and 0.995
            
            # Reshape base occupancy to make it (N_sampled_gaussians, 9)
            base_occupancy = unflatten_voronoi_features(
                base_occupancy, 
                n_voronoi_per_gaussians=9
            )  # (N_sampled_gaussians, 9)
            
            # Reset occupancy
            gaussians.reset_occupancy(
                base_occupancy=base_occupancy, 
                occupancy=base_occupancy,
                gaussian_idx=delaunay_xyz_idx, 
            )
            print(f"[INFO] Occupancy initialized.")
        else:
            raise ValueError(f"Invalid initialization method: {initialization_method}")
    
    # --------Refine the occupancy / SDF values--------

    ema_mesh_depth_loss_for_log = 0.
    ema_mesh_normal_loss_for_log = 0.
    ema_occupied_centers_loss_for_log = 0.
    ema_occupancy_labels_loss_for_log = 0.
    
    progress_bar = tqdm(range(refine_iter + 1), desc="Training progress")
    
    occupancy_labels, vert_colors = None, None

    for iteration in range(refine_iter + 1):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(18000, 3000, scale=1.0, scale2=2.0).copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))

        _random_view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(_random_view_idx)
        
        # Render Gaussians
        with torch.no_grad():
            render_pkg = render(
                viewpoint_camera=viewpoint_cam,
                pc=gaussians,
                pipe=pipe,
                bg_color=background,
                kernel_size=kernel_size,
                scaling_modifier=1,
                require_coord=False,
                require_depth=True
            )
            
            rgb = render_pkg["render"].detach()
            median_depth = render_pkg["median_depth"].detach()
            normal = render_pkg["normal"].detach()
            median_depth_to_normal = depth_double_to_normal(viewpoint_cam, median_depth)
            radii = render_pkg["radii"].detach()

            # Save rendered images periodically
            save_every = 100 
            if iteration % save_every == 0 or iteration == 1:
                render_save_dir = os.path.join(dataset.model_path, "refinement_renders")
                os.makedirs(render_save_dir, exist_ok=True)
                save_image(rgb, os.path.join(render_save_dir, f"rgb_{iteration:04d}_{viewpoint_idx:03d}.png"))
                
                # Normalize depth for visualization
                d_min, d_max = median_depth.min(), median_depth.max()
                depth_norm = (median_depth - d_min) / (d_max - d_min + 1e-5)
                save_image(depth_norm, os.path.join(render_save_dir, f"depth_{iteration:04d}_{viewpoint_idx:03d}.png"))
        
        # --- Extract the Mesh ---
        # Compute the SDF
        if delaunay_xyz_idx is not None:
            current_occupancy = gaussians.get_occupancy[delaunay_xyz_idx]  # (N_sampled_gaussians, 9)
        else:
            current_occupancy = gaussians.get_occupancy  # (N_gaussians, 9)
        current_voronoi_sdf = convert_occupancy_to_sdf(
                flatten_voronoi_features(current_occupancy)
            )  # (N_voronoi_points, )
        
        # Differentiable Marching Tetrahedra
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            vertices=voronoi_points[None],
            tets=delaunay_tets,
            sdf=current_voronoi_sdf.reshape(1, -1), # Use the computed SDF for this iteration
            scales=voronoi_scales[None]
        )
        end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
        end_scales = scale_list[0]  # (N_verts, 2, 1)
        
        norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
        verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
        faces = faces_list[0]  # (N_faces, 3)
        
        # Frustum filtering
        faces_mask = is_in_view_frustum(verts, viewpoint_cam)[faces].any(axis=1)
        
        # Filtering out large edges as in GOF
        if mesh_config["filter_large_edges"] or mesh_config["collapse_large_edges"]:
            dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
            dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
            dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
            
        if mesh_config["filter_large_edges"]:
            dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
            faces_mask = faces_mask & dmtet_face_mask
            
        if mesh_config["collapse_large_edges"]:
            min_end_points = end_points[
                np.arange(end_points.shape[0]), 
                end_sdf.argmin(dim=1).flatten().cpu().numpy()
            ]  # TODO: Do the computation only for filtered vertices
            verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)

        # Build the Mesh object
        mesh = Meshes(verts=verts, faces=faces[faces_mask])

        # --- Render the Mesh ---
        mesh_render_pkg = mesh_renderer(
            mesh,
            cam_idx=viewpoint_idx,
            return_depth=mesh_config["use_depth_loss"],
            return_normals=mesh_config["use_normal_loss"],
            use_antialiasing=True,
        )
        mesh_depth = (
            mesh_render_pkg["depth"].squeeze() 
            if mesh_config["use_depth_loss"] 
            else torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width)
        )  # (H, W)
        mesh_normal_view = (
            mesh_render_pkg["normals"].squeeze() @ viewpoint_cam.world_view_transform[:3,:3] 
            if mesh_config["use_normal_loss"] 
            else torch.zeros(viewpoint_cam.image_height, viewpoint_cam.image_width, 3)
        )  # (H, W, 3)
        rasterization_mask = mesh_depth > 0.  # (H, W)
        
        # --- Reset occupancy labels ---
        if mesh_config["use_occupancy_labels_loss"] and (iteration % mesh_config["reset_occupancy_labels_every"] == 0):
            print(f"[INFO] Resetting occupancy labels at iteration {iteration}.")
            
            occupancy_labels, vert_colors = evaluate_mesh_occupancy(
                points=voronoi_points,
                views=train_cameras,
                mesh=Meshes(verts=verts, faces=faces),
                masks=None,
                return_colors=True,
                use_scalable_renderer=mesh_config["use_scalable_renderer"],
            )
            print(f"[INFO] Points with label > 0.5: {torch.sum(occupancy_labels > 0.5) / occupancy_labels.numel()}")
        
        # --- Compute Losses ---
        
        # Mesh Depth Loss
        if mesh_config["use_depth_loss"]:
            # Use Expected Depth (blended) more heavily if median depth is noisy
            # You can tweak this ratio in your config: 0.0 is only expected_depth, 1.0 is only median_depth
            depth_ratio = mesh_config.get("depth_ratio", 0.5)
            
            gaussians_depth = (
                (1. - depth_ratio) * render_pkg["expected_depth"] 
                + depth_ratio * render_pkg["median_depth"]
            ).squeeze()  # (H, W)
            
            if mesh_config["mesh_depth_loss_type"] == "log":
                mesh_depth_loss = torch.log(1. + (mesh_depth - gaussians_depth).abs() / gaussians.spatial_lr_scale)  # (H, W)

            elif mesh_config["mesh_depth_loss_type"] == "normal":
                mesh_depth_loss = depth_double_to_normal(
                    viewpoint_cam,
                    mesh_depth.squeeze()[None],
                    gaussians_depth.squeeze()[None],
                )  # (2, 3, H, W)
                mesh_depth_loss = 1. - (mesh_depth_loss[0] * mesh_depth_loss[1]).sum(dim=0)  # (H, W)

            else:
                raise ValueError(f"Invalid mesh depth loss type: {mesh_config['mesh_depth_loss_type']}")
            
            mesh_depth_loss = mesh_config["depth_weight"] * (mesh_depth_loss * rasterization_mask).mean()
        else:
            mesh_depth_loss = torch.zeros(size=(), device=gaussians._xyz.device)

        # Mesh Normal Loss
        if mesh_config["use_normal_loss"]:
            if mesh_config["use_depth_normal"]:
                # Compute normals from Gaussian depth map
                depth_middepth_normal = depth_double_to_normal(
                    viewpoint_cam,
                    render_pkg["expected_depth"],
                    render_pkg["median_depth"]
                )
                gaussians_normal_view = (
                    (1. - mesh_config["depth_ratio"]) * depth_middepth_normal[0]
                    + mesh_config["depth_ratio"] * depth_middepth_normal[1]
                ).permute(1, 2, 0) # (H, W, 3)
            else:
                # Use rendered normals directly (already in view space)
                gaussians_normal_view = render_pkg["normal"].permute(1, 2, 0)  # (H, W, 3)

            # Compute cosine similarity loss (1 - |dot_product|)
            # Ensure normals are normalized
            # mesh_normal_view = torch.nn.functional.normalize(mesh_normal_view, dim=-1)
            # gaussians_normal_view = torch.nn.functional.normalize(gaussians_normal_view, dim=-1)
            
            # TODO: Check if the following is OK or not.
            # We flip the mesh normals to make the loss invariant to the sign of the mesh normal.
            # Indeed, we just want to make sure the planes of both the mesh and the gaussians are aligned,
            # so we don't really care about the direction of the normal.
            #
            # This might be needed in scenarios where the Delaunay triangulation is not updated for a while,
            # so that the mesh could self-intersect and have flipped normals.
            #
            # For computing the loss, we just need to use .abs() on the dot product.
            # We also explicitly flip the mesh normals for logging purposes.

            normal_dot_product = (mesh_normal_view * gaussians_normal_view).sum(dim=-1, keepdim=True)  # (H, W, 1)
            mesh_normal_loss = 1. - normal_dot_product.abs()  # (H, W, 1)
            mesh_normal_loss = mesh_config["normal_weight"] * (mesh_normal_loss * rasterization_mask.unsqueeze(-1)).mean()
            # mesh_normal = torch.sign(normal_dot_product) * mesh_normal  # (H, W, 3)
        else:
            mesh_normal_loss = torch.zeros(size=(), device=gaussians._xyz.device)
            
        # Enforce sampled Gaussian centers to be inside the mesh
        if mesh_config["enforce_occupied_centers"]:
            # Get sdf values for centers of sampled Gaussians
            gaussians_occupancy = current_occupancy[:, -1]
            occupied_centers_loss = mesh_config["occupied_centers_weight"] * (mesh_config["sdf_default_isosurface"] - gaussians_occupancy).clamp(min=0.).mean()
        else:
            occupied_centers_loss = torch.zeros(size=(), device=gaussians._xyz.device)
        
        # Occupancy labels loss
        if mesh_config["use_occupancy_labels_loss"]:
            occupancy_labels_loss = mesh_config["occupancy_labels_loss_weight"] * (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    flatten_voronoi_features(
                        gaussians.get_occupancy_logit if delaunay_xyz_idx is None
                        else gaussians.get_occupancy_logit[delaunay_xyz_idx]
                    ),
                    occupancy_labels
                )
            ) * (occupancy_labels > 0.5).float()
            occupancy_labels_loss = occupancy_labels_loss.mean()
        else:
            occupancy_labels_loss = torch.zeros(size=(), device=gaussians._xyz.device)

        # --- Return Results ---
        total_mesh_loss = (
            mesh_depth_loss 
            + mesh_normal_loss 
            + occupied_centers_loss 
            + occupancy_labels_loss
        )
        
        total_mesh_loss.backward()
        
        with torch.no_grad():
            ema_mesh_depth_loss_for_log = 0.4 * mesh_depth_loss.item() + 0.6 * ema_mesh_depth_loss_for_log
            ema_mesh_normal_loss_for_log = 0.4 * mesh_normal_loss.item() + 0.6 * ema_mesh_normal_loss_for_log
            if mesh_config["enforce_occupied_centers"]:
                ema_occupied_centers_loss_for_log = 0.4 * occupied_centers_loss.item() + 0.6 * ema_occupied_centers_loss_for_log
            if mesh_config["use_occupancy_labels_loss"]:
                ema_occupancy_labels_loss_for_log = 0.4 * occupancy_labels_loss.item() + 0.6 * ema_occupancy_labels_loss_for_log
        
            if iteration % 5 == 0:
                postfix_dict = {}
                if mesh_config["use_depth_loss"]:
                    postfix_dict["MDLoss"] = f"{ema_mesh_depth_loss_for_log:.{7}f}"
                if mesh_config["use_normal_loss"]:
                    postfix_dict["MNLoss"] = f"{ema_mesh_normal_loss_for_log:.{7}f}"
                if mesh_config["enforce_occupied_centers"]:
                    postfix_dict["OccLoss"] = f"{ema_occupied_centers_loss_for_log:.{7}f}"
                if mesh_config["use_occupancy_labels_loss"]:
                    postfix_dict["OccLabLoss"] = f"{ema_occupancy_labels_loss_for_log:.{7}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(5)
        
        # Optimizer step
        if iteration < refine_iter:
            visible = radii>0
            refine_optimizer.step(visible, radii.shape[0])
            # gaussians.optimizer.step()
            refine_optimizer.zero_grad(set_to_none = True)
            
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    # --- Build the final mesh ---
    with torch.no_grad():
        # Compute the SDF
        if delaunay_xyz_idx is not None:
            current_occupancy = gaussians.get_occupancy[delaunay_xyz_idx]  # (N_sampled_gaussians, 9)
        else:
            current_occupancy = gaussians.get_occupancy  # (N_gaussians, 9)
        current_voronoi_sdf = convert_occupancy_to_sdf(
                flatten_voronoi_features(current_occupancy)
            )  # (N_voronoi_points, )
        
        # Differentiable Marching Tetrahedra
        verts_list, scale_list, faces_list, _ = marching_tetrahedra(
            vertices=voronoi_points[None],
            tets=delaunay_tets,
            sdf=current_voronoi_sdf.reshape(1, -1), # Use the computed SDF for this iteration
            scales=voronoi_scales[None]
        )
        end_points, end_sdf = verts_list[0]  # (N_verts, 2, 3) and (N_verts, 2, 1)
        end_scales = scale_list[0]  # (N_verts, 2, 1)
        
        norm_sdf = end_sdf.abs() / end_sdf.abs().sum(dim=1, keepdim=True)
        verts = end_points[:, 0, :] * norm_sdf[:, 1, :] + end_points[:, 1, :] * norm_sdf[:, 0, :]
        faces = faces_list[0]  # (N_faces, 3)
        
        # Filtering out large edges as in GOF
        dmtet_vertex_mask = None
        dmtet_face_mask = None

        if mesh_config["filter_large_edges"] or mesh_config["collapse_large_edges"]:
            dmtet_distance = torch.norm(end_points[:, 0, :] - end_points[:, 1, :], dim=-1)
            dmtet_scale = end_scales[:, 0, 0] + end_scales[:, 1, 0]
            dmtet_vertex_mask = (dmtet_distance <= dmtet_scale)
            
        if mesh_config["filter_large_edges"]:
            dmtet_face_mask = dmtet_vertex_mask[faces].all(axis=1)
            faces_mask = dmtet_face_mask
            
        if mesh_config["collapse_large_edges"]:
            min_end_points = end_points[
                np.arange(end_points.shape[0]), 
                end_sdf.argmin(dim=1).flatten().cpu().numpy()
            ]  # TODO: Do the computation only for filtered vertices
            verts = torch.where(dmtet_vertex_mask[:, None], verts, min_end_points)
            
            # Verts should not be removed from the mesh, so we set the mask back to None.
            if not mesh_config["filter_large_edges"]:
                dmtet_vertex_mask = None
            
        # Remove out of field vertices
        if args.remove_oof_vertices:
            out_of_field_mask = identify_out_of_field_points(
                points=verts,
                views=train_cameras,
            )
            
            dmtet_vertex_mask = (
                ~out_of_field_mask if dmtet_vertex_mask is None
                else dmtet_vertex_mask & ~out_of_field_mask
            )
            
            dmtet_face_mask = (
                ~out_of_field_mask[faces].any(axis=1) if dmtet_face_mask is None 
                else dmtet_face_mask & ~out_of_field_mask[faces].any(axis=1)
            )
            
        # Compute vertex colors
        print(f"[INFO] Computing vertex colors.")
        vert_colors = evaluate_mesh_colors_all_vertices(
            views=train_cameras, 
            mesh=Meshes(verts=verts, faces=faces if dmtet_face_mask is None else faces[dmtet_face_mask]),
            masks=None,
            use_scalable_renderer=mesh_config["use_scalable_renderer"],
        )

       # Save mesh as ply file
        mesh = trimesh.Trimesh(
            vertices=verts.cpu().numpy(), 
            faces=faces.cpu().numpy(), 
            vertex_colors=vert_colors.squeeze().cpu().numpy(), 
            process=False
        )
            
        # filter
        if dmtet_vertex_mask is not None:
            mesh.update_vertices(dmtet_vertex_mask.cpu().numpy())
        if dmtet_face_mask is not None:
            mesh.update_faces(dmtet_face_mask.cpu().numpy())

        save_name = f"mesh_learnable_sdf_class_{args.target_class}.ply" if args.target_class != -1 else "mesh_learnable_sdf.ply"
        mesh.export(os.path.join(dataset.model_path, save_name))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    graphcutparams = GraphCutParams(parser)
    parser.add_argument("--iteration", default=18000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_delaunay_sites", default=-1, type=int)
    parser.add_argument("--mtet_on_cpu", action="store_true")
    # Rasterization
    parser.add_argument("--rasterizer", default="radegs", type=str, choices=["radegs", "gof"])
    # For delaunay downsampling
    parser.add_argument("--warn_until_iter", default=3000, type=int)
    parser.add_argument("--imp_metric", default='none', type=str)
    # Config file
    parser.add_argument("--config", default='default', type=str)
    # For refinement
    parser.add_argument("--refine_iter", default=1000, type=int)
    parser.add_argument("--refine_lr", default=0.05/2, type=float)
    parser.add_argument("--reset_occupancy_labels_every", default=100, type=int)
    parser.add_argument("--remove_oof_vertices", action="store_true")
    # For initialization
    parser.add_argument("--init", default="learnable", type=str)
    parser.add_argument("--sdf_default_isosurface", default=0.5, type=float)
    parser.add_argument("--transform_initial_sdf_to_linear_space", action="store_true")
    parser.add_argument("--min_occupancy_value", default=1e-10, type=float)
    parser.add_argument("--n_binary_steps_to_reset_sdf", default=8, type=int)
    parser.add_argument("--sdf_reset_linearization_n_steps", default=20, type=int)
    parser.add_argument("--target_class", default=-1, type=int, help="Extract mesh only for this class ID. -1 for all.")
    load_gc_args(parser)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # Get mesh regularization config file
    print(f"[INFO] Loading mesh regularization config from {args.config}")
    mesh_config_file = os.path.join(BASE_DIR, "configs", "mesh", f"{args.config}.yaml")
    with open(mesh_config_file, "r") as f:
        mesh_config = yaml.safe_load(f)
    mesh_config["reset_occupancy_labels_every"] = args.reset_occupancy_labels_every
    
    # Rasterization
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
    elif args.rasterizer == "gof":
        from gaussian_renderer.gof import render_gof as render
        from gaussian_renderer.gof import integrate_gof as integrate
    else:
        raise ValueError(f"Invalid rasterizer: {args.rasterizer}")
    
    if args.n_delaunay_sites > 0:
        if args.imp_metric == 'none':
            raise ValueError("imp_metric must be specified for using delaunay downsampling: Either 'indoor' or 'outdoor'")
    
    extract_mesh_with_sdf_refinement(
        model.extract(args), 
        args.iteration, 
        pipeline.extract(args), 
        graphcutparams,
        n_delaunay_sites=args.n_delaunay_sites, 
        mtet_on_cpu=args.mtet_on_cpu,
        refine_iter=args.refine_iter,
        refine_lr=args.refine_lr,
        initialization_method=args.init,
        args=args,
        mesh_config=mesh_config,
    )
    