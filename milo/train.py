import os
import sys
import torch
import torch.utils.cpp_extension
_original_load = torch.utils.cpp_extension.load
def _patched_load(*args, **kwargs):
    module = _original_load(*args, **kwargs)
    if module is not None:
        name = kwargs.get('name') or (args[0] if len(args) > 0 else None)
        if name:
            sys.modules[name] = module
    return module
torch.utils.cpp_extension.load = _patched_load

import gc
import yaml
from functools import partial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
SUBMODULES_DIR = os.path.join(ROOT_DIR, 'submodules')
sys.path.append(ROOT_DIR)
sys.path.append(SUBMODULES_DIR)
sys.path.append(os.path.join(SUBMODULES_DIR, 'Depth-Anything-V2'))

import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, L1_loss_appearance
from fused_ssim import fused_ssim

from gaussian_renderer import network_gui
from gaussian_renderer import render_imp, render_simp, render_depth, render_full
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, read_config
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

import numpy as np
import time

from utils.geometry_utils import depth_to_normal, det3x3, inverse3x3, normalize_depth, cos_weight, monosdf_normal_loss
from utils.log_utils import log_training_progress
from regularization.regularizer.depth_order import (
    initialize_depth_order_supervision,
    compute_depth_order_regularization,
)
from regularization.regularizer.mesh import (
    initialize_mesh_regularization,
    compute_mesh_regularization,
    reset_mesh_state_at_next_iteration,
)
from regularization.regularizer.normal_field import (
    initialize_normal_field,
)

def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, 
    checkpoint_iterations, checkpoint, 
    debug_from, args, 
    depth_order_config, mesh_config,
    log_interval,
):
    # ---Prepare logger--- 
    run = prepare_output_and_logger(dataset, args)
    
    # ---Initialize scene and Gaussians---
    first_iter = 0
    use_mip_filter = not args.disable_mip_filter

    n_gaussian_features = 0
    if args.use_normal_field:
        n_gaussian_features = 4

    gaussians = GaussianModel(
        sh_degree=0, 
        num_classes=args.num_classes,
        use_mip_filter=use_mip_filter, 
        learn_occupancy=args.mesh_regularization,
        use_appearance_network=args.decoupled_appearance,
        n_gaussian_features=n_gaussian_features,
    )
    if getattr(dataset, "no_depth_prior", False) and args.depth_order:
        raise ValueError("Cannot use --depth_order when --no_depth_prior is specified. Please disable depth order prior regularization if you wish to disable loading depth/normal maps.")
    scene = Scene(dataset, gaussians, resolution_scales=[1,2])
    gaussians.training_setup(opt)
    print(f"[INFO] Using 3D Mip Filter: {gaussians.use_mip_filter}")
    print(f"[INFO] Using learnable SDF: {gaussians.learn_occupancy}")
    if args.dense_gaussians:
        print("[INFO] Using dense Gaussians.")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        if args.mesh_regularization:
            if first_iter > mesh_config["start_iter"]:
                mesh_config["start_iter"] = first_iter + 1
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize culling stats
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    gaussians.init_culling(len(scene.getTrainCameras()))
    
    # Initialize 3D Mip filter
    if use_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.getTrainCameras_warn_up(first_iter + 1, args.warn_until_iter, scale=1.0, scale2=2.0).copy())

    # Additional variables
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    postfix_dict = {}
    ema_loss_for_log = 0.0
    ema_semantic_loss_for_log = 0.0
    ema_depth_normal_loss_for_log = 0.0
    
    # ---Prepare Mesh-In-the-Loop Regularization---
    if args.mesh_regularization:
        print("[INFO] Using mesh regularization.")
        mesh_renderer, mesh_state = initialize_mesh_regularization(
            scene=scene,
            config=mesh_config,
        )
    
    if args.use_normal_field:
        print("[INFO] Using Normal Field.")
        normal_field_state = initialize_normal_field(
            scene=scene,
        )

    ema_mesh_depth_loss_for_log = 0.0
    ema_mesh_normal_loss_for_log = 0.0
    ema_occupied_centers_loss_for_log = 0.0
    ema_occupancy_labels_loss_for_log = 0.0
    
    # ---Prepare Depth-Order Regularization---    
    # if args.depth_order:
    #     print("[INFO] Using depth order regularization.")
    #     print(f"        > Using expected depth with depth_ratio {depth_order_config['depth_ratio']} for depth order regularization.")
    #    depth_priors = initialize_depth_order_supervision(
    #         scene=scene,
    #         config=depth_order_config,
    #         device='cuda',
    #     )
    ema_depth_order_loss_for_log = 0.0
        
    # ---Log optimizable param groups---
    print(f"[INFO] Found {len(gaussians.optimizer.param_groups)} optimizable param groups:")
    n_total_params = 0
    for param in gaussians.optimizer.param_groups:
        name = param['name']
        n_params = len(param['params'])
        print(f"\n========== {name} ==========")
        print(f"Total number of param groups: {n_params}")
        for param_i in param['params']:
            print(f"   > Shape {param_i.shape}")
            n_total_params = n_total_params + param_i.numel()
    if gaussians.learn_occupancy:
        print(f"\n========== base_occupancy ==========")
        print(f"   > Not learnable")
        print(f"   > Shape {gaussians._base_occupancy.shape}")
    print(f"\nTotal number of optimizable parameters: {n_total_params}\n")
    
    # ---Start optimization loop---    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # ---Update SH degree---
        if iteration % 1000 == 0 and iteration>args.simp_iteration1:
            gaussians.oneupSHdegree()

        # ---Select random viewpoint---
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy()
            viewpoint_idx_stack = list(range(len(viewpoint_stack)))

        _random_view_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        viewpoint_cam = viewpoint_stack.pop(_random_view_idx)

        # ---Render scene---
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        reg_kick_on = iteration >= args.regularization_from_iter
        mesh_kick_on = args.mesh_regularization and (iteration >= mesh_config["start_iter"]) and (
            iteration == mesh_config["start_iter"] or iteration % mesh_config.get("mesh_update_interval", 1) == 0
        )
        depth_order_kick_on = args.depth_order
        normal_field_kick_on = args.use_normal_field and (iteration >= 8000)
        
        if normal_field_kick_on:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background,
                require_coord=False, require_depth=True,
                flag_max_count=False,
                render_normal_field=True
            )

        # If depth-normal regularization or mesh-in-the-loop regularization are active,
        # we use the rasterizer compatible with depth and normal rendering.
        elif reg_kick_on or mesh_kick_on:
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, background,
                require_coord=False, require_depth=True,
                flag_max_count=False,
            )
            
        # Else, if depth-order regularization is active, we use Mini-Splatting2 rasterizer 
        # but we render depth maps. This rasterizer is necessary for densification and simplification.
        elif depth_order_kick_on:
            render_pkg = render_full(
                viewpoint_cam, gaussians, pipe, background, 
                culling=gaussians._culling[:,viewpoint_cam.uid],
                compute_expected_normals=False,
                compute_expected_depth=True,
                compute_accurate_median_depth_gradient=True,
            )
            
        # If no regularization is active, we just use the default Mini-Splatting2 rasterizer.
        else:
            render_pkg = render_imp(
                viewpoint_cam, gaussians, pipe, background, 
                culling=gaussians._culling[:,viewpoint_cam.uid],
            )

        # ---Compute losses---
        semantic_loss = None
        monosdf_loss = None
        scale_factor = None
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        gt_image = viewpoint_cam.original_image.cuda()

        # Rendering loss
        if args.decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # # Semantic loss
        # if "semantic" in render_pkg and gt_semantic_mask is not None and gaussians.num_classes > 0:
        #     semantic_loss = torch.nn.functional.cross_entropy(render_pkg["semantic"].unsqueeze(0), gt_semantic_mask.unsqueeze(0))
        #     loss = loss + opt.lambda_semantic * semantic_loss

        #     visible = radii > 0

        #     # penalize gaussians with low confident semantic
        #     probs_3d = gaussians.get_semantic[visible]
        #     entropy_3d = -torch.sum(probs_3d * torch.log(probs_3d + 1e-8), dim=-1)
        #     pointwise_entropy_loss = entropy_3d.mean()
        #     loss = loss + 0.01 * pointwise_entropy_loss
            
        #     # Opacity entropy: penalizes alpha values around 0.5, rewards 0.0 and 1.0
        #     # alpha = gaussians.get_opacity[visible]
        #     # opacity_entropy_loss = - (alpha * torch.log(alpha + 1e-8) + (1 - alpha) * torch.log(1 - alpha + 1e-8)).mean()
        #     # loss = loss + 0.01 * opacity_entropy_loss

        #     if iteration % 300 == 0:
        #         xyz = gaussians.get_xyz
        #         scales = gaussians.get_scaling
        #         rots = gaussians.get_rotation
        #         opacities = gaussians.get_opacity
        #         semantics = gaussians.get_semantic
        #         with torch.no_grad():
        #             edge_index = radius_graph(xyz, r=0.05, max_num_neighbors=16, loop=False)
        #             idx_neighbor = edge_index[0] # Source nodes
        #             idx_center = edge_index[1] # Target nodes

        #             # Build Covariance Matrices
        #             R = build_rotation(rots)                
        #             S = torch.diag_embed(scales)            
        #             L = R @ S
        #             Sigma = L @ L.transpose(-1, -2)        

        #             # Gather data ONLY for the valid edges [E, ...]
        #             center_xyz = xyz[idx_center]
        #             neighbor_xyz = xyz[idx_neighbor]
                    
        #             center_Sigma = Sigma[idx_center]
        #             neighbor_Sigma = Sigma[idx_neighbor]
                    
        #             center_opacity = opacities[idx_center].squeeze(-1)
        #             neighbor_opacity = opacities[idx_neighbor].squeeze(-1)

        #             # Compute Summed Covariance for each edge [E, 3, 3]
        #             Sigma_sum = center_Sigma + neighbor_Sigma               

        #             # Invert and get determinant using optimized 3x3 functions
        #             Sigma_sum_inv = inverse3x3(Sigma_sum)             
        #             Sigma_sum_det = det3x3(Sigma_sum)             

        #             # Intersection Math
        #             diff = (center_xyz - neighbor_xyz).unsqueeze(-1) # [E, 3, 1]
                    
        #             mahalanobis_dist = diff.transpose(-2, -1) @ Sigma_sum_inv @ diff # [E, 1, 1]
        #             mahalanobis_dist = mahalanobis_dist.squeeze(-1).squeeze(-1) # [E]

        #             volume_factor = 1.0 / torch.sqrt(torch.clamp(Sigma_sum_det, min=1e-8))

        #             # Calculate absolute overlap weight per edge [E]
        #             overlap_weights = center_opacity * neighbor_opacity * volume_factor * torch.exp(-0.5 * mahalanobis_dist)
                
        #         center_semantics = semantics[idx_center]     # [E, num_classes]
        #         neighbor_semantics = semantics[idx_neighbor] # [E, num_classes]
        #         raw_mse = torch.nn.functional.mse_loss(center_semantics, neighbor_semantics, reduction='none')
        #         raw_mse = raw_mse.mean(dim=-1)
        #         semantic_knn_loss = (overlap_weights * raw_mse).sum() / xyz.shape[0]
        #         loss = loss + 0.01 * semantic_knn_loss

        #         if iteration % 300 == 0:
        #             with torch.no_grad():
        #                 num_pts = xyz.shape[0]
        #                 num_edges = edge_index.shape[1]
        #                 neighbor_counts = torch.bincount(edge_index[1], minlength=num_pts)
        #                 num_zero_neighbors = (neighbor_counts == 0).sum().item()
        #                 avg_neighbors = num_edges / num_pts
        #                 print(f"\n[Iteration {iteration}] Semantic KNN Stats: Avg Neighbors: {avg_neighbors:.2f}, Points with 0 Neighbors: {num_zero_neighbors}/{num_pts} ({num_zero_neighbors/num_pts*100:.1f}%)")
        
        # Depth-Normal Consistency & MonoSDF Normal Regularization
        depth_normal_loss = None
        monosdf_loss = None
        rendered_depth_to_normals = None
        
        monosdf_active = depth_order_kick_on and iteration <= 8000
        
        if reg_kick_on or monosdf_active:
            rendered_depth_to_normals = depth_to_normal(
                viewpoint_cam, 
                render_pkg["median_depth"],  # 1, H, W
                render_pkg["expected_depth"],  # 1, H, W
            )  # 3, H, W or 2, 3, H, W
            
        if reg_kick_on:
            rendered_normals: torch.Tensor = render_pkg["normal"]  # 3, H, W
            
            if rendered_depth_to_normals.ndim == 4:
                # If shape is 2, 3, H, W
                reg_depth_ratio = 0.6
                normal_error_map = 1. - (rendered_normals[None] * rendered_depth_to_normals).sum(dim=1)  # 2, H, W
                depth_normal_loss = args.lambda_depth_normal * (
                    (1. - reg_depth_ratio) * normal_error_map[0].mean() 
                    + reg_depth_ratio * normal_error_map[1].mean()
                )
            else:
                # If shape is 3, H, W
                depth_normal_loss = args.lambda_depth_normal * (1 - (rendered_normals * rendered_depth_to_normals).sum(dim=0)).mean()
            
            loss = loss + depth_normal_loss

        if monosdf_active:
            prior_normal = viewpoint_cam.normalmap.cuda()
            if rendered_depth_to_normals.ndim == 4:
                # Weight median and expected depth normals using reg_depth_ratio (or 0.6) to match the self-consistency combination
                reg_depth_ratio = 0.6
                rendered_depth_to_normals_cl = (
                    (1. - reg_depth_ratio) * rendered_depth_to_normals[0]
                    + reg_depth_ratio * rendered_depth_to_normals[1]
                ).permute(1, 2, 0)
            else:
                rendered_depth_to_normals_cl = rendered_depth_to_normals.permute(1, 2, 0)
                
            prior_normal_cl = prior_normal.permute(1, 2, 0)
            normal_confidence = cos_weight(rendered_depth_to_normals_cl, prior_normal_cl)
            monosdf_loss = monosdf_normal_loss(rendered_depth_to_normals_cl, prior_normal_cl, normal_confidence)
            loss = loss + 0.05 * monosdf_loss
            
        # Depth Order Regularization
        # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
        # > In the paper, MILo does not rely on any learned prior. 
        depthloss_align = viewpoint_cam.depthloss
        do_supervision_depth = None
        depth_prior_loss = None
        if depth_order_kick_on and iteration <= 8000:
            if depth_order_config["depth_ratio"] < 1.:
                depth_for_depth_order = (
                    (1. - depth_order_config["depth_ratio"]) * render_pkg["expected_depth"]
                    + depth_order_config["depth_ratio"] * render_pkg["median_depth"]
                )
            else:
                depth_for_depth_order = render_pkg["median_depth"]
                
            depth_mask = (viewpoint_cam.depthmap>0).cuda()
            gt_maskeddepth = (viewpoint_cam.depthmap.cuda() * depth_mask)

            gt_maskeddepth = normalize_depth(gt_maskeddepth)
            depth = normalize_depth(depth_for_depth_order)

            depthloss_align_tensor = torch.tensor(depthloss_align).cuda()
            scale_factor = 0.6 * torch.exp(-1 * depthloss_align_tensor)
            pixelwise_l1_loss = torch.abs(gt_maskeddepth - depth * depth_mask)
            # if viewpoint_cam.confidence_map is not None:
            #     confidence_map = viewpoint_cam.confidence_map.cuda().float()
            # else:
            #     confidence_map = torch.ones_like(gt_maskeddepth)
            weighted_loss = pixelwise_l1_loss #* confidence_map
            depth_prior_loss = weighted_loss.mean() * scale_factor
            loss = loss + depth_prior_loss

        # Mesh-In-the-Loop Regularization
        if mesh_kick_on:
            if args.detach_gaussian_rendering:
                detached_render_pkg = {
                    "render": render_pkg["render"].detach(),
                    "median_depth": render_pkg["median_depth"].detach(),
                    "expected_depth": render_pkg["expected_depth"].detach(),
                    "normal": render_pkg["normal"].detach(),
                }
            
            mesh_regularization_pkg = compute_mesh_regularization(
                iteration=iteration,
                render_pkg=detached_render_pkg if args.detach_gaussian_rendering else render_pkg,
                viewpoint_cam=viewpoint_cam,
                viewpoint_idx=viewpoint_idx,
                gaussians=gaussians,
                scene=scene,
                pipe=pipe,
                background=background,
                kernel_size=0.0,
                config=mesh_config,
                mesh_renderer=mesh_renderer,
                mesh_state=mesh_state,
                render_func=partial(render, require_coord=False, require_depth=True),
                weight_adjustment=100. / opt.iterations,
                args=args,
                integrate_func=integrate,
            )
            mesh_loss = mesh_regularization_pkg["mesh_loss"]
            mesh_depth_loss = mesh_regularization_pkg["mesh_depth_loss"]
            mesh_normal_loss = mesh_regularization_pkg["mesh_normal_loss"]
            occupied_centers_loss = mesh_regularization_pkg["occupied_centers_loss"]
            occupancy_labels_loss = mesh_regularization_pkg["occupancy_labels_loss"]
            mesh_state = mesh_regularization_pkg["updated_state"]
            mesh_render_pkg = mesh_regularization_pkg["mesh_render_pkg"]
            
            loss = loss + mesh_loss
            # torch.cuda.empty_cache()
        
        # ---Backward pass---
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if semantic_loss is not None:
                ema_semantic_loss_for_log = 0.4 * semantic_loss.item() + 0.6 * ema_semantic_loss_for_log
            # ---Logging---
            (
                postfix_dict, ema_loss_for_log, ema_semantic_loss_for_log, ema_depth_normal_loss_for_log, 
                ema_mesh_depth_loss_for_log, ema_mesh_normal_loss_for_log, 
                ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log, 
                ema_depth_order_loss_for_log
            ) = log_training_progress(
                args, iteration, log_interval, progress_bar, run,
                scene, gaussians, pipe, opt, background,
                viewpoint_idx, viewpoint_cam, render_pkg, 
                mesh_render_pkg if mesh_kick_on else None, 
                do_supervision_depth if (depth_order_kick_on and iteration <= 3000) else None,
                reg_kick_on, mesh_kick_on, depth_order_kick_on and iteration <= 3000,
                loss, depth_normal_loss if reg_kick_on else None, 
                mesh_depth_loss if mesh_kick_on else None, mesh_normal_loss if mesh_kick_on else None, 
                occupied_centers_loss if mesh_kick_on else None, occupancy_labels_loss if mesh_kick_on else None, 
                depth_prior_loss if (depth_order_kick_on and iteration <= 8000) else None,
                mesh_config if mesh_kick_on else None, 
                postfix_dict, ema_loss_for_log, ema_semantic_loss_for_log, ema_depth_normal_loss_for_log, 
                ema_mesh_depth_loss_for_log, ema_mesh_normal_loss_for_log, ema_occupied_centers_loss_for_log, ema_occupancy_labels_loss_for_log,
                ema_depth_order_loss_for_log, testing_iterations, saving_iterations, render_imp, semantic_loss,
                monosdf_loss=monosdf_loss if (depth_order_kick_on and iteration <= 8000) else None,
                scale_factor=scale_factor if (depth_order_kick_on and iteration <= 3000) else None,
            )

            # ---Densification---
            gaussians_have_changed = False
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                if gaussians._culling[:,viewpoint_cam.uid].sum()==0:
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    # normalize xy gradient after culling
                    gaussians.add_densification_stats_culling(viewspace_point_tensor, visibility_filter, gaussians.factor_culling)

                area_max = render_pkg["area_max"]
                mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != args.depth_reinit_iter:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune_mask(opt.densify_grad_threshold, 
                                                    0.005, scene.cameras_extent, 
                                                    size_threshold, mask_blur)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    gaussians_have_changed = True
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras_warn_up(
                                iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                            ).copy()
                        )
                    
                if iteration == args.depth_reinit_iter:

                    num_depth = gaussians._xyz.shape[0]*args.num_depth_factor

                    # interesction_preserving for better point cloud reconstruction result at the early stage, not affect rendering quality
                    gaussians.interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras_warn_up(
                                iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                            ).copy()
                        )
                        
                    pts, rgb = gaussians.depth_reinit(scene, render_depth, iteration, num_depth, args, pipe, background)

                    gaussians.reinitial_pts(pts, rgb)

                    gaussians.training_setup(opt)
                    gaussians.init_culling(len(scene.getTrainCameras()))
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    torch.cuda.empty_cache()
                    gaussians_have_changed = True
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras_warn_up(
                                iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                            ).copy()
                        )

                if iteration >= args.aggressive_clone_from_iter and iteration % args.aggressive_clone_interval == 0 and iteration!=args.depth_reinit_iter:
                    gaussians.culling_with_clone(scene, render_simp, iteration, args, pipe, background)
                    torch.cuda.empty_cache()
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    gaussians_have_changed = True
                    if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras_warn_up(
                                iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                            ).copy()
                        )

            # ---Pruning and simplification---
            if iteration == args.simp_iteration1:
                if args.dense_gaussians:
                    gaussians.culling_with_importance_pruning(scene, render_simp, iteration, args, pipe, background)
                else:
                    gaussians.culling_with_interesction_sampling(scene, render_simp, iteration, args, pipe, background)
                gaussians.max_sh_degree=dataset.sh_degree
                gaussians.extend_features_rest()

                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                gaussians_have_changed = True
                if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras_warn_up(
                                iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                            ).copy()
                        )
                
            if iteration == args.simp_iteration2:
                if args.dense_gaussians:
                    gaussians.culling_with_importance_pruning(scene, render_simp, iteration, args, pipe, background)
                else:
                    gaussians.culling_with_interesction_preserving(scene, render_simp, iteration, args, pipe, background)
                torch.cuda.empty_cache()
                gaussians_have_changed = True
                if use_mip_filter:
                        gaussians.compute_3D_filter(
                            cameras=scene.getTrainCameras_warn_up(
                                iteration, args.warn_until_iter, scale=1.0, scale2=2.0
                            ).copy()
                        )

            if iteration == (args.simp_iteration2+opt.iterations)//2:
                gaussians.init_culling(len(scene.getTrainCameras()))

            # ---Reset mesh state if Gaussians have changed---
            if args.mesh_regularization and gaussians_have_changed:
                mesh_state = reset_mesh_state_at_next_iteration(mesh_state)
                
            # ---Update 3D Mip Filter---
            if use_mip_filter and (
                (iteration == args.warn_until_iter)
                or (iteration % args.update_mip_filter_every == 0)
            ):
                if iteration < opt.iterations - args.update_mip_filter_every:
                    gaussians.compute_3D_filter(cameras=scene.getTrainCameras_warn_up(iteration, args.warn_until_iter, scale=1.0, scale2=2.0).copy())
                else:
                    print(f"[INFO] Skipping 3D Mip Filter update at iteration {iteration}")

            # ---Optimizer step---
            if iteration < opt.iterations:
                if gaussians.use_appearance_network:
                    gaussians.optimizer.step()
                else:
                    visible = radii>0
                    gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)

            # ---Save checkpoint---
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  
                
        if iteration % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print('Num of Gaussians: %d'%(gaussians._xyz.shape[0]))
    
    if WANDB_FOUND:
        run.finish()
    
    return 


def prepare_output_and_logger(dataset, args):    
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create WandB run       
    global WANDB_FOUND
    WANDB_FOUND = (
        WANDB_FOUND
        and (args.wandb_project is not None)
        and (args.wandb_entity is not None)
    )
    if WANDB_FOUND:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
        )
    else:
        run=None
        print("[INFO] WandB not found, skipping logging.")
    return run


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ----- Usual arguments -----
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[8000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # ----- Rasterization technique -----
    parser.add_argument("--rasterizer", type=str, default="radegs", choices=["radegs", "gof"])
    
    # ----- Mesh-In-the-Loop Regularization -----
    parser.add_argument("--no_mesh_regularization", action="store_true")
    parser.add_argument("--mesh_config", type=str, default="default")
    # Gaussians management
    parser.add_argument("--dense_gaussians", action="store_true")
    parser.add_argument("--detach_gaussian_rendering", action="store_true")

    # ----- Densification and Simplification -----
    # > Inspired by Mini-Splatting2.
    # > Used for pruning, densification and Gaussian pivots selection.
    parser.add_argument("--imp_metric", required=True, type=str, choices=["outdoor", "indoor"])
    parser.add_argument("--config_path", type=str, default="./configs/fast")
    # Aggressive Cloning
    parser.add_argument("--aggressive_clone_from_iter", type=int, default = 500)
    parser.add_argument("--aggressive_clone_interval", type=int, default = 250)
    # Depth Reinitialization
    parser.add_argument("--warn_until_iter", type=int, default = 3_000)
    parser.add_argument("--depth_reinit_iter", type=int, default=2_000)
    parser.add_argument("--num_depth_factor", type=float, default=1)
    # Simplification
    parser.add_argument("--simp_iteration1", type=int, default = 3_000)
    parser.add_argument("--simp_iteration2", type=int, default = 8_000)
    parser.add_argument("--sampling_factor", type=float, default = 0.6)
    
    # ----- Depth-Normal consistency Regularization -----
    # > Inspired by 2DGS, GOF, RaDe-GS...
    parser.add_argument("--regularization_from_iter", type=int, default = 3_000)
    parser.add_argument("--lambda_depth_normal", type=float, default = 0.05)
    
    # ----- Depth Order Regularization (Learned Prior) -----
    # > This loss relies on Depth-AnythingV2, and is not used in MILo paper.
    # > In the paper, MILo does not rely on any learned prior.
    parser.add_argument("--depth_order", action="store_true")
    parser.add_argument("--depth_order_config", type=str, default="default")

    # ----- 3D Mip Filter -----
    # > Inspired by Mip-Splatting.
    parser.add_argument("--disable_mip_filter", action="store_true", default=False)
    parser.add_argument("--update_mip_filter_every", type=int, default=100)

    # ----- Appearance Network for Exposure-aware loss -----
    # > Inspired by GOF.
    parser.add_argument("--decoupled_appearance", action="store_true")

    # ----- Logging -----
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])

    args = read_config(parser)
    args.save_iterations.append(args.iterations)
    if not -1 in args.test_iterations:
        args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    args.mesh_regularization = not args.no_mesh_regularization
    
    if args.port == -1:
        args.port = np.random.randint(5000, 9000)
        print(f"Using random port: {args.port}")
    
    # Load depth order regularization config (not used in MILo paper)
    if args.depth_order:
        # Get depth order config file
        depth_order_config_file = os.path.join(BASE_DIR, "configs", "depth_order", f"{args.depth_order_config}.yaml")
        with open(depth_order_config_file, "r") as f:
            depth_order_config = yaml.safe_load(f)
    else:
        depth_order_config = None
        
    # Load mesh-in-the-loop regularization config
    if args.mesh_regularization:
        # Get mesh regularization config file
        mesh_config_file = os.path.join(BASE_DIR, "configs", "mesh", f"{args.mesh_config}.yaml")
        with open(mesh_config_file, "r") as f:
            mesh_config = yaml.safe_load(f)
        print(f"[INFO] Using mesh regularization with config: {args.mesh_config}")
    else:
        mesh_config = None
    
    # Message for imp_metric
    print(f"[INFO] Using importance metric: {args.imp_metric}.")
    
    # Message for detach_gaussian_rendering
    if args.detach_gaussian_rendering:
        print(f"[INFO] Detaching Gaussian rendering for mesh regularization.")
    
    # Import rendering function
    print(f"[INFO] Using {args.rasterizer} as rasterizer.")
    if args.rasterizer == "radegs":
        from gaussian_renderer.radegs import render_radegs as render
        from gaussian_renderer.radegs import integrate_radegs as integrate
    elif args.rasterizer == "gof":
        from gaussian_renderer.gof import render_gof as render
        from gaussian_renderer.gof import integrate_gof as integrate
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    torch.cuda.synchronize()
    time_start=time.time()
    
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args,
        depth_order_config,
        mesh_config,
        args.log_interval,
    )

    torch.cuda.synchronize()
    time_end=time.time()
    time_total=time_end-time_start
    print('time: %fs'%(time_total))

    time_txt_path=os.path.join(args.model_path, r'time.txt')
    with open(time_txt_path, 'w') as f:  
        f.write(str(time_total)) 

    # All done
    print("\nTraining complete.")
